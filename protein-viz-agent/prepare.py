# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-12
# Summary   : This is the step between downloading a structure and doing
#             anything quantitative with it, and it is where most simulations
#             are quietly ruined. A deposited file is a record of an
#             experiment, not a model of a molecule: an NMR entry holds twenty
#             equally valid states, an X-ray entry holds side chains modelled
#             in two positions at once, and both are full of glycerol and
#             sulfate from the crystallography. Hand any of that to tleap or
#             Rosetta and it will either refuse the file or worse build
#             a system with two copies of a side chain fused through each
#             other.
#
#             Everything structural is done here by rewriting PDB records
#             column by column, with no ParmEd, MDAnalysis or Biopython. A
#             preparation step that stops working when a dependency three
#             levels down breaks is not a preparation step. `reduce` is used
#             for adding hydrogens because that is genuinely hard to do well,
#             and it is a self-contained C++ binary that the same breakage
#             does not touch. This module will not do chemistry. It does not
#             pick histidine protonation states, model missing loops, or
#             decide whether a bound lipid is part of the biology. It reports
#             each of those as something the person has to decide, which is
#             the honest division of labour: the file surgery is mechanical,
#             the chemistry is not.
# =============================================================================

import subprocess
import tempfile
from pathlib import Path

import interactions as ixn
import measure as mz
import structure_report as srep

SCHEMA_VERSION = 1

# Records that describe atoms. Everything else in a PDB file is metadata.
COORD_RECORDS = ("ATOM  ", "HETATM")

# Records dropped from the output. ANISOU and the SIG* records describe atoms
# that may no longer exist; CONECT and MASTER index atom serial numbers, which
# are renumbered here; SSBOND and LINK name residues that may have been
# removed. Keeping any of them would leave a file whose metadata contradicts
# its coordinates, which is worse than one without the metadata.
DROPPED_RECORDS = ("ANISOU", "SIGATM", "SIGUIJ", "CONECT", "MASTER",
                   "SSBOND", "LINK  ", "CISPEP", "HELIX ", "SHEET ")

# Header records worth carrying through, so the prepared file still says what
# it came from.
# Header records worth carrying through. SEQRES is deliberately NOT among
# them: this module renames residues (CYX, MET) and can drop whole chains, so
# a SEQRES block copied across would disagree with the coordinates below it,
# and anything that compares the two -- including this project's own
# composition report -- would read the disagreement as missing residues that
# are in fact present under another name. What SEQRES was useful for, the
# record of which residues the experiment never saw, is written into the
# REMARK block instead, where it stays true.
KEPT_HEADERS = ("HEADER", "TITLE ", "COMPND", "SOURCE", "EXPDTA", "CRYST1",
                "REMARK   2")

WATER_NAMES = {"HOH", "WAT", "TIP", "TIP3", "H2O", "DOD", "SOL"}

# Force-field spellings of ordinary amino acids: a protonation state or a
# bonding state, not a different molecule. They matter here because the
# component classifier works from the PDB's chemical dictionary, where CYX
# does not appear at all -- so a structure this module has just prepared for
# Amber would be read back as containing a ligand called CYX that needs GAFF
# parameters, which is both wrong and alarming.
FF_VARIANTS = {
    "CYX", "CYM",                      # disulfide-bonded and deprotonated Cys
    "HID", "HIE", "HIP",               # Amber histidine protonation states
    "HSD", "HSE", "HSP",               # the CHARMM spellings of the same
    "ASH", "GLH", "LYN", "ARN", "TYM",  # neutral Asp/Glu/Lys, deprotonated Arg/Tyr
}

# The ordinary residue each force-field spelling stands for.
FF_PARENT = {
    "CYX": "CYS", "CYM": "CYS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG", "TYM": "TYR",
}

# Selenomethionine, and how to turn it back into methionine. Selenium is used
# to phase the crystal structure, not because the protein has any; leaving it
# in means either finding parameters for it or having tleap refuse the residue.
MSE_ATOM_FIX = {"SE": (" SD ", " S")}

# Amber's names for the two halves of a disulfide. tleap will not form the
# bond from a pair of residues called CYS -- it builds two free thiols and
# leaves the sulfurs 2 Å apart, which is an immediate explosion in the first
# minimisation step.
CYX = "CYX"


# ═══════════════════════════════════════════════════════════════════════════════
# Record helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_coord(line: str) -> bool:
    return line.startswith(COORD_RECORDS)


def _resname(line: str) -> str:
    return line[17:20].strip().upper()


def _reskey(line: str) -> tuple:
    """(chain, residue number, insertion code) — the identity of a residue."""
    return (line[21], line[22:26].strip(), line[26].strip())


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def _element(line: str) -> str:
    """The element, from columns 77-78 or inferred from the atom name."""
    el = line[76:78].strip().upper()
    if el:
        return el
    name = line[12:16]
    # A leading digit or a name that starts in column 13 belongs to hydrogen or
    # a two-letter element; the PDB convention puts one-letter elements in
    # column 14, which is what makes this inferable at all.
    stripped = name.strip()
    if not stripped:
        return ""
    if stripped[0].isdigit():
        return stripped[1:2].upper()
    if name[0] != " ":
        return stripped[:2].upper()
    return stripped[0].upper()


def _is_hydrogen(line: str) -> bool:
    return _element(line) in ("H", "D")


def _occupancy(line: str) -> float:
    try:
        return float(line[54:60])
    except ValueError:
        return 1.0


def read_lines(pdb_path) -> list:
    """Every line of a PDB file, endings kept."""
    return Path(pdb_path).read_text(errors="replace").splitlines(True)


def split_models(lines: list) -> list:
    """
    Split a file into models. Returns [(model number, [lines]), ...].

    A file with no MODEL records is one model numbered 1 -- the caller should
    not have to care which kind of file it has.
    """
    models, current, number = [], [], None
    for line in lines:
        if line.startswith("MODEL "):
            try:
                number = int(line[10:14])
            except ValueError:
                number = len(models) + 1
            current = []
            continue
        if line.startswith("ENDMDL"):
            models.append((number if number is not None else len(models) + 1, current))
            current, number = [], None
            continue
        if _is_coord(line) or line.startswith("TER"):
            current.append(line)
    if not models:
        return [(1, [l for l in lines if _is_coord(l) or l.startswith("TER")])]
    if current and any(_is_coord(l) for l in current):
        models.append((number if number is not None else len(models) + 1, current))
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# Inspection
# ═══════════════════════════════════════════════════════════════════════════════

def inspect(pdb_path) -> dict:
    """
    Everything about a structure that decides how it must be prepared.

    Deliberately reports rather than fixes. Which of twenty NMR states to keep,
    whether a bound lipid is biology or detergent, which histidine is
    protonated -- none of those have a right answer this module can work out,
    and a tool that silently picks one is a tool that produces confident
    nonsense.

    Returns a dict of findings plus `issues`, a list of
    {level, title, detail} ready to render.
    """
    path = Path(pdb_path)
    lines = read_lines(path)
    models = split_models(lines)

    coords = [l for l in models[0][1] if _is_coord(l)]
    altloc_ids, altloc_residues = set(), set()
    hydrogens = 0
    insertion_codes, partial_occupancy, missing_elements = set(), 0, 0
    waters, mse, his, cyx_residues = set(), set(), set(), set()
    residues = set()

    for line in coords:
        alt = line[16].strip()
        if alt:
            altloc_ids.add(alt)
            altloc_residues.add(_reskey(line))
        if _is_hydrogen(line):
            hydrogens += 1
        if line[26].strip():
            insertion_codes.add(_reskey(line))
        occ = _occupancy(line)
        if 0 < occ < 1.0:
            partial_occupancy += 1
        if not line[76:78].strip():
            missing_elements += 1
        name = _resname(line)
        key = _reskey(line)
        if name in WATER_NAMES:
            waters.add(key)
        else:
            residues.add(key)
        if name == "MSE":
            mse.add(key)
        elif name == CYX:
            cyx_residues.add(key)
        elif name in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
            his.add(key)

    summary = _composition(path, lines)

    disulfides = _disulfides(path, lines)

    out = {
        "path": str(path),
        "models": len(models),
        "model_ids": [m for m, _ in models],
        "representative_model": representative_model(lines),
        "altloc_ids": sorted(altloc_ids),
        "altloc_residues": len(altloc_residues),
        "hydrogens": hydrogens,
        "atoms": len(coords),
        "waters": len(waters),
        "residues": len(residues),
        "insertion_codes": len(insertion_codes),
        "partial_occupancy": partial_occupancy,
        "missing_elements": missing_elements,
        "mse": len(mse),
        "cyx_present": len(cyx_residues),
        "histidines": len(his),
        "disulfides": disulfides,
        "summary": summary,
        "hetero": [],
        "gaps": [],
        "missing_residues": 0,
        "chains": [],
    }

    if summary:
        out["hetero"] = [e for e in summary["nonstandard"]
                         if e["kind"] in ("ligand", "cofactor", "ion", "additive")
                         and e["code"] not in FF_VARIANTS]
        out["chains"] = summary["chains"]
        out["missing_residues"] = summary["totals"].get("missing", 0) or 0
        out["gaps"] = [(c["chain"], a, b) for c in summary["chains"]
                       for a, b, _ in c.get("gaps", [])]
    out["issues"] = _issues(out)
    return out


def representative_model(lines: list):
    """
    The model the depositors nominated as representative, if they nominated one.

    NMR entries often carry "BEST REPRESENTATIVE CONFORMER IN THIS ENSEMBLE" in
    their REMARK 210 block -- the state the people who solved the structure
    would pick. That is a better default than "the first one", which is what
    every tool silently uses, and it costs one line of parsing to honour.
    """
    for line in lines:
        if not line.startswith("REMARK 210"):
            continue
        if "BEST REPRESENTATIVE CONFORMER" in line.upper():
            tail = line.split(":")[-1].strip()
            if tail.isdigit():
                return int(tail)
    return None


def _composition(path, lines: list):
    """
    The composition report for a file, read through standard residue names.

    A structure this module has prepared for Amber contains CYX and may
    contain HID/HIE/HIP. The composition report defines a standard residue as
    one of the twenty amino acids and nothing else, which is the right
    definition for what that report is for -- but it means a renamed cysteine
    reads as a non-polymer component sitting in the middle of a chain, and the
    chain-break detector then reports a break on either side of every
    disulfide. So the names are mapped back to the residues they stand for,
    in a temporary copy, purely for this measurement. The prepared file keeps
    the force-field names it needs; only the chemistry is read through them.
    """
    tmp = _standard_names_copy(lines)
    try:
        return srep.summarize(tmp or path)
    except Exception:
        return None
    finally:
        _discard(tmp)


def _standard_names_copy(lines: list):
    """
    A temporary copy with force-field residue names mapped back to standard
    ones, or None when the file has none. The caller must _discard() it.
    """
    if not any(_is_coord(l) and _resname(l) in FF_VARIANTS for l in lines):
        return None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
            for line in lines:
                if _is_coord(line):
                    parent = FF_PARENT.get(_resname(line))
                    if parent:
                        line = line[:17] + f"{parent:>3}" + line[20:]
                fh.write(line)
            return fh.name
    except Exception:
        return None


def _discard(path) -> None:
    """Delete a temporary file, if there is one."""
    if path:
        try:
            Path(path).unlink()
        except OSError:
            pass


def _disulfides(pdb_path, lines=None) -> list:
    """
    Disulfide-bonded cysteine pairs, as [(a_key, b_key, distance)].

    Reuses the interaction detector rather than measuring sulfur distances
    again here: one definition of a disulfide in the codebase means the bond
    the viewer draws and the bond tleap is told about are the same bond. The
    detector looks for cysteines by name, so a file where they have already
    been renamed CYX is read through a standard-named copy -- otherwise
    preparing a structure for Amber would make its disulfides disappear from
    every report, including the one that says whether the rename worked.
    """
    tmp = _standard_names_copy(lines if lines is not None else read_lines(pdb_path))
    try:
        atoms = mz.read_atoms(tmp or pdb_path)
        found = ixn.find(atoms, types=["disulfide"])
    except Exception:
        return []
    finally:
        _discard(tmp)
    out = []
    for rec in found.get("interactions", []):
        out.append({
            "a": (rec["a_key"][0], str(rec["a_key"][1]), rec["a_key"][2] or ""),
            "b": (rec["b_key"][0], str(rec["b_key"][1]), rec["b_key"][2] or ""),
            "labels": (rec["a_label"], rec["b_label"]),
            "distance": rec["distance"],
        })
    return out


def _issues(f: dict) -> list:
    """
    Turn the findings into a checklist, worst first.

    Levels: "blocker" for things that will stop a downstream tool outright,
    "warn" for things that will produce a wrong answer quietly, "note" for
    things worth knowing.
    """
    issues = []
    if f["models"] > 1:
        rep = f.get("representative_model")
        detail = ("An NMR entry is an ensemble of equally valid states. "
                  "Simulation and docking tools want one structure: pick a "
                  "state, or they will either refuse the file or silently "
                  "use the first one.")
        if rep:
            detail += (f" The depositors nominated model {rep} as the best "
                       "representative conformer, which is the sensible default.")
        issues.append({"level": "blocker",
                       "title": f"{f['models']} models in the file",
                       "detail": detail})
    if f["altloc_ids"]:
        issues.append({
            "level": "blocker",
            "title": (f"{f['altloc_residues']} residues have alternate "
                      f"locations ({', '.join(f['altloc_ids'])})"),
            "detail": ("Side chains modelled in two positions at once. Left in, "
                       "they become two overlapping copies of the same atoms, "
                       "which is an immediate clash."),
        })
    additives = [e for e in f["hetero"] if e["kind"] == "additive"]
    if additives:
        issues.append({
            "level": "warn",
            "title": ("Crystallisation additives present: "
                      + ", ".join(e["code"] for e in additives)),
            "detail": ("Glycerol, sulfate, PEG and buffer components are there "
                       "because of how the crystal was grown, not because of the "
                       "biology. Keeping them means simulating the cryoprotectant."),
        })
    if f["missing_residues"]:
        issues.append({
            "level": "warn",
            "title": f"{f['missing_residues']} residues have no coordinates",
            "detail": ("Disordered loops and termini the experiment could not "
                       "resolve. Nothing here can model them back in — the chain "
                       "will have a physical break where they belong, and tleap "
                       "will happily build a bond straight across it unless the "
                       "gap is capped or the segments are separated by TER."),
        })
    elif f["gaps"]:
        issues.append({
            "level": "warn",
            "title": f"{len(f['gaps'])} chain break(s)",
            "detail": "Consecutive residues too far apart to be bonded.",
        })
    ligands = [e for e in f["hetero"] if e["kind"] in ("ligand", "cofactor")]
    if ligands:
        issues.append({
            "level": "note",
            "title": "Ligands and cofactors: " + ", ".join(e["code"] for e in ligands),
            "detail": ("No force field has parameters for these out of the box. "
                       "Keep them and you will need antechamber/GAFF for Amber or "
                       "a .params file for Rosetta; drop them and you are "
                       "simulating the apo protein."),
        })
    if f["mse"]:
        issues.append({
            "level": "warn",
            "title": f"{f['mse']} selenomethionine (MSE) residues",
            "detail": ("Selenium is there to phase the crystal structure, not "
                       "because the protein has any. Convert them to methionine "
                       "unless you have parameters for selenium."),
        })
    if f["disulfides"]:
        issues.append({
            "level": "note",
            "title": f"{len(f['disulfides'])} disulfide bond(s)",
            "detail": ("For Amber these cysteines must be renamed CYX, or tleap "
                       "builds two free thiols with their sulfurs 2 Å apart and "
                       "the first minimisation step blows up."),
        })
    if f["histidines"]:
        issues.append({
            "level": "note",
            "title": f"{f['histidines']} histidines",
            "detail": ("Protonation state is a chemical decision this tool will "
                       "not make for you. tleap defaults every HIS to HIE "
                       "(Nε-protonated); if any of yours is HID or HIP — a metal "
                       "ligand or a catalytic residue usually is — rename it "
                       "yourself."),
        })
    if f["hydrogens"]:
        issues.append({
            "level": "note",
            "title": f"{f['hydrogens']} hydrogens already present",
            "detail": ("Fine for Amber. Rosetta rebuilds them anyway, and a "
                       "mixture of deposited and added hydrogens is the usual "
                       "cause of duplicate-atom errors."),
        })
    if f["insertion_codes"]:
        issues.append({
            "level": "note",
            "title": f"{f['insertion_codes']} residues have insertion codes",
            "detail": ("Antibody and protease numbering. Harmless in most "
                       "pipelines, but some tools silently collapse them — "
                       "renumber sequentially if a downstream tool complains."),
        })
    order = {"blocker": 0, "warn": 1, "note": 2}
    return sorted(issues, key=lambda i: order[i["level"]])


# ═══════════════════════════════════════════════════════════════════════════════
# Preparation profiles
# ═══════════════════════════════════════════════════════════════════════════════

# Defaults per downstream tool. Every one of these is a decision someone would
# otherwise make by hand and get wrong once.
PROFILES = {
    "amber": {
        "label": "Amber (tleap / pmemd)",
        "keep_waters": False, "keep_ions": True, "keep_ligands": True,
        "keep_additives": False, "hydrogens": "strip", "mse_to_met": True,
        "cys_to_cyx": True, "renumber": True,
        "note": ("tleap adds hydrogens itself from its own libraries, so the "
                 "deposited ones are stripped rather than mixed with them. "
                 "Disulfide cysteines are renamed CYX, selenomethionine becomes "
                 "methionine, and crystallisation additives are dropped. Ligands "
                 "are kept — you will need antechamber/GAFF parameters for them. "
                 "Residues are renumbered sequentially from 1, which is what makes "
                 "the disulfide bond commands in the generated leap script land on "
                 "the right residues; the original numbering is kept in the "
                 "mapping table."),
    },
    "rosetta": {
        "label": "Rosetta",
        "keep_waters": False, "keep_ions": True, "keep_ligands": True,
        "keep_additives": False, "hydrogens": "strip", "mse_to_met": True,
        "cys_to_cyx": False, "renumber": False,
        "note": ("Rosetta rebuilds hydrogens from its own chemistry, so they are "
                 "stripped here. Cysteines keep their standard name — Rosetta "
                 "detects disulfides itself. Any ligand you keep needs a .params "
                 "file made with molfile_to_params.py."),
    },
    "md_explicit": {
        "label": "MD with crystallographic waters kept",
        "keep_waters": True, "keep_ions": True, "keep_ligands": True,
        "keep_additives": False, "hydrogens": "strip", "mse_to_met": True,
        "cys_to_cyx": True, "renumber": True,
        "note": ("The same as the Amber profile but the deposited waters stay. "
                 "Worth it when a water is part of the mechanism — in an active "
                 "site or a channel — and wasted effort otherwise, since the "
                 "solvation step adds its own."),
    },
    "clean": {
        "label": "Just clean it up (keep everything else)",
        "keep_waters": True, "keep_ions": True, "keep_ligands": True,
        "keep_additives": True, "hydrogens": "keep", "mse_to_met": False,
        "cys_to_cyx": False, "renumber": False,
        "note": ("One model, one conformation, nothing else touched. For looking "
                 "at and measuring, not for simulating."),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare(pdb_path, dest, model=None, altloc: str = "occupancy",
            chains=None, keep_waters: bool = False, keep_ions: bool = True,
            keep_ligands: bool = True, keep_additives: bool = False,
            hydrogens: str = "strip", mse_to_met: bool = False,
            cys_to_cyx: bool = False, renumber: bool = False,
            finding=None, disulfides=None, provenance: str = "") -> tuple:
    """
    Write a cleaned copy of a structure, and report everything that changed.

    Args:
        pdb_path      : The structure to clean.
        dest          : Where to write the prepared copy.
        model         : Which model to keep. None means the first one.
        altloc        : "occupancy" keeps the highest-occupancy conformation of
                        each atom, "first" keeps whichever comes first, or pass
                        a letter ("A") to keep that one. "keep" leaves them.
        chains        : Chain ids to keep, or None for all of them.
        keep_waters   : Keep deposited water molecules.
        keep_ions     : Keep ions.
        keep_ligands  : Keep ligands and cofactors.
        keep_additives: Keep crystallisation additives (glycerol, PEG, sulfate).
        hydrogens     : "strip" or "keep". Adding them is a separate step —
                        see add_hydrogens().
        mse_to_met    : Convert selenomethionine to methionine.
        cys_to_cyx    : Rename disulfide-bonded cysteines to CYX for Amber.
        renumber      : Renumber residues sequentially per chain from 1.
        finding       : The inspect() result for this file, if the caller
                        already has one. Used for the disulfide list and to
                        record the original's gaps in the prepared file.
        disulfides    : Disulfide list; taken from `finding` when omitted.
        provenance    : A line recorded in the file's REMARK block.

    Returns:
        (path, report). The report says what was removed and changed, so the
        panel can show it and the file itself carries it as REMARKs.
    """
    src = Path(pdb_path)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = read_lines(src)
    models = split_models(lines)

    report = {
        "source": str(src), "output": str(dest),
        "models_found": len(models), "model_kept": None, "models_dropped": 0,
        "altloc_atoms_dropped": 0, "altloc_policy": altloc,
        "waters_removed": 0, "hydrogens_removed": 0,
        "removed_components": {}, "chains_removed": [], "chains_kept": [],
        "mse_converted": 0, "cys_renamed": 0, "renumbered": bool(renumber),
        "renumber_map": [], "atoms_before": 0, "atoms_after": 0, "notes": [],
    }

    chosen = None
    if model is None:
        chosen = models[0]
    else:
        chosen = next((m for m in models if m[0] == int(model)), None)
        if chosen is None:
            return None, {"error": f"This file has no model {model}. "
                                   f"It has: {', '.join(str(m) for m, _ in models)}."}
    report["model_kept"] = chosen[0]
    report["models_dropped"] = len(models) - 1
    body = chosen[1]
    report["atoms_before"] = sum(1 for l in body if _is_coord(l))

    keep_chains = {c.strip() for c in (chains or []) if c.strip()} or None

    # ── Which alternate location wins ────────────────────────────────────────
    best_alt = {}
    if altloc != "keep":
        for line in body:
            if not _is_coord(line) or not line[16].strip():
                continue
            key = (_reskey(line), _atom_name(line))
            alt, occ = line[16], _occupancy(line)
            if key not in best_alt:
                best_alt[key] = (alt, occ)
            elif altloc == "occupancy" and occ > best_alt[key][1]:
                best_alt[key] = (alt, occ)
            elif altloc not in ("occupancy", "first") and alt == altloc:
                best_alt[key] = (alt, occ)

    if disulfides is None and finding is not None:
        disulfides = finding.get("disulfides")
    cyx_keys = set()
    if cys_to_cyx:
        pairs = disulfides if disulfides is not None else _disulfides(src)
        for pair in pairs:
            cyx_keys.add(tuple(pair["a"]))
            cyx_keys.add(tuple(pair["b"]))

    out, seen_chain_order = [], []
    serial = 0
    renumber_map, next_resnum = {}, {}
    converted_mse = set()

    for line in body:
        if not _is_coord(line):
            continue
        resname = _resname(line)
        key = _reskey(line)
        chain = line[21]

        # ── Filters ──────────────────────────────────────────────────────────
        if keep_chains is not None and chain.strip() and chain not in keep_chains:
            if chain not in report["chains_removed"]:
                report["chains_removed"].append(chain)
            continue
        if resname in WATER_NAMES:
            if not keep_waters:
                report["waters_removed"] += 1
                continue
        else:
            kind = srep.classify_component(resname) if _is_hetero_component(line, resname) else None
            if kind == "ion" and not keep_ions:
                _count_removed(report, resname)
                continue
            if kind in ("ligand", "cofactor") and not keep_ligands:
                _count_removed(report, resname)
                continue
            if kind == "additive" and not keep_additives:
                _count_removed(report, resname)
                continue
        if hydrogens == "strip" and _is_hydrogen(line):
            report["hydrogens_removed"] += 1
            continue
        if altloc != "keep":
            alt = line[16].strip()
            if alt:
                winner = best_alt.get((key, _atom_name(line)), (alt, 0))[0]
                if alt != winner.strip():
                    report["altloc_atoms_dropped"] += 1
                    continue
                line = line[:16] + " " + line[17:]      # blank the altLoc column

        # ── Rewrites ─────────────────────────────────────────────────────────
        if mse_to_met and resname == "MSE":
            name = _atom_name(line)
            if name == "SE":
                line = line[:12] + MSE_ATOM_FIX["SE"][0] + line[16:]
                line = line[:76] + f"{MSE_ATOM_FIX['SE'][1]:>2}" + line[78:]
            line = "ATOM  " + line[6:17] + "MET" + line[20:]
            converted_mse.add(key)
            resname = "MET"
        elif cys_to_cyx and resname == "CYS" and key in cyx_keys:
            line = line[:17] + CYX + line[20:]
            report["cys_renamed"] += 1

        if not line[76:78].strip():
            el = _element(line)
            if el:
                line = line[:76] + f"{el:>2}" + line[78:]

        if renumber:
            if key not in renumber_map:
                nxt = next_resnum.get(chain, 0) + 1
                next_resnum[chain] = nxt
                renumber_map[key] = nxt
            line = line[:22] + f"{renumber_map[key]:>4} " + line[27:]

        serial += 1
        line = line[:6] + f"{serial:>5}" + line[11:]
        if chain not in seen_chain_order:
            seen_chain_order.append(chain)
        out.append(line)

    report["atoms_after"] = len(out)
    report["chains_kept"] = list(seen_chain_order)
    # Renumbering is not reversible from the file alone, so the mapping is kept
    # with the report: anything measured on the prepared structure has to be
    # talked about in the numbering of the paper it came from.
    report["renumber_map"] = [(key[0], key[1], key[2], new)
                              for key, new in renumber_map.items()]
    # Both counted per residue, which is what people mean by "how many did you
    # change" -- the loop above sees one line per atom.
    report["mse_converted"] = len(converted_mse)
    report["cys_renamed"] = len(cyx_keys) if cys_to_cyx else 0

    text = _assemble(src, lines, out, report, provenance, finding)
    dest.write_text(text)
    return str(dest), report


def _is_polymer(line: str) -> bool:
    """
    True when a residue is part of a polymer chain rather than a free molecule.

    Standard amino acids and nucleotides, plus the force-field spellings of
    amino acids, are polymer; everything else -- ligands, ions, cofactors,
    additives -- is its own molecule and needs a TER on either side of it.
    """
    name = _resname(line)
    return (name in srep.STANDARD_AA or name in srep.STANDARD_NT
            or name in FF_VARIANTS)


def _is_hetero_component(line: str, resname: str) -> bool:
    """
    True when a residue is a chemical component rather than part of a polymer.

    HETATM alone is not the test: modified amino acids are HETATM records in
    the middle of a chain, and dropping one would cut the protein in half.
    """
    if not line.startswith("HETATM"):
        return False
    return resname not in srep.STANDARD_AA and resname not in srep.STANDARD_NT


def _count_removed(report: dict, code: str) -> None:
    report["removed_components"][code] = report["removed_components"].get(code, 0) + 1


def _assemble(src, original: list, atoms: list, report: dict,
              provenance: str, finding=None) -> str:
    """
    Put the output file together: kept headers, a provenance block, atoms, TERs.

    The provenance block is the point. A prepared structure that does not say
    what was done to it is a file nobody can trust six months later, and
    "which state of the NMR ensemble is this?" is not a question the
    coordinates can answer on their own.
    """
    head = [l for l in original
            if any(l.startswith(h) for h in KEPT_HEADERS)
            and not any(l.startswith(d) for d in DROPPED_RECORDS)]

    stamp = [
        "REMARK 999 STRUCTURE PREPARED BY PARORA (prepare.py)\n",
        f"REMARK 999 SOURCE FILE: {Path(src).name}\n",
    ]
    if provenance:
        for chunk in _wrap(provenance, 68):
            stamp.append(f"REMARK 999 {chunk}\n")
    for chunk in _wrap(report_text(report, oneline=True), 68):
        stamp.append(f"REMARK 999 {chunk}\n")
    if finding:
        # SEQRES is dropped, so what it was good for is preserved here: which
        # residues the experiment never resolved, and where the chain breaks.
        if finding.get("missing_residues"):
            stamp.append(f"REMARK 999 {finding['missing_residues']} RESIDUES OF THE "
                         "DEPOSITED CONSTRUCT HAVE NO COORDINATES\n")
        for chain, a, b in (finding.get("gaps") or [])[:20]:
            stamp.append(f"REMARK 999 CHAIN BREAK: {chain} BETWEEN RESIDUES "
                         f"{a} AND {b}\n")

    # TER between molecules, not only between chains. A ligand or an ion
    # sitting at the end of a chain with no TER in front of it is read by tleap
    # as the next residue of that chain: it looks for a bond into it, finds the
    # ion has no connection point, and stops with "One sided connection.
    # Residue (CA) missing connect0 atom" -- which says nothing about the
    # missing TER that actually caused it.
    body, serial = [], 0
    for i, line in enumerate(atoms):
        body.append(line)
        serial += 1
        last = i == len(atoms) - 1
        here_key, here_poly = _reskey(line), _is_polymer(line)
        if last:
            next_chain, next_key, next_poly = None, None, None
        else:
            nxt = atoms[i + 1]
            next_chain, next_key, next_poly = nxt[21], _reskey(nxt), _is_polymer(nxt)
        residue_ends = next_key != here_key
        boundary = (last or line[21] != next_chain
                    or (residue_ends and not (here_poly and next_poly)))
        if boundary and _resname(line) not in WATER_NAMES:
            serial += 1
            body.append(f"TER   {serial:>5}      {line[17:27]}\n")
    return "".join(head + stamp + body + ["END\n"])


def _wrap(text: str, width: int) -> list:
    """Break a line of text into chunks that fit a REMARK record."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def report_text(report: dict, oneline: bool = False) -> str:
    """What the preparation did, in words."""
    if report.get("error"):
        return report["error"]
    bits = []
    if report["models_found"] > 1:
        bits.append(f"kept model {report['model_kept']} of {report['models_found']}")
    if report["altloc_atoms_dropped"]:
        bits.append(f"dropped {report['altloc_atoms_dropped']} alternate-location "
                    f"atoms ({report['altloc_policy']})")
    if report["waters_removed"]:
        bits.append(f"removed {report['waters_removed']} water atoms")
    if report["removed_components"]:
        bits.append("removed " + ", ".join(
            f"{code} ×{n}" for code, n in sorted(report["removed_components"].items())))
    if report["hydrogens_removed"]:
        bits.append(f"stripped {report['hydrogens_removed']} hydrogens")
    if report.get("hydrogens_added"):
        bits.append(f"added {report['hydrogens_added']} hydrogens")
    if report["mse_converted"]:
        bits.append(f"converted {report['mse_converted']} MSE to MET")
    if report["cys_renamed"]:
        bits.append(f"renamed {report['cys_renamed']} disulfide cysteines to CYX")
    if report["chains_removed"]:
        bits.append("removed chains " + ", ".join(report["chains_removed"]))
    if report["renumbered"]:
        bits.append("renumbered residues from 1 per chain")
    bits.append(f"{report['atoms_before']} atoms in, {report['atoms_after']} out")
    text = "; ".join(bits) + "."
    return text if oneline else text.replace("; ", "\n  · ")


# ═══════════════════════════════════════════════════════════════════════════════
# Hydrogens
# ═══════════════════════════════════════════════════════════════════════════════

def reduce_available() -> bool:
    """True when `reduce` can be run."""
    return bool(_reduce_paths()[0])


def _reduce_paths() -> tuple:
    """(reduce executable, het dictionary) from the AmberTools installation."""
    try:
        import membrane as mem
        backend = mem.find_backend()
    except Exception:
        return "", ""
    exe = backend.get("amberhome") and Path(backend["amberhome"]) / "bin" / "reduce"
    if not exe or not Path(exe).exists():
        import shutil
        found = shutil.which("reduce")
        if not found:
            return "", ""
        exe = Path(found)
    het = Path(backend.get("amberhome", "")) / "dat" / "reduce_wwPDB_het_dict.txt"
    return str(exe), str(het) if het.exists() else ""


def add_hydrogens(pdb_path, dest, flip: bool = True, his: bool = True,
                  timeout: int = 900) -> tuple:
    """
    Add hydrogens with `reduce`.

    reduce and not something simpler because placing a hydrogen is not
    geometry alone: the orientation of every OH, SH and NH3 rotor and the flip
    state of asparagine, glutamine and histidine side chains are chosen to
    satisfy hydrogen bonding, and getting them wrong buries a donor against a
    donor in the middle of the protein.

    Args:
        pdb_path: Structure to protonate.
        dest    : Where to write the protonated copy.
        flip    : Let reduce flip Asn/Gln/His where that makes better H-bonds.
        his     : Build the ring NH hydrogens on histidines.
        timeout : Seconds before giving up.

    Returns:
        (path, added, error). `added` is the number of hydrogens gained.
    """
    exe, het = _reduce_paths()
    if not exe:
        return None, 0, ("`reduce` was not found. It comes with AmberTools: "
                         "`conda create -n ambertools -c conda-forge ambertools`.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-FLIP" if flip else "-NOFLIP", "-Quiet"]
    if his:
        cmd.append("-HIS")
    if het:
        # Without the het dictionary reduce refuses to protonate anything it
        # does not recognise, which is every ligand.
        cmd += ["-DB", het]
    cmd.append(str(pdb_path))

    log = dest.with_suffix(".reduce.log")
    try:
        with open(dest, "w") as out, open(log, "w") as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, 0, f"reduce did not finish within {timeout} s."
    except Exception as e:
        return None, 0, f"Could not run reduce: {e}"

    if not dest.exists() or dest.stat().st_size == 0:
        tail = log.read_text(errors="replace")[-300:] if log.exists() else ""
        return None, 0, f"reduce produced nothing (exit {proc.returncode}). {tail.strip()}"

    before = sum(1 for l in read_lines(pdb_path) if _is_coord(l) and _is_hydrogen(l))
    after = sum(1 for l in read_lines(dest) if _is_coord(l) and _is_hydrogen(l))
    return str(dest), after - before, None


# ═══════════════════════════════════════════════════════════════════════════════
# tleap input
# ═══════════════════════════════════════════════════════════════════════════════

# Force fields and water models tleap ships with, newest first. ff19SB with
# OPC is the current recommended pairing -- ff19SB was parameterised against
# OPC, and using it with TIP3P gives back some of what it was meant to fix.
PROTEIN_FF = ["ff19SB", "ff14SB", "ff15ipq"]
WATER_MODELS = {"opc": "leaprc.water.opc", "tip3p": "leaprc.water.tip3p",
                "tip4pew": "leaprc.water.tip4pew", "spce": "leaprc.water.spce"}


def residue_order(pdb_path, lines=None) -> list:
    """
    Residues in the order they appear, as [(reskey, resname)].

    Their position in this list is the residue index under the older leap
    convention of renumbering everything from 1 on load. Current tleap does
    not do that -- see tleap_script() -- but the index is still worth having,
    because which convention a given build uses is the sort of thing that
    changes between AmberTools releases and the two numbers together let
    someone check.
    """
    order, seen = [], set()
    for line in (lines if lines is not None else read_lines(pdb_path)):
        if not _is_coord(line):
            continue
        key = _reskey(line)
        if key not in seen:
            seen.add(key)
            order.append((key, _resname(line)))
    return order


def numbering_is_sequential(pdb_path, lines=None) -> bool:
    """
    True when every chain is numbered 1, 2, 3 ... with no gaps or insertion codes.

    This is the condition under which leap's residue expressions are
    unambiguous -- see tleap_script().
    """
    per_chain = {}
    for key, _ in residue_order(pdb_path, lines):
        chain, num, icode = key
        if icode:
            return False
        try:
            n = int(num)
        except ValueError:
            return False
        expected = per_chain.get(chain, 0) + 1
        if n != expected:
            return False
        per_chain[chain] = n
    return bool(per_chain)


def tleap_script(pdb_path, finding=None, unit: str = "prot", ff: str = "ff19SB",
                 water: str = "opc", solvate: bool = True, box: float = 12.0,
                 neutralise: bool = True, salt_concentration: float = 0.0,
                 lipid: bool = False, box_shape: str = "octahedron",
                 ion_counts=None, cation: str = "Na+", anion: str = "Cl-",
                 ligand_params=None, outputs=None, combine_units=None) -> str:
    """
    Write a tleap input file for a prepared structure.

    The part worth having is the disulfide bonds. Renaming cysteines to CYX
    tells tleap the residues are in the bonded state, but it does not create
    the bond -- that takes an explicit `bond` command per disulfide, and a
    setup missing them has its sulfurs 2 Å apart with nothing holding them,
    which is an explosion in the first minimisation step.

    Which residue numbers those commands take is the trap, and the behaviour
    is worse than either of the two rules people quote. Tested against the
    installed teLeap (AmberTools 22), on trypsin, whose numbering starts at 16
    and has gaps: `prot.22` resolved to the residue at sequential position 7,
    and `prot.42` resolved to a glycine that is neither residue 42 nor
    sequential position 42. leap appears to map a residue number to a
    sequential position by subtracting the first number in the file, which is
    correct only while the numbering is dense -- and it fails silently, with
    the bond command either erroring out on a residue that has no SG or, far
    worse, bonding the wrong pair of atoms.

    So this generator does not try to outguess it. The Amber profiles
    renumber the structure sequentially from 1, which makes all three
    conventions -- file numbering, sequential position, and whatever leap
    computes -- the same number, and the bond commands then land where they
    are meant to. That is verified: with renumbering, tleap built trypsin with
    all six disulfides present in the topology; without any bond commands it
    built the same structure with twelve CYX residues and zero S-S bonds, and
    said nothing about it beyond a net-charge warning.

    When handed a file that is not sequentially numbered, the script is still
    written, with the bond lines commented out and an explanation at the top,
    rather than emitting commands that might quietly bond the wrong atoms.

    Args:
        pdb_path         : The prepared structure.
        finding          : Its inspect() result, if already computed.
        unit             : Name for the loaded unit inside leap.
        ff               : Protein force field, e.g. ff19SB.
        water            : Water model key, e.g. opc.
        solvate          : Add a solvent box.
        box              : Buffer between solute and box edge, in Å.
        neutralise       : Add counter-ions to neutralise the system.
        salt_concentration: Extra salt in molar, on top of neutralisation.
        lipid            : Also source the Lipid21 force field, for a system
                           built by the Membrane tab.
        box_shape        : "octahedron" (solvateoct) or "cubic" (solvatebox).
                           A truncated octahedron holds the same buffer in about
                           70% of the water, which is 30% of the cost of the
                           whole simulation for nothing but the box shape.
        ion_counts       : {ion name: count} to add explicitly, from
                           simulation.ion_counts(). None means neutralise only.
        cation, anion    : Ion names for neutralisation.
        ligand_params    : [(residue code, mol2 path, frcmod path)] from
                           antechamber, loaded for real rather than as comments.
        outputs          : (prmtop, inpcrd) names; defaults from the file stem.
        combine_units    : [(unit name, mol2 path)] to load separately and
                           combine with the structure. This is how a
                           parameterised ligand gets in: its mol2 carries the
                           atom types and charges antechamber worked out, and
                           combining units sidesteps having to make the atom
                           names in the PDB match the ones in the mol2.

    Returns:
        The leap input as text.
    """
    path = Path(pdb_path)
    lines = read_lines(path)
    finding = finding or inspect(path)
    order = residue_order(path, lines)
    index = {key: i + 1 for i, (key, _) in enumerate(order)}

    out = [
        "# tleap input generated by PARORA (prepare.py)",
        f"# structure: {path.name}",
        "#",
        "# Check every line before running it. The force field, the water model",
        "# and the histidine protonation states are scientific choices, not",
        "# defaults that are right for every system.",
        "",
        f"source leaprc.protein.{ff}",
        f"source {WATER_MODELS.get(water, 'leaprc.water.opc')}",
    ]
    ligands = [e for e in finding["hetero"] if e["kind"] in ("ligand", "cofactor")]
    parameterised = {code.upper(): (mol2, frcmod)
                     for code, mol2, frcmod in (ligand_params or [])}
    if ligands or parameterised:
        out += ["source leaprc.gaff2", ""]

    # Every frcmod that was generated gets loaded, whatever the structure
    # itself still contains. When a ligand is combined in from its mol2 the
    # PDB no longer mentions it at all -- and driving these lines off the PDB's
    # component list is how the parameters end up missing, which surfaces much
    # later as "No torsion terms for atom types" from leap and looks like a
    # parmchk2 failure rather than a load that never happened.
    combined = {name.upper() for name, _ in (combine_units or [])}
    for code, (mol2, frcmod) in parameterised.items():
        if frcmod:
            out.append(f"loadamberparams {Path(frcmod).name}")
        if code not in combined:
            out.append(f"{code} = loadmol2 {Path(mol2).name}")
    if parameterised:
        out.append("")

    for e in ligands:
        if e["code"].upper() in parameterised:
            continue
        out += [
            f"# {e['code']} has no parameters yet. Generate them with:",
            f"#   antechamber -i {e['code']}.sdf -fi mdl -o {e['code']}.mol2 "
            f"-fo mol2 -c bcc -nc <net charge> -at gaff2",
            f"#   parmchk2 -i {e['code']}.mol2 -f mol2 -o {e['code']}.frcmod",
            f"# loadamberparams {e['code']}.frcmod",
            f"# {e['code']} = loadmol2 {e['code']}.mol2",
        ]
    if lipid:
        out.append("source leaprc.lipid21")
    out += ["", f"{unit} = loadpdb {path.name}", ""]
    if combine_units:
        for name, mol2 in combine_units:
            out.append(f"{name} = loadmol2 {Path(mol2).name}")
        joined = " ".join([unit] + [name for name, _ in combine_units])
        out += [f"system = combine {{ {joined} }}", ""]
        unit = "system"

    pairs = finding.get("disulfides") or []
    sequential = numbering_is_sequential(path, lines)
    if pairs:
        out += [
            "# Disulfide bonds. Renaming the cysteines CYX is not enough on its",
            "# own: tleap builds the residues in their bonded form but does NOT",
            "# create the bond. Without these lines the two sulfurs sit 2 A apart",
            "# with nothing holding them, and the first minimisation step explodes.",
        ]
        if not sequential:
            out += [
                "#",
                "# *** THESE LINES ARE COMMENTED OUT ON PURPOSE ***",
                "# This structure is not numbered sequentially from 1, and leap",
                "# resolves residue numbers by subtracting the first number in the",
                "# file -- which goes wrong as soon as the numbering has a gap, and",
                "# goes wrong silently. Re-prepare with 'renumber residues' turned",
                "# on (the Amber profile does it by default) and regenerate this",
                "# script, or check each line by hand with `desc " + unit + ".N`.",
            ]
        for pair in pairs:
            a_key = (pair["a"][0], str(pair["a"][1]), pair["a"][2])
            b_key = (pair["b"][0], str(pair["b"][1]), pair["b"][2])
            ai, bi = index.get(a_key), index.get(b_key)
            prefix = "" if sequential else "# "
            if ai and bi:
                out.append(f"{prefix}bond {unit}.{a_key[1]}.SG {unit}.{b_key[1]}.SG"
                           f"    # {pair['labels'][0]} - {pair['labels'][1]}")
            else:
                out.append(f"# could not locate {pair['labels'][0]} - "
                           f"{pair['labels'][1]} in the prepared file")
        out.append("")

    if finding.get("gaps"):
        out.append("# WARNING: this structure has chain breaks where residues were "
                   "not resolved:")
        for chain, a, b in finding["gaps"][:10]:
            out.append(f"#   chain {chain}, between residues {a} and {b}")
        out.append("# tleap will build a bond straight across each one unless the "
                   "segments are")
        out.append("# separated (TER) and capped (ACE/NME), or the loop is modelled "
                   "back in first.")
        out.append("")

    out += ["check " + unit, f"charge {unit}", ""]
    if solvate:
        boxtype = {"opc": "OPCBOX", "tip3p": "TIP3PBOX", "tip4pew": "TIP4PEWBOX",
                   "spce": "SPCBOX"}.get(water, "OPCBOX")
        command = "solvateoct" if box_shape == "octahedron" else "solvatebox"
        out.append(f"{command} {unit} {boxtype} {box}")
    if neutralise:
        # Counter-ions first and separately: `addions <unit> <ion> 0` adds
        # exactly enough of one ion to cancel the net charge, and asking for
        # both in one command when the system is already neutral is a no-op
        # that some leap builds report as an error.
        out += [f"addions {unit} {cation} 0", f"addions {unit} {anion} 0"]
    if ion_counts:
        out.append("")
        out.append("# Bulk salt, placed at random rather than at the most "
                   "favourable sites:")
        for name, count in ion_counts.items():
            if count:
                out.append(f"addionsrand {unit} {name} {count}")
    prmtop, inpcrd = outputs or (f"{path.stem}.prmtop", f"{path.stem}.inpcrd")
    out += ["", f"saveamberparm {unit} {prmtop} {inpcrd}",
            f"savepdb {unit} {path.stem}_leap.pdb", "quit", ""]
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Readiness
# ═══════════════════════════════════════════════════════════════════════════════

def readiness(finding: dict, engine: str = "amber") -> list:
    """
    Check a prepared structure against what a downstream tool needs.

    Returns [{"ok": bool, "text": str, "detail": str}], in the order someone
    would work through them. This runs on the *prepared* file, so it is a
    statement about what is still true, not about what the original held.
    """
    checks = []

    def add(ok, text, detail=""):
        checks.append({"ok": ok, "text": text, "detail": detail})

    add(finding["models"] == 1,
        "One model" if finding["models"] == 1 else f"{finding['models']} models still present",
        "" if finding["models"] == 1 else "Pick a single state.")
    add(not finding["altloc_ids"],
        "No alternate locations" if not finding["altloc_ids"]
        else f"Alternate locations still present ({', '.join(finding['altloc_ids'])})")

    ligands = [e for e in finding["hetero"] if e["kind"] in ("ligand", "cofactor")]
    if engine == "amber":
        add(finding["hydrogens"] == 0 or finding["hydrogens"] > 0,
            f"{finding['hydrogens']} hydrogens present" if finding["hydrogens"]
            else "No hydrogens — tleap will add them",
            "Either is fine for tleap, as long as they are not mixed.")
        add(not ligands,
            "No ligands needing parameters" if not ligands
            else "Ligands need GAFF parameters: " + ", ".join(e["code"] for e in ligands),
            "" if not ligands else "Run antechamber + parmchk2 and load the frcmod in tleap.")
        add(not finding["mse"], "No selenomethionine" if not finding["mse"]
            else f"{finding['mse']} MSE residues — tleap has no selenium")
        renamed = finding.get("cyx_present", 0)
        if finding["disulfides"] or renamed:
            n = len(finding["disulfides"]) or renamed // 2
            add(bool(renamed),
                f"{n} disulfides, cysteines renamed CYX" if renamed
                else f"{n} disulfides still named CYS",
                "Renaming is only half of it: tleap also needs an explicit bond "
                "command for each one, in ITS residue numbering. The generated "
                "leap script below has them.")
        add(not finding["gaps"] and not finding["missing_residues"],
            "No chain breaks" if not (finding["gaps"] or finding["missing_residues"])
            else f"{finding['missing_residues'] or len(finding['gaps'])} residues "
                 "missing — the chain has gaps",
            "tleap will bond straight across a gap unless you cap it or split "
            "the segments with TER.")
        add(True, f"{finding['histidines']} histidines — tleap will treat every one as HIE",
            "Rename to HID or HIP where you know better.")
    else:
        add(finding["hydrogens"] == 0,
            "No hydrogens — Rosetta will build its own" if finding["hydrogens"] == 0
            else f"{finding['hydrogens']} hydrogens present; Rosetta rebuilds them anyway",
            "Mixed deposited and rebuilt hydrogens are the usual cause of "
            "duplicate-atom errors.")
        add(not ligands,
            "No ligands needing params files" if not ligands
            else "Ligands need .params files: " + ", ".join(e["code"] for e in ligands),
            "" if not ligands else "Make them with molfile_to_params.py.")
        add(not finding["waters"],
            "No waters" if not finding["waters"]
            else f"{finding['waters']} waters kept — most Rosetta protocols expect none")
        add(not finding["gaps"] and not finding["missing_residues"],
            "No chain breaks" if not (finding["gaps"] or finding["missing_residues"])
            else "Chain breaks present — Rosetta will treat them as cutpoints")
    return checks


def count_cyx(pdb_path) -> int:
    """Residues named CYX in a file — how readiness() knows the rename happened."""
    keys = set()
    for line in read_lines(pdb_path):
        if _is_coord(line) and _resname(line) == CYX:
            keys.add(_reskey(line))
    return len(keys)


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "structures/3PTB.pdb"
    finding = inspect(target)
    print(f"{target}: {finding['models']} model(s), {finding['atoms']} atoms, "
          f"{finding['waters']} waters, {finding['hydrogens']} hydrogens")
    for issue in finding["issues"]:
        print(f"  [{issue['level']:>7}] {issue['title']}")
