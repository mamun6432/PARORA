# =============================================================================
# Developer : Abdullah Al Mamun, Methun Kamruzzaman
# Date      : 2026-09-11
# Summary   :This is an important module in this project to analyze and add membrane
#            in a lipid bilayer or orient it across the membrane, then pack a bilayer 
#            of a chosen composition around it. This one is done through packmol-memgen,
#            which is a command-line program in the AmberTools conda env. The PACKMOL-Memgen 
#            does the packing: it ships with AmberTools, knows 250-odd lipids with their 
#            areas per lipid, and produces an Amber-ready system. It is not imported rather 
#            it is a CLI program in a different conda environment with its own AmberTools 
#            dependencies so this module shells out to it, exactly like as pymol_render.py 
#
#            The work is split into two steps with very different costs, such as the membrane 
#            first needs to be oriented correctly, which is the point of this module: 
#            Orientation: Where the membrane sits relative to the protein. OPM has this already 
#            solved for most deposited membrane proteins, so its pre-oriented coordinates are used
#            when the entry is there; otherwise MEMEMBED computes it from the sequence and structure. 
#            Either way the result carries DUM atoms marking the two bilayer planes, which is enough 
#            to see the protein sitting in the membrane and to sanity-check the orientation before 
#            spending anything on packing.
#            Packing (minutes to hours) is the second and important step here. PACKMOL placing every 
#            lipid, water and ion. few minutes for a 40-residue helix dimer on standard GPU ; a GPCR 
#            in a five-component plasma membrane should take long time. So a build is never run inline: 
#            it is launched as a background process in its own directory, and the caller polls
#            it. Anything else would freeze the app for the length of the run and lose the work on a rerun.
#
#             Two traps worth knowing is PACKMOL-Memgen's --overwrite flag does the opposite of what it
#             says for the orientation step (`if not exists(out) and not overwrite:` so passing it *skips* 
#             MEMEMBED, and the run dies later with a ZeroDivisionError on an empty structure). This module 
#             never passes it and runs each build in a fresh directory instead.
# =============================================================================

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from functools import lru_cache
from glob import glob
from pathlib import Path

import requests

SCHEMA_VERSION = 1

# OPM (Orientations of Proteins in Membranes) -- a curated database of
# membrane proteins positioned in the bilayer. When an entry is there, its
# coordinates are a published, hand-checked orientation, which beats anything
# computed locally.
OPM_SEARCH = "https://opm-back.cc.lehigh.edu/opm-backend/primary_structures"
OPM_COORDS = "https://opm-assets.storage.googleapis.com/pdb/{pdb_id}.pdb"
OPM_TIMEOUT = 25

# MEMEMBED writes the bilayer planes as DUM pseudo-atoms: N for one plane, O
# for the other. Both PACKMOL-Memgen and OPM follow this convention, so the
# same reader works on either file.
DUMMY_RESNAME = "DUM"

# Water, ion and dummy residue names in a packed system.
SOLVENT_RESNAMES = {"WAT", "HOH", "TIP3", "SOL", "T3P"}
ION_RESNAMES = {"NA+", "CL-", "K+", "MG2", "CA2", "ZN2", "NA", "CL", "K"}


# ═══════════════════════════════════════════════════════════════════════════════
# Backend discovery
# ═══════════════════════════════════════════════════════════════════════════════

# Where to look for an AmberTools installation, in order. PACKMOL_MEMGEN wins
# so a user can point at any build; the conda globs cover the normal case of
# `conda create -n ambertools -c conda-forge ambertools`, which is how this
# machine has it.
_SEARCH_GLOBS = [
    "{conda}/envs/*/bin/packmol-memgen",
    "/opt/anaconda3/envs/*/bin/packmol-memgen",
    "/opt/miniconda3/envs/*/bin/packmol-memgen",
    "{home}/anaconda3/envs/*/bin/packmol-memgen",
    "{home}/miniconda3/envs/*/bin/packmol-memgen",
    "{home}/mambaforge/envs/*/bin/packmol-memgen",
    "{amberhome}/bin/packmol-memgen",
]


@lru_cache(maxsize=1)
def find_backend() -> dict:
    """
    Locate packmol-memgen and the AmberTools installation around it.

    Returns a dict with the paths found and a `ready` flag. Everything is
    reported rather than raised: a missing backend is a normal state for this
    app -- the membrane panel says so and stays out of the way.
    """
    found = {"packmol_memgen": "", "amberhome": "", "packmol": "",
             "memembed": "", "ready": False, "detail": ""}

    explicit = os.getenv("PACKMOL_MEMGEN", "").strip()
    candidates = [explicit] if explicit else []
    if not candidates:
        for pattern in _SEARCH_GLOBS:
            path = pattern.format(conda=os.getenv("CONDA_PREFIX", "/nonexistent"),
                                  home=str(Path.home()),
                                  amberhome=os.getenv("AMBERHOME", "/nonexistent"))
            candidates.extend(sorted(glob(path)))
        which = shutil.which("packmol-memgen")
        if which:
            candidates.append(which)

    exe = next((c for c in candidates if c and Path(c).exists()), "")
    if not exe:
        found["detail"] = (
            "packmol-memgen was not found. It ships with AmberTools: "
            "`conda create -n ambertools -c conda-forge ambertools`. "
            "Set PACKMOL_MEMGEN to its path if it lives somewhere unusual.")
        return found

    found["packmol_memgen"] = exe
    bindir = Path(exe).parent
    found["amberhome"] = str(bindir.parent)
    for name in ("packmol", "memembed"):
        local = bindir / name
        found[name] = str(local) if local.exists() else (shutil.which(name) or "")

    if not found["packmol"]:
        found["detail"] = (f"Found {exe}, but no `packmol` executable beside it. "
                           "PACKMOL-Memgen writes the input file and then needs "
                           "PACKMOL itself to do the packing.")
        return found

    found["ready"] = True
    if not found["memembed"]:
        found["detail"] = ("MEMEMBED is missing, so orientation has to come from "
                           "OPM or from a structure you have already oriented.")
    return found


def available() -> bool:
    """True when a membrane system can actually be built here."""
    return find_backend()["ready"]


def backend_report() -> str:
    """One line describing what was found, for the panel and the logs."""
    b = find_backend()
    if not b["packmol_memgen"]:
        return b["detail"]
    bits = [f"packmol-memgen at {b['packmol_memgen']}"]
    if b["packmol"]:
        bits.append("packmol found")
    if b["memembed"]:
        bits.append("MEMEMBED found")
    if b["detail"]:
        bits.append(b["detail"])
    return "; ".join(bits)


def _env() -> dict:
    """
    Environment for the subprocess.

    AMBERHOME and a PATH that contains the AmberTools bin directory are both
    required: packmol-memgen locates packmol, memembed and reduce through
    them, and without them it prints "Packmol path defined but not found" and
    fails several steps later for no apparent reason.
    """
    b = find_backend()
    env = dict(os.environ)
    if b["amberhome"]:
        env["AMBERHOME"] = b["amberhome"]
        env["PATH"] = str(Path(b["amberhome"]) / "bin") + os.pathsep + env.get("PATH", "")
    return env


@lru_cache(maxsize=1)
def parmed_broken() -> bool:
    """
    True when the backend's ParmEd cannot be imported.

    Worth checking separately because the failure is silent and misleading:
    the build finishes, but the step that removes lipid tails threaded through
    aromatic rings is skipped with only a one-line warning in the log. A
    system built this way looks finished and blows up on the first MD step.
    """
    b = find_backend()
    if not b["amberhome"]:
        return False
    python = Path(b["amberhome"]) / "bin" / "python"
    if not python.exists():
        return False
    try:
        r = subprocess.run([str(python), "-c", "import parmed"],
                           capture_output=True, timeout=60, env=_env())
        return r.returncode != 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Lipid catalogue and compositions
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def lipid_catalog() -> tuple:
    """
    Every lipid the backend can pack, as ({code, charge, name, lipid21_only}).

    Parsed from `packmol-memgen --available_lipids` rather than hard-coded:
    the list grows with each AmberTools release, and a hard-coded copy would
    quietly disagree with whatever is installed.
    """
    b = find_backend()
    if not b["packmol_memgen"]:
        return ()
    try:
        # In a temporary directory: packmol-memgen writes packmol-memgen.log
        # into whatever it is run from, and merely asking it what lipids exist
        # should not drop a file in the user's project.
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([b["packmol_memgen"], "--available_lipids"],
                                 capture_output=True, text=True, timeout=120,
                                 env=_env(), cwd=tmp).stdout
    except Exception:
        return ()
    rows = []
    for line in out.splitlines():
        m = re.match(r"^([A-Z0-9]{2,5})\s+(-?\d+)\s{2,}(.+?)\s*$", line)
        if not m:
            continue
        name = m.group(3)
        only21 = "Lipid21" in name
        name = re.sub(r"\*\*\*.*$", "", name).strip()
        rows.append({"code": m.group(1), "charge": int(m.group(2)),
                     "name": name, "lipid21_only": only21})
    return tuple(rows)


def lipid_names() -> dict:
    """code → full chemical name, for labelling the pickers."""
    return {l["code"]: l["name"] for l in lipid_catalog()}


# Curated compositions. Ratios are molar and are what PACKMOL-Memgen's -r
# expects. These are readable approximations of real membranes, not a claim
# that any one bilayer has exactly this composition -- real membranes carry
# hundreds of species, and the point of a preset is to get the physics roughly
# right (charge, saturation, cholesterol content, leaflet asymmetry) with
# lipids the force field actually has parameters for.
#
# Leaflet order follows PACKMOL-Memgen's own syntax: LOWER//UPPER.
PRESETS = [
    {
        "key": "popc",
        "label": "POPC — plain bilayer",
        "lower": [("POPC", 1)],
        "upper": None,
        "note": "The default control membrane: one neutral, singly unsaturated "
                "phospholipid. Use it when the membrane is scenery rather than "
                "the subject.",
    },
    {
        "key": "popc_chol",
        "label": "POPC + cholesterol (3:1)",
        "lower": [("POPC", 3), ("CHL1", 1)],
        "upper": None,
        "note": "Cholesterol thickens and stiffens the bilayer. The cheapest way "
                "to stop a plain POPC membrane from being unrealistically fluid.",
    },
    {
        "key": "plasma_sym",
        "label": "Mammalian plasma membrane (symmetric)",
        "lower": [("POPC", 30), ("POPE", 15), ("POPS", 5), ("PSM", 15), ("CHL1", 35)],
        "upper": None,
        "note": "Phosphatidylcholine, phosphatidylethanolamine, a little "
                "phosphatidylserine, sphingomyelin and a lot of cholesterol — a "
                "reasonable average plasma membrane, with both leaflets the same.",
    },
    {
        "key": "plasma_asym",
        "label": "Mammalian plasma membrane (asymmetric)",
        "lower": [("POPC", 25), ("POPE", 35), ("POPS", 15), ("CHL1", 25)],
        "upper": [("POPC", 40), ("PSM", 25), ("CHL1", 35)],
        "note": "The real asymmetry: phosphatidylserine and "
                "phosphatidylethanolamine on the cytoplasmic (lower) side, "
                "sphingomyelin outside. This is what makes the inner leaflet "
                "negatively charged, which matters for anything that binds "
                "peripheral proteins or basic residues.",
    },
    {
        "key": "raft",
        "label": "Raft-like / neuronal (high sphingomyelin + cholesterol)",
        "lower": [("PSM", 30), ("CHL1", 40), ("POPC", 30)],
        "upper": None,
        "note": "An ordered, thick, cholesterol-rich patch. Appropriate for "
                "receptors and channels reported to partition into rafts.",
    },
    {
        "key": "ecoli",
        "label": "E. coli inner membrane (POPE:POPG 3:1)",
        "lower": [("POPE", 3), ("POPG", 1)],
        "upper": None,
        "note": "The standard bacterial model membrane: mostly "
                "phosphatidylethanolamine with an anionic fraction and no sterol.",
    },
    {
        "key": "er",
        "label": "ER-like (low cholesterol)",
        "lower": [("POPC", 60), ("POPE", 25), ("POPS", 10), ("CHL1", 5)],
        "note": "Thin and fluid, with very little sterol — the membrane most "
                "newly made proteins are actually inserted into.",
        "upper": None,
    },
    {
        "key": "mito",
        "label": "Mitochondrial inner membrane (cardiolipin-free approximation)",
        "lower": [("POPC", 40), ("POPE", 30), ("POPG", 30)],
        "upper": None,
        "note": "An approximation, and the one preset to be careful with: the "
                "real inner membrane is defined by cardiolipin, which this "
                "lipid library does not carry. POPG stands in for the anionic "
                "fraction, but it is not a four-tailed lipid and will not "
                "reproduce cardiolipin's curvature or its binding sites.",
    },
    {
        "key": "dppc",
        "label": "DPPC — gel phase at room temperature",
        "lower": [("DPPC", 1)],
        "upper": None,
        "note": "Fully saturated and ordered. Useful for phase-behaviour work, "
                "misleading as a generic membrane.",
    },
    {
        "key": "dmpc",
        "label": "DMPC — thin, short-chain",
        "lower": [("DMPC", 1)],
        "upper": None,
        "note": "A thin bilayer, close to the bicelles and nanodiscs many NMR "
                "structures of membrane proteins were solved in.",
    },
]

PRESET_BY_KEY = {p["key"]: p for p in PRESETS}


def composition_args(lower: list, upper=None) -> tuple:
    """
    Render a composition as PACKMOL-Memgen's (-l, -r) argument pair.

    Args:
        lower: [(lipid code, ratio), ...] for the lower leaflet.
        upper: the same for the upper leaflet, or None for a symmetric bilayer.

    Returns:
        (lipids string, ratios string), e.g. ("POPC:CHL1", "3:1").
    """
    def render(spec):
        return (":".join(c for c, _ in spec), ":".join(str(r) for _, r in spec))

    lip, rat = render(lower)
    if upper:
        ulip, urat = render(upper)
        return f"{lip}//{ulip}", f"{rat}//{urat}"
    return lip, rat


def describe_composition(lower: list, upper=None) -> str:
    """A composition in words, with percentages, for the report and the panel."""
    def pct(spec):
        total = sum(r for _, r in spec) or 1
        return ", ".join(f"{c} {100.0 * r / total:.0f}%" for c, r in spec)

    if upper:
        return f"lower leaflet: {pct(lower)}; upper leaflet: {pct(upper)}"
    return pct(lower)


# ═══════════════════════════════════════════════════════════════════════════════
# PDB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def first_model(src, dest) -> int:
    """
    Write only the first model of a multi-model file.

    An NMR ensemble cannot be packed into one membrane -- twenty copies of the
    protein would be packed as twenty proteins -- and the alternative is a
    failure much further downstream with nothing in the message to connect it
    to the ensemble.

    Returns:
        The number of models the source held.
    """
    lines, models, keep = [], 0, True
    for line in Path(src).read_text(errors="replace").splitlines(True):
        if line.startswith("MODEL"):
            models += 1
            keep = models == 1
            continue
        if line.startswith("ENDMDL"):
            keep = models <= 1
            continue
        if keep:
            lines.append(line)
    Path(dest).write_text("".join(lines))
    return models


def membrane_planes(pdb_path):
    """
    The z of the two bilayer planes, read from a file's DUM atoms.

    Returns (z_low, z_high), or None when the file carries no dummy atoms --
    which is how this module tells an oriented structure from a raw one.
    """
    zs = []
    try:
        for line in Path(pdb_path).read_text(errors="replace").splitlines():
            if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == DUMMY_RESNAME:
                zs.append(float(line[46:54]))
    except Exception:
        return None
    if not zs:
        return None
    return (min(zs), max(zs))


def is_oriented(pdb_path) -> bool:
    """True when the file already carries bilayer planes."""
    return membrane_planes(pdb_path) is not None


def strip_records(src, dest, drop_resnames: set, drop_dummies: bool = False) -> int:
    """
    Copy a PDB without the named residues. Returns the number of atoms dropped.

    Used to make a system the viewer can actually open: a packed box is mostly
    water, and 34,000 water atoms are 3 MB of text embedded in the page for
    something nobody looks at.
    """
    kept, dropped = [], 0
    drop = {r.upper() for r in drop_resnames}
    for line in Path(src).read_text(errors="replace").splitlines(True):
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip().upper()
            if resname in drop or (drop_dummies and resname == DUMMY_RESNAME):
                dropped += 1
                continue
        elif line.startswith("TER") and (not kept or kept[-1].startswith("TER")):
            # Every removed molecule leaves its TER behind, and eleven thousand
            # of them in a row is a file some parsers choke on.
            continue
        kept.append(line)
    Path(dest).write_text("".join(kept))
    return dropped


def system_summary(pdb_path) -> dict:
    """
    Count what is actually in a packed system.

    Lipids are counted by their phosphorus atom and sterols by residue name,
    because Amber's Lipid21 splits every phospholipid into three residues --
    a head group and two acyl tails -- so counting residues would report three
    times as many lipids as there are molecules. Each leaflet is counted
    separately by which side of the bilayer midplane the phosphorus sits on,
    which is the quickest way to see that an asymmetric composition came out
    the way it was asked for.
    """
    summary = {"atoms": 0, "protein_residues": 0, "waters": 0, "ions": {},
               "lipids": {}, "sterols": {}, "lipids_lower": 0, "lipids_upper": 0,
               "box": None, "planes": membrane_planes(pdb_path), "hetero": {}}
    try:
        text = Path(pdb_path).read_text(errors="replace")
    except Exception:
        return summary

    planes = summary["planes"]
    midplane = (planes[0] + planes[1]) / 2 if planes else 0.0
    xs, ys, zs = [], [], []
    protein_res, water_res = set(), set()

    for line in text.splitlines():
        if line.startswith("CRYST1"):
            try:
                summary["box"] = tuple(round(float(line[i:i + 9]), 2)
                                       for i in (6, 15, 24))
            except ValueError:
                pass
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        summary["atoms"] += 1
        name = line[12:16].strip()
        resname = line[17:20].strip()
        key = (line[21], line[22:27])
        try:
            x, y, z = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if resname != DUMMY_RESNAME:
            xs.append(x), ys.append(y), zs.append(z)

        upper = resname.upper()
        if upper in SOLVENT_RESNAMES:
            water_res.add(key)
        elif upper in ION_RESNAMES or (len(resname) <= 3 and resname.endswith(("+", "-"))):
            summary["ions"][resname] = summary["ions"].get(resname, 0) + 1
        elif resname == "CHL":
            if name == "C3":                      # one atom per sterol molecule
                summary["sterols"]["cholesterol"] = summary["sterols"].get("cholesterol", 0) + 1
                if z < midplane:
                    summary["lipids_lower"] += 1
                else:
                    summary["lipids_upper"] += 1
        elif name in ("P", "P31"):
            summary["lipids"][resname] = summary["lipids"].get(resname, 0) + 1
            if z < midplane:
                summary["lipids_lower"] += 1
            else:
                summary["lipids_upper"] += 1
        elif resname == DUMMY_RESNAME:
            pass
        elif name == "CA" and len(resname) == 3:
            protein_res.add(key)

    summary["protein_residues"] = len(protein_res)
    summary["waters"] = len(water_res)
    if xs:
        summary["extent"] = tuple(round(max(v) - min(v), 1) for v in (xs, ys, zs))
    return summary


def summary_text(summary: dict) -> str:
    """The system summary as one readable paragraph."""
    lipids = sum(summary["lipids"].values())
    sterols = sum(summary["sterols"].values())
    bits = [f"{summary['atoms']:,} atoms"]
    if summary["protein_residues"]:
        bits.append(f"{summary['protein_residues']} protein residues")
    if lipids:
        bits.append(f"{lipids} phospholipids (" +
                    ", ".join(f"{n} {c}" for c, n in summary["lipids"].items()) + ")")
    if sterols:
        bits.append(f"{sterols} cholesterol")
    if summary["waters"]:
        bits.append(f"{summary['waters']:,} waters")
    if summary["ions"]:
        bits.append(", ".join(f"{n} {c}" for c, n in summary["ions"].items()))
    if summary.get("extent"):
        bits.append("box ≈ {} × {} × {} Å".format(*summary["extent"]))
    line = "; ".join(bits) + "."
    if summary["lipids_lower"] or summary["lipids_upper"]:
        line += (f" Leaflets: {summary['lipids_lower']} lipids below the midplane, "
                 f"{summary['lipids_upper']} above.")
    return line


# ═══════════════════════════════════════════════════════════════════════════════
# Orientation
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def opm_lookup(pdb_id: str):
    """
    Ask OPM whether this entry has a published membrane orientation.

    Returns a dict with the membrane type and thickness, or None. Errors are
    swallowed deliberately: OPM being unreachable means falling back to
    MEMEMBED, not failing the panel.
    """
    pid = (pdb_id or "").strip().lower()
    if len(pid) != 4:
        return None
    try:
        r = requests.get(OPM_SEARCH, params={"search": pid}, timeout=OPM_TIMEOUT)
        r.raise_for_status()
        objects = r.json().get("objects") or []
    except Exception:
        return None
    for obj in objects:
        if (obj.get("pdbid") or "").lower() == pid:
            return {
                "pdb_id": pid.upper(),
                "name": obj.get("name") or "",
                "thickness": obj.get("thickness"),
                "thickness_error": obj.get("thicknesserror"),
                "tilt": obj.get("tilt"),
                "comments": obj.get("comments") or "",
                "url": OPM_COORDS.format(pdb_id=pid),
            }
    return None


def fetch_opm(pdb_id: str, dest_dir):
    """
    Download OPM's pre-oriented coordinates for an entry.

    Returns:
        (path, info, error). The file carries DUM atoms on both bilayer planes,
        so it is ready to hand to PACKMOL-Memgen with --preoriented.
    """
    info = opm_lookup(pdb_id)
    if not info:
        return None, None, f"{pdb_id.upper()} is not in OPM."
    dest = Path(dest_dir) / f"{pdb_id.upper()}_OPM.pdb"
    try:
        r = requests.get(info["url"], timeout=OPM_TIMEOUT)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:
        return None, info, f"Could not download the OPM structure: {e}"
    if not is_oriented(dest):
        return str(dest), info, ("OPM returned a structure with no membrane planes "
                                 "marked — treating it as not oriented.")
    return str(dest), info, None


def orient(pdb_path, out_dir, n_ter: str = "in", barrel: bool = False,
           keep_ligands: bool = True, timeout: int = 1800):
    """
    Orient a protein across the membrane with MEMEMBED.

    Seconds to a couple of minutes, unlike the packing step, so this one runs
    inline. MEMEMBED searches rigid-body orientations against a knowledge-based
    potential for where each residue type sits relative to a bilayer.

    Args:
        pdb_path    : Structure to orient.
        out_dir     : Directory for the oriented copy and the log.
        n_ter       : "in" or "out" — which side the first residue starts on.
        barrel      : Use beta-barrel mode for porins and outer-membrane proteins.
        keep_ligands: Carry hetero atoms through (MEMEMBED strips them itself).
        timeout     : Seconds before giving up.

    Returns:
        (path, info, error). info holds the fitted energy and the plane
        positions, so a caller can show what the fit actually did.
    """
    b = find_backend()
    if not b["memembed"]:
        return None, None, ("MEMEMBED was not found, so the orientation cannot be "
                            "computed here. Use OPM if the entry is in it, or "
                            "supply a structure that is already oriented.")
    # Absolute, because MEMEMBED runs with cwd set to the output directory --
    # a path relative to the app's own directory resolves to nothing there.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / (Path(pdb_path).stem + "_input.pdb")
    models = first_model(pdb_path, src)
    out = out_dir / (Path(pdb_path).stem + "_oriented.pdb")
    log = out_dir / "memembed.log"

    cmd = [b["memembed"], "-s", "3", "-n", n_ter, "-o", str(out)]
    if barrel:
        cmd.append("-b")
    cmd.append(str(src))
    try:
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=_env(), cwd=str(out_dir))
    except subprocess.TimeoutExpired:
        return None, None, f"MEMEMBED did not finish within {timeout} s."
    except Exception as e:
        return None, None, f"Could not run MEMEMBED: {e}"
    if proc.returncode != 0 or not out.exists():
        tail = log.read_text(errors="replace")[-400:] if log.exists() else ""
        return None, None, f"MEMEMBED failed. {tail.strip()}"

    if keep_ligands:
        _restore_hetero(pdb_path, src, out)

    info = {"planes": membrane_planes(out), "models": models, "log": str(log)}
    header = {}
    for line in out.read_text(errors="replace").splitlines():
        if not line.startswith("HEADER"):
            break
        parts = line.split()
        if len(parts) >= 3:
            header[parts[1]] = parts[2]
    info["energy"] = header.get("MEMBRANE_ENERGY")
    info["rotation"] = (header.get("X_ROTATION"), header.get("Y_ROTATION"))
    if info["planes"]:
        info["thickness"] = round(info["planes"][1] - info["planes"][0], 1)
    return str(out), info, None


def _restore_hetero(original, trimmed, oriented) -> None:
    """
    Put ligands back into an oriented structure.

    MEMEMBED cleans the PDB down to protein, so a bound ligand, cofactor or
    metal is simply gone from its output. It applies one rigid-body transform
    to everything, so the transform can be recovered from three protein atoms
    and applied to the hetero atoms that were dropped -- which is what
    PACKMOL-Memgen's own --keepligs does internally.
    """
    try:
        import numpy as np
    except Exception:
        return

    def atoms(path, hetero=False):
        out = {}
        for line in Path(path).read_text(errors="replace").splitlines(True):
            if not line.startswith("HETATM" if hetero else "ATOM"):
                continue
            resname = line[17:20].strip()
            if resname == DUMMY_RESNAME:
                continue
            key = (line[12:16], line[17:27])
            out[key] = (line, np.array([float(line[30:38]), float(line[38:46]),
                                        float(line[46:54])]))
        return out

    try:
        before = atoms(trimmed)
        after = atoms(oriented)
        shared = [k for k in before if k in after]
        if len(shared) < 3:
            return
        P = np.array([before[k][1] for k in shared])
        Q = np.array([after[k][1] for k in shared])
        pc, qc = P.mean(axis=0), Q.mean(axis=0)
        # Kabsch: the rotation that carries the original frame onto the oriented one.
        U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        if np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(axis=1).mean()) > 0.5:
            return                      # not a rigid transform — leave it alone

        moved = []
        for line, xyz in atoms(original, hetero=True).values():
            if line[17:20].strip().upper() in SOLVENT_RESNAMES:
                continue
            new = (xyz - pc) @ R.T + qc
            moved.append(line[:30] + "".join(f"{v:8.3f}" for v in new) + line[54:])
        if not moved:
            return
        text = Path(oriented).read_text(errors="replace").splitlines(True)
        body = [l for l in text if not l.startswith("END")]
        Path(oriented).write_text("".join(body + moved + ["END\n"]))
    except Exception:
        return


# ═══════════════════════════════════════════════════════════════════════════════
# Building
# ═══════════════════════════════════════════════════════════════════════════════

def build_argv(protein: str, lipids: str, ratios: str, output: str,
               preoriented: bool = False, keep_ligands: bool = True,
               salt: bool = True, salt_concentration: float = 0.15,
               salt_cation: str = "K+", salt_anion: str = "Cl-",
               water_thickness: float = 17.5, boundary_distance: float = 15.0,
               patch_xy: float = 0.0, n_ter: str = "in", barrel: bool = False,
               charmm_output: bool = False, protonate: bool = True,
               parametrize: bool = False, dry_run: bool = False,
               extra: list = None) -> list:
    """
    Assemble the packmol-memgen command line.

    Kept separate from running it so the panel can show the exact command --
    a membrane build is a long, opinionated job and the first thing anyone
    reproducing it in a paper or a cluster script needs is the command itself.

    Note what is deliberately absent: --overwrite (it makes the orientation
    step silently do nothing in the 2025.1 release, and every build here runs
    in a fresh directory anyway) and --verbose (it raises NameError in the
    same release).
    """
    b = find_backend()
    argv = [b["packmol_memgen"], "-p", protein, "-l", lipids, "-r", ratios,
            "-o", output, "--noprogress"]
    if preoriented:
        argv.append("--preoriented")
    else:
        argv += ["--n_ter", n_ter]
        if barrel:
            argv.append("--barrel")
    if keep_ligands:
        argv.append("--keepligs")
    if not protonate:
        argv.append("--notprotonate")
    if salt:
        argv += ["--salt", "--saltcon", str(salt_concentration),
                 "--salt_c", salt_cation, "--salt_a", salt_anion]
    argv += ["--dist", str(boundary_distance), "--dist_wat", str(water_thickness)]
    if patch_xy and patch_xy > 0:
        argv += ["--distxy_fix", str(patch_xy)]
    if charmm_output:
        argv.append("--charmm")
    if parametrize:
        argv.append("--parametrize")
    if dry_run:
        argv.append("--notrun")
    return argv + list(extra or [])


def start_build(protein_pdb, job_dir, lipids: str, ratios: str, label: str = "",
                **options) -> dict:
    """
    Launch a membrane build as a background process.

    Never inline: packing took eight minutes for a forty-residue helix dimer on
    the machine this was written on, and a large protein in a five-component
    membrane takes hours. The process gets its own directory because
    PACKMOL-Memgen writes a dozen intermediate files into the working
    directory and reuses them by name.

    Args:
        protein_pdb: Structure to embed. Only its first model is used.
        job_dir    : Directory to create and run in.
        lipids     : PACKMOL-Memgen lipid string, e.g. "POPC:CHL1" or "A:B//C".
        ratios     : Matching ratio string, e.g. "3:1".
        label      : Name for the job in the UI.
        options    : Passed through to build_argv().

    Returns:
        A job dict. Hand it to poll() to find out how it is getting on.
    """
    # Absolute, because the process runs with cwd set to the job directory:
    # a path relative to the app's own directory resolves to nothing there,
    # and packmol-memgen reports that as "the options were wrongly used".
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    src = job_dir / "protein.pdb"
    models = first_model(protein_pdb, src)

    preoriented = options.pop("preoriented", None)
    if preoriented is None:
        preoriented = is_oriented(src)

    out_name = "membrane_system.pdb"
    argv = build_argv(str(src), lipids, ratios, out_name,
                      preoriented=preoriented, **options)
    log_path = job_dir / "build.log"
    job = {
        "label": label or Path(protein_pdb).stem,
        "dir": str(job_dir),
        "log": str(log_path),
        "output": str(job_dir / out_name),
        "cmd": " ".join(argv),
        "lipids": lipids,
        "ratios": ratios,
        "preoriented": bool(preoriented),
        "models": models,
        "started": time.time(),
        "status": "running",
        "stage": "starting",
        "error": "",
        "pid": None,
        "proc": None,
    }
    try:
        fh = open(log_path, "w")
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(job_dir), env=_env(),
                                start_new_session=True)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"Could not start packmol-memgen: {e}"
        return job
    job["proc"] = proc
    job["pid"] = proc.pid
    (job_dir / "job.json").write_text(json.dumps(
        {k: v for k, v in job.items() if k != "proc"}, indent=2))
    return job


# What the log says, and what that means in words. Ordered: the last match in
# the log wins, so later stages override earlier ones.
_STAGES = [
    ("Preprocessing", "preparing the structure"),
    ("Orienting the protein", "orienting the protein in the membrane"),
    ("protonate", "adding hydrogens"),
    ("Estimating the volume", "measuring the protein's volume"),
    ("Running Packmol", "packing lipids"),
    ("Processing segment", "packing lipids"),
    ("All-together Packing", "final packing — the longest step"),
    ("Transforming to AMBER", "converting to Amber lipid naming"),
    ("piercing", "checking for lipid tails threaded through rings"),
    ("DONE!", "finished"),
]


def poll(job: dict) -> dict:
    """
    Update a job's status from its process and its log. Mutates and returns it.

    Both are needed: the process tells you whether it is alive, and only the
    log tells you which of the several very-different-length stages it is in.
    """
    if job.get("status") in ("finished", "failed", "cancelled"):
        return job

    text = ""
    try:
        text = Path(job["log"]).read_text(errors="replace")
    except Exception:
        pass

    stage, progress = job.get("stage", "starting"), None
    for needle, label in _STAGES:
        if needle in text:
            stage = label
    m = None
    for m in re.finditer(r"Processing segment (\d+) of (\d+)", text):
        pass
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        progress = min(0.95, done / max(total, 1))
        stage = f"packing lipids — segment {done} of {total}"
    if "All-together Packing" in text:
        stage = "final packing — the longest step"
        progress = 0.97
    job["stage"] = stage
    job["progress"] = progress
    job["elapsed"] = time.time() - job["started"]

    proc = job.get("proc")
    alive = None
    if proc is not None:
        alive = proc.poll() is None
    elif job.get("pid"):
        try:
            os.kill(job["pid"], 0)
            alive = True
        except OSError:
            alive = False
    if alive:
        return job

    # The process is gone: decide what happened from what it left behind.
    #
    # The output file is NOT the test. PACKMOL writes its working structure to
    # the output path while it is still packing -- a full-sized, plausible,
    # half-packed system appears there minutes before the run ends -- so a
    # cancelled or crashed build leaves a file that looks finished and is not.
    # packmol-memgen prints DONE! only after the final conversion, so that is
    # what counts as finished here.
    out = Path(job["output"])
    complete = "DONE!" in text
    if complete and out.exists() and out.stat().st_size > 0:
        job["status"] = "finished"
        job["stage"] = "finished"
        job["progress"] = 1.0
        job["summary"] = system_summary(out)
    elif out.exists() and out.stat().st_size > 0:
        job["status"] = "failed"
        job["partial"] = str(out)
        job["error"] = ("the run stopped before it finished. " + _failure_reason(text)
                        + " A partial system was left at " + out.name
                        + " — it is a half-packed box, not a usable one.")
    else:
        job["status"] = "failed"
        job["error"] = _failure_reason(text)
    return job


def _failure_reason(log_text: str) -> str:
    """Pull the informative line out of a failed run's log."""
    if not log_text.strip():
        return "packmol-memgen produced no output at all."
    lines = [l.strip() for l in log_text.splitlines() if l.strip()]
    for marker in ("Error", "ERROR", "error:", "Traceback"):
        for i, line in enumerate(lines):
            if marker in line:
                return " ".join(lines[i:i + 3])[:400]
    return lines[-1][:400]


def cancel(job: dict) -> dict:
    """
    Stop a running build and everything it spawned.

    start_new_session put the run in its own process group, so the signal goes
    to the group: killing only packmol-memgen would leave PACKMOL itself
    running for hours on a core nobody is watching. PACKMOL is a Fortran
    program in a tight optimisation loop and does not always take SIGTERM
    promptly, so the group is given a few seconds and then killed outright.
    The exit status is collected afterwards -- an unreaped child stays visible
    as a zombie, and poll() would go on reporting it as running.
    """
    proc = job.get("proc")
    pid = proc.pid if proc is not None else job.get("pid")
    if pid:
        for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(os.getpgid(pid), sig)
            except Exception:
                try:
                    if proc is not None:
                        proc.kill()
                except Exception:
                    pass
            if proc is None:
                time.sleep(0.2)
                break
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
    job["status"] = "cancelled"
    job["stage"] = "cancelled"
    return job


def log_tail(job: dict, lines: int = 25) -> str:
    """The last few lines of a job's log."""
    try:
        text = Path(job["log"]).read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(text[-lines:])


def viewer_copy(system_pdb, dest=None, keep_water: bool = False,
                keep_ions: bool = True):
    """
    A copy of a packed system small enough to open in the viewer.

    A packed box is mostly water -- two thirds of the atoms in a typical
    system -- and local files are embedded into the viewer's HTML as text, so
    loading the full thing means megabytes of water nobody looks at. The full
    system stays on disk for simulation; this copy is for looking at.

    Returns:
        (path, atoms dropped).
    """
    src = Path(system_pdb)
    dest = Path(dest) if dest else src.with_name(src.stem + "_view.pdb")
    drop = set()
    if not keep_water:
        drop |= SOLVENT_RESNAMES
    if not keep_ions:
        drop |= ION_RESNAMES
    dropped = strip_records(src, dest, drop)
    return str(dest), dropped


if __name__ == "__main__":
    import sys
    print(backend_report())
    if len(sys.argv) > 1:
        print(summary_text(system_summary(sys.argv[1])))
