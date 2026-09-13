# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : Geometric measurements on a loaded structure distances between
#             residues, angles, and contact shells around a residue or ligand. 
#             Coordinates are read straight out of the PDB records, as
#             everywhere else in this project, so measuring works whether or
#             not MDAnalysis imported. The other half of this module is parse_spec,
#             which accepts the ways people actually name a residue -- "12", "A/12", 
#             "12:A", "ARG12", "chain B residue 45", "12.CA", "BEN", "1UBQ/A/12" --
#             since the agent is a language model and will phrase the same
#             residue differently every time it is asked.
# =============================================================================

import math
import re

import numpy as np

import sequence_utils as squ

# Bumped when read_atoms changes the shape of what it returns, so callers that
# cache it can key on it. See structure_report.SCHEMA_VERSION.
SCHEMA_VERSION = 1

# Words people put in a residue specification that carry no information once
# the numbers and letters have been picked out.
FILLER = {"of", "in", "on", "the", "and", "to", "at", "number", "no", "num",
          "residue", "residues", "res", "resi", "resid", "resseq", "position"}

# Keywords that name what the *next* token is.
KEYWORDS = {
    "chain": "chain", "seg": "chain", "segid": "chain", "chainid": "chain",
    "atom": "atom", "name": "atom",
    "resname": "resname", "resn": "resname", "ligand": "resname",
    "structure": "structure", "model": "structure", "entry": "structure",
}

# Backbone and common atom names, used to tell "12.CA" (an alpha carbon) from
# "CA" on its own (a calcium ion) when the spelling alone is ambiguous.
BACKBONE_ATOMS = {"CA", "C", "N", "O", "CB", "OXT", "P", "C1'", "C4'"}


# ── Coordinates ──────────────────────────────────────────────────────────────

def read_atoms(pdb_path):
    """
    Read every atom of the first model as a plain dict of parallel arrays.

    Returns:
        dict with "xyz" (N×3 float array) and equal-length lists: name,
        altloc, resname, chain, resseq, icode, element, plus "kind" giving
        protein / nucleic / water / hetero per atom.
    """
    xyz, name, altloc, resname, chain, resseq, icode, element = [], [], [], [], [], [], [], []
    kinds = {}
    in_later_model = False

    with open(pdb_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL "):
                if xyz:
                    in_later_model = True
                continue
            if in_later_model or not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            raw_seq = line[22:26].strip()
            if not raw_seq:
                continue
            try:
                num = int(raw_seq)
            except ValueError:
                continue

            rn = line[17:20].strip()
            xyz.append((x, y, z))
            name.append(line[12:16].strip())
            altloc.append(line[16:17].strip())
            resname.append(rn)
            chain.append(line[21].strip() or "_")
            resseq.append(num)
            icode.append(line[26:27].strip())
            element.append(line[76:78].strip() or line[12:16].strip()[:1])
            if rn not in kinds:
                kinds[rn] = squ.classify_residue(rn)

    return {
        "xyz": np.asarray(xyz, dtype=float).reshape(-1, 3),
        "name": name, "altloc": altloc, "resname": resname, "chain": chain,
        "resseq": resseq, "icode": icode, "element": element,
        "kind": [kinds[r] for r in resname],
        "resnames_present": set(resname),
        "chains_present": sorted({c for c in chain}),
    }


def apply_matrix(atoms, matrix):
    """
    Return a copy of the atoms moved by a 4x4 superposition transform.

    Needed only when measuring *between* two structures: within one structure a
    rigid transform cancels out, but across a superposed pair the whole point
    is where one sits relative to the other.
    """
    if not matrix:
        return atoms
    m = np.asarray(matrix, dtype=float).reshape(4, 4).T
    moved = dict(atoms)
    moved["xyz"] = atoms["xyz"] @ m[:3, :3].T + m[:3, 3]
    return moved


# ── Residue specifications ───────────────────────────────────────────────────

def parse_spec(text, structure_names=(), resnames_present=()):
    """
    Turn a residue or atom specification into its parts.

    Accepts the spellings people and language models actually produce:
    "12", "A12", "A/12", "12:A", "ARG12", "chain B residue 45", "12.CA",
    "BEN", "HEM A", "1UBQ/A/12", "atom CB of residue 30".

    Args:
        text            : The specification.
        structure_names : Labels of loaded structures, so a leading "1UBQ/" is
                          read as a structure rather than a residue name.
        resnames_present: Residue names occurring in the file, so "BEN" is
                          recognised as a ligand and "CA" as calcium only when
                          the file actually contains one.

    Returns:
        dict with keys structure, chain, resseq, icode, resname, atom — each
        None when not given.
    """
    out = {"structure": None, "chain": None, "resseq": None,
           "icode": None, "resname": None, "atom": None, "extra": []}
    if not text:
        return out

    raw = str(text).strip()
    upper_structures = {s.upper() for s in structure_names}
    present = {r.upper() for r in resnames_present}

    # An atom name written with a dot ("12.CA", "A/30.CB") is unambiguous, and
    # taking it out first removes the CA-the-atom / CA-the-calcium collision
    # from everything that follows.
    if "." in raw:
        head, _, tail = raw.rpartition(".")
        candidate = tail.strip()
        if re.fullmatch(r"[A-Za-z0-9']{1,4}", candidate):
            out["atom"] = candidate.upper()
            raw = head

    tokens = [t for t in re.split(r"[\s/:,_]+", raw) if t]

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()
        i += 1
        if low in FILLER:
            continue
        if low in KEYWORDS and i < len(tokens):
            out[KEYWORDS[low]] = tokens[i].upper()
            i += 1
            continue

        up = tok.upper()

        # 4-character label of a loaded structure, e.g. "1UBQ/A/12".
        if out["structure"] is None and up in upper_structures:
            out["structure"] = up
            continue

        # A component code the file actually contains is checked before the
        # numeric patterns, because plenty of ligand codes start with a digit --
        # 03Q, 1N1, 2PE -- and would otherwise be read as residue 3 with an
        # insertion code of Q.
        if out["resname"] is None and up in present and not up.isdigit():
            out["resname"] = up
            continue

        # A plain residue number, optionally with an insertion code ("184A").
        m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", tok)
        if m and out["resseq"] is None:
            out["resseq"] = int(m.group(1))
            if m.group(2):
                out["icode"] = m.group(2).upper()
            continue

        # A residue name glued to a number: "ARG12", "HIS57".
        m = re.fullmatch(r"([A-Za-z]{1,3})(-?\d+)([A-Za-z]?)", tok)
        if m and out["resseq"] is None:
            out["resname"] = m.group(1).upper()
            out["resseq"] = int(m.group(2))
            if m.group(3):
                out["icode"] = m.group(3).upper()
            continue

        # A single character left over is a chain id.
        if out["chain"] is None and len(up) == 1:
            out["chain"] = up
            continue

        # Anything else that names a standard residue type.
        if out["resname"] is None and (up in squ.AA3_TO_1 or up in squ.NA3_TO_1):
            out["resname"] = up
            continue

        # A leftover short token with a residue number already found is an atom
        # name. It must contain a letter: a second bare number means two
        # residues were crammed into one specification, which is a mistake to
        # report rather than to silently reinterpret as an atom.
        if (out["atom"] is None and out["resseq"] is not None
                and len(up) <= 4 and re.search(r"[A-Z]", up)):
            out["atom"] = up
            continue

        # Nothing claimed this token. Kept rather than dropped so the caller can
        # say so: "residue 20 and 30" is two residues crammed into one argument,
        # and silently measuring to residue 20 would be the wrong answer given
        # confidently.
        out["extra"].append(tok)

    return out


def _residue_label(atoms, idx):
    """'ARG 12 (chain A)' for the residue containing atom index idx."""
    label = f"{atoms['resname'][idx]} {atoms['resseq'][idx]}{atoms['icode'][idx]}"
    return f"{label} (chain {atoms['chain'][idx]})"


def resolve(spec, atoms):
    """
    Find the atoms a parsed specification refers to, grouped by residue.

    Returns:
        (groups, error) where groups is a list of (label, [atom indices]) in
        file order, one entry per matching residue. An empty groups list comes
        with an error string explaining what did not match.
    """
    if spec.get("extra"):
        return [], ("could not make sense of "
                    + ", ".join(repr(t) for t in spec["extra"]))
    if (spec["resseq"] is None and not spec["resname"]
            and not spec["chain"] and not spec["atom"]):
        return [], "no residue was named"

    # Water is skipped unless it is asked for by name. Otherwise a bare "1"
    # matches HOH 1 as readily as the ligand, and the answer depends on which
    # chain the solvent happened to be numbered into.
    want_water = (spec["resname"] or "").upper() in squ.WATER_NAMES

    n = len(atoms["resseq"])
    hits = []
    for i in range(n):
        if not want_water and atoms["kind"][i] == "water":
            continue
        if spec["resseq"] is not None and atoms["resseq"][i] != spec["resseq"]:
            continue
        if spec["icode"] and atoms["icode"][i].upper() != spec["icode"]:
            continue
        if spec["chain"] and atoms["chain"][i].upper() != spec["chain"]:
            continue
        if spec["resname"] and atoms["resname"][i].upper() != spec["resname"]:
            continue
        if spec["atom"] and atoms["name"][i].upper() != spec["atom"]:
            continue
        # Alternate conformations would otherwise double every residue.
        if atoms["altloc"][i] not in ("", "A"):
            continue
        hits.append(i)

    if not hits:
        wanted = []
        if spec["resname"]:
            wanted.append(spec["resname"])
        if spec["resseq"] is not None:
            wanted.append(str(spec["resseq"]) + (spec["icode"] or ""))
        if spec["chain"]:
            wanted.append(f"chain {spec['chain']}")
        if spec["atom"]:
            wanted.append(f"atom {spec['atom']}")
        return [], "nothing matches " + " ".join(wanted or ["that"])

    groups, seen = [], {}
    for i in hits:
        key = (atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i],
               atoms["resname"][i])
        if key not in seen:
            seen[key] = (_residue_label(atoms, i), [])
            groups.append(seen[key])
        seen[key][1].append(i)
    return groups, None


# ── Measurements ─────────────────────────────────────────────────────────────

def _representative(atoms, indices):
    """The atom that stands for a residue: CA for protein, C4' / P for nucleic."""
    for want in ("CA", "C4'", "P"):
        for i in indices:
            if atoms["name"][i].upper() == want:
                return i
    return None


def distance(atoms_a, idx_a, atoms_b, idx_b):
    """
    Measure between two groups of atoms, three ways.

    Returns a dict with:
      closest  : shortest atom-to-atom distance, and which two atoms
      ca       : distance between the representative atoms (CA-CA), or None
      centre   : distance between the two centres of geometry

    All three, because they answer different questions and the difference
    between them is often the whole point: two residues whose CA atoms sit
    9 Å apart can still be hydrogen bonded through their side chains.
    """
    pa = atoms_a["xyz"][idx_a]
    pb = atoms_b["xyz"][idx_b]

    diff = pa[:, None, :] - pb[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(-1))
    flat = int(dmat.argmin())
    ia, ib = divmod(flat, dmat.shape[1])

    rep_a = _representative(atoms_a, idx_a)
    rep_b = _representative(atoms_b, idx_b)
    ca = None
    if rep_a is not None and rep_b is not None:
        ca = float(np.linalg.norm(atoms_a["xyz"][rep_a] - atoms_b["xyz"][rep_b]))

    return {
        "closest": float(dmat[ia, ib]),
        "closest_atoms": (atoms_a["name"][idx_a[ia]], atoms_b["name"][idx_b[ib]]),
        # The indices behind the closest pair, so a caller drawing the
        # measurement can put the line between the atoms it actually measured
        # rather than between the two CA atoms.
        "closest_index": (int(idx_a[ia]), int(idx_b[ib])),
        "ca": ca,
        "ca_names": (atoms_a["name"][rep_a] if rep_a is not None else None,
                     atoms_b["name"][rep_b] if rep_b is not None else None),
        "centre": float(np.linalg.norm(pa.mean(axis=0) - pb.mean(axis=0))),
    }


def angle(p1, p2, p3):
    """Angle at p2, in degrees, between the vectors to p1 and p3."""
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def dihedral(p1, p2, p3, p4):
    """
    Torsion angle about the p2-p3 axis, in degrees, signed -180 to 180.

    Returns None when the four points do not define a torsion — repeated or
    collinear points make both cross products vanish, and atan2(0, 0) is 0.0,
    which would report a perfectly ordinary-looking angle for input that has
    no angle in it at all.
    """
    b0 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    b1 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    b2 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    if (np.linalg.norm(b1) < 1e-8 or np.linalg.norm(n1) < 1e-8
            or np.linalg.norm(n2) < 1e-8):
        return None
    m = np.cross(n1, b1 / np.linalg.norm(b1))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m, n2))
    return math.degrees(math.atan2(y, x))


def contacts(atoms, target_idx, radius=4.0, include_water=False):
    """
    Every residue with an atom within `radius` of the target.

    Args:
        atoms       : atom table from read_atoms.
        target_idx  : atom indices of the residue or ligand at the centre.
        radius      : cutoff in Ångstroms.
        include_water: keep water molecules in the shell.

    Returns:
        list of dicts sorted by distance, each with label, distance, the two
        atom names, chain, resseq, resname and kind.
    """
    target_set = set(target_idx)
    target_res = {(atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i])
                  for i in target_idx}
    tp = atoms["xyz"][target_idx]

    # Only atoms inside the bounding box of the target plus the cutoff can be
    # in range, which is what keeps this fast enough for an interactive tool
    # on a structure with tens of thousands of atoms.
    lo = tp.min(axis=0) - radius
    hi = tp.max(axis=0) + radius
    inside = np.all((atoms["xyz"] >= lo) & (atoms["xyz"] <= hi), axis=1)

    best = {}
    for j in np.nonzero(inside)[0]:
        j = int(j)
        if j in target_set:
            continue
        key = (atoms["chain"][j], atoms["resseq"][j], atoms["icode"][j])
        if key in target_res:
            continue
        if not include_water and atoms["kind"][j] == "water":
            continue
        if atoms["altloc"][j] not in ("", "A"):
            continue
        d = float(np.sqrt(((tp - atoms["xyz"][j]) ** 2).sum(-1)).min())
        if d > radius:
            continue
        k = int(np.argmin(np.sqrt(((tp - atoms["xyz"][j]) ** 2).sum(-1))))
        if key not in best or d < best[key]["distance"]:
            best[key] = {
                "label": _residue_label(atoms, j),
                "distance": d,
                "atoms": (atoms["name"][target_idx[k]], atoms["name"][j]),
                "chain": atoms["chain"][j],
                "resseq": atoms["resseq"][j],
                "resname": atoms["resname"][j],
                "kind": atoms["kind"][j],
            }

    return sorted(best.values(), key=lambda c: c["distance"])
