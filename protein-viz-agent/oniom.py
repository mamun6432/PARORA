# =============================================================================
# Developer : Abdullah Al Mamun, Methun Kamruzzaman
# Date      : 2026-09-12
# Summary   : ONIOM-style QM/MM input for Gaussian: a quantum region inside a
#             molecular-mechanics protein, with the atom types and partial
#             charges the MM layer needs taken from a real Amber topology.The 
#             reason this needs the Amber topology rather than just the
#             PDB is that Gaussian's MM layer is not a force field it can work
#             out for itself. Every atom in the low layer needs an Amber atom
#             type and a partial charge, and those come from parameterising
#             the system which the Simulate tab already does with tleap,
#             including the ligand parameters from antechamber. So this module
#             reads the prmtop it produced and the matching PDB leap wrote,
#             and pairs them up atom for atom: they are in the same order by
#             construction, which is what makes the pairing safe.
#
#             This will ask user to define the chemistry. Such as which residues
#             belong in the QM layer, the charge and spin of that layer, and
#             whether electronic embedding is worth its cost are all choices,
#             and getting any of them wrong produces a number rather than an
#             error. Alternatives considered: PDB2ONIOM and the WebMO API both do
#             this. Neither is available here PDB2ONIOM is not installed,
#             and WebMO needs an account and credentials so the input is written 
#             natively, which also means it can be read and checked.
# =============================================================================

import re
from pathlib import Path

import measure as mz
import quantum as qm

SCHEMA_VERSION = 1

# Gaussian's ONIOM layers.
HIGH, MEDIUM, LOW = "H", "M", "L"

# Link-atom defaults. Gaussian needs an element, an MM type and a charge for
# the hydrogen that caps a bond crossing the layer boundary; HC with zero
# charge is the conventional choice for a C-C cut.
LINK_TYPE = "HC"
LINK_CHARGE = 0.0

# Amber's prmtop stores charges pre-multiplied by this so that energies come
# out in kcal/mol. Anything that reads charges out of one and forgets to
# divide is out by a factor of eighteen.
CHARGE_SCALE = 18.2223

ONIOM_METHODS = {
    "B3LYP/6-31G(d):Amber": "The standard first pass. Cheap enough to run on a "
                            "real site, well enough understood to defend.",
    "wB97XD/6-31G(d):Amber": "Range-separated with dispersion — better for "
                             "non-covalent interactions in a binding site.",
    "M06-2X/6-31G(d):Amber": "Good thermochemistry for organic chemistry.",
    "B3LYP/6-311+G(d,p):Amber": "Larger basis with diffuse functions, for "
                                "anionic or charge-transfer sites.",
    "PM6:Amber": "Semi-empirical high layer. For testing that the setup runs, "
                 "not for an answer.",
}

EMBEDDING = {
    "electronic": ("=EmbedCharge",
                   "The MM charges polarise the QM wavefunction. Almost always "
                   "what you want, and what makes QM/MM better than a cluster "
                   "model with a dielectric."),
    "mechanical": ("",
                   "The layers interact only through the force field. Cheaper, "
                   "and blind to the electrostatics of the protein — which is "
                   "usually the thing being studied."),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Amber topology
# ═══════════════════════════════════════════════════════════════════════════════

def read_topology(prmtop_path) -> dict:
    """
    Atom names, Amber types, charges and residue labels from a prmtop.

    Parsed directly. ParmEd would be the obvious reader and is broken by
    NumPy 2 in current AmberTools builds -- the same breakage that takes out
    pdb4amber -- and the format is a handful of named sections with a Fortran
    format line, which is not worth a dependency that falls over.
    """
    text = Path(prmtop_path).read_text(errors="replace")

    def section(name):
        m = re.search(r"%FLAG " + name + r"\s*\n%FORMAT\((.*?)\)\s*\n(.*?)(?=%FLAG|\Z)",
                      text, re.S)
        if not m:
            return []
        width = int(re.search(r"[aIEF](\d+)", m.group(1)).group(1))
        return [line[i:i + width].strip()
                for line in m.group(2).splitlines()
                for i in range(0, len(line), width)
                if line[i:i + width].strip()]

    names = section("ATOM_NAME")
    types = section("AMBER_ATOM_TYPE")
    charges = [round(float(c) / CHARGE_SCALE, 6) for c in section("CHARGE")]
    labels = section("RESIDUE_LABEL")
    pointers = [int(v) for v in section("RESIDUE_POINTER")]

    residue_of = []
    for index in range(len(names)):
        # Residue pointers are 1-based indices of each residue's first atom.
        position = 0
        for r, start in enumerate(pointers):
            if start <= index + 1:
                position = r
            else:
                break
        residue_of.append(position)

    return {
        "names": names, "types": types, "charges": charges,
        "residue_labels": labels, "residue_of": residue_of,
        "atoms": len(names),
    }


def pair_with_structure(pdb_path, topology: dict) -> dict:
    """
    Match a PDB written by leap to the topology built alongside it.

    The two are in the same atom order by construction -- leap writes both
    from the same unit -- so they are paired by position, and the pairing is
    checked rather than assumed: if the atom counts or the element sequence
    disagree, the files are from different builds and everything downstream
    would be quietly wrong.

    Returns:
        {ok, atoms: [...], error}. Each atom carries its coordinates, Amber
        type, partial charge, residue and index.
    """
    atoms = mz.read_atoms(pdb_path)
    count = len(atoms["name"])
    if count != topology["atoms"]:
        return {"ok": False, "atoms": [],
                "error": (f"The structure has {count} atoms and the topology has "
                          f"{topology['atoms']}. They are not from the same build "
                          f"— use the PDB leap wrote (savepdb), not the one it read.")}

    mismatched = sum(1 for i in range(count)
                     if atoms["name"][i].strip() != topology["names"][i].strip())
    if mismatched > count * 0.02:
        return {"ok": False, "atoms": [],
                "error": (f"{mismatched} atom names differ between the structure and "
                          f"the topology. They are not from the same build.")}

    out = []
    for i in range(count):
        x, y, z = atoms["xyz"][i]
        out.append({
            "index": i,
            "name": atoms["name"][i],
            "element": (atoms["element"][i] or atoms["name"][i][:1]).capitalize(),
            "x": float(x), "y": float(y), "z": float(z),
            "resname": atoms["resname"][i],
            "key": (atoms["chain"][i], atoms["resseq"][i], atoms["icode"][i]),
            "kind": atoms["kind"][i],
            "type": topology["types"][i],
            "charge": topology["charges"][i],
        })
    return {"ok": True, "atoms": out, "error": None, "table": atoms}


# ═══════════════════════════════════════════════════════════════════════════════
# Layers
# ═══════════════════════════════════════════════════════════════════════════════

def assign_layers(paired: dict, high_keys, high_codes=None,
                  side_chains_only: bool = False, sphere_radius: float = 0.0,
                  drop_far_solvent: bool = True) -> dict:
    """
    Decide which atoms are QM, which are MM, and which are not in the model.

    Args:
        paired          : Output of pair_with_structure().
        high_keys       : Residue keys for the QM layer.
        high_codes      : Component codes always in the QM layer (the ligand).
        side_chains_only: Put only side-chain atoms in the QM layer, cutting
                          the CA-CB bond -- the same cluster-model convention
                          quantum.py uses.
        sphere_radius   : Keep only MM atoms within this distance of the QM
                          layer. A solvated system is 30,000 atoms and Gaussian
                          does not want all of them; 15-20 A around the site is
                          the usual compromise. 0 keeps everything.
        drop_far_solvent: Drop water and ions outside the sphere even when the
                          sphere is off, since those are the bulk of the count.

    Returns:
        {atoms: [...], high: n, low: n, dropped: n, charge_high, charge_low}
        with a "layer" on every atom that stays.
    """
    import numpy as np

    codes = {c.strip().upper() for c in (high_codes or [])}
    wanted = set(high_keys or [])
    atoms = paired["atoms"]

    high_index = []
    for atom in atoms:
        is_ligand = atom["resname"].upper() in codes
        if not (is_ligand or atom["key"] in wanted):
            continue
        if (side_chains_only and not is_ligand
                and qm.is_protein(atom["resname"], atom["kind"])):
            if atom["name"] in qm.BACKBONE or atom["resname"].upper() == "GLY":
                continue
        high_index.append(atom["index"])
    high_set = set(high_index)

    keep = set(high_index)
    if high_index:
        coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
        high_coords = coords[high_index]
        # By residue, never by atom. Keeping whichever atoms happen to fall
        # inside a radius cuts residues in half, and half a residue in an MM
        # layer is a set of dangling valences the force field has no terms for
        # -- a much worse error than a slightly larger model.
        by_residue = {}
        for atom in atoms:
            by_residue.setdefault(atom["key"], []).append(atom)
        for key, members in by_residue.items():
            if any(a["index"] in high_set for a in members):
                keep.update(a["index"] for a in members)
                continue
            far_kind = (members[0]["kind"] == "water"
                        or members[0]["resname"].upper() in
                        ("NA+", "CL-", "K+", "NA", "CL", "K"))
            if sphere_radius <= 0 and not (drop_far_solvent and far_kind):
                keep.update(a["index"] for a in members)
                continue
            limit = sphere_radius if sphere_radius > 0 else 8.0
            nearest = min(
                float(np.min(np.linalg.norm(high_coords - coords[a["index"]], axis=1)))
                for a in members)
            if nearest <= limit:
                keep.update(a["index"] for a in members)

    out, charge_high, charge_low = [], 0.0, 0.0
    for atom in atoms:
        if atom["index"] not in keep:
            continue
        layer = HIGH if atom["index"] in high_set else LOW
        record = dict(atom, layer=layer)
        out.append(record)
        if layer == HIGH:
            charge_high += atom["charge"]
        else:
            charge_low += atom["charge"]

    return {
        "atoms": out,
        "high": len(high_set),
        "low": len(out) - len(high_set),
        "dropped": len(atoms) - len(out),
        "charge_high": round(charge_high, 3),
        "charge_low": round(charge_low, 3),
        "high_set": high_set,
    }


def find_boundary(layered: dict) -> list:
    """
    Bonds that cross from the QM layer into the MM layer.

    Each one becomes a link atom. Found by distance, like everything else here,
    and reported so the cuts can be looked at: a boundary through a peptide
    bond or next to a charged group is a real problem that the input file
    itself will not complain about.
    """
    import numpy as np

    atoms = layered["atoms"]
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
    high = [i for i, a in enumerate(atoms) if a["layer"] == HIGH]
    crossings = []
    for i in high:
        if atoms[i]["element"].upper() == "H":
            continue
        ri = qm.COVALENT_RADII.get(atoms[i]["element"].upper(), 0.77)
        for j, atom in enumerate(atoms):
            if atom["layer"] == HIGH or atom["element"].upper() == "H":
                continue
            rj = qm.COVALENT_RADII.get(atom["element"].upper(), 0.77)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < ri + rj + qm.BOND_TOLERANCE:
                crossings.append({"high": i, "low": j, "distance": round(d, 2),
                                  "label": (f"{atoms[i]['resname']}"
                                            f"{atoms[i]['key'][1]}"
                                            f".{atoms[i]['name']} — "
                                            f"{atom['resname']}{atom['key'][1]}"
                                            f".{atom['name']}")})
    return crossings


def topology_bonds(prmtop_path) -> list:
    """
    Every bond in the topology, as (atom i, atom j) zero-based pairs.

    Taken from the force field rather than perceived from distances. Gaussian's
    Amber layer needs to know the bonding to apply bonded terms at all, and
    letting it guess from geometry is how a strained crystallographic contact
    becomes a bond that is then minimised as one.
    """
    text = Path(prmtop_path).read_text(errors="replace")
    bonds = []
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        m = re.search(r"%FLAG " + flag + r"\s*\n%FORMAT\((.*?)\)\s*\n(.*?)(?=%FLAG|\Z)",
                      text, re.S)
        if not m:
            continue
        width = int(re.search(r"[aIEF](\d+)", m.group(1)).group(1))
        values = [int(line[i:i + width])
                  for line in m.group(2).splitlines()
                  for i in range(0, len(line), width)
                  if line[i:i + width].strip()]
        for k in range(0, len(values), 3):
            # Amber stores bonded atoms as (index - 1) * 3, a hangover from
            # Fortran coordinate arrays.
            bonds.append((values[k] // 3, values[k + 1] // 3))
    return bonds


def connectivity_block(bonds: list, atoms: list) -> str:
    """
    The Gaussian connectivity section for the atoms that made it into the model.

    Written in the order the geometry block uses, with each bond listed once
    from the lower-numbered atom, which is the format Gaussian expects.
    """
    position = {atom["index"]: i + 1 for i, atom in enumerate(atoms)}
    neighbours = {i + 1: [] for i in range(len(atoms))}
    for i, j in bonds:
        a, b = position.get(i), position.get(j)
        if a and b:
            lo, hi = (a, b) if a < b else (b, a)
            neighbours[lo].append(hi)
    lines = []
    for index in range(1, len(atoms) + 1):
        partners = sorted(neighbours[index])
        lines.append(" " + str(index) + "".join(f" {p} 1.0" for p in partners))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian ONIOM input
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_oniom_input(layered: dict, boundary: list, method: str,
                         embedding: str = "electronic",
                         charge_high: int = None, multiplicity_high: int = 1,
                         charge_low: int = None, multiplicity_low: int = 1,
                         job: str = "opt", freeze_mm: bool = True,
                         processors: int = 8, memory_gb: int = 16,
                         title: str = "", bonds=None) -> str:
    """
    Write the ONIOM input.

    The geometry block carries everything: element, Amber type and partial
    charge for each atom, a freeze flag, coordinates, the layer, and on every
    MM atom at the boundary the link atom that replaces its QM partner. That
    single block is the whole QM/MM setup, which is why it is worth reading
    before submitting a week of compute to it.

    freeze_mm freezes the MM layer. For a first calculation that is almost
    always right: the QM region relaxes inside a protein held where the
    crystallographer put it, and the optimisation finishes this decade.
    """
    atoms = layered["atoms"]
    charge_high = (round(layered["charge_high"]) if charge_high is None
                   else charge_high)
    charge_total = (round(layered["charge_high"] + layered["charge_low"])
                    if charge_low is None else charge_low)

    embed = EMBEDDING.get(embedding, EMBEDDING["electronic"])[0]
    route = f"#p ONIOM({method}){embed} {job}"
    if bonds:
        route += " geom=connectivity"

    # Gaussian wants the charge/multiplicity of the real system, then of the
    # model system, twice over (low level on real, high on model, low on model).
    charges = (f"{charge_total} {multiplicity_low} "
               f"{charge_high} {multiplicity_high} "
               f"{charge_high} {multiplicity_high}")

    lines = [f"%NProcShared={processors}", f"%Mem={memory_gb}GB",
             "%Chk=oniom.chk", route, "",
             title or f"ONIOM {method} — {layered['high']} QM atoms, "
                      f"{layered['low']} MM atoms", "",
             charges]

    link_for = {}
    for crossing in boundary:
        # The link atom is declared on the MM atom, pointing at its QM partner.
        link_for[crossing["low"]] = crossing["high"] + 1

    for i, atom in enumerate(atoms):
        frozen = 0
        if freeze_mm and "opt" in job and atom["layer"] != HIGH:
            frozen = -1
        charge = f"{atom['charge']:.6f}".rstrip("0").rstrip(".")
        label = f"{atom['element']}-{atom['type']}-{charge}"
        row = (f" {label:<24}{frozen:>3} "
               f"{atom['x']:12.6f}{atom['y']:12.6f}{atom['z']:12.6f} "
               f"{atom['layer']}")
        if i in link_for:
            row += f" H-{LINK_TYPE}-{LINK_CHARGE} {link_for[i]}"
        lines.append(row)
    lines.append("")
    if bonds:
        lines.append(connectivity_block(bonds, atoms))
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def report(layered: dict, boundary: list) -> str:
    """The layering in words, for the panel and for a note beside the input."""
    lines = [f"{layered['high']} atoms in the QM layer, {layered['low']} in the MM "
             f"layer, {layered['dropped']} left out of the model."]
    lines.append(f"Layer charges from the force field: QM {layered['charge_high']:+.3f}, "
                 f"MM {layered['charge_low']:+.3f}, total "
                 f"{layered['charge_high'] + layered['charge_low']:+.3f}.")
    fraction = layered["charge_high"] - round(layered["charge_high"])
    if abs(fraction) > 0.1:
        lines.append(
            f"NOTE: the QM layer's force-field charge is {layered['charge_high']:+.3f}, "
            f"{abs(fraction):.2f} away from a whole number. Some of that is normal — "
            "a boundary through a polarised bond always splits a charge group — but "
            "a large fraction means the cut runs through something it should not, "
            "and the QM charge has to be an integer whatever the force field says. "
            f"The input uses {round(layered['charge_high']):+d}; confirm that is the "
            "charge the chemistry implies, counting the ligand.")
    total = layered["charge_high"] + layered["charge_low"]
    if abs(total - round(total)) > 0.1:
        lines.append(
            f"NOTE: the model's total charge is {total:+.3f}, not a whole number. "
            "Truncating the MM layer to a sphere cuts the system's neutrality with "
            "it; widen the sphere, or accept that the model carries a small net "
            "charge the full system does not.")
    if boundary:
        lines.append(f"{len(boundary)} bond(s) cross the boundary and become link "
                     "atoms: " + "; ".join(c["label"] for c in boundary[:8])
                     + (" …" if len(boundary) > 8 else ""))
    else:
        lines.append("No bonds cross the boundary — the QM layer is a separate "
                     "molecule, so no link atoms are needed.")
    return "\n".join(lines)
