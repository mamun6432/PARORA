# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : Rigid-body superposition of two PDB structures, returning a 4x4
#             transform for the NGL viewer instead of rewritten coordinates.
#
#             app.py keeps every structure's deposited coordinates and hands
#             NGL a per-component transform, so a superposition is reversible,
#             costs nothing to re-render, and never leaves the on-disk file
#             disagreeing with what MDAnalysis measured. The alternative --
#             writing a moved copy of the mobile structure -- is offered as an
#             explicit export (write_transformed) rather than a side effect.
#
#             Two ways of deciding which atoms correspond are supported.
#             "resnum" pairs residues that share a residue number, which is
#             right for two structures of the same protein (apo vs holo, two
#             point mutants). "sequence" pairs them through a BLOSUM62 global
#             alignment, which is what homologues with different numbering or
#             different lengths need. "auto" tries resnum and falls back to
#             sequence when the overlap is too small to trust.
# =============================================================================

import numpy as np

import sequence_utils as squ

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align as mda_align
    MDA_AVAILABLE = True
except Exception:
    mda = None
    mda_align = None
    MDA_AVAILABLE = False

try:
    from Bio.Align import PairwiseAligner, substitution_matrices
    BIO_AVAILABLE = True
except Exception:
    PairwiseAligner = None
    substitution_matrices = None
    BIO_AVAILABLE = False

# Below this many matched residues a fit is numerically meaningless -- three
# points define a plane, and any two structures can be made to agree on a
# handful of atoms.
MIN_MATCH = 4

# "auto" only trusts residue-number matching when it explains a decent share of
# the smaller structure; below that the numbering schemes probably disagree.
RESNUM_COVERAGE = 0.5

# ...and when the residues it paired up are actually the same residues. Two
# unrelated proteins both numbered from 1 match beautifully by number and not
# at all in sequence, which yields a confident, meaningless overlay.
RESNUM_IDENTITY = 0.5

# An RMSD worse than this is not a superposition anyone should read anything
# into, whatever the matching said.
POOR_FIT_RMSD = 5.0


def available():
    """True when a superposition can actually be computed."""
    return MDA_AVAILABLE


def methods():
    """Matching methods usable in this environment, best general choice first."""
    m = ["auto", "resnum"]
    if BIO_AVAILABLE:
        m.append("sequence")
    return m


# ── Residue extraction ───────────────────────────────────────────────────────

def _select(universe, selection):
    """
    Run a selection, retrying across MDAnalysis's two names for a PDB chain.

    Which of `segid` and `chainID` carries the chain letter depends on the
    file: MDAnalysis fills segid from the PDB's chain column for most entries,
    but files written with explicit SEGID records, and some mmCIF-derived
    entries, only populate chainID. A chain selection that works on one
    structure therefore silently matches nothing on another, so the failing
    keyword is swapped and retried before giving up.
    """
    try:
        ag = universe.select_atoms(selection)
        if len(ag):
            return ag
    except Exception:
        ag = None

    for a, b in (("segid", "chainID"), ("chainID", "segid")):
        if a in selection:
            try:
                alt = universe.select_atoms(selection.replace(a, b))
                if len(alt):
                    return alt
            except Exception:
                pass
    return ag if ag is not None else universe.select_atoms("bynum 0")


def _ca_residues(universe, selection):
    """
    One representative atom per residue for the given selection.

    Returns (atoms, keys, sequence) where atoms is an AtomGroup with exactly
    one atom per residue in structure order, keys are (chain, resnum, icode)
    tuples for residue-number matching, and sequence is the one-letter string
    for sequence alignment. Restricting to a single atom per residue is what
    makes the two matching strategies comparable: both produce a residue-level
    correspondence, and the fit then runs on the same set of points.

    The insertion code belongs in the key. Antibodies and the chymotrypsin
    numbering used by the serine proteases reuse a residue number with A/B/C
    suffixes -- 3PTB has 184, 184A, 188, 188A, 221, 221A -- so a key of
    (chain, resnum) alone silently pairs the wrong residues, which shows up as
    a structure failing to superpose onto itself at 0 Å.
    """
    ag = _select(universe, selection)
    if len(ag) == 0:
        return None, [], ""

    atoms, keys, seq = [], [], []
    for res in ag.residues:
        # select_atoms may return several atoms per residue; keep the CA (or,
        # for nucleic acids, C4') so the fit is one point per residue.
        in_sel = res.atoms & ag
        pick = None
        for name in ("CA", "C4'", "P"):
            hit = in_sel.select_atoms(f"name {name}")
            if len(hit) == 1:
                pick = hit[0]
                break
        if pick is None:
            if len(in_sel) == 0:
                continue
            pick = in_sel[0]
        chain = getattr(pick, "chainID", "") or getattr(res, "segid", "") or ""
        icode = (getattr(res, "icode", "") or "").strip()
        atoms.append(pick.index)
        keys.append((str(chain).strip(), int(res.resid), icode))
        seq.append(squ.one_letter(res.resname))

    return universe.atoms[atoms], keys, "".join(seq)


# ── Matching strategies ──────────────────────────────────────────────────────

def _pair_on(keys_mob, keys_ref, project):
    """
    Pair residues whose projected keys agree, one-to-one and in order.

    Each residue is consumed at most once: the n-th mobile residue carrying a
    key is paired with the n-th reference residue carrying it. A plain
    key -> index dict would instead collapse every repeat onto a single
    reference residue and quietly poison the fit.
    """
    ref_positions = {}
    for i, k in enumerate(keys_ref):
        ref_positions.setdefault(project(k), []).append(i)

    used, pairs = {}, []
    for i, k in enumerate(keys_mob):
        pk = project(k)
        bucket = ref_positions.get(pk)
        n = used.get(pk, 0)
        if bucket and n < len(bucket):
            pairs.append((i, bucket[n]))
            used[pk] = n + 1
    return pairs


def _pair_identity(pairs, seq_mob, seq_ref):
    """
    Fraction of paired residues that are the same amino acid.

    Residue-number matching never consults the sequence, so this is what tells
    a genuine correspondence (two crystal forms of one protein) from a
    coincidence of numbering (two unrelated proteins that both start at 1).
    """
    if not pairs:
        return None
    same = sum(1 for i, j in pairs if seq_mob[i] == seq_ref[j])
    return same / len(pairs)


def _match_by_resnum(keys_mob, keys_ref):
    """
    Pair residues sharing a residue number.

    Matching is tried with and without the chain identifier and the larger
    result wins: the same protein is routinely deposited as chain A in one
    entry and chain B in another, and insisting on the chain id would then
    match nothing at all.
    """
    strict = _pair_on(keys_mob, keys_ref, lambda k: k)
    loose = _pair_on(keys_mob, keys_ref, lambda k: (k[1], k[2]))
    if len(strict) >= len(loose):
        return strict, "chain, residue number and insertion code"
    return loose, "residue number (chain ids ignored)"


def _match_by_sequence(seq_mob, seq_ref):
    """
    Pair residues through a BLOSUM62 global alignment of the two sequences.

    Returns (pairs, identity) where identity is the fraction of aligned columns
    holding the same one-letter code -- the number to look at before believing
    an RMSD between two different proteins.

    The alignment is local (Smith-Waterman), not global. A global alignment
    with free end gaps is degenerate here: for two sequences that do not
    correspond, gapping everything out costs nothing and scores better than any
    real alignment, so it returns a couple of stray pairs and the fit collapses
    onto them. Local alignment instead finds the best-matching common stretch,
    which is exactly the region a superposition should be built on -- and when
    there is no such region the match comes back short and the caller reports
    it as untrustworthy.
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    alignment = aligner.align(seq_mob, seq_ref)[0]
    blocks_mob, blocks_ref = alignment.aligned

    pairs, identical = [], 0
    for (m0, m1), (r0, r1) in zip(blocks_mob, blocks_ref):
        for k in range(m1 - m0):
            i, j = m0 + k, r0 + k
            pairs.append((i, j))
            if seq_mob[i] == seq_ref[j]:
                identical += 1

    identity = identical / len(pairs) if pairs else 0.0
    return pairs, identity


# ── Fit ──────────────────────────────────────────────────────────────────────

def _fit_matrix(mob_xyz, ref_xyz):
    """
    Least-squares rigid-body fit of mob_xyz onto ref_xyz.

    Returns (matrix4x4, rmsd). The matrix maps a mobile coordinate p to
    R @ (p - centroid_mob) + centroid_ref, expressed as a homogeneous 4x4 in
    row-major (numpy) order.
    """
    com_mob = mob_xyz.mean(axis=0)
    com_ref = ref_xyz.mean(axis=0)
    rot, rmsd = mda_align.rotation_matrix(mob_xyz - com_mob, ref_xyz - com_ref)

    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = com_ref - rot @ com_mob
    return m, float(rmsd)


def _to_column_major(m):
    """Flatten a row-major 4x4 into the column-major list THREE.Matrix4 wants."""
    return [float(v) for v in np.asarray(m).T.reshape(-1)]


def superpose(mobile_path, reference_path,
              mobile_sel="protein", reference_sel="protein",
              method="auto"):
    """
    Superpose one structure onto another and return the transform to apply.

    Neither file is modified. The caller hands `matrix` straight to NGL's
    Component.setTransform, which moves the mobile structure into the
    reference's frame while both keep their deposited coordinates.

    Args:
        mobile_path   : Path to the .pdb that will be moved.
        reference_path: Path to the .pdb that stays put.
        mobile_sel    : MDAnalysis selection limiting the mobile fit atoms
                        (e.g. "protein", "protein and segid A").
        reference_sel : MDAnalysis selection limiting the reference fit atoms.
        method        : "auto", "resnum" or "sequence" -- how residues in the
                        two structures are paired up before fitting.

    Returns:
        dict with:
          ok       : bool
          message  : one-line human-readable summary
          matrix   : 16 floats, column-major, for THREE.Matrix4.fromArray
          rmsd     : Å, over the matched atoms
          n_atoms  : number of matched residue pairs used in the fit
          identity : sequence identity over aligned columns, or None
          method   : the matching strategy actually used
    """
    fail = {"ok": False, "matrix": None, "rmsd": None,
            "n_atoms": 0, "identity": None, "method": method}

    if not MDA_AVAILABLE:
        return dict(fail, message="MDAnalysis is unavailable — cannot superpose.")
    if method == "sequence" and not BIO_AVAILABLE:
        return dict(fail, message="Sequence matching needs Biopython, which is not installed.")

    try:
        u_mob = mda.Universe(str(mobile_path))
        u_ref = mda.Universe(str(reference_path))
    except Exception as e:
        return dict(fail, message=f"Could not read the structures: {e}")

    try:
        atoms_mob, keys_mob, seq_mob = _ca_residues(u_mob, mobile_sel)
        atoms_ref, keys_ref, seq_ref = _ca_residues(u_ref, reference_sel)
    except Exception as e:
        return dict(fail, message=f"Selection error: {e}")

    if atoms_mob is None or atoms_ref is None:
        return dict(fail, message="One of the selections matched no atoms — "
                                  "check the fit selections.")

    identity = None
    used = method

    if method in ("auto", "resnum"):
        pairs, how = _match_by_resnum(keys_mob, keys_ref)
        used = f"resnum ({how})"
        identity = _pair_identity(pairs, seq_mob, seq_ref)
        smaller = min(len(keys_mob), len(keys_ref))
        thin = smaller and (len(pairs) / smaller) < RESNUM_COVERAGE
        mismatched = identity is not None and identity < RESNUM_IDENTITY
        if (method == "auto" and BIO_AVAILABLE
                and (len(pairs) < MIN_MATCH or thin or mismatched)):
            pairs, identity = _match_by_sequence(seq_mob, seq_ref)
            used = "sequence alignment (BLOSUM62)"
    else:
        pairs, identity = _match_by_sequence(seq_mob, seq_ref)
        used = "sequence alignment (BLOSUM62)"

    if len(pairs) < MIN_MATCH:
        return dict(fail, method=used, message=(
            f"Only {len(pairs)} residues could be matched between the two "
            f"structures — too few to superpose. Try method='sequence', or "
            f"narrow the fit selections to the domains that correspond."))

    idx_mob = [p[0] for p in pairs]
    idx_ref = [p[1] for p in pairs]
    mob_xyz = atoms_mob.positions[idx_mob].astype(np.float64)
    ref_xyz = atoms_ref.positions[idx_ref].astype(np.float64)

    matrix, rmsd = _fit_matrix(mob_xyz, ref_xyz)

    msg = f"RMSD {rmsd:.2f} Å over {len(pairs)} residues, matched by {used}"
    if identity is not None:
        msg += f"; {identity * 100:.0f}% sequence identity"

    # A fit over a handful of residues, or between sequences with no real
    # relationship, always succeeds numerically and always looks plausible in
    # the viewer. Say so rather than reporting a bare RMSD.
    smaller = min(len(keys_mob), len(keys_ref))
    if rmsd > POOR_FIT_RMSD:
        msg += (" — that is a poor fit; these structures probably do not "
                "correspond, or the fit selections need narrowing")
    elif identity is not None and identity < 0.20:
        msg += " — low identity, so this is a weak structural match"
    elif smaller and len(pairs) / smaller < 0.25:
        msg += (f" — only {len(pairs)} of {smaller} residues matched, so the fit "
                f"rests on a small part of the structure")

    return {"ok": True, "message": msg, "matrix": _to_column_major(matrix),
            "rmsd": rmsd, "n_atoms": len(pairs), "identity": identity,
            "method": used}


def write_transformed(pdb_path, matrix_column_major, out_path):
    """
    Write a copy of a structure with a superposition transform baked in.

    The interactive viewer does not need this -- it transforms the component
    directly -- but anything that reads coordinates back off disk does, so the
    PyMOL ray tracer and any download of the aligned structure go through here.

    Args:
        pdb_path           : Source .pdb.
        matrix_column_major: 16 floats as returned by superpose().
        out_path           : Destination .pdb.

    Returns:
        (ok, message)
    """
    if not MDA_AVAILABLE:
        return False, "MDAnalysis is unavailable."
    try:
        m = np.asarray(matrix_column_major, dtype=np.float64).reshape(4, 4).T
        u = mda.Universe(str(pdb_path))
        xyz = u.atoms.positions.astype(np.float64)
        u.atoms.positions = (xyz @ m[:3, :3].T) + m[:3, 3]
        u.atoms.write(str(out_path))
        return True, str(out_path)
    except Exception as e:
        return False, f"Could not write the transformed structure: {e}"
