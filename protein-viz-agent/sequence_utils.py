# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : Dependency-free PDB sequence extraction and residue-selection
#             helpers for the sequence browser in app.py.
#
#             Deliberately parses ATOM/HETATM records by column rather than
#             using MDAnalysis or Biopython: the sequence panel must work even
#             when MDAnalysis fails to import, and only *observed* residues are
#             selectable. SEQRES is intentionally ignored -- it lists the full
#             construct with its own numbering, which frequently disagrees with
#             the deposited coordinates (3PP0's SEQRES starts at MET 1 while
#             its first observed residue is ALA 706), so selections built from
#             SEQRES would silently point at the wrong residues.
# =============================================================================

# Bumped whenever parse_structure_residues changes the shape of what it
# returns, so callers caching its output can key on it and not serve results
# built by an older version of this module.
SCHEMA_VERSION = 1

# Three-letter → one-letter for the standard amino acids, plus the modified
# residues common enough in the PDB to be worth showing as their parent.
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Frequent modified / alternate protonation states
    "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H",
    "CSO": "C", "PTR": "Y", "SEP": "S", "TPO": "T", "KCX": "K", "MLY": "K",
}

NA3_TO_1 = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U", "DI": "I",
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "I", "T": "T",
}

WATER_NAMES = {"HOH", "WAT", "TIP", "TIP3", "H2O", "DOD", "SOL"}


def classify_residue(resname):
    """Return 'protein', 'nucleic', 'water' or 'hetero' for a residue name."""
    rn = resname.strip().upper()
    if rn in WATER_NAMES:
        return "water"
    if rn in AA3_TO_1:
        return "protein"
    if rn in NA3_TO_1:
        return "nucleic"
    return "hetero"


def one_letter(resname):
    """One-letter code for a residue, or 'X' for anything non-standard."""
    rn = resname.strip().upper()
    if rn in AA3_TO_1:
        return AA3_TO_1[rn]
    if rn in NA3_TO_1:
        return NA3_TO_1[rn]
    return "X"


def parse_structure_residues(pdb_path):
    """
    Extract the observed residues of a PDB file, grouped by chain.

    Parses by fixed column positions (the PDB format is column-oriented, not
    whitespace-delimited) and keeps residues in file order, de-duplicated by
    (chain, resseq, icode) so that multi-atom and altloc records collapse to
    one entry per residue.

    Args:
        pdb_path: Path to a .pdb file.

    Returns:
        dict mapping chain id -> list of residue dicts with keys
        resseq (int), icode (str), resname (str), one (str), kind (str).
        Chains are ordered by first appearance.
    """
    chains = {}
    seen = set()

    with open(pdb_path, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            # Column positions per the PDB v3.3 specification.
            resname = line[17:20].strip()
            chain = line[21].strip() or "_"
            raw_seq = line[22:26].strip()
            icode = line[26].strip()
            if not raw_seq:
                continue
            try:
                resseq = int(raw_seq)
            except ValueError:
                continue

            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            chains.setdefault(chain, []).append({
                "resseq": resseq,
                "icode": icode,
                "resname": resname,
                "one": one_letter(resname),
                "kind": classify_residue(resname),
            })

    return chains


def chain_summary(residues):
    """Count residues by class for one chain, for the chain picker labels."""
    counts = {"protein": 0, "nucleic": 0, "hetero": 0, "water": 0}
    for r in residues:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    return counts


def format_sequence_lines(residues, per_line=50, group=10):
    """
    Lay the polymer sequence out as numbered fixed-width lines.

    The leading number is the residue number of the first residue on that line
    (the real deposited number, which may not start at 1), so a residue can be
    located by counting along the row.

    Args:
        residues: Residue dicts for one chain, as returned by
                  parse_structure_residues.
        per_line: Residues per output line.
        group   : Insert a space every `group` residues for readability.

    Returns:
        List of formatted strings, one per line.
    """
    polymer = [r for r in residues if r["kind"] in ("protein", "nucleic")]
    if not polymer:
        return []

    lines = []
    for start in range(0, len(polymer), per_line):
        chunk = polymer[start:start + per_line]
        letters = ""
        for i, r in enumerate(chunk):
            if i and i % group == 0:
                letters += " "
            letters += r["one"]
        lines.append("%6d  %s" % (chunk[0]["resseq"], letters))
    return lines


def parse_residue_spec(spec):
    """
    Parse a residue specification such as "74-80, 95, 100-110" into ranges.

    Args:
        spec: Comma/space separated numbers and inclusive ranges.

    Returns:
        (ranges, errors) where ranges is a list of (start, end) integer tuples
        with start <= end, and errors lists the tokens that could not be parsed.
    """
    ranges = []
    errors = []
    for token in spec.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:                      # leave a leading minus alone
            head, _, tail = token.partition("-")
            try:
                lo, hi = int(head), int(tail)
            except ValueError:
                errors.append(token)
                continue
            ranges.append((min(lo, hi), max(lo, hi)))
        else:
            try:
                n = int(token)
            except ValueError:
                errors.append(token)
                continue
            ranges.append((n, n))
    return ranges, errors


def ranges_to_ngl(ranges, chain=None):
    """
    Build an NGL selection string from residue ranges and an optional chain.

    NGL writes a residue range with its chain as "74-80:A"; several such terms
    are combined with "or".

    Args:
        ranges: List of (start, end) inclusive tuples.
        chain : Chain id, or None / "_" for no chain restriction.

    Returns:
        NGL selection string, or "none" when no ranges are supplied.
    """
    if not ranges:
        return "none"
    suffix = ":%s" % chain if chain and chain != "_" else ""
    terms = []
    for lo, hi in ranges:
        terms.append(("%d%s" % (lo, suffix)) if lo == hi
                     else ("%d-%d%s" % (lo, hi, suffix)))
    return " or ".join(terms)


def residues_in_ranges(residues, ranges):
    """Return the residues whose numbers fall inside any of the given ranges."""
    if not ranges:
        return []
    return [r for r in residues
            if any(lo <= r["resseq"] <= hi for lo, hi in ranges)]
