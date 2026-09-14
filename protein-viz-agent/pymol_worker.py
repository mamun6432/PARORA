# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : Headless PyMOL ray-tracing worker. Runs inside a PyMOL-capable
#             Python interpreter (NOT the Streamlit env), reads a JSON scene
#             spec on stdin, rebuilds the NGL.js representation stack as
#             independent PyMOL objects, ray traces, and writes a PNG.
#
#             Invoked as a subprocess by pymol_render.py so that PyMOL never
#             has to be installed alongside Streamlit. Communicates results
#             back on stdout via a single RESULT_MARKER line, because PyMOL
#             writes its own chatter to stdout and cannot be fully silenced.
#
#             Target interpreter is Python 3.10 (PyMOL.app bundles conda-forge
#             3.10), so avoid 3.11+ syntax in this file.
# =============================================================================

import json
import sys

RESULT_MARKER = "__PYMOL_RENDER_RESULT__"

# ── NGL colour scheme → colour applied per element symbol ─────────────────────
# NGL's "element" scheme leaves carbon at the model colour and tints the rest.
ELEMENT_COLORS = {
    "N": "blue", "O": "red", "S": "yellow", "P": "orange",
    "F": "palegreen", "CL": "green", "BR": "firebrick", "I": "purple",
    "FE": "orange", "CA": "forest", "ZN": "grey50", "MG": "forest",
    "NA": "purple", "K": "purple", "MN": "salmon", "CU": "chocolate",
    "H": "white",
}

# Palette cycled over chains for the NGL "chainname" scheme.
CHAIN_PALETTE = [
    "skyblue", "salmon", "palegreen", "yellow", "violet",
    "orange", "slate", "lime", "hotpink", "cyan",
]

# ── NGL representation type → PyMOL show-command(s) ──────────────────────────
# Values are lists because some NGL reps are a composite in PyMOL.
REP_MAP = {
    "cartoon":     ["cartoon"],
    "surface":     ["surface"],
    "ball+stick":  ["sticks", "spheres"],
    "licorice":    ["sticks"],
    "spacefill":   ["spheres"],
    "line":        ["lines"],
    "point":       ["nonbonded"],
    "ribbon":      ["ribbon"],
    "backbone":    ["sticks"],
    "tube":        ["cartoon"],
    "rope":        ["cartoon"],
    "trace":       ["ribbon"],
    "hyperball":   ["sticks", "spheres"],
    "rocket":      ["cartoon"],
    "base":        ["sticks"],
    "helixorient": ["ribbon"],
    # NGL-only scene furniture with no PyMOL equivalent; drawn as nothing
    # rather than silently falling back to a cartoon of the whole selection.
    "axes":        [],
    "unitcell":    [],
}

# NGL cartoon variants that map onto a PyMOL cartoon sub-style.
CARTOON_STYLE = {
    "tube": "tube", "rope": "tube", "rocket": "automatic", "cartoon": None,
}

# Per-rep transparency is a different setting name for every PyMOL rep type.
TRANSPARENCY_SETTING = {
    "surface": "transparency",
    "cartoon": "cartoon_transparency",
    "sticks":  "stick_transparency",
    "spheres": "sphere_transparency",
    "ribbon":  "ribbon_transparency",
}


def ngl_to_pymol_selection(ngl):
    """
    Translate an NGL.js selection string into PyMOL selection algebra.

    Covers the full vocabulary emitted by app.py's _expression_to_ngl:
    keyword selections, ":A" chain syntax, "@1,2,3" atom-serial lists,
    "_C" element syntax, and bare / or-joined residue names.

    PyMOL's `id` selector matches the PDB atom serial number, which is exactly
    what NGL's `@serial` syntax carries, so serial lists round-trip exactly.

    Args:
        ngl: NGL selection string.

    Returns:
        Equivalent PyMOL selection string.
    """
    s = (ngl or "").strip()
    if not s:
        return "none"

    keywords = {
        "protein":  "polymer.protein",
        "nucleic":  "polymer.nucleic",
        "polymer":  "polymer",
        "ligand":   "organic and not polymer",
        "organic":  "organic",
        "water":    "solvent",
        "hetero":   "not polymer",
        "backbone": "polymer and name N+CA+C+O",
        "sidechain": "sidechain",
        "helix":    "ss H",
        "sheet":    "ss S",
        # PyMOL's dss assigns only H and S; loops are left blank, so "ss L"
        # matches nothing. Define turn as polymer that is neither helix nor sheet.
        "turn":     "polymer and not (ss H or ss S)",
        "all":      "all",
        "none":     "none",
    }
    if s.lower() in keywords:
        return keywords[s.lower()]

    # "@1,2,3" → atom serial list. Chunked so a multi-thousand-atom B-factor
    # selection does not produce one pathological selection expression.
    if s.startswith("@"):
        raw = [x.strip() for x in s[1:].split(",") if x.strip()]
        serials = [x for x in raw if x.isdigit()]
        if not serials:
            return "none"
        chunks = ["id " + "+".join(serials[i:i + 250])
                  for i in range(0, len(serials), 250)]
        return "(" + " or ".join(chunks) + ")"

    # ":A" → chain A
    if s.startswith(":") and len(s) > 1:
        return "chain " + s[1:]

    # Residue numbers and ranges, optionally chain-qualified, as produced by
    # the sequence browser: "74-80:A", "95:A", "74-80", or several joined by
    # "or". PyMOL spells these "resi 74-80 and chain A".
    resno_terms = [t.strip() for t in s.split(" or ")]
    converted = []
    for term in resno_terms:
        body, _, chain = term.partition(":")
        body = body.strip()
        chain = chain.strip()
        lo, dash, hi = body.partition("-")
        ok = (lo.strip().isdigit() and
              (not dash or hi.strip().isdigit()))
        if not ok:
            converted = []
            break
        resi = "resi %s" % (("%s-%s" % (lo.strip(), hi.strip())) if dash
                            else lo.strip())
        converted.append("(%s and chain %s)" % (resi, chain) if chain else "(%s)" % resi)
    if converted:
        return " or ".join(converted)

    # "_C" → element C
    if s.startswith("_") and len(s) > 1:
        return "elem " + s[1:]

    # "ATP" or "ATP or NAG or HEM" → residue names
    parts = [p.strip() for p in s.split(" or ")]
    if parts and all(p.replace("'", "").isalnum() for p in parts):
        return "resn " + "+".join(p.upper() for p in parts)

    # Unrecognised — hand it to PyMOL verbatim and let the caller see any error.
    return s


def is_valid_color(cmd, name):
    """
    True if PyMOL can resolve `name` to a colour.

    cmd.get_color_indices() only enumerates the ~178 menu colours and omits the
    numbered greys, so membership in that list is not a validity test.
    get_color_index() resolves the full table and returns -1 when unknown.
    """
    try:
        return cmd.get_color_index(name) >= 0
    except Exception:
        return False


def apply_color(cmd, obj, color, warnings):
    """
    Apply an NGL colour scheme or named colour to one PyMOL object.

    NGL's named schemes (element / spectrum / chainname / bfactor) have no
    direct PyMOL equivalent, so each is reproduced explicitly rather than
    delegated to pymol.util, which binds to the global cmd singleton and is
    unsafe inside a pymol2 instance.
    """
    c = (color or "element").lower()

    if c == "element":
        cmd.color("grey80", obj)
        for elem, col in ELEMENT_COLORS.items():
            cmd.color(col, "%s and elem %s" % (obj, elem))
        return

    if c in ("spectrum", "residueindex"):
        # Prefer per-residue banding on CA; fall back to all atoms for ligands.
        if cmd.count_atoms("%s and name CA" % obj) > 1:
            cmd.spectrum("count", "rainbow", "%s and name CA" % obj)
            cmd.set("cartoon_color", -1, obj)
        else:
            cmd.spectrum("count", "rainbow", obj)
        return

    if c in ("chainname", "chain"):
        chains = [ch for ch in cmd.get_chains(obj) if ch]
        if not chains:
            cmd.color("grey80", obj)
            return
        for i, ch in enumerate(chains):
            cmd.color(CHAIN_PALETTE[i % len(CHAIN_PALETTE)],
                      "%s and chain %s" % (obj, ch))
        return

    if c in ("bfactor", "b"):
        cmd.spectrum("b", "blue_white_red", obj)
        return

    if c == "sstruc":
        # Match NGL's secondary-structure palette closely enough to be readable.
        cmd.color("grey70", obj)
        cmd.color("red", "%s and ss H" % obj)
        cmd.color("yellow", "%s and ss S" % obj)
        return

    if c == "hydrophobicity":
        # Kyte-Doolittle poles: hydrophobic warm, hydrophilic cool.
        cmd.color("white", obj)
        cmd.color("orange", "%s and resn ALA+VAL+LEU+ILE+PHE+MET+TRP+CYS" % obj)
        cmd.color("skyblue", "%s and resn ARG+LYS+ASP+GLU+ASN+GLN+HIS" % obj)
        return

    if c in ("atomindex", "occupancy", "random", "chainindex"):
        spectrum_expr = {"atomindex": "count", "occupancy": "q",
                         "random": "count", "chainindex": "count"}[c]
        cmd.spectrum(spectrum_expr, "rainbow", obj)
        return

    if c == "resname":
        cmd.spectrum("count", "rainbow", "%s and name CA" % obj)
        return

    if is_valid_color(cmd, c):
        cmd.color(c, obj)
        return

    warnings.append("Unknown colour '%s' - fell back to grey80." % color)
    cmd.color("grey80", obj)


def build_and_render(spec):
    """
    Build the scene described by `spec` in a fresh PyMOL instance and ray trace it.

    Each NGL representation becomes its own PyMOL object via cmd.create(). This
    mirrors NGL's layered model and is what makes per-representation
    transparency possible at all -- PyMOL's transparency settings are
    object-scoped, not selection-scoped.

    Returns:
        Result dict with ok / out_path / warnings, or ok=False and an error.
    """
    import pymol2

    warnings = []
    p = pymol2.PyMOL()
    p.start()
    cmd = p.cmd

    try:
        cmd.feedback("disable", "all", "everything")

        # ── Load the structure ────────────────────────────────────────────────
        pdb_path = spec.get("pdb_path")
        pdb_id = spec.get("pdb_id") or "structure"
        if pdb_path:
            cmd.load(pdb_path, "master")
        else:
            cmd.fetch(pdb_id, "master", type="pdb",
                      path=spec.get("fetch_dir", "."), quiet=1)
        if cmd.count_atoms("master") == 0:
            return {"ok": False, "error": "Structure loaded but contains no atoms."}

        cmd.hide("everything", "master")
        cmd.remove("master and hydro" if spec.get("strip_hydrogens") else "none")

        # ── Global quality / style settings ───────────────────────────────────
        bg = spec.get("background", "black")
        cmd.bg_color("white" if bg == "white" else "black")
        cmd.set("ray_opaque_background", 1)

        # Surface quality and ambient occlusion dominate ray-trace cost: a
        # transparent surface over 3PP0 costs ~175 s at publication quality
        # versus a few seconds at draft. Draft is the interactive default.
        quality = str(spec.get("quality", "draft")).lower()
        publication = quality in ("publication", "high", "final")
        cmd.set("antialias", 2 if publication else 1)
        cmd.set("surface_quality", 1 if publication else 0)
        cmd.set("ray_shadows", 1 if spec.get("shadows") else 0)
        cmd.set("ambient_occlusion_mode",
                1 if (publication and spec.get("ambient_occlusion")) else 0)
        cmd.set("cartoon_sampling", 14 if publication else 7)
        cmd.set("ribbon_sampling", 10 if publication else 1)
        cmd.set("cartoon_fancy_helices", 1 if publication else 0)
        cmd.set("cartoon_highlight_color", "grey50")
        cmd.set("stick_radius", 0.15)
        cmd.set("valence", 0)
        cmd.set("specular", 0.25)

        # ── One PyMOL object per NGL representation layer ─────────────────────
        reps = spec.get("representations") or []
        built = 0
        for i, rep in enumerate(reps):
            rep_type = str(rep.get("type", "cartoon")).lower()
            ngl_sel = rep.get("selection", "all")
            pm_sel = ngl_to_pymol_selection(ngl_sel)
            obj = "rep%02d_%s" % (i, rep_type.replace("+", "_"))

            try:
                n = cmd.count_atoms("master and (%s)" % pm_sel)
            except Exception as e:
                warnings.append("Rep %d: selection '%s' rejected by PyMOL (%s)."
                                % (i, ngl_sel, e))
                continue
            if n == 0:
                warnings.append("Rep %d: selection '%s' matched 0 atoms - skipped."
                                % (i, ngl_sel))
                continue

            cmd.create(obj, "master and (%s)" % pm_sel)
            cmd.hide("everything", obj)

            shows = REP_MAP.get(rep_type, ["cartoon"])
            if not shows:
                warnings.append("Rep %d: '%s' has no PyMOL equivalent - skipped."
                                % (i, rep_type))
                cmd.delete(obj)
                continue
            if rep_type == "backbone":
                cmd.show("sticks", "%s and name N+CA+C+O" % obj)
            else:
                for sh in shows:
                    cmd.show(sh, obj)

            if rep_type == "ball+stick":
                cmd.set("sphere_scale", 0.25, obj)
            style = CARTOON_STYLE.get(rep_type)
            if style:
                cmd.cartoon(style, obj)
            if rep_type == "cartoon":
                # Cartoon needs a trace for CA-only or nucleic models.
                cmd.set("cartoon_trace_atoms", 0, obj)

            apply_color(cmd, obj, rep.get("color", "element"), warnings)

            transparency = float(rep.get("transparency", 0.0) or 0.0)
            if transparency > 0:
                for sh in shows:
                    setting = TRANSPARENCY_SETTING.get(sh)
                    if setting:
                        cmd.set(setting, transparency, obj)

            built += 1

        if built == 0:
            # Nothing survived translation - render the polymer so the user
            # gets a usable image instead of an empty frame.
            cmd.create("rep_fallback", "master and polymer")
            cmd.show("cartoon", "rep_fallback")
            cmd.spectrum("count", "rainbow", "rep_fallback and name CA")
            warnings.append("No representation translated cleanly; "
                            "rendered a default cartoon instead.")

        cmd.delete("master")

        # ── Camera ────────────────────────────────────────────────────────────
        target = spec.get("camera_target")
        if target:
            pm_target = ngl_to_pymol_selection(target)
            try:
                if cmd.count_atoms(pm_target) > 0:
                    cmd.orient(pm_target)
                    cmd.zoom(pm_target, 3.0)
                else:
                    cmd.orient()
                    cmd.zoom("all", 2.0)
            except Exception:
                cmd.orient()
                cmd.zoom("all", 2.0)
        else:
            cmd.orient()
            cmd.zoom("all", 2.0)

        # ── Ray trace ─────────────────────────────────────────────────────────
        width = int(spec.get("width", 1600))
        height = int(spec.get("height", 1200))
        out_path = spec["out_path"]

        cmd.ray(width, height)
        cmd.png(out_path, width=width, height=height,
                dpi=int(spec.get("dpi", 300)), ray=0)

        return {"ok": True, "out_path": out_path,
                "n_reps": built, "warnings": warnings}

    finally:
        try:
            p.stop()
        except Exception:
            pass


def main():
    try:
        spec = json.loads(sys.stdin.read())
    except Exception as e:
        print(RESULT_MARKER + json.dumps({"ok": False,
                                          "error": "Bad spec JSON: %s" % e}))
        return

    try:
        result = build_and_render(spec)
    except Exception as e:
        import traceback
        result = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

    sys.stdout.flush()
    print(RESULT_MARKER + json.dumps(result))


if __name__ == "__main__":
    main()
