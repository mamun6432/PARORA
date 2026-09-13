# =============================================================================
# Developer : Methun Kamruzzaman, Abdullah Al Mamun
# Date      : 2026-09-11
# Summary   : Streamlit-side bridge to headless PyMOL ray tracing.
<<<<<<< Updated upstream
#
#             PyMOL cannot be installed into the anaconda base env without a
#             ~106-package channel migration that would relink conda's own
#             libarchive/libsolv and Jupyter's zeromq. Instead this module
#             drives an already-working PyMOL interpreter (PyMOL.app's bundled
#             conda-forge Python) as a subprocess, passing the scene as JSON.
#
#             Nothing here imports pymol, so it is safe to import from app.py
#             regardless of what is installed in the Streamlit environment.
#             Subprocess start-up (~1-2 s) is negligible against ray-trace time.
=======
#             PyMOL cannot be installed into the anaconda base env without a
#             ~106-package channel migration that would relink conda's own
#             libarchive/libsolv and Jupyter's zeromq. Instead this module
#             drives a PyMOL-capable interpreter (a conda env holding
#             pymol-open-source) as a subprocess, passing the scene as JSON.
>>>>>>> Stashed changes
# =============================================================================

import json
import os
<<<<<<< Updated upstream
import shutil
=======
>>>>>>> Stashed changes
import subprocess
import sys
import time
from pathlib import Path

RESULT_MARKER = "__PYMOL_RENDER_RESULT__"
WORKER = Path(__file__).parent / "pymol_worker.py"

# Conda prefixes searched for an env containing open-source PyMOL.
CONDA_ROOTS = [
    "/opt/anaconda3/envs", "/opt/miniconda3/envs",
    os.path.expanduser("~/anaconda3/envs"), os.path.expanduser("~/miniconda3/envs"),
    os.path.expanduser("~/mambaforge/envs"), os.path.expanduser("~/miniforge3/envs"),
]

# Bundled PyMOL applications, tried only after conda envs. These ship the
# commercial "incentive" build, which burns a "For Evaluation Only" watermark
# into every ray-traced image once its licence lapses -- so open-source PyMOL
# in a conda env is always preferred when one is present.
BUNDLE_INTERPRETERS = [
    "/Applications/PyMOL.app/Contents/bin/python",
    "/Applications/PyMOL.app/Contents/bin/python3",
    "/opt/sbgrid/x86_64-linux/pymol/current/bin/python",
]

<<<<<<< Updated upstream
=======
_cached_interpreter = None

>>>>>>> Stashed changes

def _conda_candidates():
    """Interpreters from conda envs whose name mentions pymol, best first."""
    found = []
    for root in CONDA_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if "pymol" in name.lower():
                exe = os.path.join(root, name, "bin", "python")
                if os.path.exists(exe):
                    found.append(exe)
    return found

<<<<<<< Updated upstream
_cached_interpreter = None

=======
>>>>>>> Stashed changes

def find_pymol_python(force_rescan: bool = False):
    """
    Locate a Python interpreter that can import pymol2.

    Resolution order: the PYMOL_PYTHON env var, the interpreter running this
    process (in case PyMOL was later installed alongside Streamlit), any conda
    env whose name mentions pymol, and finally bundled PyMOL applications.

    Conda envs outrank bundles deliberately: bundles are the commercial build
    and watermark every render once their licence expires.

    Args:
        force_rescan: Ignore the cached result and probe again.

    Returns:
        Path to a working interpreter, or None if none can import pymol2.
    """
    global _cached_interpreter
    if _cached_interpreter and not force_rescan:
        return _cached_interpreter

    candidates = []
    env_python = os.getenv("PYMOL_PYTHON")
    if env_python:
        candidates.append(env_python)
    candidates.append(sys.executable)
    candidates.extend(_conda_candidates())
    candidates.extend(BUNDLE_INTERPRETERS)

    for exe in candidates:
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run(
                [exe, "-c", "import pymol2; print('ok')"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0 and "ok" in r.stdout:
                _cached_interpreter = exe
                return exe
        except Exception:
            continue
    return None


def pymol_available() -> bool:
    """True if a PyMOL-capable interpreter was found on this machine."""
    return find_pymol_python() is not None


def render_scene(
    pdb_path,
    pdb_id,
    representations,
    background="black",
    camera_target=None,
    width=1600,
    height=1200,
    dpi=300,
    quality="draft",
    shadows=False,
    ambient_occlusion=True,
    out_dir="renders",
    timeout=300,
):
    """
    Ray trace the current NGL scene with PyMOL and return the resulting PNG.

    The representation list is the same structure app.py keeps in
    st.session_state.representations -- NGL types, NGL selection strings and
    NGL colour schemes -- so no caller-side translation is needed. The worker
    handles the NGL -> PyMOL mapping.

    Args:
        pdb_path       : Local .pdb path; if None the worker fetches by pdb_id.
        pdb_id         : PDB accession, used for fetching and output naming.
        representations: List of {type, selection, color, transparency}.
        background     : "black" or "white".
        camera_target  : NGL selection to orient and zoom on, or None for all.
        width, height  : Ray-trace resolution in pixels.
        dpi            : PNG metadata DPI for publication output.
        quality        : "draft" for interactive use, "publication" for final
                         figures. Publication enables surface_quality, finer
                         cartoon sampling and ambient occlusion, which together
<<<<<<< Updated upstream
                         cost roughly two orders of magnitude more ray-trace
=======
                         cost roughly an order of magnitude more ray-trace
>>>>>>> Stashed changes
                         time on a transparent surface.
        shadows        : Enable ray-traced shadows.
        ambient_occlusion: Enable ambient occlusion shading.
        out_dir        : Directory for rendered PNGs.
        timeout        : Seconds before the render subprocess is killed.

    Returns:
        (ok, message, png_path). png_path is None when ok is False.
    """
    exe = find_pymol_python()
    if exe is None:
        return (False,
                "No PyMOL-capable Python found. Install PyMOL or set "
                "PYMOL_PYTHON to an interpreter that can `import pymol2`.",
                None)

    if not WORKER.exists():
        return (False, "Renderer backend missing: %s" % WORKER, None)

    if not representations:
        return (False, "Nothing to render - no active representations.", None)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_root / ("%s_%s.png" % (pdb_id or "structure", stamp))

    spec = {
        "pdb_path": str(pdb_path) if pdb_path else None,
        "pdb_id": pdb_id,
        "representations": representations,
        "background": background,
        "camera_target": camera_target,
        "width": int(width),
        "height": int(height),
        "dpi": int(dpi),
        "quality": quality,
        "shadows": bool(shadows),
        "ambient_occlusion": bool(ambient_occlusion),
        "out_path": str(out_path.resolve()),
        "fetch_dir": str(out_root.resolve()),
    }

    try:
        proc = subprocess.run(
            [exe, str(WORKER)],
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (False,
                "PyMOL render timed out after %ds. Try a lower resolution or "
                "turn off ambient occlusion." % timeout,
                None)

    # PyMOL writes its own banner and warnings to stdout, so the result is
    # recovered from the last marker-prefixed line rather than the whole stream.
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                result = json.loads(line[len(RESULT_MARKER):])
            except Exception:
                pass

    if result is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        return (False,
                "PyMOL worker returned no result (exit %d). %s"
                % (proc.returncode, " | ".join(tail)),
                None)

    if not result.get("ok"):
        return (False, "PyMOL render failed: %s" % result.get("error", "unknown"), None)

    png = Path(result["out_path"])
    if not png.exists():
        return (False, "PyMOL reported success but %s was not written." % png, None)

    msg = "Ray-traced %d representation(s) at %dx%d (%s quality)." % (
        result.get("n_reps", 0), width, height, quality)
    warnings = result.get("warnings") or []
    if warnings:
        msg += " Warnings: " + " ".join(warnings)
    return (True, msg, str(png))


def describe_backend() -> str:
    """One-line summary of which interpreter will be used, for the UI/debug log."""
    exe = find_pymol_python()
    if exe is None:
        return "PyMOL backend: not found"
    if exe == sys.executable:
        return "PyMOL backend: in-process interpreter (%s)" % exe
    return "PyMOL backend: subprocess -> %s" % exe
