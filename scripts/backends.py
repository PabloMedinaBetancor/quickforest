"""The compile step is pluggable. A backend turns a `Forest` (qf_common.py: neutral, already parsed) into an artifact file and
loads it back as a predictor. To add a faster evaluator, write a class with this interface and register it; prune, check and
optimize need no change:

    class Backend:
        name: str                  # the --backend value
        suffix: str                # artifact file extension
        def compile(self, forest: Forest, out: Path, options: dict) -> dict:   # writes `out`; returns a small info dict
        def load(self, artifact: Path, **options) -> Predictor
    Predictor.predict(x: float32 (rows, features), threads: int | None = None) -> like xgboost's predict ((rows,) for one output, else
    (rows, classes)); `threads=None` means all usable cores (check.py
    passes no thread count unless one was asked for, so an older backend with `threads=1` still works); the bars in check.py (<= 1e-6, same decisions) apply to every backend. Optional: .close().

`--backend path/to/file.py:ClassName` loads a backend without registering it here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from backend_codegen import CodegenBackend, CodegenBlockBackend, CodegenPgoBackend
from backend_flat import FlatBackend
from backend_wave import WaveAvx2Backend, WaveBackend

BACKENDS = {
    b.name: b for b in (FlatBackend(), WaveBackend(), WaveAvx2Backend(), CodegenBackend(), CodegenBlockBackend(), CodegenPgoBackend())
}


def register(backend) -> None:
    BACKENDS[backend.name] = backend


def get(name: str):
    if name in BACKENDS:
        be = BACKENDS[name]
    else:
        path, _, cls = name.partition(":")
        # a name that is no built-in backend and no .py path (a typo like codegen_pgo) is refused below, not imported
        if not path.endswith(".py"):
            raise ValueError(f"unknown backend {name!r}: one of {sorted(BACKENDS)}, or a file.py[:Class]")
        if not Path(path).is_file():
            raise ValueError(f"unknown backend {name!r}: {path} is not a file")
        spec = importlib.util.spec_from_file_location("qf_backend", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        be = getattr(mod, cls or "Backend")()
    return be


def load_artifact(artifact, **options):
    """Load a compiled artifact: the backend named in `ARTIFACT.choice.json` (written by `compile --backend auto`), else by suffix."""
    import json

    import qf_common as qf

    artifact = Path(artifact)
    choice = Path(f"{artifact}.choice.json")
    if choice.is_file():
        with qf.refusing_damaged(choice, "choice file"):
            try:
                name = json.loads(choice.read_text())["backend"]
            except (KeyError, TypeError, IndexError) as e:
                raise ValueError(f'{choice}: not a choice file (needs a "backend" entry: {type(e).__name__})') from e
        if not isinstance(name, str) or name not in BACKENDS:
            # a choice file is data: it may name only a built-in backend, never a file.py to execute
            raise ValueError(f"{choice}: unknown backend {name!r}: one of {sorted(BACKENDS)}")
        with qf.refusing_damaged(artifact, "compiled artifact"):
            return get(name).load(artifact, **options)
    with qf.refusing_damaged(artifact, "compiled artifact"):
        return (get("codegen-block") if artifact.suffix == ".so" else get("flat")).load(artifact, **options)
