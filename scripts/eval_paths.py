"""Path validation shared by the command-line evaluation scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = PROJECT_ROOT / "eval"


def _inside_eval(path: Path) -> Path:
    lexical = path.absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"evaluation paths must not use symlinks: {current}")
    resolved = path.resolve()
    try:
        resolved.relative_to(EVAL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"evaluation path must stay inside {EVAL_ROOT}") from exc
    return resolved


def input_path(path: Path) -> Path:
    """Return an existing regular file or directory inside eval, following no escape symlinks."""
    resolved = _inside_eval(path)
    if not resolved.exists():
        raise ValueError(f"evaluation input does not exist: {resolved}")
    if not (resolved.is_file() or resolved.is_dir()):
        raise ValueError(f"evaluation input must be a file or directory: {resolved}")
    return resolved


def output_path(path: Path) -> Path:
    """Return a path under eval whose parent exists or can safely be created."""
    resolved = _inside_eval(path)
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"evaluation output must be a regular file: {resolved}")
    return resolved
