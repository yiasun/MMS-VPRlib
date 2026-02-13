#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py - Unified launcher (Time-Series-Library style) for multiple model scripts.

Assumed repo layout (recommended):
  repo_root/
    run.py
    models/
      blip_classifier.py
      clip_classifier.py
      ...
    scripts/
      classification/<dataset>/*.sh

Features:
- Discover runnable scripts primarily in ./models, with fallback to repo root.
- Model key mapping:
    blip_classifier.py      -> blip
    clip_classifier.py      -> clip
    resnet_bert_fusion.py   -> resnet_bert
    vit_textplugin.py       -> vit
    LR.py                   -> lr (special)
- Runs via module.main(args) if script provides parse_args() + main(args), else subprocess fallback.
- Supports both:
    python run.py blip --data_dir ...
    python run.py --model blip --data_dir ...
"""

import argparse
import re
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_PATTERNS = (
    r".*_classifier\.py$",
    r".*_fusion\.py$",
    r".*_textplugin\.py$",
)

SPECIAL_KEYS = {
    "LR.py": "lr",  # handle uppercase LR.py -> key "lr"
}


def _is_target_script(name: str) -> bool:
    return any(re.match(pat, name, flags=re.IGNORECASE) for pat in SCRIPT_PATTERNS) or name in SPECIAL_KEYS


def _key_from_filename(filename: str) -> str:
    if filename in SPECIAL_KEYS:
        return SPECIAL_KEYS[filename]
    stem = Path(filename).stem
    stem = re.sub(r"_(classifier|fusion|textplugin)$", "", stem, flags=re.IGNORECASE)
    return stem.lower()


def discover_scripts(folder: Path) -> Dict[str, Path]:
    """
    Discover candidate python scripts in `folder` and map them to model keys.
    Returns: key -> script_path
    """
    scripts: Dict[str, Path] = {}
    if not folder.exists() or not folder.is_dir():
        return scripts

    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".py":
            continue

        name = p.name
        if not _is_target_script(name):
            continue

        key = _key_from_filename(name)

        # If multiple map to same key, prefer *_classifier.py over *_fusion/_textplugin
        if key in scripts:
            existing = scripts[key].name.lower()
            cand = name.lower()
            if ("classifier" in cand) and ("classifier" not in existing):
                scripts[key] = p
        else:
            scripts[key] = p

    return scripts


def merge_script_maps(primary: Dict[str, Path], secondary: Dict[str, Path]) -> Dict[str, Path]:
    """
    Merge two key->path maps.
    If a key exists in primary, keep it; else take from secondary.
    """
    out = dict(primary)
    for k, v in secondary.items():
        if k not in out:
            out[k] = v
    return out


def import_module_from_path(path: Path):
    """
    Import a module from a file path WITHOUT needing it to be a package.
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def run_via_main(script_path: Path, argv_tail: List[str]) -> int:
    """
    Try calling script.main(args) if available.
    - If script has parse_args() + main(args): reuse its own arg parser by patching sys.argv.
    - Otherwise: fall back to subprocess.
    """
    mod = import_module_from_path(script_path)

    if hasattr(mod, "main") and callable(getattr(mod, "main")):
        if hasattr(mod, "parse_args") and callable(getattr(mod, "parse_args")):
            old_argv = sys.argv[:]
            try:
                sys.argv = [str(script_path)] + argv_tail
                args = mod.parse_args()  # type: ignore[attr-defined]
            finally:
                sys.argv = old_argv
            mod.main(args)  # type: ignore[attr-defined]
            return 0

    return run_via_subprocess(script_path, argv_tail)


def run_via_subprocess(script_path: Path, argv_tail: List[str]) -> int:
    cmd = [sys.executable, str(script_path)] + argv_tail
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)


def resolve_script_path(repo_root: Path, models_dir: Path, script_arg: str) -> Optional[Path]:
    """
    Resolve --script argument:
    - If absolute path: use directly if exists.
    - Else look in models_dir then repo_root.
    """
    p = Path(script_arg)
    if p.is_absolute():
        return p if p.exists() else None

    cand1 = models_dir / script_arg
    if cand1.exists():
        return cand1

    cand2 = repo_root / script_arg
    if cand2.exists():
        return cand2

    # also allow passing "models/xxx.py"
    cand3 = repo_root / script_arg
    if cand3.exists():
        return cand3

    return None


def build_parser(model_keys: List[str]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified runner for multiple model scripts (scans ./models by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    p.add_argument("--list", action="store_true", help="List discovered scripts and exit.")
    p.add_argument(
        "--script",
        type=str,
        default=None,
        help='Run a specific script filename/path (overrides model key). Example: --script blip_classifier.py',
    )
    # Support both positional model and --model
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model key to run. Available: {', '.join(sorted(model_keys))}",
    )
    p.add_argument(
        "model_pos",
        nargs="?",
        default=None,
        help="(Legacy) model key as positional argument. Example: python run.py blip ...",
    )
    # Everything after is passed through to the target script
    p.add_argument("pass_through", nargs=argparse.REMAINDER, help="Arguments passed to the target script.")
    return p


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    models_dir = repo_root / "models"

    # Primary: ./models, Secondary: repo_root (fallback)
    scripts_models = discover_scripts(models_dir)
    scripts_root = discover_scripts(repo_root)
    scripts = merge_script_maps(scripts_models, scripts_root)

    parser = build_parser(list(scripts.keys()))
    args = parser.parse_args()

    if args.list:
        print("Repo root:", repo_root)
        print("Models dir:", models_dir, "(preferred)")
        if not scripts:
            print("[WARN] No runnable scripts found. Expecting *_classifier.py / *_fusion.py / *_textplugin.py")
            return 0
        print("Discovered model keys:")
        for k in sorted(scripts.keys()):
            print(f"  {k:20s} -> {scripts[k].relative_to(repo_root)}")
        return 0

    # Determine target script
    script_path: Optional[Path] = None

    if args.script:
        script_path = resolve_script_path(repo_root, models_dir, args.script)
        if script_path is None:
            print(f"[ERROR] --script not found: {args.script}")
            print("Tip: run `python run.py --list` to see discovered scripts.")
            return 2
    else:
        key = (args.model or args.model_pos)
        if not key:
            print("[ERROR] You must provide a model key (positional or --model), or use --script, or use --list.")
            return 2
        key = key.lower()
        if key not in scripts:
            print(f"[ERROR] Unknown model key: {key}")
            print("Use: python run.py --list")
            return 2
        script_path = scripts[key]

    # Clean pass-through: sometimes argparse includes a leading "--"
    argv_tail = args.pass_through
    if len(argv_tail) > 0 and argv_tail[0] == "--":
        argv_tail = argv_tail[1:]

    # Execute
    try:
        return run_via_main(script_path, argv_tail)
    except Exception as e:
        print(f"[WARN] Direct main(args) failed: {e}")
        print("[INFO] Falling back to subprocess execution.")
        return run_via_subprocess(script_path, argv_tail)


if __name__ == "__main__":
    raise SystemExit(main())
