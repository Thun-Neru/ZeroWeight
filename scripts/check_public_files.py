"""Fail closed if an unexpected file is staged/tracked for publication."""
from pathlib import Path
import subprocess
import sys

ALLOWED = {
    ".gitignore", "README.md", "AGENTS.md", "VERSION", "requirements.txt",
    "zeroweight/__init__.py", "zeroweight/__main__.py", "zeroweight/inspect.py",
    "tests/test_inspect.py", "scripts/check_public_files.py", "docs/verification.md",
}
root = Path(__file__).resolve().parents[1]
files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
files = {f for f in files if f}
unexpected = files - ALLOWED
if unexpected or not files:
    print("Publication blocked:", sorted(unexpected) if unexpected else "empty index")
    sys.exit(1)
print(f"Publication allowlist passed: {len(files)} files; no research artifacts or generators.")
