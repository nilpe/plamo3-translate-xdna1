"""Shared helpers of the release unit tests (stdlib unittest; run on Linux and on the laptop)."""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
TREE = HERE.parents[2] / "IRON" if (HERE.parents[2] / "IRON" / "iron").is_dir() else HERE.parents[2]  # release layout <root>\IRON, or the worktree itself
APP = TREE / "iron" / "applications" / "plamo3"
R83 = TREE / "experiments" / "plamo3_r83" / "source"
R45 = TREE / "experiments" / "plamo3_r45" / "source"
R43 = TREE / "experiments" / "plamo3_r43" / "source"
IS_WINDOWS = os.name == "nt"


def load_isolated(path, name):
    """Import a module from an explicit file path under a private name (does not touch sys.modules['plamo_root'] etc. of other tests)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_install(tmp, files=("plamo_root.py",)):
    """Copy plamo_root.py into <tmp>/IRON/iron/applications/plamo3 so that its ROOT is <tmp>. Returns (module, root)."""
    root = Path(tmp).resolve()
    app = root / "IRON" / "iron" / "applications" / "plamo3"
    app.mkdir(parents=True)
    for f in files:
        shutil.copy(APP / f, app / f)
    mod = load_isolated(app / "plamo_root.py", "plamo_root_" + root.name.replace("-", "_"))
    return mod, root


class TmpDirMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="h1a-", ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


def add_path(*paths):
    for p in paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
add_path(APP, R83, R45)
