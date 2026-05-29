"""Standalone test runner for steam_voi.py.

Runs every test function in tests/test_steam_voi.py without needing pytest
or the heavy ProxylFit imports (SimpleITK, etc.). Loads the module directly
by file path to bypass proxyl_analysis/__init__.py.
"""

import importlib.util
import inspect
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STEAM_PATH = ROOT / "proxyl_analysis" / "steam_voi.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the module under test.
steam_voi = _load_module("steam_voi", STEAM_PATH)


# --- Minimal pytest stub ---------------------------------------------------

class _Skipped(Exception):
    pass


class _Pytest:
    class raises:
        def __init__(self, exc):
            self.exc = exc
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__}, got nothing")
            if not issubclass(exc_type, self.exc):
                return False
            self.value = exc
            return True

    @staticmethod
    def skip(msg):
        raise _Skipped(msg)

    @staticmethod
    def fixture(*args, **kwargs):
        """No-op decorator; we invoke fixtures manually."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn):
            return fn

        return deco


pytest = _Pytest()


# Build a fixture-aware test runner that mirrors what the actual file uses.

# Load the test module — but it imports from proxyl_analysis.steam_voi
# directly, so we alias.
sys.modules["proxyl_analysis"] = type(sys)("proxyl_analysis")
sys.modules["proxyl_analysis.steam_voi"] = steam_voi
sys.modules["pytest"] = pytest  # shim

TEST_PATH = ROOT / "tests" / "test_steam_voi.py"
tests_mod = _load_module("test_steam_voi", TEST_PATH)


# Fixtures we need to provide manually (we don't have pytest's machinery).
class _Workspace:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="steam_voi_tests_"))

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def _make_fixture_values(tmp_path):
    """Return a dict {fixture_name: value} for every fixture the tests use.

    Each fixture gets its own sub-directory so they don't clobber each
    other (real pytest gives each fixture its own tmp_path; we have to
    do it manually).
    """
    sub_smf = tmp_path / "synthetic_method"
    sub_smf.mkdir()
    synthetic_method_file = tests_mod.synthetic_method_file(sub_smf)

    sub_t2 = tmp_path / "synthetic_t2_method"
    sub_t2.mkdir()
    synthetic_t2_method_file = tests_mod.synthetic_t2_method_file(sub_t2)

    sub_subj = tmp_path / "synthetic_subject"
    sub_subj.mkdir()
    synthetic_subject_file = tests_mod.synthetic_subject_file(sub_subj)

    sub_study = tmp_path / "synthetic_study_root"
    sub_study.mkdir()
    synthetic_bruker_study = tests_mod.synthetic_bruker_study(
        sub_study, synthetic_subject_file
    )

    return {
        "tmp_path": tmp_path,
        "synthetic_method_file": synthetic_method_file,
        "synthetic_t2_method_file": synthetic_t2_method_file,
        "synthetic_subject_file": synthetic_subject_file,
        "synthetic_bruker_study": synthetic_bruker_study,
    }


def _collect_tests():
    """Yield (qualname, callable, needs_fixtures) tuples."""
    for name in dir(tests_mod):
        obj = getattr(tests_mod, name)
        # Classes containing test_ methods
        if inspect.isclass(obj) and name.startswith("Test"):
            instance = obj()
            for mname in dir(instance):
                if mname.startswith("test_"):
                    method = getattr(instance, mname)
                    yield (f"{name}.{mname}", method)
        # Module-level test functions
        elif inspect.isfunction(obj) and name.startswith("test_"):
            yield (name, obj)


def _call_with_fixtures(fn, fixtures):
    sig = inspect.signature(fn)
    kwargs = {}
    for pname in sig.parameters:
        if pname == "self":
            continue
        if pname in fixtures:
            kwargs[pname] = fixtures[pname]
    return fn(**kwargs)


def main():
    passed = 0
    failed = 0
    skipped = 0
    failures = []

    for qualname, fn in _collect_tests():
        # Each test gets its own tmp_path.
        ws = _Workspace()
        try:
            fixtures = _make_fixture_values(ws.tmp)
            _call_with_fixtures(fn, fixtures)
            print(f"PASS  {qualname}")
            passed += 1
        except _Skipped as e:
            print(f"SKIP  {qualname}: {e}")
            skipped += 1
        except Exception as e:
            print(f"FAIL  {qualname}")
            failures.append((qualname, traceback.format_exc()))
            failed += 1
        finally:
            ws.cleanup()

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print()
        print("=" * 70)
        for qualname, tb in failures:
            print(f"FAIL  {qualname}")
            print(tb)
            print("-" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
