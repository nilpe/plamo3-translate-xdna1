"""Identical ABI/config/source-inventory checks after SHA-256 binary decoding."""
from runtime_manifest import *
from runtime_manifest import _record_path
from compact_catalog import read_certificate
import time
from pathlib import Path
def load(path, *, event=None, owners=None):
    path = Path(path)
    started = time.perf_counter()
    try:
        manifest = read_certificate(path, owners)
        if manifest.get("version") != VERSION:
            raise StaleManifest("runtime manifest schema changed; rebuild the bundle")
        if not __import__("plamo_root").config_equal(manifest["configuration"], configuration()):   # H1: machine keys are not compared (environment record), see plamo_root
            raise StaleManifest("runtime configuration changed; rebuild the bundle")
        if "python_abi" in manifest and not __import__("plamo_root").abi_equal(manifest["python_abi"], python_abi()):   # H1: cache tag only
            raise StaleManifest("runtime Python ABI changed; rebuild the bundle")
        roots, sources = manifest["source_roots"], manifest["sources"]
        if not isinstance(roots, list) or not all(
            isinstance(p, str) and p for p in roots
        ):
            raise StaleManifest("runtime source roots must be a list of paths")
        if not isinstance(sources, list) or not isinstance(
            manifest.get("origins", []), list
        ):
            raise StaleManifest("runtime source and origin records must be lists")
        if not isinstance(manifest["artifacts"], dict):
            raise StaleManifest("runtime artifact records must be an object")
        for record in [
            *sources,
            *manifest.get("origins", []),
            *manifest["artifacts"].values(),
        ]:
            _record_path(record)
        if event is not None:
            event(
                "validation_manifest_json_schema_config", time.perf_counter() - started
            )
        started = time.perf_counter()
        if source_files(roots) != [r["path"] for r in sources]:
            raise StaleManifest("runtime source inventory changed; rebuild the bundle")
        if event is not None:
            event("validation_source_inventory", time.perf_counter() - started)
        return manifest
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise StaleManifest(
            f"runtime manifest unavailable or malformed: {path}: {exc}"
        ) from exc
