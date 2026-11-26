# SPDX-License-Identifier: Apache-2.0
"""Install-root path model of the PLaMo 3 NPU runtime.

* ROOT is derived from this module's own location (<root>\\IRON\\iron\\applications\\plamo3\\plamo_root.py), never from an environment variable.
* Shipped manifests carry the placeholders @ROOT@ and @MODEL@ at the start of path strings. They are resolved here, after the raw envelope digest
  of the manifest has passed and before any consumer sees the object (runtime_manifest._read_manifest and compact_catalog.decode).
  A path without a placeholder is used as is, so manifests written with absolute paths still load.
* The identity receipt (state/install-receipt.json, written only by the installer, never by the runtime) is overlaid onto the manifest records
  as ntfs_identity. The comparison code is unchanged; no receipt means every file takes the existing full-hash path.
* The configuration / Python ABI comparison is relaxed in exactly one place: XRT_ROOT, XILINX_XRT and SYSTEMROOT left the compared key set
  (environment record/warn layer) and the interpreter patch level is no longer compared (cache tag and bitness still are).
Stdlib only.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "model"
STATE = ROOT / "state"
RECEIPT = STATE / "install-receipt.json"
RECEIPT_VERSION = 1
CERTIFICATE_RELPATH = "plamo3-default/r45-spec/certificate.json"

TOKENS = {"@ROOT@": ROOT, "@MODEL@": MODEL}
# Keys that older manifests still carry in `configuration`; they are ignored on both sides of the comparison and never written to os.environ.
LEGACY_CONFIG_KEYS = frozenset({"XRT_ROOT", "XILINX_XRT", "SYSTEMROOT"})

_ROOT_S = str(ROOT)
_TOKEN_PREFIXES = tuple(TOKENS)


class PathModelError(ValueError):
    pass


def resolve_token(value, error=PathModelError):
    """Resolve one string. Placeholders are only valid at the very start, followed by a separator or the end."""
    if not isinstance(value, str) or "@" not in value:
        return value
    if value[:1] != "@":
        for token in _TOKEN_PREFIXES:
            if token in value:
                raise error(f"path placeholder not at the start of the string: {value[:80]}")
        return value
    end = value.find("@", 1)
    if end < 0:
        return value
    token = value[: end + 1]
    if token not in TOKENS:
        if end > 1 and token[1:-1].replace("_", "").isalnum() and token[1:-1].isupper() and value[end + 1 : end + 2] in ("\\", "/", ""):
            raise error(f"unknown path placeholder {token}: {value[:80]}")
        return value
    rest = value[end + 1 :]
    if rest and rest[0] not in "\\/":
        raise error(f"malformed path placeholder: {value[:80]}")
    parts = [p for p in rest.replace("/", "\\").split("\\") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise error(f"parent-directory component in a placeholder path: {value[:80]}")
    base = TOKENS[token]
    out = str(base.joinpath(*parts)) if parts else str(base)
    return out


def _walk(obj, error):
    n = 0
    if isinstance(obj, dict):
        for k in list(obj):
            if isinstance(k, str) and "@" in k:
                for token in _TOKEN_PREFIXES:
                    if token in k:
                        raise error(f"path placeholder in a dictionary key: {k[:80]}")
            v = obj[k]
            if isinstance(v, str):
                if "@" in v:
                    new = resolve_token(v, error)
                    if new is not v:
                        obj[k] = new
                        n += 1
            elif isinstance(v, (dict, list)):
                n += _walk(v, error)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                if "@" in v:
                    new = resolve_token(v, error)
                    if new is not v:
                        obj[i] = new
                        n += 1
            elif isinstance(v, (dict, list)):
                n += _walk(v, error)
    return n


def rebase(obj, error=PathModelError):
    """Resolve @ROOT@/@MODEL@ in place in a decoded manifest (dict/list tree). Returns the object. A tree without placeholders is unchanged."""
    if isinstance(obj, str):
        return resolve_token(obj, error)
    _walk(obj, error)
    return obj


# ---------------------------------------------------------------- receipt overlay

_receipt_cache = {}


def _norm_key(path_s):
    return path_s.replace("\\", "/").lower()


def _load_receipt():
    """Parse and validate the receipt once per process. Returns (index|None, reason)."""
    if "r" in _receipt_cache:
        return _receipt_cache["r"]
    result = (None, "missing")
    try:
        raw = RECEIPT.read_bytes()
    except OSError:
        _receipt_cache["r"] = result
        return result
    try:
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("version") != RECEIPT_VERSION:
            raise ValueError("receipt version")
        if os.path.normcase(str(doc.get("install_root", ""))) != os.path.normcase(_ROOT_S):
            result = (None, "root")
            raise ValueError("receipt install_root differs (tree moved or copied)")
        cert = ROOT / str(doc.get("certificate_path", CERTIFICATE_RELPATH))
        if hashlib.sha256(cert.read_bytes()).hexdigest() != doc.get("certificate_sha256"):
            result = (None, "certificate")
            raise ValueError("receipt belongs to another certificate")
        index = {}
        for row in doc["ids"]:
            index[row[0].replace("\\", "/").lower()] = row
        result = (index, "ok")
    except (ValueError, KeyError, TypeError, IndexError, OSError, AttributeError):
        if result[1] == "missing":
            result = (None, "invalid")
    _receipt_cache["r"] = result
    return result


def reset_receipt_cache():
    _receipt_cache.clear()


def _identity(row):
    return dict(
        volume_serial=row[1],
        file_id_128=row[2],
        size=row[3],
        last_write_time=row[4],
        usn=row[5],
        journal_id=row[6],
    )


def apply_receipt(manifest, notify=None):
    """Overlay receipt identities onto the records of a decoded (and rebased) manifest. Only ADDS identities; never removes or edits an
    identity a manifest already carries, never writes anything. Returns the number of records overlaid."""
    if not isinstance(manifest, dict) or "sources" not in manifest or "artifacts" not in manifest:
        return 0
    legacy_identities = manifest.get("filesystem_identity_version") == 1
    index, reason = _load_receipt()
    if index is None:
        if not legacy_identities and notify is not None:
            notify(
                "PLaMo: install receipt "
                + ("is missing" if reason == "missing" else "is stale (moved or copied tree)" if reason in ("root", "certificate") else "is unreadable")
                + "; hashing all files (about 15 s). Run install.ps1 -Reattest.",
                "receipt",
            )
        return 0
    prefix = _ROOT_S.replace("\\", "/").lower() + "/"
    count = 0
    records = [*manifest["sources"], *manifest.get("origins", []), *manifest["artifacts"].values()]
    for record in records:
        if not isinstance(record, dict) or record.get("ntfs_identity") is not None:
            continue
        p = record.get("path")
        if not isinstance(p, str):
            continue
        key = p.replace("\\", "/").lower()
        if not key.startswith(prefix):
            continue
        row = index.get(key[len(prefix) :])
        if row is not None:
            record["ntfs_identity"] = _identity(row)
            count += 1
    if count:
        manifest["filesystem_identity_version"] = 1
    return count


def prepare(manifest, error=PathModelError, notify=None):
    """The two choke-point hooks call this: resolve placeholders, then overlay the receipt. Returns the manifest."""
    rebase(manifest, error)
    apply_receipt(manifest, notify)
    return manifest


# ---------------------------------------------------------------- configuration / ABI comparison

def strip_legacy_config(cfg):
    return {k: v for k, v in cfg.items() if k not in LEGACY_CONFIG_KEYS}


def config_diff(manifest_cfg, live_cfg):
    """Names of the configuration keys that differ (machine keys ignored): used in the refusal message so a mismatch is diagnosable."""
    a, b = strip_legacy_config(manifest_cfg), strip_legacy_config(live_cfg)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def config_equal(manifest_cfg, live_cfg):
    """Environment configuration comparison without the machine keys XRT_ROOT, XILINX_XRT and SYSTEMROOT."""
    return strip_legacy_config(manifest_cfg) == strip_legacy_config(live_cfg)


def abi_equal(manifest_abi, live_abi):
    """Strict: interpreter cache tag (cpython-313). The patch level is recorded and warned about by environment.py, not compared."""
    return isinstance(manifest_abi, dict) and manifest_abi.get("cache_tag") == live_abi.get("cache_tag")


def check_platform(error=PathModelError):
    """Strict ABI facts of our compiled files: 64-bit CPython 3.13 on Windows x64."""
    if sys.implementation.cache_tag != "cpython-313":
        raise error(f"Python 3.13 (64-bit) required; found {sys.version.split()[0]} ({sys.implementation.cache_tag})")
    if sys.maxsize <= 2**32:
        raise error("Python 3.13 (64-bit) required; found a 32-bit interpreter")
