"""R83: tokenizer construction for the CLI. Modes (env R77_TOK): `sync` (DEFAULT since epoch 3: flat-table tokenizer built at first use after spec-ready, no thread: the prefetch thread cost ~+8-10 ms of spec-ready), `prefetch` (opt-in probe variant; file reads + hashes in a thread started at the validation barrier, flat-table tokenizer built at
first use), `off` (the original FastPlamo3Tokenizer(manifest), nothing else). If the compact artifact is missing or any check on it fails,
the original constructor runs instead (it performs the complete original checks), so the result is always the pinned tokenizer."""
import os
from pathlib import Path

MODES = ('off', 'sync', 'prefetch')


def mode():
    value = os.environ.get('R77_TOK', 'sync')
    return value if value in MODES else 'sync'


START_POINTS = ('validated', 'root0', 'init_end')


def start_point():
    value = os.environ.get('R83_TOKSTART', 'validated')
    return value if value in START_POINTS else 'validated'


def compact_manifest(original):
    import settings83
    return Path(settings83.TOKENIZER_COMPACT) if getattr(settings83, 'TOKENIZER_COMPACT', None) else None


def start(original_manifest):
    if mode() != 'prefetch':
        return None
    path = compact_manifest(original_manifest)
    if path is None or not path.is_file():
        return None
    import tokcompact83
    return tokcompact83.Prefetch(path)


def make(original_manifest, prefetch=None):
    from fast_tokenizer import FastPlamo3Tokenizer
    import plamo_root
    if mode() == 'off':
        return FastPlamo3Tokenizer(original_manifest, checkpoint=str(plamo_root.MODEL))
    try:
        if prefetch is not None:
            return prefetch.get()
        path = compact_manifest(original_manifest)
        if path is not None and path.is_file():
            import tokcompact83
            return tokcompact83.CompactTokenizer(path)
    except Exception:
        pass
    return FastPlamo3Tokenizer(original_manifest, checkpoint=str(plamo_root.MODEL))
