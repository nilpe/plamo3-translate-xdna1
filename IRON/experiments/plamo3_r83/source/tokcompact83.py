"""R83 compact tokenizer payload: an exact, cheaper-to-load form of the pinned `FastPlamo3Tokenizer` data (iron/applications/plamo3/fast_tokenizer.py).

Why: `FastPlamo3Tokenizer(manifest)` costs ~64 ms in the one-shot CLI after spec-ready, of which marshal.loads is ~52 ms (a 195,625-entry int->int dict 33 ms, the 107,520
token strings 8 ms, two 195k-int lists 10 ms). All of it is GIL-bound Python object construction, so it cannot be hidden behind the start-up chain (R80).
What: the same trie / scores / tokens as read-only flat tables that are used straight from the file bytes (memoryview casts, zero object construction):
  * trie edges in CSR form (`first[node]..first[node+1]` index the sorted `chars`/`kids` of that node; one C `bisect` per step replaces `dict.get`),
  * `terminals` and `scores` as int64 arrays, `byte_ids` (256 ints, JSON header), the token strings as one UTF-8 blob + offsets (decoded on use).
`encode` is the base class algorithm with only the edge lookup replaced; `decode`, `chat`, `apply_chat_template`, `render_chat`, regexes are inherited unchanged.
Checks kept (same order, same exceptions): manifest format + implementation digest of the pinned fast_tokenizer.py, template digest, the five checkpoint source digests, payload size.
The compact file is verified by its own sha256 + size recorded in the manifest (`compact` entry), exactly like the marshal payload is (`build` proves the relation to the marshal
payload: every table is re-expanded and compared with the marshal data before the file is accepted).
Hash/read work (GIL-free) can run in a thread (`Prefetch`); the main thread then only casts the buffers and compiles two regexes (~1 ms).
"""
import array
import bisect
import hashlib
import json
import marshal
import re
import struct
import threading
from pathlib import Path

try:
    import fast_tokenizer as ft
except ImportError:  # pragma: no cover
    ft = None

MAGIC = b'PLAMOCT1'
SECTIONS = ('first', 'chars', 'kids', 'terminals', 'scores', 'offsets', 'blob')
KIND = {'first': 'i', 'chars': 'i', 'kids': 'i', 'terminals': 'q', 'scores': 'q', 'offsets': 'q', 'blob': 'B'}
ALIGN = 8


def _pad(n):
    return (-n) % ALIGN


def build(manifest, out_dir=None, name='tokenizer.compact'):
    """Derive the compact file from the manifest's marshal payload; prove equality of every table; write <name> and tokenizer-manifest.compact.json next to `out_dir`."""
    manifest = Path(manifest)
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    data = (manifest.parent / meta['payload']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == meta['payload_sha256']
    tokens, transitions, terminals, scores, byte_ids = marshal.loads(data)
    n = len(terminals)
    assert len(scores) == n and len(tokens) == meta['vocab_size'] and n == meta['trie_nodes']
    by_node = [[] for _ in range(n)]
    for key, child in transitions.items():
        by_node[key >> 21].append((key & 0x1FFFFF, child))
    first, chars, kids = [0], [], []
    for edges in by_node:
        edges.sort()
        for c, k in edges:
            chars.append(c)
            kids.append(k)
        first.append(len(chars))
    assert len(chars) == len(transitions)
    enc = [t.encode('utf-8') for t in tokens]
    offsets = [0]
    for e in enc:
        offsets.append(offsets[-1] + len(e))
    blob = b''.join(enc)
    arrays = dict(first=array.array('i', first), chars=array.array('i', chars), kids=array.array('i', kids), terminals=array.array('q', terminals),
                  scores=array.array('q', scores), offsets=array.array('q', offsets), blob=blob)
    for key in ('first', 'chars', 'kids'):
        assert arrays[key].itemsize == 4
    header = dict(format='plamo3-compact-trie-v1', nodes=n, edges=len(chars), vocab=len(tokens), byte_ids=list(byte_ids),
                  sizes={k: (len(arrays[k]) if k != 'blob' else len(blob)) for k in SECTIONS}, source_payload_sha256=meta['payload_sha256'])
    hb = json.dumps(header, sort_keys=True).encode('utf-8')
    out = bytearray(MAGIC + struct.pack('<Q', len(hb)) + hb)
    out += b'\0' * _pad(len(out))
    for k in SECTIONS:
        b = arrays[k].tobytes() if k != 'blob' else blob
        out += b + b'\0' * _pad(len(b))
    # proof: re-expand and compare with the marshal data
    t = load_tables(bytes(out))
    assert [t['tokens'][i] for i in range(len(tokens))] == tokens
    assert list(t['terminals']) == list(terminals) and list(t['scores']) == list(scores) and list(t['byte_ids']) == list(byte_ids)
    rebuilt = {}
    f, c, k = t['first'], t['chars'], t['kids']
    for node in range(n):
        for i in range(f[node], f[node + 1]):
            rebuilt[(node << 21) | c[i]] = k[i]
    assert rebuilt == transitions
    out_dir = Path(out_dir or manifest.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_bytes(bytes(out))
    info = dict(file=name, bytes=len(out), sha256=hashlib.sha256(bytes(out)).hexdigest(), source_payload_sha256=meta['payload_sha256'])
    target = out_dir / 'tokenizer-manifest.compact.json'
    m2 = dict(meta)
    m2['compact'] = info
    target.write_text(json.dumps(m2, indent=2, ensure_ascii=False), encoding='utf-8')
    return target, info


class _Tokens:
    """Read-only sequence of the token strings (decoded on use; the base class holds a list of the same strings)."""
    __slots__ = ('_b', '_o', '_n')

    def __init__(self, blob, offsets, n):
        self._b, self._o, self._n = blob, offsets, n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError('list index out of range')
        return bytes(self._b[self._o[i]:self._o[i + 1]]).decode('utf-8')

    def __iter__(self):
        return (self[i] for i in range(self._n))


def load_tables(data):
    view = memoryview(data)
    if bytes(view[:8]) != MAGIC:
        raise ValueError('compact tokenizer artifact has a wrong magic')
    (hl,) = struct.unpack('<Q', view[8:16])
    header = json.loads(bytes(view[16:16 + hl]))
    if header.get('format') != 'plamo3-compact-trie-v1':
        raise ValueError('compact tokenizer artifact has a wrong format')
    pos = 16 + hl
    pos += _pad(pos)
    out = {}
    for k in SECTIONS:
        size = header['sizes'][k]
        item = {'i': 4, 'q': 8, 'B': 1}[KIND[k]]
        nbytes = size * item
        part = view[pos:pos + nbytes]
        if len(part) != nbytes:
            raise ValueError('compact tokenizer artifact is truncated')
        out[k] = part if k == 'blob' else part.cast(KIND[k])
        pos += nbytes + _pad(nbytes)
    if pos != len(data):
        raise ValueError('compact tokenizer artifact has trailing bytes')
    out['tokens'] = _Tokens(out['blob'], out['offsets'], header['vocab'])
    out['byte_ids'] = header['byte_ids']
    out['header'] = header
    return out


def verified_read(manifest, *, checkpoint=None, validate_sources=True):
    """All checks of FastPlamo3Tokenizer.__init__ that read files (same order, same exceptions) + the compact artifact digest. Returns (meta, compact bytes). GIL-free
    for the hashing, so it may run in a thread."""
    manifest = Path(manifest)
    if manifest.is_dir():
        manifest = manifest / 'tokenizer-manifest.compact.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    if meta['format'] != ft.FORMAT or meta['implementation_sha256'] != ft.digest(ft.__file__):
        raise ValueError('fast tokenizer implementation changed; re-export and revalidate')
    if meta['sources']['chat_template.jinja'] != ft.TEMPLATE_SHA256:
        raise ValueError('unsupported PFML3 template')
    if validate_sources:
        import plamo_root
        directory = Path(checkpoint or plamo_root.resolve_token(meta['checkpoint']))
        for name, expected in meta['sources'].items():
            if ft.digest(directory / name) != expected:
                raise ValueError('tokenizer source changed: ' + name)
    info = meta['compact']
    blob = manifest.parent / info['file']
    if blob.parent != manifest.parent or blob.stat().st_size != info['bytes']:
        raise ValueError('fast tokenizer artifact differs from its manifest')
    data = blob.read_bytes()
    if hashlib.sha256(data).hexdigest() != info['sha256']:
        raise ValueError('fast tokenizer artifact differs from its manifest')
    return meta, data


class CompactTokenizer(ft.FastPlamo3Tokenizer if ft else object):
    def __init__(self, manifest, *, checkpoint=None, validate_sources=True, _read=None):
        meta, data = _read if _read is not None else verified_read(manifest, checkpoint=checkpoint, validate_sources=validate_sources)
        self.metadata = meta
        t = load_tables(data)
        self.tokens, self.terminals, self.scores, self.byte_ids = t['tokens'], t['terminals'], t['scores'], t['byte_ids']
        self._first, self._chars, self._kids = t['first'], t['chars'], t['kids']
        if len(self.tokens) != meta['vocab_size'] or len(self.terminals) != meta['trie_nodes']:
            raise ValueError('fast tokenizer artifact shape mismatch')
        self.special_ids = set(meta['special_ids'])
        self.legacy_added_ids = set(map(int, meta['added_tokens'])) - self.special_ids
        self._special = re.compile(r'(<\|plamo:[^|\s]{,64}\|>)')
        patterns = []
        if meta['break_repeats'] is not None:
            patterns.append(f"(.)\\2{{{meta['break_repeats'] - 1},}}")
        if meta['break_spaces'] is not None:
            patterns.append(f" {{{meta['break_spaces']},}}")
        self._breaks = re.compile(f"({'|'.join(patterns)})") if patterns else None

    def encode(self, text, *, add_special_tokens=True):
        BOUNDARY_CHAR, BOUNDARY_TOKEN_ID, UNKNOWN_SCORE = ft.BOUNDARY_CHAR, ft.BOUNDARY_TOKEN_ID, ft.UNKNOWN_SCORE
        text = self._special.sub(BOUNDARY_CHAR + r'\1' + BOUNDARY_CHAR, text)
        if self._breaks is not None:
            text = self._breaks.sub(BOUNDARY_CHAR + r'\1' + BOUNDARY_CHAR, text)
        n = len(text)
        costs, lengths, chosen = [0] * (n + 1), [0] * n, [-1] * n
        terminal, scores = self.terminals, self.scores
        first, chars, kids = self._first, self._chars, self._kids
        bisect_left = bisect.bisect_left
        for start in range(n - 1, -1, -1):
            node, best, best_length, token = 0, 1 << 60, 0, -1
            for end in range(start, n):
                c = ord(text[end])
                hi = first[node + 1]
                i = bisect_left(chars, c, first[node], hi)
                if i == hi or chars[i] != c:
                    break
                node = kids[i]
                candidate = terminal[node]
                if candidate >= 0:
                    cost, length = costs[end + 1] - scores[node], end - start + 1
                    if cost < best or (cost == best and length > best_length):
                        best, best_length, token = cost, length, candidate
            unknown = costs[start + 1] - UNKNOWN_SCORE
            if unknown < best:
                best, best_length, token = unknown, 1, -1
            costs[start], lengths[start], chosen[start] = best, best_length, token
        result, position = [], 0
        while position < n:
            token = chosen[position]
            if token < 0:
                result.extend(self.byte_ids[byte] for byte in text[position].encode('utf-8'))
            elif token != BOUNDARY_TOKEN_ID:
                result.append(token)
            position += lengths[position]
        if add_special_tokens:
            if self.metadata['add_bos_token']:
                result.insert(0, self.metadata['bos_id'])
            if self.metadata['add_eos_token']:
                result.append(self.metadata['eos_id'])
        return result


class Prefetch:
    """Runs the file reads and hashes (`verified_read`) in a daemon thread; `get()` joins and builds the tokenizer (errors re-raised at that point, like the constructor)."""

    def __init__(self, manifest, **kw):
        self._manifest, self._kw = manifest, kw
        self._result = self._error = None
        self._thread = threading.Thread(target=self._run, name='r83-tok-prefetch', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._result = verified_read(self._manifest, **self._kw)
        except BaseException as error:  # re-raised in get()
            self._error = error

    def get(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return CompactTokenizer(self._manifest, _read=self._result)
