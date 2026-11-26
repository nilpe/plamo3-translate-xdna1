"""Pinned Phoenix TXN parser and measurement-only same-channel completion reduction."""

from array import array
from collections import Counter

SIZES = {0: (5, 6), 1: (3, 4), 3: (6, 7), 4: (6, 7), 128: (1, 4), 129: (1, 12)}
QUEUES = {0x1D204: (0, 0), 0x1D20C: (0, 1), 0x1D214: (1, 0), 0x1D21C: (1, 1)}


def parse(raw):
    w = array("I", raw)
    assert w[:2] == array("I", [0x06030100, 0x104]) and w[3] == len(raw)
    at = 4
    parts = []
    while at < len(w):
        op = w[at]
        field, minimum = SIZES[op]
        n = w[at + field] // 4
        assert n >= minimum and (op == 1 or n == minimum) and at + n <= len(w)
        parts.append(list(w[at : at + n]))
        at += n
    assert len(parts) == w[2]
    return parts


def encode(parts):
    data = [v for p in parts for v in p]
    return array(
        "I", [0x06030100, 0x104, len(parts), 16 + 4 * len(data)] + data
    ).tobytes()


def address(p):
    return p[6] if p[0] == 129 else p[2]


def location(addr):
    return (addr >> 25, (addr >> 20) & 31, addr & 0xFFFFF)


def kind(p):
    op = p[0]
    if op == 128:
        return "completion_wait"
    if op == 129:
        return "ddr_patch"
    col, row, reg = location(address(p))
    if op == 4:
        return (
            "ready_lock_poll" if row >= 2 and 0x1F000 <= reg < 0x1F100 else "mask_poll"
        )
    if row == 0 and 0x1D000 <= reg < 0x1D200:
        return "bd_write"
    if row == 0 and reg in QUEUES:
        return "task_push"
    if row >= 2 and 0x1F000 <= reg < 0x1F100:
        return "lock_write"
    if row >= 2 and reg < 0x10000:
        return "rtp_write"
    return f"op{op}_r{row}_reg{reg:x}"


def counts(raw):
    ps = parse(raw)
    c = Counter(kind(p) for p in ps)
    return dict(
        instructions=len(ps),
        bytes=len(raw),
        counts=dict(c),
        write_words=sum(
            len(p) - 4 if p[0] == 1 else 1 for p in ps if p[0] in (0, 1, 3)
        ),
    )


def queue_key(p):
    if p[0] != 0:
        return None
    col, row, reg = location(p[2])
    return (col, *QUEUES[reg]) if row == 0 and reg in QUEUES else None


def reduce_completion(raw):
    """Only remove earlier task tokens within a contiguous final await block.
    Last completion on each FIFO-ordered DMA channel covers every predecessor.
    All group boundaries, last channel waits and BD lifetimes are preserved.
    """
    ps = parse(raw)
    out = []
    segment = []
    saved = 0

    def emit(chunk):
        nonlocal saved
        waits = [p for p in chunk if p[0] == 128]
        if not waits:
            return chunk
        first = next(i for i, p in enumerate(chunk) if p[0] == 128)
        if any(p[0] != 128 for p in chunk[first:]):
            raise ValueError("await block not contiguous")
        pushes = {}
        for i, p in enumerate(chunk[:first]):
            k = queue_key(p)
            if k is not None:
                if not p[4] & 0x80000000:
                    raise ValueError("unexpected un-tokened task")
                pushes.setdefault(k, []).append(i)
        expected = Counter({k: len(v) for k, v in pushes.items()})
        actual = Counter()
        keys = []
        for p in waits:
            direction = p[2] & 255
            row = (p[2] >> 8) & 255
            col = (p[2] >> 16) & 255
            nrow = (p[3] >> 8) & 255
            ncol = (p[3] >> 16) & 255
            chan = (p[3] >> 24) & 255
            if row != 0 or nrow != 1 or ncol != 1:
                raise ValueError("not a single shim completion")
            k = (col, direction, chan)
            actual[k] += 1
            keys.append(k)
        if actual != expected:
            raise ValueError(f"task/await mismatch {actual} != {expected}")
        for inds in pushes.values():
            for i in inds[:-1]:
                chunk[i] = chunk[i].copy()
                chunk[i][4] &= 0x7FFFFFFF
                saved += 1
        # Keep the last occurrence for each channel, in the original wait order.
        last = {k: i for i, k in enumerate(keys)}
        return chunk[:first] + [p for i, p in enumerate(waits) if last[keys[i]] == i]

    for p in ps:
        if segment and segment[-1][0] == 128 and p[0] != 128:
            out.extend(emit(segment))
            segment = []
        segment.append(p)
    out.extend(emit(segment))
    return encode(out), saved


def contiguous_kicks(raw, available_bytes):
    """Diagnostic rewrite of the exact 512x4-byte ignored kick, within input BO."""
    ps = parse(raw)
    changed = 0
    remove = set()
    for i, p in enumerate(ps):
        if p[0] != 1 or len(p) != 12 or p[4] != 1:
            continue
        if i + 4 >= len(ps):
            continue
        patch, early, control, last = ps[i + 1 : i + 5]
        if not (
            patch[0] == 129
            and patch[8] == 0
            and patch[10:] == [0, 0]
            and early[0] == 0
            and queue_key(early) is not None
            and early[4] & 0xFFFF0000 == 0x00FF0000
            and control[0] == 3
            and last[0] == 0
            and last[2] == early[2]
            and last[4] == early[4] | 0x80000000
        ):
            continue
        assert available_bytes >= 2048, "contiguous kick exceeds source allocation"
        ps[i] = p.copy()
        ps[i][4] = 512
        ps[i + 4] = last.copy()
        ps[i + 4][4] &= ~0x00FF0000
        remove.add(i + 2)
        changed += 1
    return encode([p for i, p in enumerate(ps) if i not in remove]), changed
