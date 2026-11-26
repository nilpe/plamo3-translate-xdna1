#!/usr/bin/env python3
"""L9 router equivalence (workstation, HF tokenizer == sealed tokenizer): 2,000 generated (text, cap, -Long) cases. (1) long_core.decide equals an independent implementation
of the rule of DESIGN 3.2; (2) the wrapper pre-filter is SAFE: whenever needs_router() says 'not needed' the exact verdict is 'default' (so a request is never wrongly kept off long mode);
(3) the pre-filter is not vacuous (it skips most small requests)."""
import json, random, sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
from pathlib import Path
import long_core as C
sys.path.insert(0, '/home/ykogi/workspace/plamo3-fast-inference-by-xdna/poc/pd256k/p')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/ykogi/workspace/plamo3-fast-inference-by-xdna/poc/pd256k/p/model', trust_remote_code=True)
def chat_len(text, src, dst):
    return len(tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True, target_lang=dst, source_lang=src))
def indep(P, cap, force):
    if force: return 'long'
    if P >= 2046: return 'long'
    if cap is not None and P + cap > 2048: return 'long'
    return 'default'
rnd = random.Random(9)
ja = open('data/docs.json', encoding='utf-8').read()
docs = json.load(open('data/docs.json', encoding='utf-8'))
src_ja = ''.join(d['text'] for d in docs if d['src'] == 'ja')[:400000]
src_en = ''.join(d['text'] for d in docs if d['src'] == 'en')[:400000]
rare = ''.join(chr(rnd.randrange(0x4e00, 0x9fff)) for _ in range(2000)) + ''.join(chr(rnd.randrange(0x1f300, 0x1f64f)) for _ in range(500)) + ''.join(chr(rnd.randrange(0x20000, 0x2a6df)) for _ in range(500))
n_skip = n_total = n_long = 0; bad = []
for k in range(2000):
    kind = rnd.choice(['ja', 'en', 'rare', 'mix'])
    pool = {'ja': src_ja, 'en': src_en, 'rare': rare, 'mix': src_ja[:50000] + src_en[:50000] + rare}[kind]
    L = int(rnd.choice([5, 20, 100, 300, 600, 900, 1200, 1500, 1800, 2500, 4000, 8000]) * rnd.random() + 1)
    st = rnd.randrange(0, max(1, len(pool) - L)); text = pool[st:st + L].strip() or 'a'
    src = 'ja' if kind != 'en' else 'en'; dst = 'en' if src == 'ja' else 'ja'
    cap = rnd.choice([None, None, 1, 64, 256, 800, 1500, 2047]); force = rnd.random() < 0.1
    P = chat_len(text, src, dst)
    got = C.decide(P, cap, force); want = indep(P, cap, force)
    n_total += 1; n_long += got == 'long'
    if got != want: bad.append(('decide', k, P, cap, force, got, want))
    nb = len(text.encode('utf-8'))
    if not C.needs_router(nb, cap, force):
        n_skip += 1
        if got != 'default': bad.append(('prefilter-unsafe', k, P, nb, cap, force, got))
        if P > nb + C.TEMPLATE_OVERHEAD: bad.append(('overhead-bound', k, P, nb))
res = dict(cases=n_total, long_verdicts=n_long, prefilter_skips=n_skip, failures=bad[:10], n_fail=len(bad), PASS=not bad and n_skip > 200 and n_long > 200)
print(json.dumps(res)); sys.exit(0 if res['PASS'] else 1)
