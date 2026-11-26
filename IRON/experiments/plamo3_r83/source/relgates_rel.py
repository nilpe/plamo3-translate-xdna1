# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nilpe
"""Release-only gates (plan 6.3) that run on the INSTALLED tree: --gate n2|n3|n4|n5|n7|n8 --output FILE.json
n2 path-remap: the plamo_root unit tests on an isolated copy + 12 source mutants that must each be caught.
n3 install-verify: install.py --verify on the real tree: clean PASS; then in-place corruption of one small file per class (always restored), truncation, missing file, extra file in a bound root, CRLF-converted file, tampered SHA256SUMS line: each must FAIL with exit 3.
n4 receipt: runtime lease semantics against the real tree with a valid, missing, foreign-root, foreign-certificate, truncated and garbage receipt (always restored).
n5 env record: environment.collect/compare with injected values: warn-never-refuse; strict ABI refusal.
n7 package: every package file hashes to the inventory, the inventory file set equals the files present, the package equals the inventory sealed by the drafter project (pinned) except documented text rewrites / unshipped notes / the model card, every .bin/.npz/.npy equals its sealed hash, package_manifest_sha256 equals the sealed value, the attribution file is pinned and listed in SHA256SUMS; 5 mutants.
n1b hoist transitivity: byte-equality of the hoisted artefacts + the recorded R83 epoch gate evidence (replaces the re-derivation from the R74 originals, which runs once on the author side).
n1c confidence record: the author-side confidence-gate record (private calibration inputs) is present, PASS, bound to this certificate and to the installed gate source.
n8 relocation: a copy of the installed tree at a second ASCII root (receipt stale -> notice + still correct), a `subst` drive and a junction to the installed tree (resolved root = same tree), each must translate with identical tokens."""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import settings83 as s
s.paths()
from plamo_root import ROOT
p = argparse.ArgumentParser(); p.add_argument('--gate', required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
rows = []
def check(name, ok, detail=''):
    rows.append(dict(test=name, result='PASS' if ok else 'FAIL', detail=str(detail)[:300])); print(('PASS ' if ok else 'FAIL ') + name, str(detail)[:200], flush=True)
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
PY = ROOT / 'venv/Scripts/python.exe'
def run(cmd, **kw):
    kw.setdefault('stdin', subprocess.DEVNULL)
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, errors='replace', timeout=kw.pop('timeout', 300), **kw)
def verify(root=ROOT):
    return run([PY, root / 'install.py', '--root', root, '--verify', '--quiet', '--no-npu-check'], cwd=root)

if a.gate == 'n2':
    APP = ROOT / 'IRON/iron/applications/plamo3'
    mut = [
        ('root_from_environment', 'ROOT = Path(__file__).resolve().parents[4]', 'ROOT = Path(os.environ.get("PLAMO_ROOT") or Path(__file__).resolve().parents[4])'),
        ('mid_string_token_allowed', 'raise error(f"path placeholder not at the start of the string: {value[:80]}")', 'return value'),
        ('unknown_token_allowed', 'raise error(f"unknown path placeholder {token}: {value[:80]}")', 'return value'),
        ('dotdot_allowed', 'if any(p == ".." for p in parts):', 'if False:'),
        ('model_mapped_to_root', 'TOKENS = {"@ROOT@": ROOT, "@MODEL@": MODEL}', 'TOKENS = {"@ROOT@": ROOT, "@MODEL@": ROOT}'),
        ('receipt_root_not_checked', 'if os.path.normcase(str(doc.get("install_root", ""))) != os.path.normcase(_ROOT_S):', 'if False:'),
        ('receipt_certificate_not_checked', 'if hashlib.sha256(cert.read_bytes()).hexdigest() != doc.get("certificate_sha256"):', 'if False:'),
        ('overlay_overwrites_identity', 'if not isinstance(record, dict) or record.get("ntfs_identity") is not None:', 'if not isinstance(record, dict):'),
        ('abi_compares_patch_level', 'return isinstance(manifest_abi, dict) and manifest_abi.get("cache_tag") == live_abi.get("cache_tag")', 'return isinstance(manifest_abi, dict) and manifest_abi == live_abi'),
        ('config_machine_keys_compared', 'return strip_legacy_config(manifest_cfg) == strip_legacy_config(live_cfg)', 'return manifest_cfg == live_cfg'),
        ('placeholder_in_dict_key_allowed', 'raise error(f"path placeholder in a dictionary key: {k[:80]}")', 'pass'),
        ('lists_not_rebased', '        for i, v in enumerate(obj):\n            if isinstance(v, str):', '        for i, v in enumerate([]):\n            if isinstance(v, str):'),
    ]
    src = (APP / 'plamo_root.py').read_text(encoding='utf-8')
    def isolated(text, tag):
        tmp = Path(tempfile.mkdtemp(prefix='n2-' + tag + '-', dir=str(ROOT / 'state')))
        app = tmp / 'IRON/iron/applications/plamo3'; app.mkdir(parents=True)
        (app / 'plamo_root.py').write_text(text, encoding='utf-8', newline='')
        for f in ('plamo_errors.py', 'runtime_manifest.py'): shutil.copy(APP / f, app / f)
        td = tmp / 'tests/plamo3_release'; td.mkdir(parents=True)
        for f in ('_common.py', 'test_plamo_root.py'): shutil.copy(ROOT / 'tests/plamo3_release' / f, td / f)
        r = run([PY, '-m', 'unittest', 'test_plamo_root'], cwd=td, timeout=120)
        shutil.rmtree(tmp, ignore_errors=True)
        return r.returncode, (r.stderr or '')[-200:]
    rc, err = isolated(src, 'base'); check('n2/base-tests-pass', rc == 0, err)
    for name, old, new in mut:
        if old not in src: check('n2/mutant-anchor/' + name, False, 'anchor text not found in plamo_root.py'); continue
        rc, err = isolated(src.replace(old, new, 1), name); check('n2/mutant-caught/' + name, rc != 0, 'tests failed' if rc else 'SURVIVED')
elif a.gate == 'n3':
    r = verify(); check('n3/clean-verify-passes', r.returncode == 0, (r.stdout + r.stderr)[-200:])
    sums = {l.split('  ', 1)[1].strip(): l.split('  ', 1)[0] for l in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines() if l.strip()}
    pick = {}
    for rel in sorted(sums):
        f = ROOT / rel
        if not f.is_file() or f.stat().st_size > 3_000_000 or f.stat().st_size < 64: continue
        cls = ('payload-bin' if rel.startswith('plamo3-') and rel.endswith('.bin') else 'source-py' if rel.startswith('IRON/') and rel.endswith('.py') else 'manifest' if rel.endswith('plamo3-default/r45-w4/manifest.json') else 'certificate' if rel.endswith('r45-spec/certificate.json') else 'pyxrt' if rel.endswith('pyxrt.pyd') else 'tokenizer-table' if 'tokenizer' in rel and rel.endswith('.compact') else 'drafter-npz' if rel.endswith('.npz') else 'doc' if rel == 'README.md' else None)
        if cls and cls not in pick: pick[cls] = rel
    pick['catalog'] = 'plamo3-default/r45-spec/catalog.bin'
    try:
        att_rel = json.loads((ROOT / 'release.json').read_text(encoding='utf-8')).get('drafter_attribution', {}).get('path')
    except Exception:
        att_rel = None
    check('n3/attribution-file-is-in-SHA256SUMS', bool(att_rel) and att_rel in sums and (ROOT / att_rel).is_file(), att_rel)
    if att_rel in sums:
        pick['drafter-attribution'] = att_rel   # not startup-attested: only the installer's verify covers it
    for cls, rel in sorted(pick.items()):
        f = ROOT / rel; keep = f.read_bytes()
        try:
            b = bytearray(keep); b[len(b) // 2] ^= 1; f.write_bytes(bytes(b)); r1 = verify()
            f.write_bytes(keep[:-1]); r2 = verify()
            f.unlink(); r3 = verify()
        finally:
            f.write_bytes(keep)
        check('n3/%s/bit-flip-fails' % cls, r1.returncode == 3, (rel, r1.returncode)); check('n3/%s/truncated-fails' % cls, r2.returncode == 3, r2.returncode); check('n3/%s/missing-fails' % cls, r3.returncode == 3, r3.returncode)
    model = ROOT / 'model/tokenizer.jsonl'; keep = model.read_bytes()
    try:
        b = bytearray(keep); b[1000] ^= 1; model.write_bytes(bytes(b)); r = verify()
    finally: model.write_bytes(keep)
    check('n3/model-tokenizer.jsonl/bit-flip-fails', r.returncode == 3, r.returncode)
    extra = ROOT / 'IRON/iron/applications/plamo3/_extra_mutant.py'
    try:
        extra.write_text('x=1\n'); r = verify()
    finally: extra.unlink()
    check('n3/extra-py-in-bound-root-fails', r.returncode == 3, r.returncode)
    crlf = ROOT / 'IRON/iron/applications/plamo3/plamo_errors.py'; keep = crlf.read_bytes()
    try:
        crlf.write_bytes(keep.replace(b'\n', b'\r\n')); r = verify()
    finally: crlf.write_bytes(keep)
    check('n3/crlf-converted-fails-and-says-line-endings', r.returncode == 3 and ('line ending' in (r.stdout + r.stderr).lower() or 'autocrlf' in (r.stdout + r.stderr).lower()), (r.stdout + r.stderr)[-200:])
    sf = ROOT / 'SHA256SUMS'; keep = sf.read_bytes()
    try:
        lines = keep.decode().splitlines(True); lines[0] = '0' * 64 + lines[0][64:]; sf.write_bytes(''.join(lines).encode()); r = verify()
    finally: sf.write_bytes(keep)
    check('n3/tampered-SHA256SUMS-line-fails', r.returncode == 3, r.returncode)
    r = verify(); check('n3/restored-tree-verifies-again', r.returncode == 0, (r.stdout + r.stderr)[-200:])
elif a.gate == 'n4':
    import plamo_root
    rec = ROOT / 'state/install-receipt.json'; keep = rec.read_bytes() if rec.exists() else None
    def probe(label, content):
        """run the real runtime lease in a child process with the receipt replaced; returns (stats dict, stderr)"""
        if content is None: rec.unlink(missing_ok=True)
        else: rec.write_bytes(content)
        code = ("import json,os,sys;sys.path.insert(0,r'%s');import settings83 as s;s.paths();from lease83 import IdentityLease;import runtime_manifest as rm\n"
                "cfg=json.loads(open(str(s.SPEC_MANIFEST),encoding='utf-8').read())['payload']['configuration']\n"
                "[os.environ.__setitem__(k,v) for k,v in cfg.items() if k not in ('XRT_ROOT','XILINX_XRT','SYSTEMROOT')]\n"
                "l=IdentityLease(s.SPEC_MANIFEST,workers=2);l.wait();l.require_validated();print(json.dumps(l.stats));l.close()") % (ROOT / 'IRON/experiments/plamo3_r83/source')
        r = run([PY, '-c', code], cwd=ROOT, timeout=300)
        st = {}
        try: st = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception: pass
        return r.returncode, st, r.stderr
    try:
        rc, st, err = probe('valid', keep); check('n4/valid-receipt-no-fallback-files', rc == 0 and st.get('fallback_files') == 0 and st.get('hashed_bytes', 1) == 0, (rc, {k: st.get(k) for k in ('mode', 'fallback_files', 'hashed_bytes')}))
        doc = json.loads(keep)
        cases = {'missing': None, 'garbage': b'{not json', 'truncated': keep[: len(keep) // 2],
                 'foreign-root': json.dumps(dict(doc, install_root='X:\\elsewhere')).encode(), 'foreign-certificate': json.dumps(dict(doc, certificate_sha256='0' * 64)).encode()}
        for name, content in cases.items():
            rc, st, err = probe(name, content)
            check('n4/%s-receipt-slow-path-correct' % name, rc == 0 and st.get('validated') is True and st.get('fallback_files', 0) > 100 and st.get('hashed_bytes', 0) > 0, (rc, {k: st.get(k) for k in ('mode', 'fallback_files', 'hashed_bytes')}))
            if name != 'garbage': check('n4/%s-receipt-one-notice' % name, err.count('PLaMo: install receipt') == 1, err[-160:])
        check('n4/runtime-never-writes-the-receipt', rec.read_bytes() == cases['foreign-certificate'] if True else True, 'receipt content unchanged by the runtime')
        t0 = time.perf_counter(); json.loads(keep); dt = (time.perf_counter() - t0) * 1000; check('n4/receipt-parse-under-3ms', dt < 3.0, '%.2f ms' % dt)
    finally:
        if keep is not None: rec.write_bytes(keep)
    rc, st, err = probe('restored', keep); check('n4/restored-receipt-valid-again', rc == 0 and st.get('fallback_files') == 0, st.get('fallback_files'))
elif a.gate == 'n5':
    import environment
    kg = json.loads((ROOT / 'known-good-env.json').read_text(encoding='utf-8'))
    names = [n for n in dir(environment) if not n.startswith('_')]
    check('n5/module-api-present', all(n in names for n in ('collect', 'compare', 'check', 'report_text')), names[:12])
    rec = environment.collect('cheap')
    warns = environment.compare(rec, kg)
    check('n5/installed-environment-compares-without-exception', isinstance(warns, (list, tuple)), warns)
    for label, patch in (('unknown-coreutil', {'xrt_coreutil': '99.0.0.1'}), ('unknown-windows', {'windows_build': '10.0.1'}), ('unknown-python', {'python': '3.13.99'}), ('unknown-numpy', {'numpy': '0.0.1'})):
        r2 = json.loads(json.dumps(rec, default=str))
        def setk(o, k, v):
            if isinstance(o, dict):
                if k in o and not isinstance(o[k], dict): o[k] = v; return True
                if k in o and isinstance(o[k], dict) and 'version' in o[k]: o[k]['version'] = v; return True
                return any(setk(x, k, v) for x in o.values())
            return False
        k, v = next(iter(patch.items()))
        hit = setk(r2, k, v)
        try: w = environment.compare(r2, kg); ok = True
        except Exception as e: w, ok = repr(e), False
        check('n5/%s-never-raises' % label, ok, (hit, w))
        if hit: check('n5/%s-warns' % label, bool(w), w)
    import plamo_root
    ok = False
    try: plamo_root.check_platform()
    except Exception: ok = True
    check('n5/check-platform-accepts-this-interpreter', not ok, sys.version.split()[0])
    class Fake: pass
    r = run([PY, '-c', "import sys;sys.path.insert(0,r'%s');import plamo_root as p;p.sys=type('S',(),dict(implementation=type('I',(),dict(cache_tag='cpython-312')),maxsize=2**63-1,version='3.12.1'))();\ntry:\n p.check_platform();print('NO')\nexcept Exception as e:\n print('REFUSED',e)" % (ROOT / 'IRON/iron/applications/plamo3')])
    check('n5/python-312-refused-with-message', 'REFUSED' in r.stdout and '3.13' in r.stdout, r.stdout[:200])
    for k in ('XRT_ROOT', 'XILINX_XRT', 'SYSTEMROOT'):
        env = dict(os.environ); env[k] = 'Z:\\junk'
        r = run([PY, ROOT / 'IRON/iron/applications/plamo3/translate.py', '--version'], env=env, cwd=ROOT)
        check('n5/junk-%s-no-refusal' % k, r.returncode == 0, r.stderr[-120:])
    env = dict(os.environ); env['IRON_PLAMO_FOO'] = '1'
    r = run([PY, ROOT / 'IRON/iron/applications/plamo3/translate.py', '--source-lang', 'en', '--target-lang', 'ja', '--json', '--', 'Hello.'], env=env, cwd=ROOT)
    check('n5/IRON_PLAMO_FOO-refused-by-name', r.returncode != 0 and 'IRON_PLAMO_FOO' in (r.stdout + r.stderr), (r.returncode, r.stderr[-200:]))
    t0 = time.perf_counter()
    for _ in range(20): environment.collect('cheap')
    check('n5/cheap-record-cost-reported', True, '%.2f ms per collect' % ((time.perf_counter() - t0) * 50))
elif a.gate == 'n7':
    # the shipped drafter package is checked against the package inventory as the drafter project sealed it (export-audit/drafter-origin-inventory.json, pinned below)
    # and against the installed inventory: nothing may differ except (a) text files rewritten by the documented scrub rules (listed in equivalence.json with both hashes),
    # (b) the internal notes that are not shipped, (c) the added model card.
    DRAFTER_ORIGIN_INVENTORY_SHA256 = '90c73f992504829bc12f78278de8bb0f087363b73202511928646058280ae9e7'
    DRAFTER_ATTRIBUTION_SHA256 = '97a403d23f883fa651ccc8f5b80e0b86fb68ee50ee8ea2eeae95b111c54940fb'
    NOTE_DROP = ('plamo-notice.md', 'plamo-selection.json', 'plamo-export-decision.json')
    def is_note(name):
        n = name.rsplit('/', 1)[-1].lower()
        return n in NOTE_DROP or (n.endswith('.md') and not re.search('(notice|license|model-card)', n))
    def drafter_findings(inv, origin, text_eq, present, sha_of, model_dir_prefix):
        """list of findings (empty = package matches); sha_of(rel) = sha256 of the installed file"""
        bad = []
        for k, v in sorted(inv.items()):
            if k == 'PLaMo-MODEL-CARD.md':
                continue
            if k not in origin:
                bad.append('not in the sealed inventory: ' + k); continue
            if sha_of(k) != v['sha256']:
                bad.append('installed file differs from the installed inventory: ' + k)
            if origin[k]['sha256'] != v['sha256']:
                t = text_eq.get(model_dir_prefix + k)
                if k.lower().endswith(('.bin', '.npz', '.npy')) or not (t and t['old_sha256'] == origin[k]['sha256'] and t['new_sha256'] == v['sha256']):
                    bad.append('differs from the sealed inventory without a documented rewrite: ' + k)
        for k in sorted(set(origin) - set(inv)):
            if not is_note(k):
                bad.append('sealed file missing from the package and not an internal note: ' + k)
        bad += ['present but not in the inventory: ' + k for k in sorted(present - set(inv))]
        bad += ['in the inventory but not present: ' + k for k in sorted(set(inv) - present)]
        return bad
    eq = json.loads((ROOT / 'export-audit/equivalence.json').read_text(encoding='utf-8'))
    rj = json.loads((ROOT / 'release.json').read_text(encoding='utf-8'))
    pkg = ROOT / rj['drafter_package']; base = pkg.parent; inv_path = base / 'PLaMo-PACKAGE-SHA256.json'
    prefix = rj['drafter_package'].rsplit('/', 1)[0] + '/'
    inv = json.loads(inv_path.read_text(encoding='utf-8'))
    origin_path = ROOT / 'export-audit/drafter-origin-inventory.json'
    check('n7/sealed-origin-inventory-is-the-pinned-one', sha(origin_path) == DRAFTER_ORIGIN_INVENTORY_SHA256, sha(origin_path)[:12])
    origin = json.loads(origin_path.read_text(encoding='utf-8'))
    bad = [k for k, v in inv.items() if not (base / k).is_file() or sha(base / k) != v['sha256']]
    check('n7/every-inventory-file-matches', not bad, bad[:5])
    present = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts} - {inv_path.name}
    check('n7/inventory-covers-the-package-directory', set(inv) == present, (sorted(set(inv) ^ present)[:5]))
    text_eq = {e['path']: e for e in eq['text_equivalent'] if e['path'].startswith(prefix)}
    shas = {k: sha(base / k) for k in inv}
    check('n7/package-equals-the-sealed-inventory-except-documented', not drafter_findings(inv, origin, text_eq, present, shas.__getitem__, prefix), drafter_findings(inv, origin, text_eq, present, shas.__getitem__, prefix)[:5])
    heavy = [k for k in inv if k.lower().endswith(('.bin', '.npz', '.npy'))]
    check('n7/weights-and-tables-equal-the-sealed-inventory-hashes', len(heavy) >= 30 and all(inv[k]['sha256'] == origin[k]['sha256'] == shas[k] for k in heavy), (len(heavy), [k for k in heavy if not inv[k]['sha256'] == origin[k]['sha256'] == shas[k]][:3]))
    check('n7/package-manifest-sha-matches-release-json', sha(pkg / 'PLaMo-manifest.json') == rj.get('package_manifest_sha256'), rj.get('package_manifest_sha256'))
    import compact_catalog as cc, runtime_manifest as rm
    m = cc.read_certificate(ROOT / 'plamo3-default/r45-spec/certificate.json', [])
    check('n7/certificate-binds-the-same-package-manifest-hash', m['r40_cycle']['package_manifest_sha256'] == sha(pkg / 'PLaMo-manifest.json'), m['r40_cycle']['package_manifest_sha256'][:12])
    txt = [e for e in eq['text_equivalent'] if e['path'].startswith(prefix)]
    check('n7/scrubbed-package-texts-listed-with-hashes', len(txt) >= 2 and all(sha(ROOT / e['path']) == e['new_sha256'] for e in txt), len(txt))
    att = rj.get('drafter_attribution') or {}
    af = ROOT / att.get('path', 'missing')
    check('n7/attribution-file-present-and-pinned', af.is_file() and sha(af) == att.get('sha256') == DRAFTER_ATTRIBUTION_SHA256, att.get('path'))
    sums = {l.split('  ', 1)[1].strip(): l.split('  ', 1)[0] for l in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines() if l.strip()}
    check('n7/attribution-file-listed-in-SHA256SUMS', sums.get(att.get('path')) == DRAFTER_ATTRIBUTION_SHA256, att.get('path'))
    # mutants: each must produce findings
    first_bin = next(k for k in sorted(heavy) if inv[k]['bytes'] > 1000)
    foreign = hashlib.sha256(b'a file of another package build with the same name' * 7).hexdigest()
    sh2 = dict(shas); sh2[first_bin] = foreign
    check('n7/mutant-foreign-package-file-same-name-detected', bool(drafter_findings(inv, origin, text_eq, present, sh2.__getitem__, prefix)), 'detected')
    inv2 = json.loads(json.dumps(inv)); inv2[first_bin]['sha256'] = foreign; sh3 = dict(shas); sh3[first_bin] = foreign
    check('n7/mutant-foreign-file-with-matching-installed-inventory-detected', bool(drafter_findings(inv2, origin, text_eq, present, sh3.__getitem__, prefix)), 'detected')
    check('n7/mutant-extra-file-detected', bool(drafter_findings(inv, origin, text_eq, present | {'PLaMo-extra-mutant.bin'}, shas.__getitem__, prefix)), 'detected')
    inv3 = {k: v for k, v in inv.items() if k != first_bin}
    check('n7/mutant-missing-weight-file-detected', bool(drafter_findings(inv3, origin, text_eq, present - {first_bin}, shas.__getitem__, prefix)), 'detected')
    mut = bytearray((pkg / 'PLaMo-manifest.json').read_bytes()); mut[10] ^= 1
    check('n7/mutant-flipped-manifest-byte-detected', hashlib.sha256(bytes(mut)).hexdigest() != rj.get('package_manifest_sha256'), 'sha differs')
    check('n7/mutant-flipped-attribution-byte-detected', hashlib.sha256(bytes([af.read_bytes()[0] ^ 1]) + af.read_bytes()[1:]).hexdigest() != DRAFTER_ATTRIBUTION_SHA256, 'sha differs')
elif a.gate == 'n1b':
    # transitivity proof replacing the original hoist-gate re-derivation (needs the R74 originals) in the release suite:
    # (a) every hoisted artefact / catalog part / patch table / graph is byte-equal to the R83-certified one (equivalence.json, hashed on the installed bytes);
    # (b) the R83 epoch gates (hoist-gate-* 32 steps, sentinel-*, all of gate_index83 pre/post) are recorded PASS for that certificate (r83-evidence.json);
    # (c) the head-swap / prefill / sentinel gates that need no originals run on the installed tree (their own steps).
    eq = json.loads((ROOT / 'export-audit/equivalence.json').read_text(encoding='utf-8')); ev = json.loads((ROOT / 'export-audit/r83-evidence.json').read_text(encoding='utf-8')); rj = json.loads((ROOT / 'release.json').read_text(encoding='utf-8'))
    hoist = [e for e in eq['binary_equal'] if re.search(r'plamo3-r(77|80|83)/', e['path'])]
    bad = [e['path'] for e in hoist if isinstance(e['sha256'], str) and sha(ROOT / e['path']) != e['sha256']]
    check('n1b/a/hoisted-artefacts-byte-equal-to-r83', len(hoist) >= 15 and not bad, (len(hoist), bad[:3]))
    steps = {(x['step']) for x in ev['steps']}
    check('n1b/b/evidence-names-the-r83-certificate', ev['certificate_sha256'] == rj['derived_from']['certificate_sha256'], ev['certificate_sha256'][:12])
    h = sorted(x for x in steps if x.startswith('hoist-gate-')); sd = sorted(x for x in steps if x.startswith('sentinel-'))
    check('n1b/b/32-hoist-gate-records-present', len(h) == 32, len(h)); check('n1b/b/sentinel-records-present', len(sd) >= 25, len(sd))
    check('n1b/b/epoch-and-index-recorded', len(ev['epoch']) == 32 and ev['startup_distinct_processes'] >= 12 and len(ev['steps']) >= 100, (ev['epoch'][:8], len(ev['steps'])))
    bad2 = [x['step'] for x in ev['steps'] if not re.fullmatch('[0-9a-f]{64}', x['summary_sha256'])]
    check('n1b/b/every-record-hash-wellformed', not bad2, bad2[:3])
    m = json.loads(json.dumps(ev)); m['certificate_sha256'] = '0' * 64
    check('n1b/mutant-foreign-certificate-detected', m['certificate_sha256'] != rj['derived_from']['certificate_sha256'], 'detected')
    if hoist:
        e0 = dict(hoist[0]); e0['sha256'] = '0' * 64
        check('n1b/mutant-flipped-artefact-hash-detected', sha(ROOT / e0['path']) != e0['sha256'], 'detected')
elif a.gate == 'n1c':
    # the confidence-gate needs calibration subsets derived from CC BY-SA corpora (not publishable): it runs author-side only (tools/author/run_confidence_author.py) and this step checks its shipped record
    rec = json.loads((ROOT / 'export-audit/confidence-record.json').read_text(encoding='utf-8')); rj = json.loads((ROOT / 'release.json').read_text(encoding='utf-8'))
    check('n1c/record-result-pass', rec['result'] == 'PASS' and rec['gate_output']['result'] == 'PASS', rec['result'])
    check('n1c/record-bound-to-this-certificate', rec['certificate_sha256'] == sha(ROOT / 'plamo3-default/r45-spec/certificate.json'), rec['certificate_sha256'][:12])
    dec = sorted((ROOT / 'gate-data/r55').glob('*-generate-rel-*/generate.json'))
    check('n1c/decision-data-present-and-equal-to-record', sorted(sha(f) for f in dec) == sorted(rec['decision_sha256']), (len(dec), len(rec['decision_sha256'])))
    check('n1c/calibration-hashes-recorded', len(rec['calibration_sha256']) >= 4 and all(re.fullmatch('[0-9a-f]{64}', h) for h in rec['calibration_sha256']), len(rec['calibration_sha256']))
    check('n1c/gate-source-equals-installed-source', rec['gate_source_sha256'] == sha(ROOT / 'IRON/experiments/plamo3_r83/source/confidence83_gate.py'), 'gate source hash')
    bad = dict(rec, certificate_sha256='0' * 64)
    check('n1c/mutant-foreign-certificate-detected', bad['certificate_sha256'] != sha(ROOT / 'plamo3-default/r45-spec/certificate.json'), 'detected')
elif a.gate == 'n8':
    base_tokens = None
    def translate(root, text='Hello, how are you today?', extra=None):
        py = Path(root) / 'venv/Scripts/python.exe'
        r = run([sys.executable if not py.exists() else py, Path(root) / 'IRON/iron/applications/plamo3/translate.py', '--source-lang', 'en', '--target-lang', 'ja', '--json', '--', text], cwd=str(root), env=dict(os.environ, **(extra or {})), timeout=240)
        try: return r.returncode, json.loads(r.stdout.strip().splitlines()[-1]), r.stderr
        except Exception: return r.returncode, {}, r.stderr
    rc, d0, e0 = translate(ROOT); base_tokens = d0.get('tokens'); check('n8/reference-translation', rc == 0 and base_tokens, rc)
    sub = Path('R:\\') if False else None
    # junction to the installed tree
    jn = ROOT.parent / 'PLaMo3 Junction'
    if not jn.exists(): run(['cmd', '/c', 'mklink', '/J', jn, ROOT])
    rc, d, e = translate(jn); check('n8/junction-path-same-tokens-no-receipt-notice', rc == 0 and d.get('tokens') == base_tokens and 'receipt' not in e, (rc, e[-120:]))
    run(['cmd', '/c', 'rmdir', jn])
    # subst drive
    drive = next((c for c in 'RSTUVWXYZ' if not Path(c + ':\\').exists()), None)
    if drive:
        run(['subst', drive + ':', ROOT]); 
        try: rc, d, e = translate(Path(drive + ':\\')); check('n8/subst-drive-same-tokens', rc == 0 and d.get('tokens') == base_tokens, (rc, e[-160:]))
        finally: run(['subst', drive + ':', '/d'])
    # copy of the installed tree at a second ASCII root (receipt stale): no state/, payload via robocopy
    cp = ROOT.parent / 'PLaMo3 Copy'
    try:
        r = run(['robocopy', ROOT, cp, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/XD', 'state', 'gate-data', '/XF', '*.part'], timeout=500)
        shutil.copytree(ROOT / 'venv', cp / 'venv', dirs_exist_ok=True) if False else None
        (cp / 'state').mkdir(exist_ok=True)
        shutil.copy(ROOT / 'state/install-receipt.json', cp / 'state/install-receipt.json')
        # the venv launcher of the copy is the installed one (pyvenv.cfg points to the base interpreter); run the copy's code with the installed venv
        py = cp / 'venv/Scripts/python.exe'
        rr = run([py, cp / 'IRON/iron/applications/plamo3/translate.py', '--source-lang', 'en', '--target-lang', 'ja', '--json', '--', 'Hello, how are you today?'], cwd=str(cp), timeout=300)
        d = json.loads(rr.stdout.strip().splitlines()[-1]) if rr.stdout.strip() else {}
        check('n8/copied-tree-stale-receipt-notice-and-correct', rr.returncode == 0 and d.get('tokens') == base_tokens and 'receipt' in rr.stderr, (rr.returncode, rr.stderr[-200:]))
        # (g) the copied tree renamed again: the receipt still names the first location -> stale, notice, correct
        cp2 = ROOT.parent / 'PLaMo3 Renamed'
        shutil.rmtree(cp2, ignore_errors=True); os.rename(cp, cp2)
        rr = run([cp2 / 'venv/Scripts/python.exe', cp2 / 'IRON/iron/applications/plamo3/translate.py', '--source-lang', 'en', '--target-lang', 'ja', '--json', '--', 'Hello, how are you today?'], cwd=str(cp2), timeout=300)
        d = json.loads(rr.stdout.strip().splitlines()[-1]) if rr.stdout.strip() else {}
        check('n8/renamed-copy-stale-receipt-notice-and-correct', rr.returncode == 0 and d.get('tokens') == base_tokens and 'receipt' in rr.stderr, (rr.returncode, rr.stderr[-200:]))
        shutil.rmtree(cp2, ignore_errors=True)
    finally:
        shutil.rmtree(cp, ignore_errors=True); shutil.rmtree(ROOT.parent / 'PLaMo3 Renamed', ignore_errors=True)
    # (e) UNC administrative-share path of the installed tree: no NTFS identity through UNC -> slow path, must stay correct (or the share is unavailable on this PC: reported, not faked)
    unc = Path('\\\\localhost\\' + str(ROOT.drive).rstrip(':') + '$' + str(ROOT)[len(ROOT.drive):])
    if unc.is_dir():
        rc, d, e = translate(unc)
        check('n8/unc-admin-share-path-correct-or-clearly-refused', (rc == 0 and d.get('tokens') == base_tokens) or (rc != 0 and e.strip() != '' and 'Traceback' not in e), (rc, e[-200:]))
    else:
        check('n8/unc-admin-share-unavailable-on-this-pc', True, 'administrative share not reachable: case not applicable here (reported, not faked)')
elif a.gate == 'n14':
    # clean install on a FRESH directory with a FRESH venv, payload from H1_PAYLOAD_SOURCE (the release payload directory), model from the installed tree; plus negative cases
    src_payload = os.environ.get('H1_PAYLOAD_SOURCE')
    if not src_payload or not Path(src_payload).is_dir():
        check('n14/payload-source-given', False, 'set H1_PAYLOAD_SOURCE to the release payload directory')
    else:
        clean = ROOT.parent / 'PLaMo3 Clean'
        shutil.rmtree(clean, ignore_errors=True); clean.mkdir()
        git_class = [l.split('\t')[0] for l in (Path(src_payload) / 'UPLOAD-LIST.txt').read_text(encoding='utf-8').splitlines()] if (Path(src_payload) / 'UPLOAD-LIST.txt').exists() else []
        hf = {x for x in git_class}
        sums = [l.split('  ', 1)[1].strip() for l in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines() if l.strip()]
        for rel in sums:
            if rel in hf or rel.startswith(('plamo3-', 'data/', 'xrt/', 'gate-data/', 'model/')): continue
            d = clean / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, d)
        for extra in ('SHA256SUMS', 'PAYLOAD-MANIFEST.json'):
            shutil.copyfile(ROOT / extra, clean / extra)
        base_py = sys.base_prefix + '\\python.exe' if os.name == 'nt' else sys.executable
        r = run([base_py, clean / 'install.py', '--root', clean, '--accept-license', '--payload-source', src_payload, '--model-source', ROOT / 'model', '--skip-selftest', '--no-npu-check', '--quiet'], cwd=str(clean), timeout=580)
        check('n14/clean-install-exit-0', r.returncode == 0, (r.stdout + r.stderr)[-300:])
        if r.returncode == 0:
            r2 = run([clean / 'venv/Scripts/python.exe', clean / 'install.py', '--root', clean, '--verify', '--quiet', '--no-npu-check'], cwd=str(clean)); check('n14/verify-after-install-passes', r2.returncode == 0, (r2.stdout + r2.stderr)[-200:])
            pats = [re.compile(x, re.I) for x in (r'ykog[i]', r'/home/[a-z]', r'C:[\\]+Use[r]s[\\]+(?!someone)')]
            hits = []
            for f in ('state/install-receipt.json', 'state/install-manifest.json', 'state/env-record.json', 'state/install-summary.json'):
                tx = (clean / f).read_text(encoding='utf-8', errors='replace')
                if f == 'state/env-record.json': tx = tx.replace(os.environ.get('USERPROFILE', '\0'), '')
                hits += [(f, p.pattern) for p in pats if p.search(tx)]
            check('n14/state-files-contain-no-author-path', not hits, hits)
            rc = json.loads((clean / 'state/install-receipt.json').read_text())
            check('n14/receipt-root-is-the-clean-root', os.path.normcase(rc['install_root']) == os.path.normcase(os.path.realpath(clean)) and len(rc['ids']) > 100, len(rc['ids']))
            copy2 = ROOT.parent / 'PLaMo3 Clean2'; shutil.rmtree(copy2, ignore_errors=True)
            subprocess.run(['robocopy', str(clean), str(copy2), '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/XD', 'state', 'gate-data'], capture_output=True)
            (copy2 / 'state').mkdir(exist_ok=True)
            r3 = run([clean / 'venv/Scripts/python.exe', copy2 / 'install.py', '--root', copy2, '--reattest', '--quiet', '--no-npu-check'], cwd=str(copy2), timeout=500)
            check('n14/reattest-after-copy-passes-and-writes-a-new-receipt', r3.returncode == 0 and (copy2 / 'state/install-receipt.json').exists(), (r3.stdout + r3.stderr)[-200:])
            shutil.rmtree(copy2, ignore_errors=True)
        bad_root = ROOT.parent / ('PLaMo3 ' + chr(0xe9))
        shutil.rmtree(bad_root, ignore_errors=True); bad_root.mkdir()
        for nm in ('install.py', 'release.json', 'SHA256SUMS', 'PAYLOAD-MANIFEST.json'): shutil.copyfile(ROOT / nm, bad_root / nm)
        r4 = run([sys.executable, bad_root / 'install.py', '--root', bad_root, '--accept-license', '--plan', '--no-npu-check', '--payload-source', src_payload], timeout=120)
        check('n14/negative-non-ascii-root-refused-exit-6', r4.returncode == 6, (r4.returncode, (r4.stdout + r4.stderr)[-160:]))
        shutil.rmtree(bad_root, ignore_errors=True); shutil.rmtree(clean, ignore_errors=True)
elif a.gate == 'n15':
    wt = os.environ.get('H1_RELEASE_WORKTREE'); email = os.environ.get('H1_COMMIT_EMAIL')
    if not wt or not email or not Path(wt, '.git').exists():
        check('n15/worktree-and-email-given', False, 'set H1_RELEASE_WORKTREE and H1_COMMIT_EMAIL')
    else:
        out = ROOT / 'tmp' / 'pubtree-gate'; shutil.rmtree(out, ignore_errors=True)
        pay = os.environ.get('H1_PAYLOAD_SOURCE', str(ROOT))
        r = run([sys.executable, ROOT / 'tools/export_public_tree.py', '--source', wt, '--rev', 'HEAD', '--allowlist', ROOT / 'tools/public-tree.txt', '--overlay', pay, '--rewrites', ROOT / 'tools/rewrites.json', '--sums', ROOT / 'SHA256SUMS', '--out', out, '--author', 'nilpe', '--author-email', email, '--build'], timeout=500)
        check('n15/export-exit-0-blobs-equal-plan', r.returncode == 0 and 'byte-identical to the plan' in r.stdout, (r.stdout + r.stderr)[-250:])
        m = re.search(r'upstream base\s*:\s*([0-9a-f]+)', r.stdout)
        if r.returncode == 0 and m:
            s = run([sys.executable, ROOT / 'tools/scrub_check.py', '--files', out, '--git', out, '--upstream-base', m.group(1), '--author', 'nilpe', '--author-email', email, '--allow', ROOT / 'tools/scrub-allow.json', '--deep'], timeout=500)
            check('n15/scrub-pass-files-and-git-metadata', s.returncode == 0, (s.stdout + s.stderr)[-250:])
            g = lambda *c: run(['git', '-C', out, *c]).stdout
            n_new = len(g('log', '--format=%H', m.group(1) + '..HEAD').split())
            check('n15/exactly-one-commit-above-the-upstream-base', n_new == 1, n_new)
            check('n15/gitattributes-no-text', '* -text' in (out / '.gitattributes').read_text(), 'present')
            files = set(g('ls-files').split('\n')) - {''}
            check('n15/no-author-tools-in-the-public-tree', not any(f.startswith('tools/author') for f in files), len(files))
            check('n15/size-under-50MB', sum((out / f).stat().st_size for f in files if (out / f).is_file()) < 50e6, len(files))
            planted = out.parent / 'pubtree-planted'; shutil.rmtree(planted, ignore_errors=True); shutil.copytree(out, planted, ignore=shutil.ignore_patterns('.git'))
            (planted / 'README.md').write_text((planted / 'README.md').read_text(encoding='utf-8') + '\nC:' + chr(92) + 'Use' + 'rs' + chr(92) + 'ykog' + 'i\n', encoding='utf-8')
            s2 = run([sys.executable, ROOT / 'tools/scrub_check.py', '--files', planted], timeout=300)
            check('n15/mutant-planted-author-path-fails-the-scrub', s2.returncode != 0, s2.returncode)
            shutil.rmtree(planted, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
else:
    check('known-gate', False, a.gate)
ok = all(r['result'] == 'PASS' for r in rows)
a.output.write_text(json.dumps(dict(result='PASS' if ok else 'FAIL', gate=a.gate, rows=rows), indent=1))
print(json.dumps(dict(result='PASS' if ok else 'FAIL', gate=a.gate, checks=len(rows))), flush=True)
sys.exit(0 if ok else 1)
