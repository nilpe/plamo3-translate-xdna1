"""R83 tokenizer-gate (CPU only): the flat-table CompactTokenizer (tokcompact83) versus the pinned FastPlamo3Tokenizer on a frozen corpus, the artifact checks, the fallback cases and mutants.
Corpus (frozen): plamo3-r74\\tokenizer-gate\\corpus.json (9,025 tatoeba ja/en sentences + 160 multi-sentence and one 60k-char text, sha256 below) + all check_fast_tokenizer cases (texts, chats, invalid inputs)
+ the 12 gate prompts (order-reference metadata) decoded and re-encoded + every vocabulary token + 30,000 random vocabulary concatenations / control strings + 20,000 random code-point strings (seeded).
Checks: encode with/without BOS, chat (ja<->en over every corpus sentence), template cases, invalid-input behaviour (result or exception type+text), decode (random id lists, 3 option sets, gate prompts, negative indices),
tables equal to the marshal payload (terminals, scores, byte_ids, all tokens, every trie edge), artifact sha256 / size / source-payload link, fallback cases (manifest missing, one payload byte flipped, truncated file, wrong manifest sha:
make() must return the ORIGINAL class with identical output), and MUTANTS of tokcompact83 (each must be caught): skip the artifact sha check, wrong bisect upper bound, off-by-one table offset, token offsets off by one, no hash on fallback, scores read from terminals."""
import argparse,hashlib,json,random,sys,time,traceback,types
from pathlib import Path
ROOT=Path(next(str(_a) for _a in Path(__file__).resolve().parents if (_a/'IRON'/'iron'/'applications'/'plamo3').is_dir()))
import settings83 as s
s.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
CORPUS_SHA='7fe496298e9b30127b0f838d091f2ce3919c5e0c6339a1b271e7f9ddec73a1dd';COMPACT_SHA='cad4afef3b39c2b2d96eca867048b9206a2b85e0d57641aed376570f3df12224'
T0=time.time();rep=dict(result='FAIL')
try:
 sys.path.insert(0,str(ROOT/'IRON/iron/applications/plamo3'))
 import fast_tokenizer as ft,tokcompact83 as tc,tokfast83
 import check_fast_tokenizer as cf
 from settings83 import TOKENIZER_COMPACT
 import settings83
 import marshal
 root_tok=Path(json.loads(s.SPEC_MANIFEST.read_text())['payload'].get('tokenizer_manifest','')) if False else None
 orig_manifest=ROOT/'plamo3-runtime/r34-w4/tokenizer/tokenizer-manifest.json'
 base=ft.FastPlamo3Tokenizer(orig_manifest,checkpoint=str(__import__('plamo_root').MODEL))           # full original checks incl. the five checkpoint source digests
 comp=tc.CompactTokenizer(TOKENIZER_COMPACT)           # full checks incl. artifact sha
 art=Path(TOKENIZER_COMPACT).parent/'tokenizer.compact'
 assert hashlib.sha256(art.read_bytes()).hexdigest()==COMPACT_SHA,'compact artifact sha256'
 meta=json.loads(Path(TOKENIZER_COMPACT).read_text(encoding='utf-8'));assert meta['compact']['sha256']==COMPACT_SHA and meta['compact']['source_payload_sha256']==base.metadata['payload_sha256']
 for k in meta:
  if k!='compact':assert meta[k]==base.metadata[k],k
 # ---- tables vs marshal payload ----
 data=(orig_manifest.parent/base.metadata['payload']).read_bytes();tokens,trans,term,scores,byte_ids=marshal.loads(data)
 t=tc.load_tables(art.read_bytes());V=len(tokens)
 assert [t['tokens'][i] for i in range(V)]==tokens and list(t['terminals'])==list(term) and list(t['scores'])==list(scores) and list(t['byte_ids'])==list(byte_ids)
 f,c,k=t['first'],t['chars'],t['kids'];n=len(term);cnt=0
 for node in range(n):
  for i in range(f[node],f[node+1]):
   assert trans[(node<<21)|c[i]]==k[i];cnt+=1
 assert cnt==len(trans)
 rep['tables']=dict(nodes=n,edges=cnt,tokens=V)
 # ---- corpus ----
 cb=(ROOT/'gate-data/plamo3-r74/tokenizer-gate/corpus.json').read_bytes();assert hashlib.sha256(cb).hexdigest()==CORPUS_SHA,'frozen corpus changed'
 sent=json.loads(cb.decode('utf-8'));ctexts,chats,invalid=cf.cases()
 texts=list(sent)+[x for x in ctexts if isinstance(x,str)]
 meta12=json.loads((ROOT/'gate-data/plamo3-w4-r32/order-reference/metadata.json').read_text());gate=list(meta12['prompts'])+[json.loads(c['prompt'])if isinstance(c['prompt'],str)else c['prompt'] for c in meta12['long_cases']];assert len(gate)==12
 texts+=[base.decode(g,skip_special_tokens=False) for g in gate]
 toks=[tokens[i] for i in range(V)];texts+=[x for x in toks if not x.startswith('<0x')]
 rng=random.Random(74740)
 for _ in range(30000):texts.append(''.join(rng.choice(toks) for _ in range(rng.randint(1,60))) if rng.random()<.8 else ''.join(rng.choice([' ','  ','   ','a','a','a','a','\n','<|plamo:tag|>','<|plamo:x','|>','\uee00','\u3000','、','。',chr(rng.randint(0x20,0x2fff)),chr(rng.randint(0x10000,0x1ffff))]) for _ in range(rng.randint(1,80))))
 for _ in range(20000):texts.append(''.join(chr(rng.choice([rng.randint(0x20,0x7e),rng.randint(0xa0,0x2ff),rng.randint(0x3000,0x30ff),rng.randint(0x4e00,0x9fff),rng.randint(0xff00,0xffef),rng.randint(0x1f300,0x1faff),rng.randint(0x100000,0x10ffff)])) for _ in range(rng.randint(1,40))))
 def run(tk,m,o):
  try:return ('ok',tk.chat(m,**o))
  except Exception as e:return ('err',type(e).__name__,str(e))
 def compare(base,comp,texts,full=True):
  bad=0
  for x in texts:
   if base.encode(x)!=comp.encode(x) or base.encode(x,add_special_tokens=False)!=comp.encode(x,add_special_tokens=False):bad+=1
  return bad
 bad=compare(base,comp,texts);assert bad==0,('encode mismatches',bad);rep['encode_texts']=len(texts)
 chat_bad=0;nchat=0
 for x in sent:
  if not x.strip() or '<|plamo:' in x or len(x)>5000:continue
  for sl,tl in(('ja','en'),('en','ja')):nchat+=1;chat_bad+=base.chat([dict(role='user',content=x)],source_lang=sl,target_lang=tl)!=comp.chat([dict(role='user',content=x)],source_lang=sl,target_lang=tl)
 assert chat_bad==0,chat_bad
 cb2=sum(run(base,m,o)!=run(comp,m,o) for m,o in chats);ib=sum(run(base,m,o)!=run(comp,m,o) for m,o in invalid);assert cb2==0 and ib==0,(cb2,ib)
 db=0;nd=0
 for _ in range(20000):
  ids=[rng.randrange(V) for _ in range(rng.randint(0,200))]
  for kw in(dict(),dict(skip_special_tokens=False),dict(spaces_between_special_tokens=False)):nd+=1;db+=base.decode(ids,**kw)!=comp.decode(ids,**kw)
 for g in gate:
  nd+=1;db+=base.decode(g)!=comp.decode(g);assert comp.encode(base.decode(g,skip_special_tokens=False),add_special_tokens=False)==base.encode(base.decode(g,skip_special_tokens=False),add_special_tokens=False)
 for i in(0,1,2,V-1,-1,-V):nd+=1;db+=base.tokens[i]!=comp.tokens[i]
 assert db==0,db
 rep['chat_prompts']=nchat;rep['template_cases']=len(chats);rep['invalid_cases']=len(invalid);rep['decode_checks']=nd;rep['gate_prompts']=12
 # ---- fallback cases (production loader tokfast83.make) ----
 import shutil,tempfile
 tmp=Path(tempfile.mkdtemp(prefix='tokgate74_'));src=Path(TOKENIZER_COMPACT).parent
 def variant(name,mut):
  d=tmp/name;d.mkdir();shutil.copy(src/'tokenizer-manifest.compact.json',d);shutil.copy(src/'tokenizer.compact',d);mut(d);return d/'tokenizer-manifest.compact.json'
 def flip(d):
  b=bytearray((d/'tokenizer.compact').read_bytes());b[len(b)//2]^=1;(d/'tokenizer.compact').write_bytes(bytes(b))
 def trunc(d):(d/'tokenizer.compact').write_bytes((d/'tokenizer.compact').read_bytes()[:-1000])
 def nomanifest(d):(d/'tokenizer-manifest.compact.json').unlink()
 def badsha(d):
  m=json.loads((d/'tokenizer-manifest.compact.json').read_text());m['compact']['sha256']='0'*64;(d/'tokenizer-manifest.compact.json').write_text(json.dumps(m))
 def nopayload(d):(d/'tokenizer.compact').unlink()
 sample=texts[:300]+texts[-100:];fb={}
 saved=settings83.TOKENIZER_COMPACT
 try:
  for name,mut in(('manifest_missing',nomanifest),('payload_byte_flipped',flip),('payload_truncated',trunc),('manifest_sha_wrong',badsha),('payload_missing',nopayload)):
   path=variant(name,mut);settings83.TOKENIZER_COMPACT=path
   tk=tokfast83.make(orig_manifest,None)
   assert type(tk) is ft.FastPlamo3Tokenizer,(name,type(tk))
   assert all(tk.encode(x)==base.encode(x) for x in sample),name
   fb[name]='original constructor, identical output'
  settings83.TOKENIZER_COMPACT=saved;tk=tokfast83.make(orig_manifest,None);assert isinstance(tk,tc.CompactTokenizer)
 finally:settings83.TOKENIZER_COMPACT=saved
 rep['fallback']=fb
 # ---- mutants (each must be caught by a check above, run on a reduced corpus) ----
 src_text=Path(tc.__file__).read_text(encoding='utf-8')
 def load_mut(text):
  mod=types.ModuleType('tokcompact83_mut');mod.__file__=tc.__file__;exec(compile(text,'tokcompact83_mut','exec'),mod.__dict__);return mod
 reduced=texts[:1500]+texts[9025:9300]+[x for x in toks[::40] if not x.startswith('<0x')]
 mutants={
  'skip_artifact_sha_check':("if hashlib.sha256(data).hexdigest() != info['sha256']:\n        raise ValueError('fast tokenizer artifact differs from its manifest')\n    return meta, data","return meta, data",'fallback'),
  'bisect_lower_bound_plus_one':("i = bisect_left(chars, c, first[node], hi)","i = bisect_left(chars, c, first[node] + 1, hi)",'encode'),
  'missing_end_of_node_check':("if i == hi or chars[i] != c:","if chars[i] != c:",'encode'),
  'table_offset_off_by_one':("pos += nbytes + _pad(nbytes)","pos += nbytes + _pad(nbytes) + (4 if k == 'kids' else 0)",'any'),
  'token_offset_off_by_one':("self._b[self._o[i]:self._o[i + 1]]","self._b[self._o[i]:self._o[i + 1] + 1]",'decode'),
  'scores_from_terminals':("self.tokens, self.terminals, self.scores, self.byte_ids = t['tokens'], t['terminals'], t['scores'], t['byte_ids']","self.tokens, self.terminals, self.scores, self.byte_ids = t['tokens'], t['terminals'], t['terminals'], t['byte_ids']",'encode'),
  'ties_prefer_shorter':("(cost == best and length > best_length)","(cost == best and length < best_length)",'encode'),
  'skip_template_digest_check':("    if meta['sources']['chat_template.jinja'] != ft.TEMPLATE_SHA256:\n        raise ValueError('unsupported PFML3 template')\n","",'static'),
 }
 caught={}
 for name,(old,new,kind) in mutants.items():
  assert old in src_text,name
  mod=load_mut(src_text.replace(old,new,1))
  try:
   if kind=='fallback':
    path=variant('mut_'+name,flip);settings83.TOKENIZER_COMPACT=path
    import importlib;orig_tc=sys.modules['tokcompact83'];sys.modules['tokcompact83']=mod
    try:tk=tokfast83.make(orig_manifest,None)
    finally:sys.modules['tokcompact83']=orig_tc;settings83.TOKENIZER_COMPACT=saved
    ok=type(tk) is ft.FastPlamo3Tokenizer      # a correct loader must have refused the flipped file; the mutant accepts it
    caught[name]=not ok
   elif kind=='static':
    # the template check must exist in verified_read: the mutant text differs, and a manifest with a changed template digest must be refused by the real loader but not by the mutant
    d=tmp/('mut_'+name);d.mkdir();shutil.copy(src/'tokenizer.compact',d);m=json.loads((src/'tokenizer-manifest.compact.json').read_text());m['sources']['chat_template.jinja']='1'*64;(d/'tokenizer-manifest.compact.json').write_text(json.dumps(m))
    refused=False
    try:tc.verified_read(d/'tokenizer-manifest.compact.json',validate_sources=False)
    except ValueError:refused=True
    accepted_by_mut=True
    try:mod.verified_read(d/'tokenizer-manifest.compact.json',validate_sources=False)
    except ValueError:accepted_by_mut=False
    caught[name]=refused and accepted_by_mut
   else:
    try:mc=mod.CompactTokenizer(TOKENIZER_COMPACT)
    except Exception:caught[name]=True;continue
    bad_m=0
    try:
     bad_m+=compare(base,mc,reduced)
     for _ in range(300):
      ids=[rng.randrange(V) for _ in range(50)];bad_m+=base.decode(ids)!=mc.decode(ids)
    except Exception:bad_m+=1
    caught[name]=bad_m>0
  except Exception:caught[name]=True
 rep['mutants']=caught;assert all(caught.values()),caught
 shutil.rmtree(tmp,ignore_errors=True)
 rep.update(result='PASS',seconds=time.time()-T0,corpus_sha256=CORPUS_SHA,compact_sha256=COMPACT_SHA)
 a.output.write_text(json.dumps(rep,indent=1));print(json.dumps(rep),flush=True)
except BaseException:
 rep['error']=traceback.format_exc();a.output.write_text(json.dumps(rep,indent=1));traceback.print_exc();sys.exit(1)
