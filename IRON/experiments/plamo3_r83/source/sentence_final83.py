"""R83 gate: the controller's SENTENCE_FINAL_IDS table equals the table re-derived from the bound tokenizer vocabulary."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import settings83 as s
s.paths()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
try:
 from fast_tokenizer import FastPlamo3Tokenizer
 manifest=__import__('plamo_root').resolve_token(json.loads(s.W4_MANIFEST.read_text())['payload']['tokenizer_manifest'])
 tokenizer=FastPlamo3Tokenizer(manifest,checkpoint=str(__import__('plamo_root').MODEL));tokens=tokenizer.tokens
 assert len(tokens)==107520
 def raw(x):return x if isinstance(x,bytes) else x.encode('utf-8','surrogateescape')
 TERM=(b'\xe3\x80\x82',b'\xef\xbc\x81',b'\xef\xbc\x9f',b'.',b'!',b'?')
 derived=[i for i,x in enumerate(tokens) if raw(x).endswith(TERM)]
 source=Path(__file__).with_name('loop83.py');node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)and n.targets[0].id=='SENTENCE_FINAL_IDS')
 table=list(ast.literal_eval(node.value));assert table==derived and len(set(table))==len(table),'table differs from tokenizer derivation'
 eos=json.loads(s.W4_MANIFEST.read_text())['payload']['model']['eos'];assert not set(eos)&set(table),'EOS ids must never be sentence-final'
 # Representative checks through the real decoder.
 for text in('ありがとうございます。','Thank you.','Really?','Stop!'):
  ids=tokenizer.encode(text,add_special_tokens=False);assert ids[-1] in set(table),(text,ids)
 report=dict(result='PASS',ids=len(table),sha256=hashlib.sha256(json.dumps(table).encode()).hexdigest(),vocab=len(tokens),eos=eos,scope='controller table == tokenizer-derived table (all 107520 pieces); EOS excluded; real encodings of final-punctuation texts end in table ids')
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
except BaseException:
 import traceback;e=traceback.format_exc();print(e,flush=True);a.output.write_text(json.dumps(dict(result='FAIL',error=e),indent=2));sys.exit(1)
