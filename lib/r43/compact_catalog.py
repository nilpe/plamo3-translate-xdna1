"""Typed exact-case catalogs. Every key/slice is validated, without per-row JSON objects."""
from collections.abc import Mapping
import hashlib,json,marshal,struct,time
from pathlib import Path
import numpy as np
MAGIC='PLaMo-R43-catalog-v1'
class Cases(Mapping):
 def __init__(self,record):
  required={'arity','count','artifact','codec','fields','keys','slices'}
  if type(record)is not dict or set(record)!=required:raise ValueError('invalid compact case record')
  self.record=record;self.n=record['count'];self.arity=record['arity'];self.artifact=record['artifact'];self.codec=record['codec'];self.fields=record['fields']
  if type(self.n)is not int or not 1<=self.n<=1000000 or type(self.arity)is not int or not 0<=self.arity<=8:raise ValueError('invalid compact case dimensions')
  if type(record['keys'])is not bytes or type(record['slices'])is not bytes or len(record['keys'])!=self.n*self.arity*8 or len(record['slices'])!=self.n*16:raise ValueError('invalid compact byte extent')
  if type(self.fields)is not list or len(set(self.fields))!=len(self.fields)or not {'artifact','nbytes'}<=set(self.fields)<= {'artifact','offset','nbytes','decoded_nbytes','codec'}:raise ValueError('invalid compact part fields')
  if type(self.artifact)is not str or type(self.codec)is not str:raise ValueError('invalid compact artifact/codec')
  self.keys=np.frombuffer(record['keys'],'<i8').reshape(self.n,self.arity);self.parts=np.frombuffer(record['slices'],dtype=[('offset','<u8'),('nbytes','<u4'),('decoded','<u4')]);self.cache={};self.axes=None;self.index=None
 def validate(self,params,artifacts):
  if self.arity!=len(params):raise ValueError('compact key arity differs')
  if self.artifact not in artifacts:raise ValueError('compact case references unknown artifact')
  if self.codec=='zlib'and'decoded_nbytes'not in self.fields:raise ValueError('compressed compact slice needs decoded size')
  if self.codec not in('raw','zlib'):raise ValueError('unsupported compact instruction codec')
  if self.arity:
   order=np.lexsort(tuple(self.keys[:,i]for i in range(self.arity-1,-1,-1)))
   if not np.array_equal(order,np.arange(self.n)):raise ValueError('compact keys not in exact numeric order')
   if self.n>1 and np.any(np.all(self.keys[1:]==self.keys[:-1],axis=1)):raise ValueError('duplicate compact keys')
   axes=[np.unique(self.keys[:,i])for i in range(self.arity)];product=1
   for axis in axes:product*=len(axis)
   if product==self.n:
    self.axes=[]
    for axis in axes:
     step=int(axis[1]-axis[0])if len(axis)>1 else 1
     if step<=0 or not np.array_equal(axis,np.arange(len(axis),dtype=np.int64)*step+axis[0]):self.axes=None;break
     self.axes.append((int(axis[0]),step,len(axis)))
  elif self.n!=1:raise ValueError('duplicate empty compact keys')
  off=self.parts['offset'];
  if'offset'not in self.fields and np.any(off):raise ValueError('implicit compact offset must be zero')
  if'codec'not in self.fields and self.codec!='raw':raise ValueError('implicit compact codec must be raw')
  size=self.parts['nbytes'];decoded=self.parts['decoded'];limit=artifacts[self.artifact]['nbytes']
  # Subtraction avoids uint64 addition overflow. Same bounds as _check_slice.
  if type(limit)is not int or limit<0 or np.any(off>limit)or np.any(size>limit-off):raise ValueError('compact instruction slice exceeds artifact')
  if np.any(decoded==0)or np.any(decoded>64*1024*1024)or np.any(decoded%4):raise ValueError('compact instructions must be bounded uint32 words')
  if self.codec=='zlib'and np.any(size==0):raise ValueError('empty compact compressed slice')
  if self.codec=='raw'and np.any(decoded!=size):raise ValueError('raw compact decoded size mismatch')
  return self.n
 def __len__(self):return self.n
 def __iter__(self):
  for row in self.keys:yield ','.join(str(int(v))for v in row)
 def __getitem__(self,key):
  try:
   if type(key)is not str:raise KeyError(key)
   values=tuple(int(v)for v in key.split(','))if key else()
   if len(values)!=self.arity or ','.join(map(str,values))!=key:raise KeyError(key)
   if not values:row=0
   elif self.axes is not None:
    row=0
    for value,(start,step,n)in zip(values,self.axes):
     delta=value-start
     if delta<0 or delta%step or delta//step>=n:raise KeyError(key)
     row=row*n+delta//step
   else:
    if self.index is None:self.index={tuple(map(int,k)):i for i,k in enumerate(self.keys)}
    row=self.index[values]
   if row not in self.cache:
    part=self.parts[row];d={'artifact':self.artifact,'nbytes':int(part['nbytes'])}
    if'offset'in self.fields:d['offset']=int(part['offset'])
    if'codec'in self.fields:d['codec']=self.codec
    if'decoded_nbytes'in self.fields:d['decoded_nbytes']=int(part['decoded'])
    self.cache[row]=d
   return self.cache[row]
  except(ValueError,OverflowError):raise KeyError(key)from None

def pack(manifest):
 import copy
 m=copy.deepcopy(manifest);packed=0
 for desc in m['instructions'].values():
  cases=desc['cases'];parts=list(cases.values());fields=sorted(parts[0]);artifact=parts[0]['artifact'];codec=parts[0].get('codec','raw');arity=len(desc['params'])
  if not set(fields)<={'artifact','offset','nbytes','decoded_nbytes','codec'}or any(sorted(p)!=fields or p['artifact']!=artifact or p.get('codec','raw')!=codec for p in parts):continue
  rows=[]
  for key,p in cases.items():
   values=tuple(int(v)for v in key.split(','))if key else()
   if len(values)!=arity or ','.join(map(str,values))!=key:raise ValueError('original case key is not canonical')
   for field in('offset','nbytes','decoded_nbytes'):
    if field in p and(type(p[field])is not int or p[field]<0):raise ValueError('invalid original integer field')
   rows.append((values,p))
  rows.sort(key=lambda row:row[0]);keys=b''.join(struct.pack('<'+'q'*arity,*v)for v,p in rows);slices=b''.join(struct.pack('<QII',p.get('offset',0),p['nbytes'],p.get('decoded_nbytes',p['nbytes']))for v,p in rows);desc['cases']={'r43_cases':dict(arity=arity,count=len(rows),artifact=artifact,codec=codec,fields=fields,keys=keys,slices=slices)};packed+=len(rows)
 return marshal.dumps(dict(magic=MAGIC,manifest=m),4),packed

def decode(blob):
 data=marshal.loads(blob,allow_code=False)
 if type(data)is not dict or set(data)!= {'magic','manifest'}or data['magic']!=MAGIC:raise ValueError('unknown binary catalog')
 m=data['manifest']
 if type(m)is not dict:raise ValueError('binary manifest must be an object')
 for desc in m['instructions'].values():
  if type(desc['cases'])is dict and set(desc['cases'])=={'r43_cases'}:desc['cases']=Cases(desc['cases']['r43_cases'])
 # H1: resolve @ROOT@/@MODEL@ placeholders and overlay the install receipt; strictly after the caller's digest check of `blob`. Paths without a placeholder are unchanged.
 import plamo_root
 def notify(message,key=None):
  try:
   from plamo_errors import note
   note(message,key=key)
  except Exception:pass
 plamo_root.prepare(m,ValueError,notify)
 return m

def expand(m):
 import copy
 result=dict(m);result['instructions']={k:{**d,'cases':dict(d['cases'].items())}for k,d in m['instructions'].items()};return result

def read_certificate(path,owners):
 import runtime_manifest as base
 c=base._read_manifest(path)
 if c.get('kind')!=MAGIC:raise base.StaleManifest('not an R43 compact certificate')
 values={}
 for name in('catalog','expanded_manifest'):
  record=c[name];file=base._record_path(record);stream=base._open_retained(file);owners.append(stream);h=hashlib.sha256();chunks=[];total=0
  while True:
   b=stream.read(1024*1024)
   if not b:break
   h.update(b);total+=len(b)
   if name=='catalog':chunks.append(b)
  if total!=record['nbytes']or h.hexdigest()!=record['sha256']:raise base.StaleManifest('compact content hash mismatch: '+name)
  if name=='catalog':values['blob']=b''.join(chunks)
 m=decode(values['blob']);m['manifest_sha256']=c['expanded_payload_sha256'];return m
