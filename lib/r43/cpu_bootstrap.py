"""Batched CPU bootstrap. Norm/RoPE instruction policy is retained; BLAS is gated."""
import numpy as np

def inv_batch(target,total,cols):
 fp=target.fp;bf=target.bf;mul=target.mul2;limbs=target.limbs
 sh,sl=limbs(total);ih,il=limbs(fp(1/128 if cols==128 else(0x7fffffff//cols)*2.**-31));mean=mul(sh,ih,sl,ih,fp(1e-6));mean=mul(sh,il,sl,il,mean);mh,ml=limbs(mean);u=(np.uint32(0x5f3759df)-(mean.view(np.uint32)>>1))>>16;y0=(u<<16).view(np.float32);my0=bf(mul(mh,y0,ml,y0));v=mul(-my0,y0,0,0,1);half=((u-np.uint32(0x80))<<16).view(np.float32);y1=bf(mul(bf(v),half,0,0,y0));sqh,sql=limbs(fp(y1*y1));v=mul(-mh,sql,-ml,sql,1);v=mul(-mh,sqh,-ml,sqh,v);rh,rl=limbs(v);rsq=bf(fp(rh*rh));y38=bf(fp(y1*.375));half=fp(y1*.5);y=mul(rh,half,rl,half,y1);y=mul(rsq,y38,0,0,y);return limbs(y)

def rms_batch(target,x,gamma):
 x=np.asarray(x,np.float32);N,cols=x.shape;acc=np.zeros((N,4,16),np.float32)
 for i in range(cols//32):
  z=x[:,i*32:(i+1)*32];slot=i%4 if cols//32>=8 else 0;acc[:,slot]=target.mul2(z[:,:16],z[:,:16],z[:,16:],z[:,16:],acc[:,slot])
 sums=target.aligned_add(target.aligned_add(acc[:,0],acc[:,1]),target.aligned_add(acc[:,2],acc[:,3]))
 for n in(8,4,2,1):sums=target.aligned_add(sums[:,:n],sums[:,n:2*n])
 hi,lo=inv_batch(target,sums[:,0],cols);norm=target.bf(target.mul2(x,hi[:,None],x,lo[:,None]));return target.bf(target.single_mac(norm,np.asarray(gamma,np.float32)[None]))

class Bootstrap:
 def __init__(self,model,target,norm=None):
  self.m=model;self.t=target;self.norm=norm or(lambda x,g:rms_batch(target,x,g));self.weights={};self.profile=[]
  for n in('feature.down.weight','feature.up.weight','mix.down.weight','mix.up.weight','blocks.0.mixer.qkv_proj.down.weight'):
   q,s=model.quant[n];self.weights[n]=(np.ascontiguousarray(q,np.float32),np.ascontiguousarray(s,np.float32))
  q,s=model.quant['blocks.0.mixer.qkv_proj.up.weight'];self.weights['kv']=(np.ascontiguousarray(q[-256:],np.float32),np.ascontiguousarray(s[-256:],np.float32))
 def linear(self,name,x):
  q,s=self.weights[name];return self.t.bf((np.asarray(x,np.float32)@q.T)*s[None])
 def __call__(self,raw,tokens):
  import time
  begin=time.perf_counter();m=self.m;t=self.t;raw=t.bf(raw);tokens=np.asarray(tokens,np.int64);x=raw[:-1].reshape(-1,6144);stages={}
  def stage(name,fn):
   at=time.perf_counter();
   if getattr(self,'cancelled',lambda:False)():raise InterruptedError('priming cancelled at stage boundary')
   out=fn();stages[name]=(time.perf_counter()-at)*1000;return out
  if len(x)==0:return dict(last_taps=raw[-1],position=0,layers=[(np.empty((1,0,128),np.float32),np.empty((1,0,128),np.float32))])
  x=stage('feature_down',lambda:self.linear('feature.down.weight',x));x=stage('feature_up',lambda:self.linear('feature.up.weight',x));x=np.concatenate((x,np.asarray(m.embedding[tokens[1:]],np.float32)),1);x=stage('mix_down',lambda:self.linear('mix.down.weight',x));x=stage('mix_up',lambda:self.linear('mix.up.weight',x));gamma=t.bf(m.w['blocks.0.pre_mixer_norm.weight']+np.float32(1));x=stage('pre_norm',lambda:self.norm(x,gamma));x=stage('qkv_down',lambda:self.linear('blocks.0.mixer.qkv_proj.down.weight',x));kv=stage('kv_up',lambda:self.linear('kv',x));k,v=kv[:,:128],kv[:,128:];gamma=t.bf(m.w['blocks.0.mixer.k_norm.weight']+np.float32(1));k=stage('k_norm',lambda:self.norm(k,gamma));pos=np.arange(len(k),dtype=np.float32)[:,None];angle=pos*m.w['rope.0.inv_freq'];c=np.float32(m.meta['rotary'][0]['concentration']);table=np.stack((t.bf(np.cos(angle)*c),t.bf(np.sin(angle)*c)),-1).reshape(-1,128);k=stage('rope',lambda:t.bf(np.concatenate((t.mul2(k[:,:64],table[:,::2],-k[:,64:],table[:,1::2]),t.mul2(k[:,64:],table[:,::2],k[:,:64],table[:,1::2])),1)));self.profile.append(dict(rows=len(k),total_ms=(time.perf_counter()-begin)*1000,stages_ms=stages));return dict(last_taps=raw[-1],position=len(tokens)-1,layers=[(k[None],v[None])])
