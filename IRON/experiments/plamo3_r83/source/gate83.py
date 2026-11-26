"""R83 confidence of the native drafter's 512 exact candidate logits and the gate comparison. Pure functions.
REFERENCE (the R57 definition, unchanged): top1_prob = 1 / sum_i exp(l_i - l_max) in float64 (math.exp, math.fsum: the sum is exactly rounded, so it does not depend on the order).
margin = l_top1 - l_top2 in float64 (0.0 when tied). A non-finite maximum or a NaN/inf logit gives NaN, which never fires the gate (fail closed to M2).
The gate fires iff conf < tau (strict); tau None never fires.

FAST PATH (R83, `confidence(logits,'top1_prob',tau)` with tau given): the same decision with vectorised NumPy float64.
 d_i = l_i - l_max (float64, the very same IEEE subtraction as the reference), S_np = numpy pairwise float64 sum of numpy.exp(d_i).
 If S_np is farther than REL_BOUND (relative) from 1/tau the result is 1/S_np, which lies on the same side of tau as the reference value (proof below); otherwise the exact reference runs.
 Proof. Let T = sum exp(d_i) (real numbers), n = len <= 4096, u = 2**-53. Assume (documented libm / NumPy accuracy, stated with a large margin) that numpy.exp and math.exp return exp(d)(1+theta), |theta| <= EPS_EXP = 2**-48 (16 ulp); the additive subnormal/underflow error is <= 2**-1074 per term, below 1e-300 in total, while T >= 1 because the maximum term is exp(0)=1.
  numpy sum of n floating terms (any association order; pairwise is better than the sequential bound): |S_np - sum(e~_i)| <= gamma_{n-1} * sum|e~_i|, gamma_k = k*u/(1-k*u) (Higham, Lemma 8.4/Thm 4.2), so S_np = T(1+a), |a| <= (1+EPS_EXP)(1+gamma_{n-1}) - 1.
  reference: S_ref = fsum(e_i) is the exactly rounded sum of math.exp terms, S_ref = T(1+b), |b| <= (1+EPS_EXP)(1+u) - 1.
  Hence |S_np/S_ref - 1| <= (1+|a|)(1+|b|)... <= 2*EPS_EXP + gamma_{n-1} + 2u + tiny = 5.7e-14 + 7.1e-15 + 2.2e-16 = 6.5e-14 for n = 512.  REL_BOUND = 2**-40 = 9.1e-13 is 14 times larger.
  The reference decision is fl(1/S_ref) < fl(tau), where fl(1/x) and fl(tau) carry a relative error <= u = 1.1e-16 << REL_BOUND. If S_np > (1/tau)(1+REL_BOUND) then S_ref > (1/tau)(1+REL_BOUND)(1-6.5e-14) so fl(1/S_ref) < tau(1-8e-13) < fl(tau): both fire. If S_np < (1/tau)(1-REL_BOUND) both do not fire. In the band the exact reference value is returned, so the decision is the reference decision by construction.
 Finite check: x.max() and x.min() are both finite iff no NaN and no +-inf is present (NaN propagates through max/min). An empty or longer-than-4096 input uses the exact path.
 The returned float (fast case) is 1/S_np: within 7e-14 relative of the reference conf and on the same side of tau; this is what production logs as `conf` (the exact reference value is logged when the band was hit). margin always uses the reference."""
import heapq,math
import numpy as np
STATS=('top1_prob','margin')
EPS_EXP=2.0**-48
REL_BOUND=2.0**-40
MAX_FAST_N=4096
def _values(logits):
 return [float(x)for x in logits]
def reference_confidence(logits,stat):
 xs=_values(logits)
 if not xs or any(x!=x or math.isinf(x)for x in xs):return float('nan')
 if stat=='top1_prob':
  m=max(xs);return 1.0/math.fsum([math.exp(x-m)for x in xs])
 if stat=='margin':
  top=heapq.nlargest(2,xs);return top[0]-top[1]if len(top)==2 else float('nan')
 raise ValueError('unknown gate statistic '+str(stat))
def error_bound(n):
 u=2.0**-53;g=(n-1)*u/(1-(n-1)*u);return 2*EPS_EXP+g+2*u+2*EPS_EXP*g
def fast_top1(logits,tau):
 """Returns (conf, used_exact_fallback). tau must be a positive float."""
 x=np.asarray(logits,np.float64)
 if x.ndim!=1 or not 0<x.size<=MAX_FAST_N:return reference_confidence(logits,'top1_prob'),True
 hi=float(x.max());lo=float(x.min())
 if not(math.isfinite(hi)and math.isfinite(lo)):return float('nan'),False
 s=float(np.exp(x-hi).sum());mid=1.0/tau
 if abs(s-mid)>REL_BOUND*mid:return 1.0/s,False
 return reference_confidence(logits,'top1_prob'),True
def confidence(logits,stat,tau=None):
 if stat=='top1_prob'and tau is not None and tau>0:return fast_top1(logits,tau)[0]
 return reference_confidence(logits,stat)
def gate_fires(conf,tau):
 return tau is not None and conf<tau
