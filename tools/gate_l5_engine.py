"""L5 launcher: runs the existing Milestone-6 vLLM-criterion engine step (vllm_gate.py engine) on the SHIPPED long_engine (module substitution: `engine` := long_engine), with the
shipped GPU code objects. argv: <root> <out> <model_dir> [extra vllm_gate args]"""
import os, sys
root, out, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(root, 'long'))
import long_engine
sys.modules['engine'] = long_engine
pdir = r'C:\Users\ykogi\npu\poc-pd256k\p'
os.chdir(pdir); sys.path.insert(0, pdir)
co = os.path.join(root, 'plamo3-long', 'prefill')
sys.argv = ['vllm_gate.py', 'engine', '--variant', 'v2', '--out', out, '--model-dir', model_dir,
            '--co', os.path.join(co, 'plamo3_prefill.co'), '--wmma-co', os.path.join(co, 'plamo3_wmma.co'), '--attn-co', os.path.join(co, 'attn2.co'),
            '--attn-full-co', os.path.join(co, 'attn2_full.co'), '--fused-co', os.path.join(co, 'plamo3_fused.co'), '--gemm-co', os.path.join(co, 'plamo3_gemm2.co')] + sys.argv[4:]
import vllm_gate
vllm_gate.main()
