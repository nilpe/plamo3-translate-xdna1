"""L12 isolation proof helper: runs the DEFAULT CLI (translate.py of the install) in-process under a Python audit hook that appends every file open to a trace file, so the gate can show that
no file under long\\ or plamo3-long\\ (nor the long manifests) is opened by the default path. argv: <install root> <trace file> <translate args...>"""
import os, runpy, sys
root, trace = sys.argv[1], sys.argv[2]
f = open(trace, 'a', encoding='utf-8', buffering=1)
def hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            f.write(os.fsdecode(args[0]) + '\n')
        except Exception:
            pass
sys.addaudithook(hook)
sys.argv = [os.path.join(root, 'IRON', 'iron', 'applications', 'plamo3', 'translate.py')] + sys.argv[3:]
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
runpy.run_path(sys.argv[0], run_name='__main__')
