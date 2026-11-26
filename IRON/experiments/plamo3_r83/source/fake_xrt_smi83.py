"""Fake xrt-smi for the turbo-switch gates (a Python script, not an executable). Keeps its pmode in $FAKE_XRT_DIR/mode.txt and logs every call to calls.log.
Failure injection (env): FAKE_XRT_FAIL_EXAMINE=1, FAKE_XRT_FAIL_CONFIGURE=1 (exit 1, nothing changes), FAKE_XRT_ERR_APPLY=1 (exit 1 + error text but the mode DOES change, the R63 behaviour),
FAKE_XRT_SILENT_NOOP=1 (exit 0 but nothing changes), FAKE_XRT_FAIL_TO=<mode> (configure to that mode fails), FAKE_XRT_DELAY_MS=<n> (every call sleeps n ms before acting; makes the background-acquire races deterministic)."""
import os,sys,time
d=os.environ['FAKE_XRT_DIR'];mp=os.path.join(d,'mode.txt')
def mode():
    try:return open(mp).read().strip()
    except OSError:return 'default'
args=sys.argv[1:]
marker=os.environ.get('FAKE_XRT_MARKER','')
with open(os.path.join(d,'calls.log'),'a') as f:f.write('%.3f %d %s marker=%d\n'%(time.time(),os.getppid(),' '.join(args),int(bool(marker) and os.path.exists(marker))))
print('[xrt-smi] build banner on stdout',flush=True);print('banner on stderr',file=sys.stderr)
time.sleep(int(os.environ.get('FAKE_XRT_DELAY_MS','0'))/1000.0)
if args[:1]==['examine']:
    if os.environ.get('FAKE_XRT_FAIL_EXAMINE')=='1':print('[xrt-smi] ERROR: examine failed',file=sys.stderr);sys.exit(1)
    print('-----------------------------\n[0004:00:01.1] : NPU Phoenix\n-----------------------------\nPlatform\n  Name                   : NPU Phoenix \n  Power Mode             : %s \n  Total Columns          : 5 \n\nEstimated Power          : N/A'%mode().capitalize());sys.exit(0)
if args[:2]==['configure','--pmode']:
    new=args[2]
    if os.environ.get('FAKE_XRT_FAIL_CONFIGURE')=='1' or os.environ.get('FAKE_XRT_FAIL_TO')==new:print('[xrt-smi] ERROR: Failed to escape (0xc0000001)',file=sys.stderr);sys.exit(1)
    if os.environ.get('FAKE_XRT_SILENT_NOOP')=='1':print('Power mode is set to '+new);sys.exit(0)
    open(mp,'w').write(new)
    if os.environ.get('FAKE_XRT_ERR_APPLY')=='1':print('[xrt-smi] ERROR: Failed to escape (0xc0000001)',file=sys.stderr);sys.exit(1)
    print('Power mode is set to '+new);sys.exit(0)
print('unknown',file=sys.stderr);sys.exit(2)
