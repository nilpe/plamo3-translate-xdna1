"""R83: the shorter GIL switch interval is used only in a declared window of speculative startup.

Measured (R83): pyxrt holds the GIL for the whole of device_open, xclbin_load/register and every hw_context_create (a 1 ms Python
heartbeat thread makes no progress inside any of these calls, at either interval), so a shorter interval can only matter around
them, never inside them. The window therefore opens ONLY when the early-XRT thread has finished the last hw_context
(startup event 'xclbin_hw_contexts') and closes (restores the previous interval) when speculative construction ends, on every exit
path: success, speculative failure (before the serial fallback initialization), any exception, and DefaultRuntime construction end.
Only thread-scheduling latency changes: no bytes, order, arithmetic or validation scope."""
import sys,threading,time
TRIGGER='xclbin_hw_contexts'
_active=None;_lock=threading.Lock()

class Window:
    trigger=TRIGGER
    def __init__(self,short_s):
        self.short_s=short_s;self.previous=sys.getswitchinterval();self.armed=False;self.opened=False;self.closed=False
        self.seen=[];self.begin=time.perf_counter();self.record={}
    def arm(self):
        with _lock:self.armed=True
    def notify(self,phase):
        with _lock:
            if not self.armed or self.closed:return
            if self.opened:return
            if phase==self.trigger:
                self.previous=sys.getswitchinterval();sys.setswitchinterval(self.short_s);self.opened=True
                self.record.update(opened=True,trigger=phase,short_s=sys.getswitchinterval(),opened_after_start_ms=(time.perf_counter()-self.begin)*1000,seen_before_open=list(self.seen))
            else:self.seen.append(phase)
    def close(self):
        with _lock:
            was=self.opened;self.armed=False
            if was and not self.closed:sys.setswitchinterval(self.previous)
            self.closed=True
            self.record.update(opened=was,closed=True,restored_s=sys.getswitchinterval())
        return self.record

def install(window):
    global _active
    with _lock:_active=window
def uninstall(window):
    global _active
    with _lock:
        if _active is window:_active=None
def notify(phase):
    window=_active
    if window is not None:window.notify(phase)

def assert_declared_window(startup):
    """Real-run check (smoke/timing/startup gates): the window opened exactly at xclbin_hw_contexts, after every device/xclbin/hw_context phase, and was restored."""
    window=startup['switch_window'];names=[e['phase'] for e in startup['events']]
    assert window['opened'] and window['closed'] and window['trigger']==TRIGGER,window
    assert names.count(TRIGGER)==1,names
    before=names[:names.index(TRIGGER)]
    assert window['seen_before_open']==before,'window opened out of order'
    during=[n for n in names if n=='device_open' or n.startswith(('xclbin_load','xclbin_register','hw_context_create','kernel_binding'))]
    assert during and all(n in before for n in during),'device/xclbin/hw_context phase outside the pre-window region'
    assert [n for n in names[names.index(TRIGGER)+1:] if n=='device_open' or n.startswith(('xclbin_load','xclbin_register','hw_context_create','kernel_binding'))]==[],'XRT context phase after the window opened'
    assert window['restored_s']==startup['switch_interval_default_s']==startup['switch_interval_restored_s'] and window['short_s']<startup['switch_interval_default_s']
    return window
