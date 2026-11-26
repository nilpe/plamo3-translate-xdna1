"""Windows working-set/private peaks and laptop physical-memory samples."""
import ctypes
from ctypes import wintypes
import os

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t)for n in('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
class Physical(ctypes.Structure):
    _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(n,ctypes.c_ulonglong)for n in('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]

def process(pid):
    api=ctypes.windll.kernel32
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.OpenProcess(0x410,False,pid)
    if not handle:return dict(pid=pid,unavailable=True)
    try:
        c=Counters();c.cb=ctypes.sizeof(c)
        fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not fn(handle,ctypes.byref(c),ctypes.sizeof(c)):raise ctypes.WinError()
        return dict(pid=pid,rss_bytes=c.WorkingSetSize,peak_rss_bytes=c.PeakWorkingSetSize,private_bytes=c.PrivateUsage,peak_pagefile_bytes=c.PeakPagefileUsage)
    finally:api.CloseHandle(handle)

def snapshot(worker=None):
    p=Physical();p.dwLength=ctypes.sizeof(p)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(p)):raise ctypes.WinError()
    parent=process(os.getpid());child=process(worker.child.pid)if worker is not None and worker.healthy()else None
    return dict(parent=parent,worker=child,rss_total_bytes=parent.get('rss_bytes',0)+(child or{}).get('rss_bytes',0),physical_total_bytes=p.ullTotalPhys,physical_available_bytes=p.ullAvailPhys,physical_used_bytes=p.ullTotalPhys-p.ullAvailPhys,physical_load_percent=p.dwMemoryLoad,commit_used_bytes=p.ullTotalPageFile-p.ullAvailPageFile)

class Sampler:
    """Bounded aggregates only; no growing history can masquerade as app RSS."""
    def __init__(self,worker_getter):
        import threading
        self.worker_getter=worker_getter;self.stop_event=threading.Event();self.samples=0
        self.peak_physical_used_bytes=self.peak_rss_total_bytes=self.peak_commit_used_bytes=0
        self.minimum_available_bytes=None
        self.thread=threading.Thread(target=self.run,daemon=True,name='r45-memory-observer')
    def run(self):
        while not self.stop_event.is_set():
            row=snapshot(self.worker_getter());self.samples+=1
            self.peak_physical_used_bytes=max(self.peak_physical_used_bytes,row['physical_used_bytes'])
            self.peak_rss_total_bytes=max(self.peak_rss_total_bytes,row['rss_total_bytes'])
            self.peak_commit_used_bytes=max(self.peak_commit_used_bytes,row['commit_used_bytes'])
            value=row['physical_available_bytes'];self.minimum_available_bytes=value if self.minimum_available_bytes is None else min(self.minimum_available_bytes,value)
            self.stop_event.wait(.05)
    def start(self):self.thread.start()
    def finish(self):
        self.stop_event.set();self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        return dict(samples=self.samples,requested_period_ms=50,peak_sampled_physical_used_bytes=self.peak_physical_used_bytes,minimum_sampled_physical_available_bytes=self.minimum_available_bytes,peak_sampled_parent_plus_worker_RSS_bytes=self.peak_rss_total_bytes,peak_sampled_system_commit_bytes=self.peak_commit_used_bytes,scope='bounded aggregate observer, includes startup; GIL may stretch actual sample intervals; exact individual PeakWorkingSet counters are also in block snapshots')
