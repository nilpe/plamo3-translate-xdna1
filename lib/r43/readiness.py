"""Windows process creation timestamp, measured without launching a helper."""
def process_age_ms():
 import ctypes,time
 class FT(ctypes.Structure):_fields_=[('lo',ctypes.c_uint32),('hi',ctypes.c_uint32)]
 create,exit,kernel,user=FT(),FT(),FT(),FT();api=ctypes.windll.kernel32;api.GetCurrentProcess.restype=ctypes.c_void_p;api.GetProcessTimes.argtypes=[ctypes.c_void_p]*5;ok=api.GetProcessTimes(api.GetCurrentProcess(),ctypes.byref(create),ctypes.byref(exit),ctypes.byref(kernel),ctypes.byref(user));assert ok;epoch=(create.hi<<32)+create.lo-116444736000000000;return(time.time_ns()-epoch*100)/1e6
