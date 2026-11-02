import ctypes,mmap,os
lib=ctypes.CDLL(None,use_errno=True);lib.syscall.restype=ctypes.c_long
m=mmap.mmap(-1,260*1024*1024,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,prot=mmap.PROT_READ|mmap.PROT_WRITE);b=ctypes.addressof(ctypes.c_char.from_buffer(m));a=2*1024*1024;start=(b+a-1)&~(a-1);length=128*1024*1024;mask=ctypes.c_ulong(1)
for maxnode in [2,64,1024]:
 ctypes.set_errno(0);r=lib.syscall(ctypes.c_long(237),ctypes.c_void_p(start),ctypes.c_ulong(length),ctypes.c_int(2),ctypes.byref(mask),ctypes.c_ulong(maxnode),ctypes.c_uint(0));print('maxnode',maxnode,'rv',r,'errno',ctypes.get_errno(),flush=True)
