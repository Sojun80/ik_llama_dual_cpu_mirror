import ctypes,mmap
lib=ctypes.CDLL(None,use_errno=True);lib.syscall.restype=ctypes.c_long;m=mmap.mmap(-1,260*1024*1024,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,prot=mmap.PROT_READ|mmap.PROT_WRITE);b=ctypes.addressof(ctypes.c_char.from_buffer(m));a=2*1024*1024;start=(b+a-1)&~(a-1)
for node in [0,1]:
 mask=ctypes.c_ulong(1<<node);ctypes.set_errno(0);r=lib.syscall(ctypes.c_long(237),ctypes.c_void_p(start+node*128*1024*1024),ctypes.c_ulong(128*1024*1024),ctypes.c_int(2),ctypes.byref(mask),ctypes.c_ulong(2),ctypes.c_uint(0));print('node',node,'result',r,'errno',ctypes.get_errno())
