import mmap,os,subprocess
size=256*1024*1024
m=mmap.mmap(-1,size,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,prot=mmap.PROT_READ|mmap.PROT_WRITE)
for i in range(0,size,4096): m[i:i+1]=b'X'
print('pid',os.getpid(),'bytes',size,flush=True)
subprocess.run(['numastat','-p',str(os.getpid())])
for l in open('/proc/self/numa_maps'):
 if 'anon=65536' in l:print(l.strip())
