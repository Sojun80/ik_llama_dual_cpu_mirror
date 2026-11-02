import struct,json,sys,pathlib
p=pathlib.Path(sys.argv[1]);f=p.open('rb')
def r(n):
 b=f.read(n)
 if len(b)!=n:raise EOFError
 return b
def u32():return struct.unpack('<I',r(4))[0]
def u64():return struct.unpack('<Q',r(8))[0]
def string():return r(u64()).decode('utf-8','replace')
def value(t):
 sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
 if t in sizes:return r(sizes[t])
 if t==8:return string()
 if t==9:
  typ=u32();n=u64()
  if typ in sizes:f.seek(n*sizes[typ],1);return None
  for _ in range(n):value(typ)
  return None
 raise ValueError(t)
magic=r(4);ver=u32();nt=u64();nk=u64();assert magic==b'GGUF';meta={}
for i in range(nk):
 key=string();typ=u32();v=value(typ)
 if any(x in key for x in ['expert_count','expert_used_count','block_count','architecture']):meta[key]=v.hex() if isinstance(v,bytes) else v
tensors=[]
for i in range(nt):
 name=string();nd=u32();dims=[u64() for _ in range(nd)];typ=u32();offset=u64()
 if 'exps' in name or 'expert' in name:tensors.append({'name':name,'dimensions':dims,'type':typ,'offset':offset})
print(json.dumps({'version':ver,'tensor_count':nt,'metadata':meta,'expert_tensors':tensors},indent=2))
