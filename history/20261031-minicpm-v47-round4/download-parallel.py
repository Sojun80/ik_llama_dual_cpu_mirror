import concurrent.futures, subprocess, pathlib, os
p=pathlib.Path(__file__).parent
size=21166255008
url="https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf/resolve/cf943fbe32f840b20ca05ccc146e7dc72baf9a9b/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf"
def fetch(i):
 a=size*i//8;b=size*(i+1)//8-1; f=p/f"part-{i:02}"
 with (p/f"part-{i:02}.log").open("w") as log: subprocess.run(["curl","-fLsS","--retry","4","-r",f"{a}-{b}","-o",str(f),url+f"?part={i}"],stdout=log,stderr=log,check=True)
 assert f.stat().st_size==b-a+1
 print(i,"complete",flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:list(e.map(fetch,range(8)))
with (p/"model.complete.gguf").open("wb") as out:
 for i in range(8):
  with (p/f"part-{i:02}").open("rb") as inp:
   while True:
    b=inp.read(8*1024*1024)
    if not b:break
    out.write(b)
print("assembled",flush=True)
