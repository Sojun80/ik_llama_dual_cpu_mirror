import os,subprocess,pathlib,json,time,urllib.request,signal,traceback
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';order=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92]
while len(order)<48:
 counts=[sum(48+i*8<=x<56+i*8 for x in order) for i in range(6)];k=min(range(6),key=lambda i:counts[i]);order.append(next(x for x in range(48+k*8,56+k*8) if x not in order))
cpustr=','.join(map(str,order));(p/'scaling-order.json').write_text(json.dumps(order))
for n in [24,32,48]:
 port=19173;base='http://127.0.0.1:'+str(port);binary=o/'ik-build/bin/llama-server';cmd=['numactl','--physcpubind=48-95','--membind=1',str(binary),'-m',m,'--no-mmap','-ngl','0','-t',str(n),'-tb','48','--cpu-range',cpustr,'-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'ikscale-{n}-command.json').write_text(json.dumps(cmd,indent=2));print('START',n,time.time(),flush=True)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
 with (p/f'ikscale-{n}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   for count in [156,524]:
    prompt='The following numeric pattern repeats: 1, 2, 3, 4. '*count+'Continue the pattern:';payload={'prompt':prompt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False};x=req('/completion',payload);ref=json.loads((o/f'server-compare-upstream-{count}-0.json').read_text())['response'];out={'count':count,'response':x,'matches_upstream':x.get('content')==ref.get('content')};(p/f'ikscale-{n}-{count}.json').write_text(json.dumps(out,indent=2));print('DONE',n,count,'match',out['matches_upstream'],'timings',x.get('timings'),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
 print('STOPPED',n,flush=True)
(p/'IKSCALE_COMPLETE').write_text('complete\n')
