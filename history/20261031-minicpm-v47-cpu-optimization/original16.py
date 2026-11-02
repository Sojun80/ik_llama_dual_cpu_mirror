import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';order=[0,1,4,2,3,5,6,7]
for n in [16]:
 cpus=[48+8*j+i for i in order for j in range(6)][:n];mask='0x111113131313000000000000';port=19169;base='http://127.0.0.1:'+str(port)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=300))
 cmd=['numactl','--physcpubind=48-95','--membind=1',str(s/'build/bin/llama-server'),'-m',str(s/'model.complete.gguf'),'--load-mode','none','-ngl','0','-t',str(n),'-C',mask,'--cpu-strict','1','-tb','48','-Cb','0xffffffffffff000000000000','--cpu-strict-batch','1','-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'candidate-original-{n}-command.json').write_text(json.dumps(cmd,indent=2));print('START',n,flush=True)
 with (p/f'candidate-original-{n}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll() is not None:raise RuntimeError(f'server exited {srv.returncode}')
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   (p/f'candidate-original-{n}-smaps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text())
   with (p/f'candidate-original-{n}-numastat.txt').open('w') as f:subprocess.run(['numastat','-p',str(srv.pid)],stdout=f)
   for count in [156,524]:
    prompt='The following numeric pattern repeats: 1, 2, 3, 4. '*count+'Continue the pattern:';payload={'prompt':prompt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False};r=req('/completion',payload);reference=json.loads((p/f'server-compare-upstream-{count}-0.json').read_text())['response'];x={'response':r,'reference_content_equal':r.get('content')==reference.get('content')};(p/f'candidate-original-{n}-{count}.json').write_text(json.dumps(x,indent=2));print('DONE',n,count,'match',x['reference_content_equal'],'timings',r.get('timings'),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'ORIGINAL_SERVER_FINISHED').write_text('complete\n')
