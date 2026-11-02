import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';cpus='48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92'; results={}
for engine in ['upstream','ik']:
 port=19164 if engine=='upstream' else 19165;base='http://127.0.0.1:'+str(port)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=900))
 binary=s/'build/bin/llama-server' if engine=='upstream' else p/'ik-build/bin/llama-server';load=['--load-mode','none'] if engine=='upstream' else ['--no-mmap'];cmd=['numactl','--physcpubind='+cpus,'--membind=1',str(binary),'-m',str(s/'model.complete.gguf'),*load,'-ngl','0','-t','16','-tb','16','-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'server-compare-{engine}-command.json').write_text(json.dumps(cmd,indent=2));print('START',engine,flush=True)
 with (p/f'server-compare-{engine}.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll() is not None:raise RuntimeError(f'{engine} exited {srv.returncode}')
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   for count in [156,524]:
    prompt='The following numeric pattern repeats: 1, 2, 3, 4. '*count+'Continue the pattern:';payload={'prompt':prompt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False};toks=req('/tokenize',{'content':prompt,'add_special':False})['tokens'];assert len(toks)+64<16384
    for rep in range(2):
     r=req('/completion',payload); result={'engine':engine,'count':count,'rep':rep,'tokens':toks,'response':r};(p/f'server-compare-{engine}-{count}-{rep}.json').write_text(json.dumps(result,indent=2));print('DONE',engine,count,rep,'prompt',len(toks),'timings',r.get('timings'),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'SERVER_COMPARE_FINISHED').write_text('complete\n')
