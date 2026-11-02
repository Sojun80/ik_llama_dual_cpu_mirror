import os,subprocess,pathlib,json,time,urllib.request,signal,traceback
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';cores=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92];prompt='A sequence begins 1, 2, 3, 4. Continue:'
for mode in ['unique','collision','collision_passive']:
 port={'unique':19176,'collision':19177,'collision_passive':19178}[mode]
 vals=cores.copy()
 if mode!='unique':vals[8]=vals[2]
 env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1'
 if mode=='collision_passive':env['OMP_WAIT_POLICY']='PASSIVE';env['GOMP_SPINCOUNT']='0'
 cmd=['numactl','--physcpubind=48-95','--membind=1',str(o/'ik-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','16','-tb','16','--cpu-range',','.join(map(str,vals)),'-c','4096','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'collision-{mode}-command.json').write_text(json.dumps({'cmd':cmd,'wait_policy':env.get('OMP_WAIT_POLICY'),'spin_count':env.get('GOMP_SPINCOUNT')},indent=2));print('START',mode,time.time(),flush=True)
 with (p/f'collision-{mode}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(240):
    if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
    try:urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   payload={'prompt':prompt,'temperature':0,'n_predict':8,'seed':1234,'cache_prompt':False,'stream':False};r=urllib.request.Request('http://127.0.0.1:'+str(port)+'/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'});start=time.monotonic()
   try:
    with urllib.request.urlopen(r,timeout=150) as resp:x=json.load(resp)
    (p/f'collision-{mode}-result.json').write_text(json.dumps(x,indent=2));print('DONE',mode,'seconds',round(time.monotonic()-start,2),'timings',x.get('timings'),'content',repr(x.get('content')),flush=True)
   except Exception as e:print('REQUEST_FAILURE',mode,repr(e),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'COLLISION_COMPLETE').write_text('complete\n')
