import os,subprocess,pathlib,json,time,urllib.request,signal,traceback
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';cpu=','.join(map(str,json.loads((p/'scaling-order.json').read_text())));cmd=['numactl','--physcpubind=48-95','--membind=1',str(p/'ik-q8k-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','32','-tb','48','--cpu-range',cpu,'--no-fused-moe','--no-fused-up-gate','--ctx-checkpoints','0','-c','16384','--host','127.0.0.1','--port','19242','--parallel','1'];(p/'q8knockpt-test-command.json').write_text(json.dumps(cmd,indent=2));print('START',flush=True)
def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:19242'+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
with (p/'q8knockpt-test-server.log').open('w') as log:
 srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:
  for i in range(300):
   if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
   try:urllib.request.urlopen('http://127.0.0.1:19242/health',timeout=2);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('startup timeout')
  for case in ['short','medium','long']:
   prompt='A fixed sequence: one two three four five.' if case=='short' else (p/f'corpus-{case}.txt').read_text();payload={'prompt':prompt,'temperature':0,'n_predict':16,'seed':1234,'cache_prompt':False,'stream':False,'ignore_eos':True,'n_probs':20};r=req('/completion',payload);(p/f'q8knockpt-test-{case}.json').write_text(json.dumps(r,indent=2));print('DONE',case,'timings',r.get('timings'),'text',repr(r.get('content')),flush=True)
  (p/'Q8KNOCKPT_TEST_COMPLETE').write_text('complete\n')
 except Exception as e:print('ERROR',repr(e),traceback.format_exc(),flush=True)
 finally:
  try:os.killpg(srv.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:srv.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
