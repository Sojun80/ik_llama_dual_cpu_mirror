import os,subprocess,pathlib,json,time,urllib.request,signal,traceback
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';cpus='48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92';cmd=['numactl','--physcpubind='+cpus,'--membind=1',str(o/'ik-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','16','-tb','16','-c','16384','--host','127.0.0.1','--port','19170','--parallel','1'];(p/'repro-command.json').write_text(json.dumps(cmd,indent=2));prompt='The following numeric pattern repeats: 1, 2, 3, 4. '*524+'Continue the pattern:';payload={'prompt':prompt,'temperature':0,'n_predict':8,'seed':1234,'cache_prompt':False,'stream':True};(p/'repro-request.json').write_text(json.dumps(payload,indent=2));signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('240-second request deadline')))
with (p/'repro-server.log').open('w') as log:
 srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);(p/'repro-server.pid').write_text(str(srv.pid));print('server pid',srv.pid,flush=True)
 try:
  for i in range(300):
   if srv.poll()!=None:raise RuntimeError('server exited')
   try:urllib.request.urlopen('http://127.0.0.1:19170/health',timeout=2);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('startup timeout')
  req=urllib.request.Request('http://127.0.0.1:19170/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'});print('REQUEST_START',time.time(),flush=True);signal.alarm(240)
  with urllib.request.urlopen(req,timeout=260) as resp,(p/'repro-stream.txt').open('w') as out:
   for line in resp:
    out.write(line.decode());out.flush();print('EVENT',time.time(),line[:180],flush=True)
  signal.alarm(0);print('REQUEST_FINISHED',time.time(),flush=True)
 except Exception as e:print('ERROR',repr(e),traceback.format_exc(),flush=True)
 finally:
  signal.alarm(0)
  try:os.killpg(srv.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:srv.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
  print('SERVER_STOPPED',srv.returncode,flush=True)
