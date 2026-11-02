import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';cpu=','.join(map(str,json.loads((p/'scaling-order.json').read_text())));cmd=['numactl','--physcpubind=48-95','--membind=1',str(o/'ik-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','32','-tb','48','--cpu-range',cpu,'-c','16384','--host','127.0.0.1','--port','19191','--parallel','1'];(p/'pinned-profile-command.json').write_text(json.dumps(cmd,indent=2));print('START',flush=True)
with (p/'pinned-profile-server.log').open('w') as log:
 srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:
  for i in range(300):
   if srv.poll()!=None:raise RuntimeError('server exited')
   try:urllib.request.urlopen('http://127.0.0.1:19191/health',timeout=2);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('startup timeout')
  for case in ['medium','long']:
   txt=(p/f'corpus-{case}.txt').read_text();payload={'prompt':txt,'temperature':0,'n_predict':512,'seed':1234,'cache_prompt':False,'stream':True,'ignore_eos':True};r=urllib.request.Request('http://127.0.0.1:19191/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'});print('REQUEST',case,flush=True);profile=None
   with urllib.request.urlopen(r,timeout=300) as resp,(p/f'pinned-profile-{case}-stream.txt').open('w') as out:
    for line in resp:
     out.write(line.decode());out.flush()
     if profile is None and line.startswith(b'data: '):
      with (p/f'pinned-profile-{case}-capture.log').open('w') as cap:
       profile=subprocess.Popen(['sudo','-n','perf','record','-F','79','-g','-p',str(srv.pid),'-o',str(p/f'pinned-profile-{case}.data'),'--','sleep','8'],stdout=cap,stderr=subprocess.STDOUT)
   if profile:
    profile.wait(timeout=20)
    with (p/f'pinned-profile-{case}-report.txt').open('w') as report:subprocess.run(['sudo','-n','perf','report','--stdio','-i',str(p/f'pinned-profile-{case}.data'),'--no-children','--percent-limit','1'],stdout=report,stderr=subprocess.STDOUT)
   print('DONE',case,flush=True)
 finally:
  try:os.killpg(srv.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:srv.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'PROFILE_COMPLETE').write_text('complete\n')
