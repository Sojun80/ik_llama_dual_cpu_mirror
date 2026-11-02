import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';env['IK_EXPERT_NUMA']='1';env['IK_EXPERT_NUMA_SCHED']='1';n0=[16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45];n1=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92];order=n0+n1+[i for i in range(48,96) if i not in n1][:16];cmd=['numactl','--physcpubind=16-95','--membind=1',str(p/'ik-numa-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','32','-tb','48','--cpu-range',','.join(map(str,order)),'-c','16384','--host','127.0.0.1','--port','19241','--parallel','1'];(p/'numa-profile-command.json').write_text(json.dumps(cmd,indent=2));print('START',flush=True)
with (p/'numa-profile-server.log').open('w') as log:
 srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:
  for i in range(300):
   if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
   try:urllib.request.urlopen('http://127.0.0.1:19241/health',timeout=2);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('startup timeout')
  with (p/'numa-profile-numastat.txt').open('w') as f:subprocess.run(['numastat','-p',str(srv.pid)],stdout=f)
  for case in ['medium','long']:
   txt=(p/f'corpus-{case}.txt').read_text();payload={'prompt':txt,'temperature':0,'n_predict':256,'seed':1234,'cache_prompt':False,'stream':True,'ignore_eos':True,'n_probs':5};r=urllib.request.Request('http://127.0.0.1:19241/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'});print('REQUEST',case,flush=True);profile=None;contents=[]
   with urllib.request.urlopen(r,timeout=300) as resp,(p/f'numa-profile-{case}-stream.txt').open('w') as out:
    for line in resp:
     out.write(line.decode());out.flush()
     if line.startswith(b'data: '):
      try:
       event=json.loads(line[6:]);contents.append(event.get('content',''))
       if profile is None and event.get('content'):
        with (p/f'numa-profile-{case}-capture.log').open('w') as cap:profile=subprocess.Popen(['sudo','-n','perf','record','-F','79','-g','-p',str(srv.pid),'-o',str(p/f'numa-profile-{case}.data'),'--','sleep','8'],stdout=cap,stderr=subprocess.STDOUT)
      except (json.JSONDecodeError,UnicodeDecodeError):pass
   if profile:
    profile.wait(timeout=20)
    with (p/f'numa-profile-{case}-report.txt').open('w') as report:subprocess.run(['sudo','-n','perf','report','--stdio','-i',str(p/f'numa-profile-{case}.data'),'--no-children','--percent-limit','1'],stdout=report,stderr=subprocess.STDOUT)
   text=''.join(contents);ref=json.loads((p/f'corpus-ik-{case}-0.json').read_text())['response']['content'];(p/f'numa-profile-{case}-compare.json').write_text(json.dumps({'matches_ik':text==ref,'text':text,'reference':ref},indent=2));print('DONE',case,'matches_ik',text==ref,flush=True)
 finally:
  try:os.killpg(srv.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:srv.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'NUMA_PROFILE_COMPLETE').write_text('complete\n')
