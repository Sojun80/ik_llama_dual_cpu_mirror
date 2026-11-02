import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';cpu=','.join(map(str,json.loads((p/'scaling-order.json').read_text())))
for idx,(name,extra) in enumerate([('spin100',{'GOMP_SPINCOUNT':'100'}),('passive',{'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'})]):
 env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';env.update(extra);port=19246+idx;base='http://127.0.0.1:'+str(port);cmd=['numactl','--physcpubind=48-95','--membind=1',str(o/'ik-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','32','-tb','48','--cpu-range',cpu,'-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'wait-{name}-command.json').write_text(json.dumps({'cmd':cmd,'env':extra},indent=2));print('START',name,flush=True)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
 with (p/f'wait-{name}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   for case in ['medium','long']:
    txt=(p/f'corpus-{case}.txt').read_text();payload={'prompt':txt,'temperature':0,'n_predict':256,'seed':1234,'cache_prompt':False,'stream':False,'ignore_eos':True,'n_probs':5};x=req('/completion',payload);ref=json.loads((p/f'corpus-ik-{case}-0.json').read_text())['response'];v={'response':x,'matches_ik':x.get('content')==ref.get('content')};(p/f'wait-{name}-{case}.json').write_text(json.dumps(v,indent=2));print('DONE',name,case,'match',v['matches_ik'],'timings',x.get('timings'),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'WAITPOLICY_COMPLETE').write_text('complete\n')
