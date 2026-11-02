import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';node0=[16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45];node1=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92];order=node0+node1+[i for i in range(48,96) if i not in node1][:16];cpus=','.join(map(str,order))
for idx,mode in enumerate(['node1','interleave']):
 port=19188+idx;base='http://127.0.0.1:'+str(port);policy='--membind=1' if mode=='node1' else '--interleave=0,1';cmd=['numactl','--physcpubind=16-95',policy,str(o/'ik-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','32','-tb','48','--cpu-range',cpus,'-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'ikdual-{mode}-command.json').write_text(json.dumps(cmd,indent=2));print('START',mode,flush=True)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
 with (p/f'ikdual-{mode}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   (p/f'ikdual-{mode}-smaps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text());(p/f'ikdual-{mode}-numa-maps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/numa_maps').read_text());
   with (p/f'ikdual-{mode}-numastat.txt').open('w') as f:subprocess.run(['numastat','-p',str(srv.pid)],stdout=f)
   for count in [156,524]:
    txt='The following numeric pattern repeats: 1, 2, 3, 4. '*count+'Continue the pattern:';payload={'prompt':txt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False};x=req('/completion',payload);ref=json.loads((o/f'server-compare-upstream-{count}-0.json').read_text())['response'];v={'response':x,'matches_upstream':x.get('content')==ref.get('content')};(p/f'ikdual-{mode}-{count}.json').write_text(json.dumps(v,indent=2));print('DONE',mode,count,'match',v['matches_upstream'],'timings',x.get('timings'),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'IKDUAL_COMPLETE').write_text('complete\n')
