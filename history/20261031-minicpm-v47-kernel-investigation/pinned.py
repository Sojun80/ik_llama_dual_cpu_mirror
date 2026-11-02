import os,subprocess,pathlib,json,time,urllib.request,signal,traceback
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';base16=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92];cpus=base16+[i for i in range(48,96) if i not in base16];cpu=','.join(map(str,cpus));cmd=['numactl','--physcpubind=48-95','--membind=1',str(o/'ik-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','16','-tb','48','--cpu-range',cpu,'-c','16384','--host','127.0.0.1','--port','19172','--parallel','1'];(p/'pinned-command.json').write_text(json.dumps({'cmd':cmd,'ordered_cpus':cpus},indent=2));print('COMMAND',cmd,flush=True)
def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:19172'+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
with (p/'pinned-server.log').open('w') as log:
 srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);(p/'pinned-server.pid').write_text(str(srv.pid)); print('PID',srv.pid,flush=True)
 try:
  for i in range(300):
   if srv.poll()!=None:raise RuntimeError('server exited')
   try:urllib.request.urlopen('http://127.0.0.1:19172/health',timeout=2);break
   except Exception:time.sleep(1)
  else:raise RuntimeError('startup timeout')
  (p/'pinned-smaps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text())
  with (p/'pinned-numastat.txt').open('w') as f:subprocess.run(['numastat','-p',str(srv.pid)],stdout=f)
  for count in [156,524,156,524]:
   idx=len(list(p.glob('pinned-result-*.json')));prompt='The following numeric pattern repeats: 1, 2, 3, 4. '*count+'Continue the pattern:';payload={'prompt':prompt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False};print('START',idx,count,time.time(),flush=True);x=req('/completion',payload); ref=json.loads((o/f'server-compare-upstream-{count}-0.json').read_text())['response'];out={'count':count,'response':x,'matches_upstream':x.get('content')==ref.get('content')};(p/f'pinned-result-{idx}.json').write_text(json.dumps(out,indent=2));print('DONE',idx,count,'match',out['matches_upstream'],'timings',x.get('timings'),flush=True)
  (p/'PINNED_COMPLETE').write_text('complete\n')
 except Exception as e:print('ERROR',repr(e),traceback.format_exc(),flush=True)
 finally:
  try:os.killpg(srv.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:srv.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
  print('SERVER_STOPPED',srv.returncode,flush=True)
