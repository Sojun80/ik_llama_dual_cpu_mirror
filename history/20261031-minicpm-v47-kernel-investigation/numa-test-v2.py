import os,subprocess,pathlib,json,time,urllib.request,signal,re,traceback
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');m='/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf';node0=[16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45];node1=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92];order=node0+node1+[i for i in range(48,96) if i not in node1][:16];cpu=','.join(map(str,order))
for idx,mode in enumerate(['global','local']):
 env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';env['IK_EXPERT_NUMA']='1'
 if mode=='local':env['IK_EXPERT_NUMA_SCHED']='1'
 port=19240+idx;base='http://127.0.0.1:'+str(port);cmd=['numactl','--physcpubind=16-95','--membind=1',str(p/'ik-numa-build/bin/llama-server'),'-m',m,'--no-mmap','-ngl','0','-t','32','-tb','48','--cpu-range',cpu,'-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'numa-proto-v2-{mode}-command.json').write_text(json.dumps({'cmd':cmd,'env':{k:env[k] for k in ['GLIBC_TUNABLES','IK_EXPERT_NUMA']},'local_scheduler':mode=='local'},indent=2));print('START',mode,flush=True)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
 with (p/f'numa-proto-v2-{mode}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   (p/f'numa-proto-v2-{mode}-smaps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text());(p/f'numa-proto-v2-{mode}-numa-maps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/numa_maps').read_text());stat=subprocess.check_output(['numastat','-p',str(srv.pid)],text=True);(p/f'numa-proto-v2-{mode}-numastat.txt').write_text(stat);print('PLACEMENT',mode,stat.splitlines()[-1],flush=True)
   line=stat.splitlines()[-1].split(); node0mb=float(line[1]); node1mb=float(line[2]);
   if node0mb<4000: print('INVALID_PLACEMENT',mode,node0mb,node1mb,flush=True);break
   for count in [156,524]:
    txt='The following numeric pattern repeats: 1, 2, 3, 4. '*count+'Continue the pattern:';payload={'prompt':txt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False};x=req('/completion',payload);ref=json.loads((o/f'server-compare-ik-{count}-0.json').read_text())['response'] if (o/f'server-compare-ik-{count}-0.json').exists() else json.loads((o/f'server-compare-upstream-{count}-0.json').read_text())['response'];v={'response':x,'matches_reference':x.get('content')==ref.get('content')};(p/f'numa-proto-v2-{mode}-{count}.json').write_text(json.dumps(v,indent=2));print('DONE',mode,count,'match',v['matches_reference'],'timings',x.get('timings'),flush=True)
  except Exception as e: print('ERROR',mode,repr(e),traceback.format_exc(),flush=True);break
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'NUMA_V2_TEST_FINISHED').write_text('finished, inspect placement and errors\n')
