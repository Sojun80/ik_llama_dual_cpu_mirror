import os,subprocess,pathlib,json,time,urllib.request,signal,random,statistics
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';original='0x111113131313000000000000';allmask='0xffffffffffff000000000000';order=json.loads((p/'scaling-order.json').read_text());cpu=','.join(map(str,order))
scores={}
for n in [24,32,48]:
 vals=[]
 for count in [156,524]:
  f=p/f'ikscale-{n}-{count}.json';v=json.loads(f.read_text());
  if not v['matches_upstream']:raise RuntimeError(f'candidate {n} has mismatched output')
  vals.append(v['response']['timings']['predicted_per_second'])
 scores[n]=sum(vals)
best=max(scores,key=scores.get);(p/'chunk4-selection.json').write_text(json.dumps({'candidate':best,'raw_decode_sum':scores,'warning':'screening only; 1 sample per candidate'},indent=2));print('SELECT',best,scores,flush=True)
words='river forest silver stone quiet meadow cloudy paper candle orange window spring shadow lantern ocean mountain valley bird garden thunder distant circle copper velvet marble winter summer autumn morning planet comet crystal flower honey island orchard purple yellow bright wind water meadow mirror garden valley silent open field star'.split(); cases={}
for name,n,seed in [('medium',2650,2812),('long',8900,9436)]:
 r=random.Random(seed);txt='A fixed collection of ordinary words follows.\n'+' '.join(r.choice(words) for _ in range(n))+'\nContinue with a coherent fictional passage:';cases[name]=txt
for engine in ['ik']:
 port=19179;base='http://127.0.0.1:'+str(port)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=300))
 if engine=='upstream':binary=s/'build/bin/llama-server';options=['--load-mode','none','-t','16','-C',original,'--cpu-strict','1','-tb','48','-Cb',allmask,'--cpu-strict-batch','1']
 else:binary=p/'ik-chunk4-build/bin/llama-server';options=['--no-mmap','-t',str(best),'-tb','48','--cpu-range',cpu]
 cmd=['numactl','--physcpubind=48-95','--membind=1',str(binary),'-m',str(s/'model.complete.gguf'),'-ngl','0',*options,'-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'chunk4-{engine}-command.json').write_text(json.dumps(cmd,indent=2));print('START',engine,time.time(),flush=True)
 with (p/f'chunk4-{engine}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll()!=None:raise RuntimeError(f'{engine} exited {srv.returncode}')
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   (p/f'chunk4-{engine}-smaps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text())
   for name,txt in cases.items():
    tokens=req('/tokenize',{'content':txt,'add_special':False})['tokens'];assert len(tokens)+256<16384,(name,len(tokens))
    for rep in range(2):
     payload={'prompt':txt,'temperature':0,'n_predict':256,'seed':1234,'cache_prompt':False,'stream':False,'ignore_eos':True,'n_probs':5};print('REQUEST',engine,name,rep,'tokens',len(tokens),flush=True);x=req('/completion',payload);result={'tokens':tokens,'response':x};(p/f'chunk4-{engine}-{name}-{rep}.json').write_text(json.dumps(result,indent=2));print('DONE',engine,name,rep,'timings',x.get('timings'),'truncated',x.get('truncated'),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'CHUNK4_TEST_COMPLETE').write_text('complete\n')
