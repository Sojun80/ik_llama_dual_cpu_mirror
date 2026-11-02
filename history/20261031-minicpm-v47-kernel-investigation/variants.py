import os,subprocess,pathlib,json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation');o=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';order=json.loads((p/'scaling-order.json').read_text());cpu=','.join(map(str,order));mask=hex(sum(1<<i for i in order[:32]));prompt=(p/'corpus-long.txt').read_text()
variants=[('upstream32',[]),('ik-default',[]),('ik-no-fmoe',['--no-fused-moe']),('ik-no-fug',['--no-fused-up-gate']),('ik-no-mmad',['--no-fused-mul-multiadd']),('ik-no-fa',['--no-flash-attn']),('ik-all-off',['--no-fused-moe','--no-fused-up-gate','--no-fused-mul-multiadd','--no-flash-attn']),('upstream32-ub128',['-ub','128']),('ik-ub128',['-ub','128']),('upstream32-kvf32',['-ctk','f32','-ctv','f32']),('ik-kvf32',['-ctk','f32','-ctv','f32']),('ik-no-checkpoints',['--ctx-checkpoints','0'])]
for idx,(name,flags) in enumerate(variants):
 upstream=name.startswith('upstream32');port=19200+idx;base='http://127.0.0.1:'+str(port);binary=(s/'build/bin/llama-server') if upstream else (o/'ik-build/bin/llama-server');options=['--load-mode','none','-C',mask,'--cpu-strict','1','-Cb','0xffffffffffff000000000000','--cpu-strict-batch','1'] if upstream else ['--no-mmap','--cpu-range',cpu];cmd=['numactl','--physcpubind=48-95','--membind=1',str(binary),'-m',str(s/'model.complete.gguf'),'-ngl','0','-t','32','-tb','48',*options,*flags,'-c','16384','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'variant-{name}-command.json').write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=240))
 with (p/f'variant-{name}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll()!=None:raise RuntimeError('server exited '+str(srv.returncode))
    try:urllib.request.urlopen(base+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   payload={'prompt':prompt,'temperature':0,'n_predict':16,'seed':1234,'cache_prompt':False,'stream':False,'ignore_eos':True,'n_probs':20};r=req('/completion',payload);(p/f'variant-{name}-result.json').write_text(json.dumps(r,indent=2));print('DONE',name,'timings',r.get('timings'),'output',repr(r.get('content')),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(p/'VARIANTS_COMPLETE').write_text('complete\n')
