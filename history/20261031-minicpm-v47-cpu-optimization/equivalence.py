import os, subprocess, pathlib, json,time,urllib.request,signal
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';cpus='48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92'; prompt='The following numeric pattern repeats: 1, 2, 3, 4. '*64+'Continue the pattern:';payload={'prompt':prompt,'temperature':0,'n_predict':64,'seed':1234,'cache_prompt':False,'stream':False}; (p/'equivalence-request.json').write_text(json.dumps(payload,indent=2));results={}
def req(path,obj):return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:'+str(port)+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=300))
for engine in ['upstream','ik']:
 port=19162 if engine=='upstream' else 19163
 binary=s/'build/bin/llama-server' if engine=='upstream' else p/'ik-build/bin/llama-server'; load=['--load-mode','none'] if engine=='upstream' else ['--no-mmap'];cmd=['numactl','--physcpubind='+cpus,'--membind=1',str(binary),'-m',str(s/'model.complete.gguf'),*load,'-ngl','0','-t','16','-tb','16','-c','4096','--host','127.0.0.1','--port',str(port),'--parallel','1'];(p/f'equiv-{engine}-command.json').write_text(json.dumps(cmd,indent=2));print('START',engine,flush=True)
 with (p/f'equiv-{engine}-server.log').open('w') as log:
  srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for i in range(300):
    if srv.poll() is not None:raise RuntimeError(f'{engine} exited {srv.returncode}')
    try:urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=2);break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   (p/f'equiv-{engine}-smaps.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text());
   with (p/f'equiv-{engine}-numastat.txt').open('w') as f:subprocess.run(['numastat','-p',str(srv.pid)],stdout=f)
   tokenized=req('/tokenize',{'content':prompt,'add_special':False});answer=req('/completion',payload);results[engine]={'prompt_tokens':tokenized.get('tokens'),'answer':answer};(p/f'equiv-{engine}-result.json').write_text(json.dumps(results[engine],indent=2)); print('DONE',engine,'prompt tokens',len(tokenized.get('tokens',[])),'content chars',len(answer.get('content','')),flush=True)
  finally:
   try:os.killpg(srv.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:srv.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait()
(a,b)=(results['upstream'],results['ik']); compare={'prompt_tokens_equal':a['prompt_tokens']==b['prompt_tokens'],'output_text_equal':a['answer'].get('content')==b['answer'].get('content'),'upstream_content':a['answer'].get('content'),'ik_content':b['answer'].get('content'),'note':'Synthetic deterministic smoke only; not a correctness or benchmark suite.'};(p/'equivalence-comparison.json').write_text(json.dumps(compare,indent=2));print(json.dumps(compare),flush=True)
