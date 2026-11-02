import pathlib,json,time,subprocess,urllib.request,hashlib,os,signal,sys,traceback
P=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4'); O=pathlib.Path('/home/ubuntu/llm-evaluations/20261029-tiny-log-intelligence-round3'); R=pathlib.Path('/home/ubuntu/llm-evaluations/20261029-tiny-log-intelligence-round4'); base='http://127.0.0.1:19161'
def req(path,obj): return urllib.request.urlopen(urllib.request.Request(base+path,json.dumps(obj).encode(),{'Content-Type':'application/json'}),timeout=7200)
def write(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')
assert subprocess.check_output(['sha256sum',str(P/'model.complete.gguf')],text=True).split()[0]=='9ecf91e814d7e9adc64325b1b65f8628a9b1bb8da4f61a92869ca7489bfd888e'
subprocess.run(['sha256sum','-c',str(R/'datasets.sha256')],cwd=O,check=True)
cmd=['numactl','--physcpubind=48-95','--membind=1',str(P/'build/bin/llama-server'),'-m',str(P/'model.complete.gguf'),'--load-mode','none','-ngl','0','-t','16','-C','0x111113131313000000000000','--cpu-strict','1','-tb','48','-Cb','0xffffffffffff000000000000','--cpu-strict-batch','1','-c','32768','--host','127.0.0.1','--port','19161','--parallel','1']
write(P/'command.json',cmd); env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';out=(P/'server.log').open('w');s=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
try:
 for i in range(600):
  if s.poll() is not None: raise RuntimeError('server exited '+str(s.returncode))
  try:
   urllib.request.urlopen(base+'/health',timeout=2);break
  except Exception:time.sleep(1)
 else:raise RuntimeError('startup timeout')
 kw={'enable_thinking':False}; sampling={'temperature':1,'top_p':.95,'top_k':0,'min_p':0.,'repeat_penalty':1.,'presence_penalty':0.}
 def run(label,messages,seed,budget):
  rendered=json.load(req('/apply-template',{'messages':messages,'chat_template_kwargs':kw}))['prompt']; tokens=len(json.load(req('/tokenize',{'content':rendered,'add_special':False}))['tokens']); assert tokens+budget+2<=32768
  (P/(label+'-rendered.txt')).write_text(rendered)
  if rendered[-200:].rfind('<think>') > rendered[-200:].rfind('</think>'):raise RuntimeError('unclosed thinking opener in rendered prompt')
  payload={'messages':messages,**sampling,'seed':seed,'max_tokens':budget,'stream':True,'cache_prompt':False,'chat_template_kwargs':kw,'return_progress':True};write(P/(label+'-request.json'),payload)
  events=[];content='';reason='';start=time.monotonic()
  with req('/v1/chat/completions',payload) as response:
   for line in response:
    line=line.decode().strip()
    if not line.startswith('data: '):continue
    if line[6:]=='[DONE]':break
    e=json.loads(line[6:]);events.append(e);d=(e.get('choices') or [{}])[0].get('delta',{});content+=d.get('content') or '';reason+=d.get('reasoning_content') or ''
  last=events[-1] if events else {};finish=(last.get('choices') or [{}])[0].get('finish_reason'); result={'finish_reason':finish,'elapsed_seconds':time.monotonic()-start,'prompt_tokens':tokens,'timings':last.get('timings',{}),'content':content,'reasoning':reason,'events':events};write(P/(label+'-result.json'),result);(P/(label+'-answer.txt')).write_text(content);print(label,finish,len(content),len(reason),result['timings'],flush=True);return result
 smoke=run('smoke',[{'role':'user','content':'A service logged an error and then restarted successfully. What is known, and what is unknown? Answer briefly.'}],1234,256)
 if smoke['finish_reason']!='stop' or not smoke['content'].strip() or smoke['reasoning'].strip() or '<think>' in smoke['content']:raise RuntimeError('smoke gate failed')
 (P/'SMOKE_PASSED').write_text('stop, nonempty final answer, no thinking output\n')
 prompt=(R/'task-instructions.txt').read_text().rstrip()
 for scenario in ['small','large','normal','ambiguous']:
  dataset=(O/'datasets'/f'{scenario}.log').read_text(); messages=[{'role':'user','content':prompt+'\n\nLogs:\n'+dataset}]
  for seed in [1234,2345,3456]:run(f'{scenario}-{seed}',messages,seed,8192)
 (P/'EVALUATION_FINISHED').write_text('12 requests completed; inspect finish reasons and grade manually\n')
finally:
 os.killpg(s.pid,signal.SIGTERM)
 try:s.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(s.pid,signal.SIGKILL);s.wait()
 out.close()
