import os,subprocess,json,pathlib
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';order=[0,1,4,2,3,5,6,7]
for n in [32,16,48]:
 c=[48+8*j+i for i in order for j in range(6)][:n];mask=hex(sum(1<<i for i in c)); name=f'verify-{n}';cmd=['numactl','--physcpubind=48-95','--membind=1',str(s/'build/bin/llama-bench'),'-m',str(s/'model.complete.gguf'),'-ngl','0','-lm','none','-t',str(n),'-C',mask,'--cpu-strict','1','-p','0','-n','64','-d','2800,9475','-r','2','-o','jsonl','--progress'];(p/(name+'-command.json')).write_text(json.dumps(cmd,indent=2));print('START',n,flush=True)
 with (p/(name+'.jsonl')).open('w') as out,(p/(name+'.log')).open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err,env=env)
 print('DONE',n,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
(p/'VERIFY_FINISHED').write_text('complete\n')
