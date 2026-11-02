import os,subprocess,json,pathlib,time
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization'); s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1'
order=[0,1,4,2,3,5,6,7]
for n,b,ub in [(48,2048,512),(48,2048,128),(48,2048,256),(48,2048,1024),(48,1024,512),(32,2048,512)]:
 cpus=[48+8*ccd+offset for offset in order for ccd in range(6)][:n];mask=hex(sum(1<<x for x in cpus)); name=f'prefill-t{n}-b{b}-ub{ub}'
 cmd=['numactl','--physcpubind=48-95','--membind=1',str(s/'build/bin/llama-bench'),'-m',str(s/'model.complete.gguf'),'-ngl','0','-lm','none','-t',str(n),'-C',mask,'--cpu-strict','1','-p','2048','-n','0','-d','0','-b',str(b),'-ub',str(ub),'-r','2','-o','jsonl','--progress']
 (p/(name+'-command.json')).write_text(json.dumps({'cmd':cmd,'cpus':cpus},indent=2));print('START',name,flush=True)
 with (p/(name+'.jsonl')).open('w') as out,(p/(name+'.log')).open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err,env=env)
 print('DONE',name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
(p/'PREFILL_FINISHED').write_text('complete\n')
