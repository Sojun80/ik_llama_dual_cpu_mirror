import os, subprocess, json, pathlib, time
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization'); s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4'); env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1'
order=[0,1,4,2,3,5,6,7]
def cpus(n):return [48+8*ccd+offset for offset in order for ccd in range(6)][:n]
def mask(xs):return hex(sum(1<<x for x in xs))
for n in [16,8,32,12,48,24]:
 c=cpus(n); cmd=['numactl','--physcpubind=48-95','--membind=1',str(s/'build/bin/llama-bench'),'-m',str(s/'model.complete.gguf'),'-ngl','0','-lm','none','-t',str(n),'-C',mask(c),'--cpu-strict','1','-p','0','-n','64','-d','2800','-r','2','-o','jsonl','--progress']
 (p/f'threads-{n}-command.json').write_text(json.dumps({'cmd':cmd,'cpus':c,'env':{'GLIBC_TUNABLES':env['GLIBC_TUNABLES']}},indent=2)+'\n'); print('START',n,time.time(),flush=True)
 with (p/f'threads-{n}.jsonl').open('w') as out,(p/f'threads-{n}.log').open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err,env=env)
 print('DONE',n,r.returncode,time.time(),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
(p/'THREAD_SWEEP_FINISHED').write_text('complete\n')
