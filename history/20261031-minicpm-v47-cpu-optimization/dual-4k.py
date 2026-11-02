import os,subprocess,json,pathlib
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=0'
n0=[16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45]; n1=[48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92]; c=n0+n1; mask=hex(sum(1<<i for i in c))
for mode in ['node1','interleave']:
 policy='--membind=1' if mode=='node1' else '--interleave=0,1';name='dual-4k-'+mode;cmd=['numactl','--physcpubind=16-95',policy,str(s/'build/bin/llama-bench'),'-m',str(s/'model.complete.gguf'),'-ngl','0','-lm','none','-t','32','-C',mask,'--cpu-strict','1','-p','0','-n','64','-d','2800','-r','2','-o','jsonl','--progress']; (p/(name+'-command.json')).write_text(json.dumps({'cmd':cmd,'cores':c},indent=2));print('START',name,flush=True)
 with (p/(name+'.jsonl')).open('w') as out,(p/(name+'.log')).open('w') as err:
  process=subprocess.Popen(cmd,stdout=out,stderr=err,env=env)
  import time
  time.sleep(25)
  with (p/(name+'-numastat.txt')).open('w') as n:subprocess.run(['numastat','-p',str(process.pid)],stdout=n)
  with (p/(name+'-smaps.txt')).open('w') as m:
   try:m.write(pathlib.Path(f'/proc/{process.pid}/smaps_rollup').read_text())
   except FileNotFoundError:pass
  if process.wait():raise RuntimeError(name+' failed')
 print('DONE',name,flush=True)
(p/'DUAL_4K_FINISHED').write_text('complete\n')
