#!/usr/bin/env python3
"""Flint 40B-A2.5B CPU baseline on gigabyte5, same method as the MiniCPM-V 4.7 baseline
(20261031-minicpm-v47-kernel-investigation/corpus.py + corpus-ik-command.json):
stock ik_llama.cpp 13cffc0 (the preserved baseline binary), NUMA node 1 only, 32 decode threads,
48 batch threads, same cpu order, -c 16384, same word-salad corpora, raw /completion,
temperature 0, n_predict 512, ignore_eos, cache_prompt false, GLIBC hugetlb.

usage: flintbench.py OUT --threads 32 [--threads 24 ...] [--rounds 2] [--n-predict 512] [--cases medium,long]
One server per decode-thread count; configs alternate per round to control drift.
"""
import argparse, json, os, pathlib, signal, subprocess, time, urllib.request

E = pathlib.Path('/home/ubuntu/llm-evaluations')
D = E / '20261104-flint40b-cpu-baseline'
K = E / '20261031-minicpm-v47-kernel-investigation'
BIN = E / '20261031-minicpm-v47-cpu-optimization/ik-build/bin/llama-server'
MODEL = D / 'flint-int4-q4_1.gguf'
ORDER = '48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61,69,77,85,93,54,62,70,78,86,94,55,63,71,79,87,95'
PORT = 19195


def others():
    out = subprocess.run(['pgrep', '-a', '-x', 'llama-server'], capture_output=True, text=True).stdout.strip()
    svc = subprocess.run(['systemctl', 'is-active', 'casebook-llm', 'casebook', 'casebookharness'],
                         capture_output=True, text=True).stdout.split()
    return out, svc


def req(path, obj, timeout=1800):
    r = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(obj).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def run(out, threads, rnd, cases, n_predict):
    procs, svc = others()
    assert not procs and 'active' not in svc, f'other LLM work running: {procs} {svc}'
    cmd = ['numactl', '--physcpubind=48-95', '--membind=1', str(BIN), '-m', str(MODEL), '-ngl', '0', '--no-mmap',
           '-t', str(threads), '-tb', '48', '--cpu-range', ORDER, '-c', '16384',
           '--host', '127.0.0.1', '--port', str(PORT), '--parallel', '1']
    env = dict(os.environ, GLIBC_TUNABLES='glibc.malloc.hugetlb=1')
    tag = f't{threads}-r{rnd}'
    (out / f'command-{tag}.json').write_text(json.dumps(cmd, indent=1))
    res = {}
    with open(out / f'server-{tag}.log', 'w') as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            t0 = time.time()
            while True:
                if srv.poll() is not None:
                    raise RuntimeError(f'server exited {srv.returncode}')
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2); break
                except Exception:
                    time.sleep(1)
                if time.time() - t0 > 900:
                    raise RuntimeError('startup timeout')
            (out / f'smaps-{tag}.txt').write_text(pathlib.Path(f'/proc/{srv.pid}/smaps_rollup').read_text())
            (out / f'numastat-{tag}.txt').write_text(
                subprocess.run(['numastat', '-p', str(srv.pid)], capture_output=True, text=True).stdout)
            for name in cases:
                txt = (K / f'corpus-{name}.txt').read_text()
                toks = req('/tokenize', {'content': txt, 'add_special': False})['tokens']
                x = req('/completion', {'prompt': txt, 'temperature': 0, 'n_predict': n_predict, 'seed': 1234,
                                        'cache_prompt': False, 'stream': False, 'ignore_eos': True, 'n_probs': 5})
                (out / f'{tag}-{name}.json').write_text(json.dumps({'tokens': len(toks), 'response': x}, indent=1))
                t = x['timings']
                res[name] = t
                print(f"{tag:8s} {name:6s} prompt_tokens={t['prompt_n']} decode={t['predicted_per_second']:.2f} "
                      f"prompt={t['prompt_per_second']:.1f} n={t['predicted_n']}", flush=True)
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL); srv.wait()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--threads', type=int, action='append', required=True)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--n-predict', type=int, default=512)
    ap.add_argument('--cases', default='medium,long')
    a = ap.parse_args()
    out = D / a.out
    out.mkdir(parents=True, exist_ok=False)
    cases = a.cases.split(',')
    allr = {}
    for r in range(a.rounds):
        order = a.threads if r % 2 == 0 else list(reversed(a.threads))
        for th in order:
            allr.setdefault(th, []).append(run(out, th, r, cases, a.n_predict))
    print('SUMMARY')
    summ = {}
    for th, rs in allr.items():
        for c in cases:
            d = [x[c]['predicted_per_second'] for x in rs]
            p = [x[c]['prompt_per_second'] for x in rs]
            summ[f't{th}-{c}'] = dict(decode=d, prompt=p, prompt_tokens=rs[0][c]['prompt_n'])
            print(f"t{th:<3d} {c:6s} tokens={rs[0][c]['prompt_n']}  decode mean {sum(d)/len(d):.2f} {[round(v, 2) for v in d]}"
                  f"  prompt mean {sum(p)/len(p):.1f}")
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
