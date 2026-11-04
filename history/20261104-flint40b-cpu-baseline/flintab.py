#!/usr/bin/env python3
"""Flint 40B-A2.5B A/B harness (same payload as flintbench.py: word-salad corpora, raw /completion, temperature 0,
ignore_eos, cache_prompt false, GLIBC hugetlb). Configs are interleaved per round; one model instance at a time.

  flintab.py OUT --config NAME=MODE:THREADS[:ENV=V,ENV=V] [...] [--rounds 2] [--n-predict 512] [--cases medium,long]
             [--probs]   (n_predict 32, n_probs 5; compares every config's tokens and probabilities with the first)
  [--extra NAME="--flag value ..."]  extra server args for a config

MODE: stock1  = preserved stock ik 13cffc0, node 1 only, -tb 48, baseline cpu order
      r5node1 = round-5 build (build-mirror3), node 1 only, -tb 48, baseline cpu order
      r5mirror= round-5 build, --numa mirror on cores 16-95, -t = -tb = THREADS, tools/m64pair.txt order (THREADS=64)
"""
import argparse, json, os, pathlib, shlex, signal, subprocess, time, urllib.request

E = pathlib.Path('/home/ubuntu/llm-evaluations')
D = E / '20261104-flint40b-cpu-baseline'
K = E / '20261031-minicpm-v47-kernel-investigation'
R5 = E / '20261031-minicpm-v47-round5'
STOCK = E / '20261031-minicpm-v47-cpu-optimization/ik-build/bin/llama-server'
MODEL = D / 'flint-int4-q4_1.gguf'
ORDER = '48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61,69,77,85,93,54,62,70,78,86,94,55,63,71,79,87,95'
PORT = 19196


def req(path, obj, timeout=1800):
    r = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(obj).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def command(mode, threads, binary):
    if mode in ('stock1', 'r5node1'):
        b = STOCK if mode == 'stock1' else R5 / 'build-mirror3/bin/llama-server'
        return ['numactl', '--physcpubind=48-95', '--membind=1', str(binary or b), '-m', str(MODEL), '-ngl', '0',
                '--no-mmap', '-t', str(threads), '-tb', '48', '--cpu-range', ORDER]
    if mode == 'r5mirror':
        cpus = (R5 / 'tools/m64pair.txt').read_text().strip()
        return ['numactl', '--physcpubind=16-95', str(binary or R5 / 'build-mirror3/bin/llama-server'), '-m', str(MODEL),
                '-ngl', '0', '--no-mmap', '--numa', 'mirror', '--numa-mirror', 'all', '-t', str(threads),
                '-tb', str(threads), '--cpu-range', cpus]
    raise ValueError(mode)


def run(out, name, spec, extra, rnd, cases, n_predict, probs):
    procs = subprocess.run(['pgrep', '-a', '-x', 'llama-server'], capture_output=True, text=True).stdout.strip()
    svc = subprocess.run(['systemctl', 'is-active', 'casebook-llm', 'casebook', 'casebookharness'],
                         capture_output=True, text=True).stdout.split()
    assert not procs and 'active' not in svc, f'other LLM work running: {procs} {svc}'
    parts = spec.split(':', 2)
    mode, threads = parts[0], int(parts[1])
    envx = dict(kv.split('=', 1) for kv in parts[2].split(',')) if len(parts) > 2 and parts[2] else {}
    binary = envx.pop('BIN', None)
    cmd = command(mode, threads, binary) + ['-c', '16384', '--host', '127.0.0.1', '--port', str(PORT), '--parallel', '1'] \
        + shlex.split(extra.get(name, ''))
    env = dict(os.environ, GLIBC_TUNABLES='glibc.malloc.hugetlb=1', **envx)
    tag = f'{name}-r{rnd}'
    (out / f'command-{tag}.json').write_text(json.dumps({'cmd': cmd, 'env': envx}, indent=1))
    res = {}
    with open(out / f'server-{tag}.log', 'w') as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            t0 = time.time()
            while True:
                if srv.poll() is not None:
                    raise RuntimeError(f'{name}: server exited {srv.returncode}')
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2); break
                except Exception:
                    time.sleep(1)
                if time.time() - t0 > 900:
                    raise RuntimeError('startup timeout')
            for case in cases:
                txt = (K / f'corpus-{case}.txt').read_text()
                payload = {'prompt': txt, 'temperature': 0, 'n_predict': 32 if probs else n_predict, 'seed': 1234,
                           'cache_prompt': False, 'stream': False, 'ignore_eos': True}
                if probs:
                    payload['n_probs'] = 5
                x = req('/completion', payload)
                (out / f'{tag}-{case}.json').write_text(json.dumps(x, indent=1))
                t = x['timings']
                res[case] = dict(t=t, content=x.get('content'), probs=x.get('completion_probabilities'))
                print(f"{tag:14s} {case:6s} prompt_n={t['prompt_n']} decode={t['predicted_per_second']:.2f} "
                      f"prompt={t['prompt_per_second']:.1f}", flush=True)
            st = pathlib.Path(f"/proc/{srv.pid}/status").read_text()
            print(tag, [l.replace(chr(9), " ").strip() for l in st.splitlines() if l.startswith(("VmHWM", "VmRSS"))], flush=True)
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
    ap.add_argument('--config', action='append', required=True)
    ap.add_argument('--extra', action='append', default=[])
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--n-predict', type=int, default=512)
    ap.add_argument('--cases', default='medium,long')
    ap.add_argument('--probs', action='store_true')
    a = ap.parse_args()
    out = D / a.out
    out.mkdir(parents=True, exist_ok=False)
    cases = a.cases.split(',')
    cfg = dict(c.split('=', 1) for c in a.config)
    extra = dict(e.split('=', 1) for e in a.extra)
    names = list(cfg)
    allr = {n: [] for n in names}
    for r in range(1 if a.probs else a.rounds):
        for n in (names if r % 2 == 0 else list(reversed(names))):
            allr[n].append(run(out, n, cfg[n], extra, r, cases, a.n_predict, a.probs))
    print('SUMMARY')
    summ = {}
    for n in names:
        for c in cases:
            d = [x[c]['t']['predicted_per_second'] for x in allr[n]]
            p = [x[c]['t']['prompt_per_second'] for x in allr[n]]
            summ[f'{n}|{c}'] = dict(decode=d, prompt=p)
            line = f"{n:12s} {c:6s} decode mean {sum(d)/len(d):.2f} {[round(v, 2) for v in d]}  prompt mean {sum(p)/len(p):.1f}"
            if a.probs and n != names[0]:
                ref = allr[names[0]][0][c]
                cur = allr[n][0][c]
                toks_same = [e['id'] if 'id' in e else e.get('token') for e in ref['probs']] == \
                            [e['id'] if 'id' in e else e.get('token') for e in cur['probs']]
                line += f"  | vs {names[0]}: content same={ref['content'] == cur['content']} tokens same={toks_same} " \
                        f"probs bitwise={ref['probs'] == cur['probs']}"
            print(line)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
