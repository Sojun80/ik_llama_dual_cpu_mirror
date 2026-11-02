#!/usr/bin/env python3
"""Round-5 KV-reuse correctness test: one server, several requests with prompt caching ON, so later requests keep a
cached prefix and rewrite the KV rows after it (the case IK_R5_FA_SHADOW's write log must catch). Records the top-5
probabilities of every generated token so two configs can be compared bitwise, request by request.

usage: seqtest.py --out runs/NN-name --config name=ENV=V,ENV=V ... [--n-predict 160]
Mirror layout tools/m64pair.txt with the recommended IK_R5 environment plus the config's variables.
"""
import argparse, json, os, pathlib, random, signal, subprocess, time, urllib.request

R = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
MODEL = R.parent / '20261031-minicpm-v47-round4/model.complete.gguf'
BIN = R / 'build-mirror3/bin/llama-server'
PORT = 19187
ENV = dict(IK_R5_DYN='1', IK_R5_FA='1', IK_R5_HUGE='1', IK_R5_PINONCE='1', IK_R5_FMOE='1', IK_R5_FMOE_UG='1',
           IK_R5_NONUMAB='1', IK_R5_PF='1', GLIBC_TUNABLES='glibc.malloc.hugetlb=1')

rnd = random.Random(11)
words = 'disk error retry kernel module network packet timeout queue flush cache page memory thread lock'.split()
LONG = ' '.join(rnd.choice(words) for _ in range(3000))
A = 'Summarize the following log-like text in one paragraph.\n\n' + LONG + '\n\nSummary:'
B = 'Write a short story about a lighthouse keeper who collects lost umbrellas.'
REQUESTS = [
    ('A', A),                                                    # fresh long prompt
    ('A2', A[:-len('Summary:')] + 'List three keywords:'),       # same long prefix, different ending: rows rewritten
    ('B', B),                                                    # unrelated short prompt: nearly everything rewritten
    ('A3', A),                                                   # back to A: prefix only up to B's divergence
]


def http(path, obj=None, timeout=900):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}' + path, data, {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run(name, env_extra, out, n_predict):
    cpus = (R / 'tools/m64pair.txt').read_text().strip()
    cmd = ['numactl', '--physcpubind=16-95', str(BIN), '-m', str(MODEL), '-ngl', '0', '--no-mmap', '--numa', 'mirror',
           '--numa-mirror', 'all', '-t', '64', '-tb', '64', '--cpu-range', cpus, '-c', '16384', '--host', '127.0.0.1',
           '--port', str(PORT), '--parallel', '1']
    env = dict(os.environ, **ENV, **env_extra)
    recs = []
    with open(out / f'server-{name}.log', 'w') as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            for _ in range(600):
                if srv.poll() is not None:
                    raise RuntimeError(f'server exited {srv.returncode}')
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2); break
                except Exception:
                    time.sleep(1)
            for tag, prompt in REQUESTS:
                x = http('/completion', {'prompt': prompt, 'n_predict': n_predict, 'temperature': 0, 'n_probs': 5,
                                         'cache_prompt': True, 'seed': 1234})
                recs.append(dict(req=tag, content=x.get('content'), probs=x.get('completion_probabilities'),
                                 tokens_cached=x.get('tokens_cached'), timings=x.get('timings', {})))
                t = x.get('timings', {})
                print(f"{name:8s} {tag:3s} prompt_n={t.get('prompt_n')} cached={x.get('tokens_cached')} "
                      f"decode={t.get('predicted_per_second', 0):.2f}", flush=True)
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL); srv.wait()
    (out / f'run-{name}.json').write_text(json.dumps(recs, indent=1))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', action='append', required=True)
    ap.add_argument('--n-predict', type=int, default=160)
    a = ap.parse_args()
    out = R / a.out
    out.mkdir(parents=True, exist_ok=False)
    res = {}
    for c in a.config:
        n, v = c.split('=', 1)
        env_extra = dict(kv.split('=', 1) for kv in v.split(',')) if v != 'none' else {}
        res[n] = run(n, env_extra, out, a.n_predict)
    names = list(res)
    print('SUMMARY')
    for n in names[1:]:
        for ra, rb in zip(res[names[0]], res[n]):
            print(f"{n} vs {names[0]} {ra['req']:3s}: content same={ra['content'] == rb['content']}  "
                  f"probs bitwise same={ra['probs'] == rb['probs']}")
    print('DONE')


if __name__ == '__main__':
    main()
