#!/bin/bash
# placemap.sh OUT MODE   (MODE = single | mirror)
# Starts one server, records numa_maps after load, runs the 2.8K corpus request (512 tokens),
# records numa_maps again, stops the server.
set -u
OUT=$1; MODE=$2
R=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5
KI=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
MODEL=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf
PORT=19188
mkdir -p "$OUT"
if [ "$MODE" = mirror ]; then
    env GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 numactl --physcpubind=16-95 \
        $R/build-mirror3/bin/llama-server -m $MODEL -ngl 0 --no-mmap --numa mirror --numa-mirror all \
        -t 64 -tb 64 --cpu-range $(cat $R/tools/m64.txt) -c 16384 --host 127.0.0.1 --port $PORT --parallel 1 > "$OUT/server.log" 2>&1 &
else
    env GLIBC_TUNABLES=glibc.malloc.hugetlb=1 numactl --physcpubind=48-95 --membind=1 \
        /home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization/ik-build/bin/llama-server -m $MODEL -ngl 0 --no-mmap \
        -t 32 -tb 48 --cpu-range 48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61,69,77,85,93,54,62,70,78,86,94,55,63,71,79,87,95 \
        -c 16384 --host 127.0.0.1 --port $PORT --parallel 1 > "$OUT/server.log" 2>&1 &
fi
SRV=$!
for i in $(seq 600); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
python3 $R/tools/numamap_summary.py $SRV > "$OUT/after-load.txt"
python3 - "$KI/corpus-medium.txt" $PORT > "$OUT/request.json" <<'PY'
import json, sys, urllib.request
txt = open(sys.argv[1]).read()
req = urllib.request.Request(f'http://127.0.0.1:{sys.argv[2]}/completion',
      json.dumps({'prompt': txt, 'temperature': 0, 'n_predict': 512, 'cache_prompt': False, 'ignore_eos': True}).encode(),
      {'Content-Type': 'application/json'})
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=900))['timings']))
PY
python3 $R/tools/numamap_summary.py $SRV > "$OUT/after-request.txt"
grep -E "mirror|NUMA" "$OUT/server.log" | head -5 > "$OUT/mirror-log.txt"
kill $SRV; wait $SRV 2>/dev/null
echo "== after load"; cat "$OUT/after-load.txt"; echo "== after request"; cat "$OUT/after-request.txt"; cat "$OUT/request.json"
