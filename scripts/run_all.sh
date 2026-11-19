#!/usr/bin/env bash
# End-to-end driver, one function per plan stage. usage: bash scripts/run_all.sh <stage0|stage1|stage2|stage3|stage4|stage5|all>
# GPU plan: GPU0 = training / encoders, GPU1 = vLLM servers. Long jobs: run inside tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
export MMS_STORAGE=${MMS_STORAGE:-$PWD/storage}
VLM=${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}
TR="torchrun --nproc_per_node ${NGPU:-2}"

start_vlm() {  # $1 model  $2 port  $3 name  -> background server, waits until ready
  bash scripts/serve_vlm.sh "$1" "$2" "$3" 1 > "$MMS_STORAGE/vllm_$2.log" 2>&1 &
  echo $! > "$MMS_STORAGE/vllm_$2.pid"
  until curl -sf "http://127.0.0.1:$2/v1/models" > /dev/null; do sleep 10; done
  echo "vLLM $3 ready on :$2"
}
stop_vlm() { [ -f "$MMS_STORAGE/vllm_$1.pid" ] && kill "$(cat "$MMS_STORAGE/vllm_$1.pid")" 2>/dev/null || true; }

stage0() {  # week 1: env, data, CLIP zero-shot baseline, modality gap
  python scripts/check_env.py
  torchrun --nproc_per_node 2 scripts/check_dist.py
  bash scripts/download_data.sh week1
  python -m mmsearch.eval.eval_clip --model openai/clip-vit-large-patch14 --dataset coco_test --folds 1 5
  python -m mmsearch.eval.eval_clip --model openai/clip-vit-large-patch14 --dataset flickr_test --folds 1
  python -m mmsearch.eval.modality_gap --dataset coco_test
}

stage1() {  # weeks 2-3: loss ablation (E2) then negative-sampling ladder (E1)
  bash scripts/download_data.sh stage1
  for loss in infonce siglip triplet; do
    $TR -m mmsearch.train.train_clip --config configs/clip_coco.yaml loss=$loss name=clip_${loss}_inbatch
  done
  for neg in random hard mixed; do
    $TR -m mmsearch.train.train_clip --config configs/clip_coco.yaml negatives=$neg name=clip_hn_$neg
  done
  for run in clip_infonce_inbatch clip_siglip_inbatch clip_triplet_inbatch clip_hn_random clip_hn_hard clip_hn_mixed; do
    python -m mmsearch.eval.eval_clip --model "$MMS_STORAGE/runs/clip/$run/best" --dataset coco_test --folds 1 5
  done
}

stage2() {  # weeks 4-6: ColQwen2 reproduction, OCR for BM25, ViDoRe benchmark, index ablation, badcases
  bash scripts/download_data.sh stage2
  $TR -m mmsearch.train.train_colpali --config configs/colpali_vidore.yaml
  start_vlm "$VLM" 8000 qwen-vl
  python -m mmsearch.data.ocr --datasets vidore_arxivqa vidore_docvqa vidore_infovqa vidore_tabfquad vidore_tatdqa \
    vidore_shift vidore_synth_ai vidore_synth_energy vidore_synth_gov vidore_synth_health
  python -m mmsearch.eval.run_benchmark --config configs/vidore_benchmark.yaml
  stop_vlm 8000
  python -m mmsearch.eval.run_benchmark --config configs/index_ablation.yaml
  latest=$(ls -td "$MMS_STORAGE"/runs/benchmark/index_ablation_* | head -1)
  python -m mmsearch.eval.plots tradeoff --results "$latest/results.json" --metric ndcg@5
  main=$(ls -td "$MMS_STORAGE"/runs/benchmark/vidore_main_* | head -1)
  python -m mmsearch.eval.badcase --run-dir "$main" --method "colqwen2-v1.0(ref)" --compare bm25-ocr
}

stage3() {  # weeks 7-8: RAG end-to-end + attribution + k sensitivity
  bash scripts/download_data.sh stage3
  start_vlm "$VLM" 8000 qwen-vl
  python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml
  python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml budget.strategy=rank_decay name=rag_docvqa_rankdecay
  python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml dataset=chartqa_test name=rag_chartqa
  stop_vlm 8000
}

stage4() {  # weeks 9-10: query understanding
  bash scripts/download_data.sh stage4
  start_vlm "$VLM" 8000 qwen-vl
  python -m mmsearch.query.translate --datasets vidore_all --langs zh ja
  python -m mmsearch.query.synth build --config configs/query_rewrite.yaml
  stop_vlm 8000
  echo ">>> label $MMS_STORAGE/runs/query/qrewrite/human_check.csv, then: python -m mmsearch.query.synth validate --file ..."
  $TR -m mmsearch.train.train_qlora --config configs/query_rewrite.yaml
  GPU_UTIL=0.45 start_vlm "$VLM" 8000 qwen-vl
  GPU_UTIL=0.45 start_vlm "$MMS_STORAGE/runs/query/qrewrite_sft/merged" 8002 rewriter
  python -m mmsearch.query.run_rewrite_eval --config configs/query_rewrite.yaml
  stop_vlm 8002; stop_vlm 8000
}

stage5() {  # weeks 11-12: LLM judge, service + load test, one-command benchmark
  start_vlm "$VLM" 8000 qwen-vl
  main=$(ls -td "$MMS_STORAGE"/runs/benchmark/vidore_main_* | head -1)
  python -m mmsearch.eval.llm_judge judge --run-dir "$main" --method "colqwen2-v1.0(ref)" --n-queries 200
  echo ">>> label $main/judge/human_annotation.csv, then: python -m mmsearch.eval.llm_judge agree --run-dir $main"
  MMS_SERVICE_CONFIG=configs/service.yaml CUDA_VISIBLE_DEVICES=0 uvicorn mmsearch.service.app:app --port 8080 > "$MMS_STORAGE/service.log" 2>&1 &
  SVC=$!
  until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 5; done
  python -m mmsearch.service.loadtest --url http://127.0.0.1:8080 --dataset vidore_docvqa --mode multivector
  kill $SVC
  stop_vlm 8000
}

case "${1:-}" in
  stage0|stage1|stage2|stage3|stage4|stage5) "$1" ;;
  all) stage0; stage1; stage2; stage3; stage4; stage5 ;;
  *) echo "usage: $0 <stage0|stage1|stage2|stage3|stage4|stage5|all>"; exit 1 ;;
esac
