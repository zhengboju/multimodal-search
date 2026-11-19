#!/usr/bin/env bash
# Data download + conversion (ModelScope). Plan §3.1: week 1 only COCO + Flickr30K + ViDoRe.
# usage: bash scripts/download_data.sh [week1|stage1|stage2|stage3|stage4|all]
set -euo pipefail
cd "$(dirname "$0")/.."
P="python -m mmsearch.data.preprocess"
STAGE=${1:-week1}

week1() {
  $P coco --splits test            # Karpathy 5K test (~0.8 GB)
  $P flickr --splits test          # Karpathy 1K test
  $P vidore --subsets all          # ViDoRe v1, 10 subsets (~3 GB)
}
stage1() { $P coco --splits train; }                       # ~13 GB
stage2() { $P colpali_train --shards "${SHARDS:-35}"; }    # 35 shards ≈ 50k pairs ≈ 22 GB; SHARDS=all for 118k
stage3() {
  $P docvqa --split validation     # 5.3k QA over ~1.3k pages (answers available, unlike ViDoRe test)
  $P chartqa --split test
  $P infovqa --split validation
}
stage4() { $P colpali_pool --n-pages "${POOL_PAGES:-20000}" --shards "${SHARDS:-35}"; }

case "$STAGE" in
  week1) week1 ;;
  stage1) stage1 ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  stage4) stage4 ;;
  all) week1; stage1; stage2; stage3; stage4 ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
