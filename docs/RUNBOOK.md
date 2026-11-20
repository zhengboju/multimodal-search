# GPU 机器执行与学习手册

> 配套：[README.md](../README.md)（架构与设计取舍）、[执行计划](../多模态检索项目执行计划.md)（12 周排期）、[REPORT.md](REPORT.md)（结果填写模板）。
> 适用机器：2 × H20 96GB（NVLink）、Python 3.11.11、torch 2.10、transformers 5.17、vLLM 0.19.1，**HuggingFace 不通**。

---

## 0. 一页速览

```bash
git clone https://github.com/zhengboju/multimodal-search.git && cd multimodal-search
export MMS_STORAGE=/data/mmsearch            # 大容量盘；默认 ./storage
bash scripts/setup_env.sh                    # 只补装缺失依赖，不碰 torch/vllm
bash scripts/run_all.sh stage0               # 数据 + 环境验收 + CLIP 零样本基线
```

三条铁律：

1. **模型/数据集只走 ModelScope**（`MMS_HUB_ORDER=modelscope,hf`），HF 仅作镜像兜底；下载一次后本地就有，后续离线可用。
2. **训练用 GPU0，vLLM 用 GPU1**（`scripts/serve_vlm.sh` 第 4 个参数默认 1），四个阶段并行也不互相抢显存。
3. **test split 只跑一次**：模型选择一律用 dev（CLIP 从 train 留出 5k 图；ColQwen2 用 colpali_train_set 自带 test 切分）。每次 test 评测都会写进 `storage/runs/test_ledger.jsonl`，超过 10 次会告警。

时间预算（全职、双卡、按本文顺序执行）：

| 阶段 | 内容 | 纯执行 | 含学习 | 主要卡点 |
|---|---|---|---|---|
| 0 | 环境 + 数据 + CLIP 零样本 | 3–5 h（多为下载） | 1 天 | 下载耗时 |
| 1 | CLIP 微调 + 负样本/loss 消融（6 组） | 20–26 h | 3–4 天 | 显存/吞吐摸底 |
| 2 ★ | ColQwen2 复现 + OCR + ViDoRe + 索引消融 | 1.5–2 天 | 8–12 天 | 复现偏差排查 |
| 3 | 多模态 RAG + 归因 | 4–6 h | 2–3 天 | vLLM 服务调参 |
| 4 | Query 理解（合成→SFT→下游评估） | 8–12 h | 4–5 天 | 合成数据质量 |
| 5 | 评估体系 + 服务压测 + 报告 | 6–10 h | 3–4 天 | 报告写作 |

**先跑通最小闭环（约 30 分钟，除首次模型下载外）**，再排长任务：

```bash
bash scripts/download_data.sh week1                 # COCO 5K test + Flickr 1K + ViDoRe（~4 GB）
python -m mmsearch.eval.run_benchmark --config configs/vidore_benchmark.yaml \
  datasets=[vidore_docvqa] max_queries=50 latency.enabled=false
```

跑通后再把 `datasets` 换回 `vidore_all`、去掉 `max_queries`。

---

## 1. 第一次上手（Day 1，约 2 小时）

### 1.1 目录布局

```
MMS_STORAGE/                      # 默认 ./storage，建议软链到大盘：ln -s /data/mmsearch storage
├── models/                       # ModelScope 下载的权重（CLIP 1.7G、ColQwen2 4.4G、Qwen2.5-VL-7B ~16G）
├── raw/                          # 原始数据集（parquet / webdataset tar）
├── processed/<name>/             # 统一格式：corpus.jsonl / queries.jsonl / qrels.tsv / images/ （+ ocr.jsonl）
├── cache/emb/<encoder_tag>/      # 嵌入缓存（doc 侧按语料名+doc 列表哈希；query 侧按文本集合哈希）
└── runs/                         # 训练 run（config.yaml/metrics.jsonl/summary.json）、benchmark、report
```

`storage/processed/` 里的名字就是所有命令里的 `--dataset`：

| 名字 | 来源 | 规模 | 用途 |
|---|---|---|---|
| `vidore_arxivqa` … `vidore_synth_health`（10 个） | ViDoRe v1 test | 70–1000 页/子集 | 阶段 2 主评测 |
| `coco_test` / `coco_train` / `flickr_test` | clip-benchmark wds（Karpathy） | 5K / 113k / 1K | 阶段 0–1 |
| `colpali_train_pool` | colpali_train_set 抽样 2 万页 | 20k 页 | 阶段 4 检索奖励池（训练侧） |
| `docvqa_validation` / `infovqa_validation` / `chartqa_test` | DocVQA / InfoVQA / ChartQA | 1.3k/0.5k/2.5k 页 | 阶段 3（有答案） |
| `vidore_pool`（可选） | 上述语料合并 | 2 万+ 页 | 索引延迟实验（1k 页太小看不出差异） |

### 1.2 环境自检要怎么看

```bash
python scripts/check_env.py
```

- `GPUs: 2` 且两行 `<96 GB> bf16=True` —— 否则先查 `nvidia-smi` 与容器透传。
- `OK flash_attn`：训练会走 `flash_attention_2`；若为 `--`，代码自动降级 `sdpa`（慢 ~15%，不影响正确性）。
- `-- peft` → `pip install peft`；`-- faiss` → 只用 PLAID-lite 就够，`FaissTokenIndex` 可跳过。
- `-- modelscope unreachable`：查代理/内网 DNS，这一步不通后面全卡。
- `/dev/shm: 64 MB` —— **容器默认值，会让多进程 DataLoader 直接崩**（见排障表 `unable to allocate shared memory`）。代码已自动降级为线程池（只影响速度，不影响结果）；想跑满速就让运维用 `--shm-size=8g` 重启容器，或 `export MMS_NUM_WORKERS=...` 手动指定。

```bash
torchrun --nproc_per_node 2 scripts/check_dist.py
# 期望：all_reduce OK on 2 ranks；all_reduce 1 GiB: ~10-20 ms, bus bandwidth ≈ 200-400 GB/s
```
带宽低于 50 GB/s 说明没走 NVLink（PCIe 兜底），2 卡训练的 gather 会成为瓶颈 —— 消融实验尽量单卡跑大步数，或减小 `per_device_batch`。

### 1.3 预下载（把网络风险提前暴露）

```bash
python -m mmsearch.utils.hub --model \
  AI-ModelScope/clip-vit-large-patch14 \
  vidore/colqwen2-v1.0-hf \
  Qwen/Qwen2-VL-2B-Instruct \
  Qwen/Qwen2.5-VL-7B-Instruct
```
断点续传：重跑同一条命令即可（`storage/models/<id>` 有 `.mms_complete` 哨兵则跳过）。

### 1.4 长任务纪律

```bash
tmux new -s mm            # 所有训练/服务都放 tmux
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 5   # 另开窗口盯
tail -f storage/runs/clip/clip_infonce_inbatch/metrics.jsonl   # 每 20 step 一行
```
每个 run 目录都有 `config.yaml`（本次全部超参，含覆盖项）和 `meta.json`（git commit + 开始时间）—— 复现只需要这两样。

---

## 2. 阶段执行手册

每个阶段格式：**目标 → 命令 → 产物 → 验收 → 卡点**。命令均在仓库根目录执行。

### 阶段 0｜基建与基线（第 1 周）

```bash
bash scripts/download_data.sh week1                      # ~4 GB
python -m mmsearch.eval.eval_clip --dataset coco_test   --folds 1 5   # 5K 全量 + COCO 1K×5 折
python -m mmsearch.eval.eval_clip --dataset flickr_test --folds 1
python -m mmsearch.eval.modality_gap --dataset coco_test              # D4：模态鸿沟
```

产物：`storage/runs/clip_eval/<model>_<dataset>/results.md`、`storage/runs/analysis/modality_gap_*/{report.md,gap.json,cosine_hist.png,conditions.png}`。

模态鸿沟的读法（结论由数字决定，不要预设）：看 `translate`（只平移文本云 = 真正的去鸿沟）的 Δ——**掉很多**说明鸿沟轴本身就承载匹配信号（匹配对的偏移 68% 落在该轴上），事后减掉等于删信息，只能在训练目标里解决；**几乎不动**说明鸿沟不是召回瓶颈，掉点来自居中附带的副作用。再看 `margin`（matched−unmatched）在所有条件里的变化：它才是排序真正可用的“判别预算”，`translate` 若把 0.17 压到 0.08，召回必然塌。`center_txt` / `center_img` 的不对称告诉你哪一侧更脆弱（文本云更分散 0.59 vs 图像 0.70，两边作为候选/查询时的行为不同）。报告里写“margin 0.17 对质心偏移 1.26”这个比值，它才是双塔在这份数据上的天花板。

**验收（论文水平，ViT-L/14，COCO 5K）**：i2t R@1 ≈ 56–58，t2i R@1 ≈ 36–37；Flickr 1K t2i R@1 ≈ 85–87。
不达标按此顺序查：① `preprocessor_config.json` 的 mean/std 是否被沿用（代码已自动读）② resize 是否 `Resize(短边)+CenterCrop`（不是直接 Resize 到 224）③ 文本是否截断到 77 ④ 是否忘了 L2 归一化。

**卡点**：COCO test 分片 5 个 tar（~0.8 GB）下载慢 → 提前挂后台；`wds` 的一个样本可能带 5 条 caption，代码按图片 sha1 合并（日志里 `captions/image≈5.0` 才算对）。

---

### 阶段 1｜CLIP 微调与消融（第 2–3 周）

```bash
bash scripts/download_data.sh stage1          # coco_train，~13 GB

# E2：三种 loss（其余不变）
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml loss=infonce name=clip_infonce_inbatch
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml loss=siglip  name=clip_siglip_inbatch
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml loss=triplet name=clip_triplet_inbatch

# E1：负样本阶梯（loss 固定 infonce）
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml negatives=random name=clip_hn_random
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml negatives=hard   name=clip_hn_hard
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml negatives=mixed  name=clip_hn_mixed

# 统一在测试集上评一次（coco_test 同时出 5K 与 1K×5）
for r in clip_infonce_inbatch clip_siglip_inbatch clip_triplet_inbatch clip_hn_random clip_hn_hard clip_hn_mixed; do
  python -m mmsearch.eval.eval_clip --model "$MMS_STORAGE/runs/clip/$r/best" --dataset coco_test --folds 1 5
done
```

产物：`storage/runs/clip/<name>/{best,last}/`（HF 格式，可直接给 `eval_clip` 或 `DenseRetriever` 用）、`metrics.jsonl`。

**先建干净的测试集，再评第一组**（阶段 1 起口径固定，否则后面无法横向比较）：

```bash
bash scripts/download_data.sh stage1
python -m mmsearch.data.check_leak --a coco_train --b coco_test --write-clean coco_test_clean
# 之后所有消融都用这条命令评： --dataset coco_test_clean --folds 1 5
```
`check_leak` 的判据是 **“(文本, 图片) 对重叠 = 0”**，不是“文本零重复”——COCO 的短 caption 天生会在不同图之间复用（实测 4.5%），那属于正常。实测剔除后 t2i **+2.1 点**（这些 query 上存在跨模态负迁移），因此 clean 集比原始集更适合当基准。

**先测吞吐再定 batch**（计划里的硬要求）：加 `per_device_batch=128 log_every=10` 跑 50 step，看日志里的 `img/s`、`peak_mem_gb`，按显存 ×0.85 线性放大。参考：bs 512×2 卡 + 梯度检查点约 60–70 GB/卡；不开检查点会 OOM。
CLIP 每 epoch 约 110k/1024 ≈ 108 step，单 step 1.5–2.5 s ⇒ 每 epoch 4–6 分钟，8 epoch + 每 epoch 一次 5k dev 编码 ≈ 45–60 分钟/组。6 组 ≈ 6 h（若发现 8 epoch 早就在 dev 上平台，可降到 5 epoch）。

**验收**：mixed 相对 inbatch 的 t2i R@1 ≥ +3 点，且 `hard ≥ random`（否则说明挖出来的“难负样本”其实是假负例，去看 `rank_range` 下界）。

**卡点**
- DDP 报 `Expected to have finished reduction in the prior iteration` 或参数未参与计算 → `siglip/triplet` 已自动冻结 `logit_scale`；自己改 loss 时记得保持。
- 显存抖动 → `grad_ckpt: true` 别关；`num_workers=12` 若 CPU 核少会饿死 GPU，看 `img/s` 是否掉。
- 难负样本挖掘每 epoch 一次，2 卡各编码一半再 all-gather（`dist_encode`），约 1–2 分钟。

---

### 阶段 2｜文档视觉检索（第 4–6 周）★ 核心

**2.1 复现 ColQwen2**

```bash
bash scripts/download_data.sh stage2                       # colpali_train 35 分片 ≈ 50k 对，~22 GB；SHARDS=all 为 118k
torchrun --nproc_per_node 2 -m mmsearch.train.train_colpali --config configs/colpali_vidore.yaml
```

先做一次 20 分钟摸底（正式训练前必做）。判断瓶颈只看日志里两个数：
* `data-wait`（每 10 步打印）——这一步里有多少时间在等数据。正常 <15%；>40% 就去调 `MMS_NUM_WORKERS`（解码线程数，建议 `CPU 核数 ÷ 卡数`），而不是调模型。
* `img/s` 与 `peak_mem_gb`——按显存 ×0.85 反推能加到多大的 `per_device_batch`。
```bash
torchrun --nproc_per_node 2 -m mmsearch.train.train_colpali --config configs/colpali_vidore.yaml \
  data.max_train=2000 per_device_batch=8 epochs=1 log_every=5 eval_every=100000 eval_at_start=false
```
看 `pairs/s` 和 `peak_mem_gb`，然后线性放大 `per_device_batch`（32 时约 55–70 GB/卡，760 token/页）。总步数 = 50k × epochs ÷ (bs × 2)。**这一步决定后面所有时间估算，务必先跑。**

产物：`storage/runs/colpali/colqwen2_repro/{step*/final}/`（`adapter/` + `proj.pt` + `encoder_config.json`，`ColQwen2Encoder(checkpoint=...)` 直接加载）。
训练中每 500 step 会在 dev（colpali_train_set 自带 test 切分）上算 nDCG@5 —— 这是模型选择依据，**不是** ViDoRe。

**2.2 OCR（BM25 与混合检索的前提）**

```bash
bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000 qwen-vl 1     # GPU1，等到 /v1/models 有响应
python -m mmsearch.data.ocr --datasets vidore_arxivqa vidore_docvqa vidore_infovqa vidore_tabfquad \
  vidore_tatdqa vidore_shift vidore_synth_ai vidore_synth_energy vidore_synth_gov vidore_synth_health
```
写 `<dataset>/ocr.jsonl`，`load_dataset` 会自动把它当作 doc text。约 7000 页 × ~1.5 s / 64 并发 ≈ 10–20 分钟。
**没有 OCR 也能跑**：`bm25-ocr` 在配置里是 `optional: true`，缺文本会抛错→被跳过，依赖它的 hybrid 行也会被自动跳过（日志里有 warning），不会中断整个 benchmark。

**2.3 ViDoRe 评测（第一次真正的成绩单）**

```bash
python -m mmsearch.eval.run_benchmark --config configs/vidore_benchmark.yaml
cat $(ls -td storage/runs/benchmark/vidore_main_* | head -1)/results.md
```
表里有 4–6 行：`colqwen2-v1.0(ref)`（官方权重，用来验证你的评测流水线）→ `colqwen2-repro`（你的复现）→ `clip-L14(dense)` → `bm25-ocr` → 两个 hybrid。

**先看 ref 行**：官方 `vidore/colqwen2-v1.0-hf` 在 ViDoRe v1 上 avg nDCG@5 ≈ 89（以 model card / leaderboard 为准）。**ref 行对得上，才说明你的预处理、padding、MaxSim、qrels 全对**，此时 repro 行的差距才是训练问题。
repro 差的排查顺序：① 图像分辨率（`max_pixels: 768*28*28`，与参考一致）② query augmentation（10 个 `<|endoftext|>`）③ doc prompt（"Describe the image."）④ 学习率（论文 5e-5；我们默认 2e-4，先在 dev 上扫）⑤ LoRA 是否漏训 projection（`proj.pt` 是随机初始化，必须训）。

**2.4 索引与压缩消融（E3/E4）**

```bash
python -m mmsearch.eval.run_benchmark --config configs/index_ablation.yaml
latest=$(ls -td storage/runs/benchmark/index_ablation_* | head -1)
python -m mmsearch.eval.plots tradeoff --results "$latest/results.json" --metric ndcg@5
python -m mmsearch.eval.badcase --run-dir <vidore_main run> --method "colqwen2-v1.0(ref)" --compare bm25-ocr
```
17 个配置全部复用同一份 doc 嵌入缓存，所以主要开销是编码（第一次 ~20–40 分钟）+ 每个配置的压缩/建索引（kmeans 最慢）。
**注意**：1k 页的子集上延迟差异被启动开销淹没。要支撑“层次索引降 P99”的结论，必须上 `vidore_pool`（2 万+ 页）：
```bash
python -m mmsearch.data.preprocess pool --name vidore_pool \
  --sources vidore_arxivqa vidore_docvqa vidore_infovqa vidore_tabfquad vidore_tatdqa vidore_shift \
            vidore_synth_ai vidore_synth_energy vidore_synth_gov vidore_synth_health \
            colpali_train_pool \
  --queries-from vidore_docvqa vidore_tatdqa vidore_synth_gov
python -m mmsearch.eval.run_benchmark --config configs/index_ablation.yaml datasets=[vidore_pool]
```

**验收**：ref 行与公开数字差 ≤ 2 点（证明流水线可信）；repro 与 ref 差 ≤ 2–3 点（证明复现成功）；压缩/层次索引至少有 1 个配置在 nDCG 掉 ≤ 1 点内把体积或 P99 砍 40%+。

**卡点**
- `--dataset vidore_all` 展开成 10 个子集，每个都重新评测；单子集调试用 `datasets=[vidore_docvqa]`。
- 显存：确保没同时开着 vLLM（GPU1 被占满时 `max_pixels` 编码会 OOM）。
- `auto` CLI 覆盖里 list 要用 YAML 内联语法：`datasets=[vidore_docvqa,vidore_tatdqa]`。

---

### 阶段 3｜多模态 RAG（第 7–8 周）

```bash
bash scripts/download_data.sh stage3                 # docvqa_validation / infovqa_validation / chartqa_test
bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000 qwen-vl 1

python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml                          # k=1,2,3,5,8 + oracle + 闭卷
python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml budget.strategy=rank_decay name=rag_docvqa_rankdecay
python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml dataset=chartqa_test max_queries=1000 name=rag_chartqa
python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml rerank.top_n=10 name=rag_docvqa_rerank   # VLM 重排
```
产物：`storage/runs/rag/<run>/{report.md, results.json, k_sensitivity.png, attribution_k3.png, per_query.jsonl}`。

单次规模：1000 题 × 7 条件 = 7000 次生成，32 并发、每次 1–3 s ⇒ 15–30 分钟（含图像编码）。
**先跑 100 题（`max_queries=100`）确认 prompt 与答案抽取正确**，再跑全量 —— 答案抽取错了会浪费一整轮。

**验收**：`oracle ≈ 满分`（否则是 VLM 读图/答案抽取问题，不是检索问题）；`k` 曲线有最优点（E5）；归因里 retrieval-attributable 与 generation-side 各占一定比例，能写进报告。

**读归因的姿势**：`context_distraction` 高 → 少给页数或换 `rank_decay` 预算；`generation_error` 高 → 换更强 VLM 或提高分辨率（`budget.total_tokens`）；`retrieval_miss` 高 → 回阶段 2。
**注意**：ViDoRe 的 docvqa/infovqa **test** 没有人能看到的答案，`run_rag_eval` 会拒绝没有 `answers` 的数据集 —— 所以这里用的是 `lmms-lab/DocVQA` 的 validation 划分。

---

### 阶段 4｜Query 理解（第 9–10 周）

```bash
bash scripts/download_data.sh stage4                  # colpali_train_pool（2 万页）
bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000 qwen-vl 1        # teacher

# 1) 多语言评测集（ViDoRe 查询翻成中/日，语料不变）
python -m mmsearch.query.translate --datasets vidore_all --langs zh ja

# 2) 合成 SFT 数据：teacher 只看 query（不看金页，防答案泄漏），N 个候选 → 检索奖励筛选
python -m mmsearch.query.synth build --config configs/query_rewrite.yaml
cat $(ls -td storage/runs/query/qrewrite_* | head -1)/synth_stats.json

# 3) 人工校验 1000 条（plan 要求可用率 ≥ 70%）
#    编辑 <run>/human_check.csv 的 valid(1/0) 与 error_type 两列，然后：
python -m mmsearch.query.synth validate --file <run>/human_check.csv

# 4) LoRA SFT（会占 GPU0；若 teacher 还在 GPU1 上不受影响）
torchrun --nproc_per_node 2 -m mmsearch.train.train_qlora --config configs/query_rewrite.yaml

# 5) 下游评估：起两个 rewriter（零样本 :8000 / 微调 :8002），比较 original / replace / concat / entities / 等长随机扩写
GPU_UTIL=0.45 bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000 qwen-vl 1
GPU_UTIL=0.45 bash scripts/serve_vlm.sh "$MMS_STORAGE/runs/query/qrewrite_sft/merged" 8002 rewriter 1
python -m mmsearch.query.run_rewrite_eval --config configs/query_rewrite.yaml
```

产物：`synth_stats.json`（保留/丢弃/改进统计）、`sft_*.jsonl`、`qrewrite_sft/{adapter,merged}`、`report.md`（含 Δ 与配对置换检验 p 值）。

**验收**：改写后 R@10 ≥ +2 点，且 `concat ≫ rand_concat`（等长随机扩写几乎不涨）—— 证明收益来自语义补全而非查询变长。若 `rand_concat` 也涨 2 点，说明你的收益全部来自长度，必须重新设计。

**卡点**
- `synth_stats.json` 里 `dropped_worse` 占比过高（>50%）→ teacher 太弱或 prompt 太保守，换 `Qwen2.5-72B` 级别的模型做 teacher。
- 可用率 < 70% → 按 `human_check.csv` 的 error_type 分布改 prompt（最典型：`meaning_changed`，即改写引入了原查询没有的约束）。
- SFT 后模型只会输出 JSON 而忘了语言标注 → 检查 `lang` 标签是否来自真实来源（代码里用 ground-truth 覆盖 teacher 判定）。

---

### 阶段 5｜评估体系与工程（第 11–12 周）

```bash
# LLM-as-judge 与人工一致性
run=$(ls -td storage/runs/benchmark/vidore_main_* | head -1)
python -m mmsearch.eval.llm_judge judge --run-dir "$run" --method "colqwen2-v1.0(ref)" --n-queries 200
#   标注 $run/judge/human_annotation.csv 的 human_label 列（0/1/2）后：
python -m mmsearch.eval.llm_judge agree --run-dir "$run"      # kappa / quadratic kappa / 混淆矩阵

# 在线服务 + 压测
MMS_SERVICE_CONFIG=configs/service.yaml CUDA_VISIBLE_DEVICES=0 uvicorn mmsearch.service.app:app --port 8080
curl -s localhost:8080/health
curl -s localhost:8080/search -H 'content-type: application/json' \
  -d '{"query":"total revenue in 2023","k":5,"mode":"hybrid"}' | python -m json.tool | head -40
python -m mmsearch.service.loadtest --url http://127.0.0.1:8080 --dataset vidore_docvqa \
  --concurrency 1 2 4 8 16 32 64 --duration 30 --mode multivector     # 出 qps_latency.png

# 一条命令复现全部评测
python -m mmsearch.eval.run_benchmark --config configs/benchmark_all.yaml
```
服务里 `/search` 的 `timings` 会分离 `encode_ms` / `search_ms` / `rerank_ms` / `total_ms` —— 压测时先看哪个环节先饱和（通常 query 编码先饱和，micro-batching 就是为了压它）。

**验收**：kappa ≥ 0.6（3 级）；QPS-P99 曲线单调可解释；`benchmark_all.yaml` 一条命令能复跑全部表格。

---

## 3. 学习路线（与阶段对齐，每周 6–8 h 阅读 + 实验）

每周配一份产出：**一张表 + 一段结论 + 一个能讲 3 分钟的坑**。做不到这三样，说明这周只跑了代码没学到东西。

### 第 1 周｜CLIP 与对比学习（对应阶段 0）

- 读：CLIP 论文（§2 双塔结构、§3.1 对比目标）、SigLIP（sigmoid 损失为什么小 batch 更稳）。
- 手算：`ViT-L/14 @224` → 14×14=196 patch + 1 CLS = 197 token；`(196+1)×1024` 的视觉 token 预算。
- 自问：① 为什么 CLIP 用对称 InfoNCE 而不是单向？② 温度系数 τ 在训练中怎么变（`logit_scale` clamp 到 100）？③ 模态鸿沟为什么存在、它对你后面的多向量检索意味着什么？
- 实验：`modality_gap` 的 cos 分布图里，matched 与 random 的分布重叠多少？这个重叠就是双塔的上限。

### 第 2–3 周｜难负样本与消融方法论（阶段 1）

- 读：ANCE（为什么用当前模型挖、为什么要刷新）、VSE++（max-violation triplet）、"False negatives in COCO"（为什么跳 top-3）。
- 自问：① 难负样本为什么容易变成假负例？② in-batch 负样本数量与 batch size 的关系（global batch 1024 ⇒ 每行 1023 个负样本）？③ 为什么 `hard only`（不加 in-batch）通常更差？
- 产出：消融阶梯表（inbatch→random→hard→mixed），每格只改一个变量。

### 第 4–6 周｜Late interaction 与多向量索引（阶段 2，最重要）

- 读：ColBERT（MaxSim 定义，`s(q,d)=Σ_i max_j q_i·d_j`）、ColBERTv2（残差压缩）、**PLAID**（三阶段：质心候选→质心交互打分→全精度精排）、**ColPali**（VLM patch 直接当文档 token，为什么不需要 OCR）。
- 手算：一页 768 token × 128 维 × 2 字节 = 197 KB；10 万页 = 19 GB —— 这就是为什么必须压缩/建索引，也是面试第一个问题。
- 自问：① MaxSim 为什么天然容忍“答案只占页面一小块”？② 质心交互（centroid interaction）为什么能近似 MaxSim，误差来源是什么？③ 层次索引的粗向量用 k-means 池化，和 PLAID 的质心码本有什么本质区别（一个降低候选数，一个降低单页 token 数）？④ `nprobe` 与 `n_rerank` 各自的 recall/延迟权衡。
- 产出：三条曲线（recall vs 体积、recall vs P99、Pareto 前沿）+ 一句结论：“在 nDCG 掉 0.8 点的前提下 P99 从 X 降到 Y”。

### 第 7–8 周｜视觉 RAG 与归因（阶段 3）

- 读：Qwen2-VL（动态分辨率的 token 计算 `(H/28)×(W/28)`）、ColPali 的 RAG 实验、DocVLM/视觉 RAG 综述。
- 自问：① 一张 A4 页面在 `max_pixels=768×28²` 下是多少 token？② 为什么 k 大会降准确率（context distraction 的机制）？③ 视觉 token 预算和 top-k 是两个独立的旋钮，先调哪个？
- 产出：k 敏感性曲线 + 归因饼图 + 一句“检索侧贡献 A%、生成侧 B%”。

### 第 9–10 周｜Query 理解（阶段 4）

- 读：query rewriting / pseudo-relevance feedback 经典方法、LLM 合成数据的质量校验实践。
- 自问：① 为什么不能用人工/LLM 打分直接评改写质量（bias 与不可比）？② 等长随机扩写对照为什么能证明“收益来自语义”？③ 多语言改写对跨语言检索的收益来自翻译还是来自归一化到英文？
- 产出：Δ 表 + 显著性 + 失败案例 3 个。

### 第 11–12 周｜评估体系与工程（阶段 5）

- 读：pytrec_eval 的指标定义、LLM-as-judge 一致性（kappa 与 quadratic weighted kappa 的差别）、ANN-Benchmarks 的 recall-latency 报告方式。
- 自问：① nDCG 的线性增益 vs 指数增益对“分级相关性”有什么影响？② 为什么 nDCG@5 是这个项目的主动指标（而不是 R@10）？③ 压测时 QPS 上不去，先看 GPU 利用率还是 CPU（preprocess）还是 micro-batching？
- 产出：一条命令复现 + 报告 + 简历段落。

### 参考清单（按重要度）

| 主题 | 文献/代码 | 必读章节 |
|---|---|---|
| 文档视觉检索 | ColPali (2407.01449) / `illuin-tech/colpali` | §2 架构、§4 实验、§5 存储分析 |
| 多向量 ANN | PLAID (2205.09707) / `stanford-futuredata/ColBERT` | §3 三阶段检索 |
| late interaction | ColBERT (2004.12832) | §3 MaxSim |
| 对比学习 | SigLIP (2303.15343) | §2 sigmoid loss |
| 双塔 | CLIP (2103.00020) | §2.1–2.3 |
| VLM 底座 | Qwen2-VL (2409.12191) | §2 动态分辨率 |
| 多模态 embedding 评测 | MMEB (2407.08516) | §3 任务分类 |
| 检索评测 | BEIR / pytrec_eval 文档 | 指标定义 |

---

## 4. 面试讲法映射（每个实验一句话）

| 实验 | 一句话结论模板 |
|---|---|
| E1 负样本 | “难负样本把 t2i R@1 从 A 提到 B；随机负样本（同数量）只到 C，说明收益来自**难度**而非**数量**。” |
| E2 loss | “batch 1024 下 SigLIP 与 InfoNCE 持平，但 SigLIP 在小 batch（256）默认更稳；triplet 收敛快但最终低 X 点。” |
| E3 压缩 | “kmeans64 + int8 把索引从 19 GB 压到 2.3 GB，nDCG@5 只掉 0.9 点 —— 这是我们的工作点。” |
| E4 层次索引 | “页级粗筛 + 块级精排，在 2 万页上 P99 从 X ms 降到 Y ms，nDCG 不掉；本质是用一层额外打分换取候选数减少一个量级。” |
| E5 上下文 | “k 的最优点在 3，超过之后 context distraction 使准确率下降；提高单页分辨率比增大 k 更划算。” |
| E6 改写 | “改写后在 R@10 上 +X 点，而等长随机扩写只有 +0.3，p<0.01 —— 收益来自语义补全。” |
| 评估体系 | “LLM-as-judge 与人工 3 级标注 kappa=Z；并用 judge 发现 qrels 漏标 N 条。” |

---

## 5. 排障手册

| 症状 | 原因 | 处理 |
|---|---|---|
| 训练很慢、`nvidia-smi` 里 GPU 利用率 0 | 数据管道喂不上：DDP + 小 `/dev/shm` 时旧版回退会变成 `num_workers=0`（串行解码 512 张 JPEG） | 已修（`sampler` 也走线程池）。看训练日志每 10 步的 `data-wait`：>40% 说明解码仍是瓶颈 → 提高 `MMS_NUM_WORKERS`（线程数，按 `CPU 核数/卡数` 给）；<15% 说明 GPU 才是瓶颈，可以加大 `per_device_batch` |
| `RuntimeError: unable to allocate shared memory(shm) for file </torch_...>` | 容器 `/dev/shm` 太小（Docker 默认 64 MB），DataLoader 工作进程靠共享内存传 batch | 代码已修：推理走线程池（不用 shm），训练在 shm 不足时自动降级为线程池。彻底解决：`docker run --shm-size=8g`。自查：`python scripts/check_env.py` 的 `/dev/shm` 行；临时指定：`export MMS_NUM_WORKERS=0`（串行）或 `8` |
| `modelscope unreachable` / 下载卡住 | 代理、内网 DNS | `curl -I https://www.modelscope.cn`；重跑命令自动续传；改 `MMS_HUB_ORDER=hf` + `HF_ENDPOINT=https://hf-mirror.com` 兜底 |
| `Could not find ... colqwen2` 或 `ColQwen2ForRetrieval` ImportError | transformers < 4.53 | 该机为 5.17，正常；若降级过，`pip install -U transformers`（别动 torch） |
| CLIP 零样本 R@1 只有 30% | mean/std、resize 方式、忘了归一化 | 用 `storage/models/.../preprocessor_config.json` 的 mean/std（代码已读）；`Resize(短边)+CenterCrop` |
| DDP 卡在 `init_process_group` / NCCL timeout | 卡号被占用、P2P 不通 | `export NCCL_P2P_DISABLE=1` 试一次；`CUDA_VISIBLE_DEVICES` 只给要用的两张卡 |
| `Expected to have finished reduction` | 有参数未参与 loss | 换 loss 时保持 `logit_scale` 冻结/参与一致（见 `ClipTrainModule`） |
| CUDA OOM（训练） | batch 太大 | 降 `per_device_batch`；确认 `grad_ckpt: true`；ColPali 降 `max_pixels` 到最后手段 |
| CUDA OOM（推理/评测） | GPU1 上还挂着 vLLM | `nvidia-smi` 看进程；停 vLLM 或用 `CUDA_VISIBLE_DEVICES=0` 隔离 |
| vLLM 起不来：端口占用 / `Address already in use` | 旧进程 | `lsof -i:8000` 或 `kill $(cat $MMS_STORAGE/vllm_8000.pid)` |
| vLLM `limit-mm-per-prompt` 报错 | 一次给的图超过上限 | `scripts/serve_vlm.sh` 里调大 `"image": 12`；同时受 `--max-model-len` 约束 |
| RAG 输出被截断/答非所问 | prompt 太松、max_new_tokens 太小 | `generator.max_new_tokens=64` 够短答案；确认 system prompt 未被覆盖 |
| `bm25-ocr` 被跳过 | 没有 `ocr.jsonl` | 跑 `mmsearch.data.ocr`；或直接忽略（hybrid 行会一起跳过） |
| `benchmark` 里 `colqwen2-repro` 被跳过 | 训练还没产出 checkpoint | 正常（`optional: true`）；训练完把路径对上或去掉 `optional` |
| `faiss` ImportError | 未安装 | `pip install --no-deps faiss-cpu`；或只用 PLAID-lite |
| `pytrec_eval` 编译失败 | 缺 gcc | 可跳过（我们自实现指标，单测已交叉验证） |
| 图表中文变方框 | 无中文字体 | 图里标签本就是英文；要中文则装 `fonts-noto-cjk` 并设 `rcParams['font.sans-serif']` |
| 磁盘爆 | raw+processed+cache 重复 | `du -sh storage/*`；`storage/raw` 里 `clip-benchmark` 的 tar 与 `processed/` 的 jpg 是二份，够用后可删 raw |
| 评测数字每次都不一样 | 没固定种子 / 索引非确定性 | `seed` 已在 config；PLAID 的 kmeans 有 seed；`torch.use_deterministic_algorithms` 一般不需要（排序稳定） |

---

## 6. 每周检查清单（贴显示器）

- [ ] 本周实验有假设、有变量、有结论（一句话写进 `REPORT.md`）
- [ ] 结果已进 `storage/runs/<run>/results.md` + 报告表格，注明 run 目录与 commit
- [ ] 代码已 commit（分支/提交信息写清阶段与变量）
- [ ] 至少记录 1 个 badcase 及归因（`badcases_*.md`）
- [ ] `test_ledger.jsonl` 没有异常增长（同一个 test 集别反复调参）
- [ ] 能用 3 句话向非同行讲清本周做了什么

---

## 7. 产物索引（写报告时直接查）

| 报告里的表/图 | 来源 |
|---|---|
| CLIP 零样本/微调 R@1/R@5/rsum | `storage/runs/clip_eval/*/results.md` |
| 模态鸿沟统计 | `storage/runs/analysis/modality_gap_*/gap.json` |
| loss / 负样本消融 | `storage/runs/clip/<name>/summary.json` + `clip_eval/*` |
| ViDoRe 方法×子集表 | `storage/runs/benchmark/vidore_main_*/results.md`（含 `avg` 列） |
| 压缩/索引帕累托 | `storage/runs/benchmark/index_ablation_*/results.md` + `tradeoff_ndcg5.png` |
| Badcase / 失败聚类 | `.../badcases_<method>_vs_<compare>.md` |
| RAG 条件准确率、k 曲线、归因 | `storage/runs/rag/<run>/{report.md,results.json,k_sensitivity.png,attribution_k3.png}` |
| 改写收益与显著性 | `storage/runs/query/qrewrite_eval_*/report.md` |
| LLM-judge 一致性 | `<benchmark run>/judge/agreement.json` |
| QPS–P99 | `storage/runs/loadtest/*/{loadtest.md,qps_latency.png}` |
