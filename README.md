# mmsearch — 面向 LLM 的多模态检索系统

复现并改进 **ColPali 式文档视觉检索**（ColQwen2），扩展为完整的多模态检索系统：
query 理解 → 多路召回（BM25 / CLIP 双塔 / 多向量 late interaction）→ 索引筛选与压缩 → VLM 重排 → 多图 RAG 生成，
并配套可量化的评估体系（Recall/nDCG/MRR + 延迟/体积/QPS + badcase 归因 + LLM-as-judge 一致性）。

对应执行计划：[多模态检索项目执行计划.md](多模态检索项目执行计划.md)。实验报告模板：[docs/REPORT.md](docs/REPORT.md)。
**GPU 机器上怎么跑、每个阶段学什么：[docs/RUNBOOK.md](docs/RUNBOOK.md)（执行与学习手册）**。

## 1. 架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│ service   FastAPI /search /rag /page  · query 编码 micro-batching · 压测       │  mmsearch/service
├──────────────────────────────────────────────────────────────────────────┤
│ eval      metrics · run_benchmark(配置驱动) · eval_clip · badcase · llm_judge │  mmsearch/eval
│           · plots(帕累托) · ledger(测试集使用审计) · modality_gap              │
├──────────────────────────────────────────────────────────────────────────┤
│ rag/query RAG(视觉 token 预算) · 失败归因 · k 敏感性 │ 改写/翻译/合成/下游评估   │  mmsearch/rag, query
├──────────────────────────────────────────────────────────────────────────┤
│ retrieve  sparse(BM25,OCR 文本) · dense(CLIP) · multivector(ColQwen2+MaxSim) │  mmsearch/retrieve
│           · fusion(RRF/加权/zscore) · rerank(VLM yes-prob)                    │
├──────────────────────────────────────────────────────────────────────────┤
│ index     MultiVectorStore(CSR) · Exhaustive · PLAID-lite · Hierarchical     │  mmsearch/index
│           · FAISS token ANN · 压缩(kmeans/pca/prune/int8, 可组合)            │
├──────────────────────────────────────────────────────────────────────────┤
│ data      统一格式(corpus/queries/qrels) · 预处理 · VLM-OCR · 难负样本挖掘     │  mmsearch/data
├──────────────────────────────────────────────────────────────────────────┤
│ model     CLIPWrapper · ColQwen2Encoder(hf 参考 / custom 复现) · losses       │  mmsearch/models
└──────────────────────────────────────────────────────────────────────────┘
 utils: ModelScope 优先的 hub 解析 · YAML 配置(_base_/覆盖) · DDP 工具 · 实验追踪(JSONL/wandb)
 llm:   vLLM OpenAI 兼容客户端(并发/重试) · Qwen2-VL 视觉 token 预算
```

换 backbone 只动 `models/`，换索引只动 `index/`；所有实验通过 `configs/*.yaml` 驱动，产物落在 `storage/runs/`。

## 2. 环境（2×H20，已预装 torch 2.10 / transformers 5.17 / vLLM 0.19）

```bash
bash scripts/setup_env.sh          # 只补装缺失的: peft matplotlib pytest pytrec-eval faiss-cpu；pip install -e . --no-deps
torchrun --nproc_per_node 2 scripts/check_dist.py   # 阶段 0 验收：双卡 all-reduce + NVLink 带宽
```

**HuggingFace 不通 → 模型与数据集一律 ModelScope 优先**（`mmsearch/utils/hub.py`，`MMS_HUB_ORDER=modelscope,hf`，HF 走 `HF_ENDPOINT` 镜像兜底）。已核实 ModelScope 上可用：

| 用途 | ID |
|---|---|
| CLIP | `AI-ModelScope/clip-vit-large-patch14`（代码里写 `openai/clip-vit-large-patch14`，自动映射） |
| ColQwen2 参考模型 | `vidore/colqwen2-v1.0-hf`（transformers 原生 `ColQwen2ForRetrieval`） |
| ColQwen2 复现底座 | `Qwen/Qwen2-VL-2B-Instruct` |
| 生成 / 重排 / OCR / 改写 | `Qwen/Qwen2.5-VL-7B-Instruct`（3B 作在线消融） |
| COCO / Flickr30K（Karpathy 划分） | `clip-benchmark/wds_mscoco_captions`, `clip-benchmark/wds_flickr30k` |
| ViDoRe v1（10 子集） | `vidore/{arxivqa,docvqa,infovqa,tabfquad}_test_subsampled`, `vidore/tatdqa_test`, `vidore/shiftproject_test`, `vidore/syntheticDocQA_*_test` |
| 文档检索训练集 | `vidore/colpali_train_set`（ColPali 论文 118k 训练对，含 DocMatix 部分） |
| RAG 评测 | `lmms-lab/DocVQA`（DocVQA + InfographicVQA，validation 有答案）, `HuggingFaceM4/ChartQA` |

存储根目录 `MMS_STORAGE`（默认 `./storage`，建议软链到大盘）。

## 3. 按阶段运行

`bash scripts/run_all.sh stage0..stage5` 依次执行下表；每一步也可单独跑。GPU 分工：GPU0 训练/编码，GPU1 vLLM。

| 阶段 | 关键命令 | 产出 |
|---|---|---|
| 0 基建 | `bash scripts/download_data.sh week1`；`python -m mmsearch.eval.eval_clip --dataset coco_test --folds 1 5`；`python -m mmsearch.eval.modality_gap` | CLIP zero-shot 基线表（COCO 5K/1K、Flickr 1K），模态鸿沟统计图 |
| 1 CLIP 微调 | `torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml loss=siglip` / `negatives=mixed` | E2 loss 消融、E1 负样本阶梯（inbatch→random→hard→mixed） |
| 2 ★ 文档视觉检索 | `torchrun ... -m mmsearch.train.train_colpali --config configs/colpali_vidore.yaml`；`python -m mmsearch.data.ocr ...`；`python -m mmsearch.eval.run_benchmark --config configs/vidore_benchmark.yaml`；`... configs/index_ablation.yaml`；`python -m mmsearch.eval.plots tradeoff ...` | ViDoRe 方法×子集表、E3 压缩 / E4 层次索引帕累托图、badcase 报告 |
| 3 多模态 RAG | `bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000`；`python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml` | 条件准确率（k / oracle / 闭卷）、k 敏感性曲线（E5）、失败归因饼图 |
| 4 Query 理解 | `python -m mmsearch.query.synth build ...`；`torchrun ... -m mmsearch.train.train_qlora ...`；`python -m mmsearch.query.run_rewrite_eval ...` | 合成数据统计 + 人工可用率、E6 下游检索收益（含等长随机扩写对照与显著性） |
| 5 评估体系/工程 | `python -m mmsearch.eval.run_benchmark --config configs/benchmark_all.yaml`；`python -m mmsearch.eval.llm_judge judge/agree`；`uvicorn mmsearch.service.app:app`；`python -m mmsearch.service.loadtest` | 一条命令复现全部评测、Cohen's kappa、QPS–P99 曲线 |

## 4. 关键设计

- **统一数据格式**（`data/schema.py`）：`corpus.jsonl / queries.jsonl / qrels.tsv`，与 BEIR/pytrec_eval 同构；派生数据集（翻译 query）用 `corpus_from` 共享语料与嵌入缓存。`RetrievalSample` 即计划 §3.2 的样本结构。
- **评测协议定死**：`eval/metrics.py` 自实现（与 pytrec_eval 同约定，单测交叉校验）；每次在 test split 上评估都写入 `storage/runs/test_ledger.jsonl`，模型选择只用 dev（CLIP 从 train 留出 5k 图，ColQwen2 用训练集自带 test 切分）。
- **多向量索引**：`MultiVectorStore` 以 CSR 存全部 token 向量；`ExhaustiveIndex` 按长度分桶 padding 做精确 MaxSim；`PLAIDLiteIndex` = 质心码本 + 倒排 → 质心交互近似分 → top-N 精排；`HierarchicalIndex` = 每页 k-means 粗向量先粗筛再精排（计划 Week 6 “层次化多向量”）。压缩器（kmeans/pca/prune/int8）可链式组合，query 侧自动做同样的投影。
- **难负样本（ANCE）**：每个 epoch 用当前模型重挖，跳过 top-3 降低 COCO 假负例，并屏蔽与正样本重合的负样本；`random` 负样本作对照，区分“更难”与“更多”。
- **ColQwen2 复现**：Qwen2-VL-2B + LoRA(LM only) + 128 维投影 + query augmentation tokens，跨卡 gather 文档做 in-batch 负样本，pairwise / InfoNCE 两种 loss；`hf` 后端加载官方 `colqwen2-v1.0-hf` 作为评测流水线的对齐基准。
- **RAG 归因**：每个 query 落入唯一桶：`retrieval_miss`（oracle 能答对、检索没命中）/ `context_distraction`（命中但多页干扰，oracle 能答对）/ `generation_error`（给了正确页仍答错）/ `both_fail` / `correct_without_hit`。比计划里的三分法多拆出“上下文干扰”，直接指导上下文组织策略。
- **改写评估**：不直接评改写质量，而是用下游检索指标反推；训练数据用“检索奖励拒绝采样”（teacher 只看 query、不看金页，防泄漏），评测含 replace/concat/entities 三种组合方式与**等长随机扩写**对照，并做配对置换检验。

## 5. 与执行计划的偏差（有意为之）

| 计划 | 实际 | 原因 |
|---|---|---|
| torch 2.4 / transformers 4.45 / Python 3.10 | 沿用服务器 torch 2.10 / transformers 5.17 / Python 3.11 | 已预装且更新；ColQwen2 需要 transformers ≥ 4.53 |
| DocMatix 流式子集 | `vidore/colpali_train_set` 前 35 个分片（≈5 万对） | ColPali 论文的原始训练集（已含 DocMatix 部分），ModelScope 可直接下载，复现可比性更好 |
| PaliGemma-3B / Qwen2-VL-2B 二选一 | Qwen2-VL-2B（ColQwen2） | 更现代、ViDoRe 上更强、与 vLLM 生态一致 |
| RAG 生成 Qwen2-VL-7B | Qwen2.5-VL-7B | 同尺寸更强，vLLM 原生支持 |
| RAG 评测在 ViDoRe 上 | DocVQA/InfoVQA validation、ChartQA test | ViDoRe 的 docvqa/infovqa test 答案不公开 |
| faiss-gpu 多向量索引 | 自研 PLAID-lite（torch GPU）为主，FAISS 为可选对照 | 无需额外 CUDA 依赖；更贴近 ColBERTv2/PLAID 的真实做法，可讲性更强 |
| wandb 必须 | 默认本地 JSONL（`metrics.jsonl/summary.json/config.yaml/git commit`），`MMS_WANDB=1` 时同步 wandb | 服务器外网受限 |

## 6. 测试

```bash
pytest -q        # CPU 即可：指标、损失、MaxSim、各索引与压缩器的正确性、BM25/融合、RAG 指标、基准流水线端到端
```
