# 实验报告（模板，随实验逐步填写）

> 规则：每个实验只改一个变量；固定种子；test split 只跑一次（见 `storage/runs/test_ledger.jsonl`）。
> 表格数字直接从 `storage/runs/**/results.md` 拷贝，注明 run 目录与 commit。

## 0. 基建与基线（Week 1）

| 模型 | 数据 | 协议 | i2t R@1 | i2t R@5 | t2i R@1 | t2i R@5 | rsum |
|---|---|---|---|---|---|---|---|
| CLIP ViT-L/14 zero-shot | COCO | 5K | | | | | |
| CLIP ViT-L/14 zero-shot | COCO | 1K×5 | | | | | |
| CLIP ViT-L/14 zero-shot | Flickr30K | 1K | | | | | |

验收：COCO 5K t2i/i2t R@1 落在论文区间（i2t≈56–58，t2i≈36–37）；否则先查 resize/normalize。
模态鸿沟：质心距离 = __，去鸿沟后检索变化 = __（`runs/analysis/modality_gap_*`）。

## 1. 图文检索微调（Week 2–3）

**E2 loss**（negatives=inbatch，global batch 1024）

| loss | COCO 1K t2i R@1 | i2t R@1 | rsum | 峰值显存 | epoch 耗时 | 训练稳定性备注 |
|---|---|---|---|---|---|---|
| InfoNCE | | | | | | |
| SigLIP | | | | | | |
| Triplet (VSE++) | | | | | | |

**E1 负样本阶梯**（loss=InfoNCE）

| 负样本 | t2i R@1 | Δ vs inbatch | 挖掘耗时/epoch |
|---|---|---|---|
| in-batch | | – | – |
| + random（对照） | | | |
| hard only | | | |
| in-batch + hard（mixed） | | | |

结论：__（验收：R@1 相比 baseline ≥ +3）

## 2. 文档视觉检索（Week 4–6）

**ViDoRe v1 nDCG@5**（`configs/vidore_benchmark.yaml`）

| method | arxivqa | docvqa | infovqa | tabfquad | tatdqa | shift | ai | energy | gov | health | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| colqwen2-v1.0 (官方, 对齐评测流水线) | | | | | | | | | | | |
| colqwen2-repro (ours) | | | | | | | | | | | |
| CLIP-L/14 dense | | | | | | | | | | | |
| BM25 (VLM-OCR) | | | | | | | | | | | |
| hybrid RRF | | | | | | | | | | | |

验收：repro 与同数据训练的参考差距 ≤ 2 点；若不达标，写“复现偏差分析”（分辨率 / prompt / batch / lr）。

**E3 压缩 / E4 层次索引**（`configs/index_ablation.yaml` + `plots tradeoff`）

| method | nDCG@5 | R@5 | 体积 MB | search P50 ms | search P99 ms |
|---|---|---|---|---|---|
| exact fp16 | | | | | |
| int8 | | | | | |
| pca64 / pca32 | | | | | |
| kmeans64 / 32 / 16 | | | | | |
| prune salience 50% / 25%（random 25% 对照） | | | | | |
| kmeans64+pca64+int8 | | | | | |
| PLAID-lite p2 / p4 / approx | | | | | |
| hierarchical k8 n50 / n100 | | | | | |

![tradeoff](../storage/runs/benchmark/<index_ablation_run>/tradeoff_ndcg5.png)

Badcase（`badcases_*.md`）：失败桶分布 __；主要失败模式 __；与 BM25 互补性 __。

## 3. 多模态 RAG（Week 7–8）

| 条件 | DocVQA ANLS | ChartQA relaxed acc |
|---|---|---|
| closed book | | |
| k=1 / 2 / 3 / 5 / 8 | | |
| oracle page | | |

归因 @k=3：retrieval_miss __% / context_distraction __% / generation_error __% / both_fail __%（占错误）。
上下文组织改进（uniform → rank_decay 预算 / + VLM rerank）：端到端 +__。

## 4. Query 理解（Week 9–10）

合成数据：候选 __ 条 → 保留 __（改进 __ / easy __ / 丢弃更差 __）；人工校验 1000 条可用率 __%（阈值 70%）。

| lang | condition | R@10 | Δ | p | 平均词数 |
|---|---|---|---|---|---|
| en | original | | – | – | |
| en | sft:concat | | | | |
| en | sft:rand_concat（等长对照） | | | | |
| zh / ja | original → sft:replace | | | | |

结论：收益来自语义补全而非变长 ⇔ concat ≫ rand_concat（且显著）。

## 5. 评估体系与工程（Week 11–12）

LLM-as-judge vs 人工：n=__，kappa(3 级)=__，quadratic kappa=__，binary kappa=__；judge 判 2 分但不在 qrels 的疑似漏标 __ 条。

压测（PLAID-lite, ColQwen2 bf16 vs int8）：

| 并发 | QPS | P50 | P99 |
|---|---|---|---|

## 6. 反思
- 如果重做：先建评估集与 badcase 标注规范，再训练模型。
