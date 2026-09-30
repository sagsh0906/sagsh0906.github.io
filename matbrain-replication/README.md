# MatBrain 复现（Mat-R1 + Mat-T1 + Mat-MCP）

独立复现：Shi *et al.*, "A collaborative agent with two lightweight synergistic models for autonomous crystal materials research", *Nature Machine Intelligence* **8**, 1441–1456 (2026), doi:[10.1038/s42256-026-01298-6](https://doi.org/10.1038/s42256-026-01298-6)。

> 这不是作者的官方代码。论文公开了代码（Zenodo doi:10.5281/zenodo.21507186）和数据（Mat-252K-SFT doi:10.57967/hf/9652，Mat-20K-RL doi:10.57967/hf/9653）。有条件时请先对照官方实现；本仓库按论文正文和 Methods 从头实现了整套流程，并在下文列出所有论文没写清、需要我们自行决定的地方。

---

## 1. 复现范围一览

| 论文组件 | 本仓库实现 | 状态 |
|---|---|---|
| Mat-MCP 工具生态（检索 / 生成 / 校验 / 相图 / 性质 / 仿真） | `matbrain/mcp/`：31 个工具，Pydantic 参数 schema，MCP server（streamable-http / stdio），Docker + K8s | ✅ 可运行；pymatgen 类工具已在 CPU 上测试，ML/DFT/生成模型后端需按需安装 |
| CIF 句柄传递（artefact handle） | `matbrain/mcp/artifacts.py`（`cif://<id>`） | ✅ 已测试 |
| Mat-T1 复合奖励（式 1–5） | `matbrain/rl/rewards.py` + verl 入口 `verl_reward.py` | ✅ 已测试（与手算值一致） |
| Mat-T1 DAPO 多轮强化学习 | `training/rl/run_mat_t1_dapo.sh`（verl v0.6.1，vLLM async agent loop） | ⚙️ 脚本就绪，需 8×H800（或按小规模脚本缩小） |
| Mat-R1 全参 SFT | `training/sft/run_mat_r1_megatron.sh`（Megatron-SWIFT 3.10，TP2/EP8） | ⚙️ 脚本就绪，需 8×H800 |
| 基座模型选择（初始 LM loss + GPQA/HLE） | `training/sft/base_model_probe.py`、`scripts/plot_base_model_selection.py` | ⚙️ 需 GPU |
| 数据：MP 抓取、DOI 关联、时间切分 | `matbrain/data/mp_fetch.py`、`split.py` | ✅ 切分已测试；抓取需 `MP_API_KEY` |
| 五级泄漏审计 → 1,662 条干净测试集 | `matbrain/data/leakage.py` | ✅ 已测试（五个级别各有用例） |
| MinerU 转换 + 三模型多数投票分类 + LangExtract 抽取 | `matbrain/data/literature.py` | ⚙️ 需要 PDF 与 LLM API |
| generate-distil（V3 出题 → R1 蒸馏 → R1 0–5 打分过滤） | `matbrain/data/generate_distil.py` | ⚙️ 需要 DeepSeek API；`--template-only` 可离线跑 |
| Mat-20K-RL 构建 | `matbrain/data/rl_dataset.py` | ✅ 已离线测试 |
| MatBrain 双模型 LangGraph 状态机 | `matbrain/agent/graph.py`、`executor.py` | ✅ 已测试（含回退、强制作答、max_iterations=6） |
| 泄漏受控基准（分类 / 回归 / 结构设计 7 指标） | `matbrain/data/benchmark.py`、`matbrain/eval/` | ✅ 指标与流程已测试；模型对比需部署模型 |
| 决策树脚本控制器基线 | `matbrain/eval/decision_tree.py` | ✅ |
| 熵分析（轨迹、15 点平滑、分段熵、KDE） | `matbrain/analysis/entropy.py` | ✅ 数值部分已测试；生成需 GPU |
| NRR 案例：MatterGen 生成 3 万结构 → 七步筛选 → V/(V+M) 排序 | `mattergen_generate` 工具 + `matbrain/screening/funnel.py` | ✅ 筛选逻辑已测试；生成与性质计算需 GPU |
| 电化学数据处理（式 6–8） | `matbrain/experiments/nrr.py` | ✅ |
| CoV4S8 合成、XRD、STEM、NRR 电化学测试 | — | ❌ 湿实验，无法用代码复现 |

## 2. 算力与外部资源

| 阶段 | 论文配置 | 最低可行替代 |
|---|---|---|
| Mat-R1 SFT（Qwen3-30B-A3B，全参，2 epoch） | 8×H800 80GB | `run_mat_r1_lora_small.sh`：Qwen3-4B LoRA，单卡 24GB |
| Mat-T1 RL（Qwen3-14B，DAPO，~1,250 step） | 8×H800 80GB | `run_mat_t1_small.sh`：Qwen3-1.7B，2×24GB 或 1×80GB |
| MatBrain 推理 | RTX 4090 工作站（约 1.5 万美元） | `deploy/serve_models.sh`（4×4090；或量化到更少卡） |
| 工具后端 | Docker + K8s | CPU 可跑 pymatgen 类工具；MatGL/CHGNet 建议 GPU；MatterGen、CrystaLLM 需 GPU |
| 外部服务 | MP / OQMD / Bing / SearXNG / DeepSeek / GPT-4o / Claude | `MP_API_KEY`、`DEEPSEEK_API_KEY` 等；Bing Web Search API 已于 2025-08 停用，`web_search` 会自动退回 SearXNG |

## 3. 目录结构

```
matbrain/
  chem.py                 # 晶体学公共函数：CIF 解析/哈希、有效性、电荷平衡、对称性、AFLOW 式原型标签、Vegard、容忍因子
  prompts.py              # Mat-R1 / Mat-T1 / 推理节点提示词
  mcp/                    # Mat-MCP：registry（Pydantic 校验）、artifacts（CIF 句柄）、server、tools/*
  agent/                  # llm 后端、Think-then-Act 解析、Mat-T1 执行器（运行时参数校验层）、LangGraph 图、CLI
  rl/                     # 复合奖励、verl reward 入口、verl 原生工具类
  data/                   # MP 抓取、时间切分、泄漏审计、文献处理、generate-distil、基准构建、RL 数据集
  eval/                   # 指标、决策树基线、基准 runner
  analysis/entropy.py     # 熵诊断（Fig. 4）
  screening/funnel.py     # 七步筛选（Fig. 6b）
  experiments/nrr.py      # NRR 产率 / 法拉第效率
training/sft/             # Megatron-SWIFT 脚本、数据混合、基座 probe
training/rl/              # verl DAPO 脚本、工具配置生成
deploy/                   # Dockerfile、docker-compose、K8s、vLLM 启动脚本
configs/                  # MatBrain 运行、基准系统、数据生成 LLM 配置
scripts/                  # 离线演示、参考条目快照、报告与画图、合成玩具数据
tests/                    # pytest（CPU，无需网络）
```

## 4. 快速开始（CPU、离线）

```bash
cd matbrain-replication
python -m venv .venv && source .venv/bin/activate
pip install -e ".[agent,data,dev]"
pytest -q                          # 45 个测试
python scripts/demo_offline.py     # 用脚本化策略回放 Fig. 5b 的 CsPb(Cl0.2Br0.4I0.4)3 案例，工具调用是真实执行的
```

演示会走完整的 MatBrain 回路：第 1 轮 Mat-T1 用非化学计量式调用 CrystaLLM 被工具拒绝 → Mat-R1 判断为无序混合卤化物问题并给出新指令 → 第 2 轮用 Vegard 定律、VCA 占位建模、结构校验、容忍因子 → Mat-R1 FINISH，句柄被解析为最终 CIF。输出与论文一致：a = 6.025476 Å、V = 218.76 Å³、t = 0.859、μ = 0.587。

> 如果系统 Python 由 Debian 管理，`pip install pymatgen` 可能在编译 `bibtexparser` 时失败（`install_layout` 报错）；用 venv 安装即可。

离线流水线冒烟测试（合成玩具数据，数值没有物理意义）：

```bash
python scripts/make_toy_records.py --out /tmp/toy.jsonl
python -m matbrain.data.split --records /tmp/toy.jsonl --out-dir /tmp/splits --test-size 6 --val-size 4
python -m matbrain.data.leakage --train /tmp/splits/train.jsonl --test /tmp/splits/test.jsonl --out-dir /tmp/audit
python -m matbrain.data.benchmark --clean /tmp/audit/test_clean.jsonl --out /tmp/tasks.jsonl
python -m matbrain.eval.runner --tasks /tmp/tasks.jsonl --mode decision_tree --out-dir /tmp/results
python scripts/report_benchmark.py --dir /tmp/results
```

## 5. 完整复现路线

> 先做小规模验证：一张 80GB 显卡、3–5 天、约 ¥1–3 千跑通全流程，逐步命令和验收清单见 [docs/small_scale_validation.md](docs/small_scale_validation.md)。

### 5.1 数据（Mat-SFT → Mat-252K-SFT、Mat-20K-RL、基准）

```bash
export MP_API_KEY=...  DEEPSEEK_API_KEY=...  OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
# (1) MP 结构 + 性质 + DOI（约 15.3 万条）
python -m matbrain.data.mp_fetch --out data/mp/mp_records.jsonl
# (2) 时间切分：去掉无 created_at 的条目，最新 2000 条为测试集，再往前 2000 条为验证集
python -m matbrain.data.split --records data/mp/mp_records.jsonl --out-dir data/splits
# (3) 五级泄漏审计 → data/audit/test_clean.jsonl（论文：2000 → 1662）
python -m matbrain.data.leakage --train data/splits/train.jsonl --test data/splits/test.jsonl --out-dir data/audit
# (4) 基准任务（分类 / 回归 / 结构设计）
python -m matbrain.data.benchmark --clean data/audit/test_clean.jsonl --out data/benchmark/tasks.jsonl
# (5) 文献：MinerU 转 Markdown → 三模型投票分类 → LangExtract 抽取合成/应用段落
python -m matbrain.data.literature convert  --pdf-dir papers/ --out-dir data/markdown
python -m matbrain.data.literature classify --md-dir data/markdown --out data/chunks.jsonl
python -m matbrain.data.literature extract  --chunks data/chunks.jsonl --out data/extractions.jsonl
# (6) generate-distil：文献部分 + MP 部分（只用训练集）
python -m matbrain.data.generate_distil literature --chunks data/chunks.jsonl --out data/sft_raw/lit.jsonl
python -m matbrain.data.generate_distil mp --records data/splits/train.jsonl --out data/sft_raw/mp.jsonl
# (7) Mat-R1 训练混合：Mat-252K-SFT + Mixture-of-Thoughts/science（17.3 万）+ 自我认知数据
python training/sft/prepare_sft_data.py --mat-sft data/sft_raw --mot-science data/Mixture-of-Thoughts/science --out-dir data/sft
# (8) Mat-20K-RL（verl parquet；输入 CIF 预先注册为句柄）
MATBRAIN_ARTIFACT_DIR=data/rl/artifacts python -m matbrain.data.rl_dataset --sft data/sft_raw/mp.jsonl --out-dir data/rl
```

如果直接用官方发布的 Mat-252K-SFT / Mat-20K-RL，可以跳过 (5)–(8) 中对应步骤；`prepare_sft_data.py` 支持 messages / ShareGPT / Alpaca / QA 格式。

### 5.2 基座选择（Fig. 2b）

```bash
python training/sft/base_model_probe.py --data data/sft/mat_r1_train.jsonl \
  --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-30B-A3B ... --out results/base_model_loss.csv
python scripts/plot_base_model_selection.py --loss results/base_model_loss.csv --scores my_gpqa_hle.csv
```

### 5.3 Mat-R1（SFT）

```bash
bash training/sft/run_mat_r1_megatron.sh          # 论文设置
bash training/sft/run_mat_r1_lora_small.sh        # 单卡缩小版
```

### 5.4 Mat-MCP 与 Mat-T1（RL）

```bash
docker build -f deploy/Dockerfile -t mat-mcp . && docker run -p 8000:8000 -e MP_API_KEY=$MP_API_KEY mat-mcp
python training/rl/make_tool_config.py --mode local  --out training/rl/tool_config/mat_mcp_local.yaml   # 进程内调用
python training/rl/make_tool_config.py --mode remote --out training/rl/tool_config/mat_mcp_remote.yaml --mcp-url http://mat-mcp:8000/mcp
MATBRAIN_ARTIFACT_DIR=data/rl/artifacts bash training/rl/run_mat_t1_dapo.sh
```

训练时 wandb 里的 `reward/turns|think|format|syntax` 对应 Fig. 2g,h。

### 5.5 部署 MatBrain

```bash
bash deploy/serve_models.sh                      # vLLM：Mat-T1 :8001（hermes 工具解析），Mat-R1 :8002
matbrain --config configs/matbrain.yaml "Analyse the thermodynamic stability of Li2ZrCl6" --trace-out trace.json
```

### 5.6 评测（Fig. 3）

```bash
python scripts/build_reference_entries.py --records data/splits/train.jsonl --out data/reference_entries.jsonl
export MATBRAIN_REFERENCE_ENTRIES=$PWD/data/reference_entries.jsonl   # 相图参考集只含训练集，避免泄漏
R="python -m matbrain.eval.runner --tasks data/benchmark/tasks.jsonl --config configs/benchmark.yaml"
for s in Mat-R1 Mat-R1-base GPT-5 Gemini-2.5-Pro DeepSeek-R1; do $R --system $s --mode direct; done   # Fig. 3a-c
for s in Mat-T1 Mat-T1-base GPT-5 Gemini-2.5-Pro DeepSeek-R1; do $R --system $s --mode tool; done     # Fig. 3d-f
$R --mode decision_tree
$R --system MatBrain --mode matbrain                                                                    # Fig. 3g-i
python scripts/report_benchmark.py --dir results/benchmark
```

### 5.7 熵分析（Fig. 4）

```bash
python -m matbrain.analysis.entropy generate --model <Mat-T1 或基座路径> --prompts data/entropy_prompts.jsonl --out results/entropy/mat_t1.jsonl --limit 200
python -m matbrain.analysis.entropy plot --inputs results/entropy/*.jsonl --out-dir results/entropy/figs
```

### 5.8 NRR 催化剂发现（Fig. 6）

通过 MatBrain 下达任务，或直接调用工具与筛选脚本：

```bash
# 三个体系各 1 万个结构，多卡并行（MATBRAIN_GPUS=0,1,2,3）
python -c "from matbrain.mcp import load_registry as L; r=L(); [print(r.call('mattergen_generate', {'chemical_system': s, 'num_samples': 10000, 'num_workers': 4}).data) for s in ['Fe-V-S','Co-V-S','Ni-V-S']]"
cat /tmp/mattergen_*/handles.txt > data/nrr_handles.txt
MATBRAIN_ARTIFACT_DIR=... python -m matbrain.screening.funnel --handles-file data/nrr_handles.txt --out-dir results/nrr_screening
python -m matbrain.experiments.nrr --absorbance 0.21 --charge 1.3 --e-ag-agcl -0.747   # 实验数据处理
```

## 6. 超参数对照（均取自论文 Methods）

| 项目 | 论文 | 本仓库位置 |
|---|---|---|
| Mat-R1 基座 / 框架 | Qwen3-30B-A3B，Megatron-Swift 3.10，BF16 | `run_mat_r1_megatron.sh` |
| 并行 | TP=2，EP=8，MoE aux loss 1e-3，Flash-Attn 2，全量重计算 | 同上 |
| 优化 | GBS 16，max len 16,384，packing，AdamW，1e-6→2e-5 预热 5%，余弦降到 1e-6，2 epoch | 同上（`lr_warmup_init` 经 `megatron_extra_kwargs` 传入） |
| Mat-T1 基座 / 框架 | Qwen3-14B，verl v0.6.1，vLLM TP=4，FSDP + Ulysses SP=4 + CPU offload | `run_mat_t1_dapo.sh` |
| RL 超参 | clip [0.2, 0.28]，KL β=0，lr 1e-6 恒定，GBS 16，G=8，1 epoch | 同上 |
| 预算 | prompt 16,384，response 3,072，最多 4 轮 user/assistant | 同上 |
| 奖励 | w=(0.1, 0.3, 0.25, 0.35)，k=4，λ=500，α=0.4，β=0.6 | `matbrain/rl/rewards.py` |
| MatBrain | LangGraph，max_iterations=6，asyncio，Pydantic 运行时校验 | `matbrain/agent/graph.py` |
| 基准 | 12 次工具调用预算，temp=0，禁用目标库检索 | `matbrain/eval/runner.py` |
| 筛选 | stol 0.3 / ltol 0.2 / 5°，Ehull ≤ 0.025 eV/atom，带隙 < 2.0 eV | `matbrain/screening/funnel.py` |

## 7. 论文没写清的地方与本复现的处理

1. **R_turns 在阈值之间的取值**：式 2 只给了 n = ⌈k/4⌉、⌈k/2⌉、⌈3k/4⌉、≥k 几个点；k 不为 4 时，阈值之间取下一档。k=4 时和论文完全一致。
2. **没有中间步骤时（K=0）式 4 第一项如何算**：取 0，即不调用工具直接回答时只能拿到 β 部分。
3. **"超过 4 分"**：默认 `score > 4`（`--non-strict` 改为 `≥ 4`）。
4. **think 长度 L 的 token 计数**：设 `MATBRAIN_REWARD_TOKENIZER` 用 Qwen3 分词器精确计数；不设时用近似计数（仅用于测试）。
5. **语法奖励中的句柄**：句柄只有在同一轨迹里之前的工具返回（或 prompt）中出现过才算合法，防止模型编造句柄。这比论文描述更严格一点，但符合"可执行"的本意。
6. **DAPO 的动态采样**：verl v0.6.1 的 `recipe/dapo` 训练器只支持单轮同步 rollout，多轮工具调用必须走 `main_ppo` 的 async agent loop。我们因此用 `main_ppo` 实现 DAPO 的 decoupled clip、token 级损失聚合和零 KL；组内奖励全相同的样本 GRPO 优势为 0、不产生梯度，这正是动态采样要剔除的组。复合奖励是连续值，这种组很少。
7. **max_assistant_turns=4 的副作用**：verl 在第 4 个 assistant 回合后不再执行工具，所以"4 轮工具 + 最终回答"放不进预算，最优策略实际上是 3 轮工具 + 1 轮回答。这是论文超参数本身带来的，我们照原样保留。
8. **原型标签**：若安装了 `aviary`，直接用 Matbench Discovery 的实现；否则用自带的 AFLOW 风格实现（化学计量 + Pearson 符号 + 空间群 + 各元素 Wyckoff 序列 + 化学体系），对原胞 / 惯用胞 / 超胞保持不变，但不做 Wyckoff 等价设置的规范化。
9. **Fermi 能工具**：论文说大多数控制器用了"同一个确定性后端"，但没说是什么。这里提供 `predict_property_custom`，可以接入你自己在 MP efermi 上训练的 MatGL 模型（`MATBRAIN_EFERMI_MODEL`）。
10. **提示词**：论文未公开，`matbrain/prompts.py` 按 Methods 的描述重写。
11. **CrystaLLM / MatterGen 封装**：分别调用上游的 `bin/sample.py` 和 `mattergen-generate` CLI；上游参数可能随版本变化，使用前请对照你的版本。
12. **势函数与 MatGL 模型**：默认势函数用独立的 `chgnet` 包（v0.3.0，权重随 pip 包附带，MPtrj 训练，能量与 MP 兼容，E_hull 依赖这一点）。MatGL ≥ 3.0 的权重全部改从 Hugging Face `materialyze` 下载，名称也变了，且其 M3GNet/TensorNet 势函数改为 MatPES 训练、与 MP2020 修正不兼容；代码会依次尝试新旧名称，也可以用 `MATBRAIN_<类型>_<模型>_MODEL` 环境变量指定。
13. **Ehull 与泄漏**：相图参考集用训练集快照（`MATBRAIN_REFERENCE_ENTRIES`），否则在线 MP 参考集里可能包含测试集材料本身。ML 能量默认加 MP2020 修正后再与 MP 参考比较。
14. **MatBrain 入口**：Methods 写的是从执行节点（Mat-T1）开始；Fig. 1b 画了 Mat-R1 的初始分析。默认按 Methods，`start_with_analysis: true` 切换到 Fig. 1b 的流程。

## 8. 复现时可对照的论文数值

| 指标 | 论文 |
|---|---|
| Mat-R1 金属性 balanced acc. | 0.807（基座 0.525） |
| Mat-R1 MAE：形成能 / Ehull / Fermi 能 | 0.319 / 0.193 / 1.322（形成能基座 1.740） |
| Mat-R1 结构设计：valid / charge-balanced / complete success | 0.850 / 0.700 / 0.475（基座 complete success 0.010） |
| 平均熵：Qwen3-30B-A3B → Mat-R1 | 0.673 → 0.483 |
| 平均熵：Qwen3-14B → Mat-T1 | 0.878 → 0.974 |
| 协作时 Mat-R1 `<answer>` 段熵 | 0.12 → 0.05；Mat-T1 KDE 峰值约 3.3 bits |
| RL：平均 rollout 奖励、总序列长度 | 约 0.95；约 12k → 19k tokens |
| 泄漏审计 | 2000 → 1662 条 |
| 筛选漏斗 | 30,000 → 26,973 → 21,954 → 10,452 → 10,128 → 42 → 42 → 38 |
| CoV4S8 NRR | −0.55 V 时 NH3 产率 34.6 µg h⁻¹ mg⁻¹；−0.35 V 时 FE 4.3% |

本环境（无 GPU、只能访问 PyPI）中已经验证的内容：45 个单元测试；CHGNet 能量、弛豫、磁矩与 E_hull 工具的真实调用；verl v0.6.1 的 MCP 客户端逻辑能解析并调用 Mat-MCP 的全部工具；Fig. 5b 的 Vegard 晶格常数、体积、t、μ；V2Fe6S4 被价态过滤剔除；CsPb(Cl0.2Br0.4I0.4)3 的整数近似 Cs5Pb5Cl3Br6I6；MCP server 往返调用；奖励函数与手算一致；玩具数据上的切分→审计→基准→评测→报告全流程。**模型训练、模型对比和熵分析的数值都还没有跑，需要 GPU 与数据。**
