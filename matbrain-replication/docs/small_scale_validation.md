# 小规模验证执行手册（方案 A）

用一台 80GB 显卡的机器、3–5 天、约 ¥1–3 千，把完整流程跑一遍：数据 → Mat-R1-mini（SFT）→ Mat-T1-mini（多轮 RL）→ MatBrain → 评测 → 熵分析。

目的是确认代码、数据和训练信号都正常，并检查几个定性现象是否和论文一致。**不追求论文里的数值**：这里用的是 4B 和 1.7B 小模型，论文是 30B 和 14B。

| 阶段 | 耗时（1×A100/H800 80GB） | 产出 |
|---|---|---|
| 0–2 环境、模型、数据下载 | 2–3 小时 | 三个 conda 环境、基座模型、训练数据 |
| 3 MP 数据、时间切分、泄漏审计 | 1–1.5 小时 | 训练/测试切分、审计报告、100 条基准记录 |
| 4 启动 Mat-MCP 并自检 | 20 分钟 | 工具服务、自检全部 OK |
| 5 Mat-R1-mini（Qwen3-4B LoRA SFT） | 4–8 小时 | 合并后的 Mat-R1-mini |
| 6 Mat-T1-mini（Qwen3-1.7B 多轮 RL，300 步） | 12–24 小时 | 合并后的 Mat-T1-mini、奖励曲线 |
| 7 部署与评测 | 4–6 小时 | 6 组评测结果与报告 |
| 8 熵分析（可选） | 1–2 小时 | 熵轨迹图、KDE 图 |

GPU 约 25–45 卡时。按 A100/H800 每卡时 ¥8–12 计，跑一遍 GPU 约 ¥200–550；预算留到 ¥1–3 千，是给重跑和可选的 API 调用。

---

## 0. 准备

**机器**（任选一种）：
- 推荐：1 × A100 或 H800 80GB，≥ 16 核 CPU，≥ 128 GB 内存，≥ 300 GB 磁盘。
- 省钱：2 × RTX 4090 24GB。SFT 要把 `--max_length` 降到 4096，RL 设 `NGPUS=2`。

**账号**：
- Materials Project API key（免费）：https://next-gen.materialsproject.org/api
- Hugging Face：用于下载模型和数据。国内机器先执行 `export HF_ENDPOINT=https://hf-mirror.com`。
- 可选：wandb（看训练曲线）、DeepSeek API key（第 7 步的商业模型参照，约 ¥50–150）。

**CrystaLLM 不是必需的**：小规模验证只用"性质预测"类任务。结构生成任务需要另外安装 CrystaLLM，放到论文规模时再做。

## 1. 安装三个环境（约 1 小时）

训练框架之间的 PyTorch 版本互相冲突，所以分成三个环境：

| 环境 | 用途 |
|---|---|
| `mb-tools` | Mat-MCP 工具（CHGNet、MatGL）、数据处理、评测、熵分析 |
| `mb-swift` | ms-swift 做 SFT 和 LoRA 合并 |
| `mb-verl` | verl 0.6.1 做 RL，vLLM 部署模型 |

```bash
git clone -b claude/paper-replication-fmcnp6 https://github.com/sagsh0906/sagsh0906.github.io.git
cd sagsh0906.github.io/matbrain-replication
mkdir -p logs models data results checkpoints

# 所有环境变量放进 env.sh；以后每开一个终端，先 conda activate 再 source env.sh
echo "export ROOT=$PWD" > env.sh
cat >> env.sh <<'EOF'
export HF_ENDPOINT=https://hf-mirror.com            # 海外机器删掉这行
export MP_API_KEY=你的_Materials_Project_key
export MATBRAIN_ARTIFACT_DIR=$ROOT/data/artifacts
export MATBRAIN_REFERENCE_ENTRIES=$ROOT/data/reference_entries.jsonl
export SMALL_TOOLS="validate_structure analyze_symmetry check_charge_balance store_structure get_structure_cif relax_structure predict_energy predict_formation_energy predict_bandgap predict_magnetic_moments phase_diagram_ehull"
export MATBRAIN_TOOL_POOL=analyze_symmetry,check_charge_balance,get_structure_cif,phase_diagram_ehull,predict_bandgap,predict_energy,predict_formation_energy,predict_magnetic_moments,relax_structure,store_structure,validate_structure
cd $ROOT
EOF
source env.sh

# 1) 工具 / 数据 / 评测
conda create -n mb-tools python=3.11 -y && conda activate mb-tools
pip install -e ".[agent,data,tools,analysis,dev]"
pytest -q                                   # 应全部通过

# 2) SFT
conda create -n mb-swift python=3.11 -y && conda activate mb-swift
pip install "ms-swift==3.10.*"
pip install flash-attn --no-build-isolation # 编译慢；装不上就在第 5 步加 --attn_impl sdpa

# 3) RL + 部署（verl 0.6.1 要求 vLLM 0.8.5–0.11.0）
conda create -n mb-verl python=3.11 -y && conda activate mb-verl
pip install "verl[vllm,gpu]==0.6.1" fastmcp
pip install -e .                            # 只装基础依赖，奖励函数要用
```

`verl[gpu]` 会编译 flash-attn，失败的话改用 verl 官方 Docker 镜像（见 verl v0.6.1 安装文档），在镜像里再执行最后两行。

## 2. 下载模型和数据（约 1 小时）

```bash
conda activate mb-tools
huggingface-cli download Qwen/Qwen3-4B   --local-dir models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
huggingface-cli download open-r1/Mixture-of-Thoughts --repo-type dataset \
    --include "science/*" --local-dir data/Mixture-of-Thoughts
```

**官方 Mat-252K-SFT**：在浏览器打开 https://doi.org/10.57967/hf/9652 ，跳转后的 Hugging Face 页面上写着仓库名（形如 `组织名/数据集名`），然后：

```bash
huggingface-cli download <仓库名> --repo-type dataset --local-dir data/Mat-252K-SFT
```

拿不到也不影响跑通流程，第 5 步有替代方案。

## 3. MP 数据、时间切分、泄漏审计（约 1–1.5 小时）

```bash
conda activate mb-tools && source env.sh

# 全库约 15 万条，30–60 分钟。免费；抓全库才能按论文做时间切分，也才有完整的相图参考集
python -m matbrain.data.mp_fetch --out data/mp/mp_records.jsonl

# 最新 2000 条为测试集，再往前 2000 条为验证集
python -m matbrain.data.split --records data/mp/mp_records.jsonl --out-dir data/splits

# 五级泄漏审计，16 核约 15–30 分钟
python -m matbrain.data.leakage --train data/splits/train.jsonl --test data/splits/test.jsonl \
    --out-dir data/audit --workers 16
cat data/audit/audit_summary.md              # 论文：2000 条剔除 338 条，剩 1662 条

# 相图参考集（只含训练集，防泄漏）
python scripts/build_reference_entries.py --records data/splits/train.jsonl --out data/reference_entries.jsonl

# 小规模基准：从干净测试集抽 100 条记录，约 700 个任务
python -m matbrain.data.benchmark --clean data/audit/test_clean.jsonl \
    --out data/benchmark/tasks_small.jsonl --sample 100

# RL 用的查询（以及拿不到官方数据时的 SFT 替代数据）：随机抽 5000 条训练记录生成模板问答
python -m matbrain.data.generate_distil mp --records data/splits/train.jsonl \
    --out data/sft_raw/mp_template.jsonl --template-only --limit 5000
```

## 4. 启动 Mat-MCP 并自检工具（约 20 分钟）

小规模验证只开放 11 个工具，这样提示词更短，RL 也更快：

```bash
conda activate mb-tools && source env.sh

# 首次运行会从 Hugging Face 下载 MatGL 权重；必需工具全部 OK 才继续
python scripts/check_tools.py

# 工具服务跑在 CPU 上，避免和训练抢显存。--flat-schemas 是 verl 的 MCP 客户端必需的
CUDA_VISIBLE_DEVICES="" nohup python -m matbrain.mcp.server --port 8000 --flat-schemas \
    --tools $SMALL_TOOLS > logs/mcp.log 2>&1 &
python scripts/check_tools.py --mcp-url http://127.0.0.1:8000/mcp
```

如果 `predict_formation_energy` 或 `predict_bandgap` 报找不到模型，说明你装的 MatGL 版本里模型改了名。先查可用名称：

```bash
python -c "import matgl; print(matgl.get_available_pretrained_models())"
```

再用环境变量指定，例如 `export MATBRAIN_BANDGAP_MEGNET_MODEL=<带 BandGap 的名称>`、`export MATBRAIN_EFORM_MEGNET_MODEL=<带 Eform 的名称>`，然后重启工具服务。

## 5. Mat-R1-mini：LoRA SFT（约 4–8 小时）

```bash
conda activate mb-tools && source env.sh
# 每个来源最多 1.5 万条，加 500 条自我认知数据，共约 3 万条。
# 如果打印 "Mat-252K-SFT: 0"，说明官方数据的字段格式没被识别，把一条样例发给我适配
python training/sft/prepare_sft_data.py --mat-sft data/Mat-252K-SFT \
    --mot-science data/Mixture-of-Thoughts/science --limit 15000 --out-dir data/sft_small
# 拿不到官方数据时：把 --mat-sft 换成 data/sft_raw/mp_template.jsonl（没有推理过程，只能验证流程）

conda activate mb-swift
MODEL=$ROOT/models/Qwen3-4B OUT=$ROOT/checkpoints/Mat-R1-mini \
TRAIN_DATA=$ROOT/data/sft_small/mat_r1_train.jsonl VAL_DATA=$ROOT/data/sft_small/mat_r1_val.jsonl \
    bash training/sft/run_mat_r1_lora_small.sh          # 4090：末尾加 --max_length 4096

# 合并 LoRA（目录名按实际输出填写）
swift export --adapters checkpoints/Mat-R1-mini/<v0-日期时间>/checkpoint-<最后一步> --merge_lora true
# 得到 .../checkpoint-<最后一步>-merged，把路径记进 env.sh（按实际目录名改）：
echo 'export R1_MINI=$ROOT/checkpoints/Mat-R1-mini/v0-日期时间/checkpoint-最后一步-merged' >> env.sh
```

**看什么**：输出目录里的 `logging.jsonl` 或 tensorboard。训练 loss 和验证 loss 都应持续下降。论文 30B 全参训练是从约 1.0 降到约 0.5；小模型加 LoRA 降幅会小一些。

## 6. Mat-T1-mini：多轮 RL（约 12–24 小时）

```bash
conda activate mb-tools && source env.sh
# 4800 条 = 300 步 × 每步 16 条；只保留小工具集能算出来的 5 类性质，均匀抽样
python -m matbrain.data.rl_dataset --sft data/sft_raw/mp_template.jsonl --out-dir data/rl_small \
    --n 4800 --val 100 --uniform --artifact-dir $MATBRAIN_ARTIFACT_DIR \
    --tasks property:formation_energy_per_atom property:energy_above_hull property:band_gap property:is_metal property:is_magnetic

python training/rl/make_tool_config.py --mode remote --out training/rl/tool_config/small_remote.yaml \
    --mcp-url http://127.0.0.1:8000/mcp --include $SMALL_TOOLS
# 最后一行打印的 MATBRAIN_TOOL_POOL 应与 env.sh 里的一致：语法奖励只认这 11 个工具

conda activate mb-verl && source env.sh
MODEL_PATH=$ROOT/models/Qwen3-1.7B \
TRAIN_FILE=$ROOT/data/rl_small/mat20k_rl_train.parquet VAL_FILE=$ROOT/data/rl_small/mat20k_rl_val.parquet \
TOOL_CONFIG=$ROOT/training/rl/tool_config/small_remote.yaml NGPUS=1 \
    bash training/rl/run_mat_t1_small.sh trainer.total_training_steps=300 trainer.logger='["console","wandb"]'
# 没有 wandb 就改成 trainer.logger='["console"]'；2×4090 设 NGPUS=2

# 合并成 Hugging Face 格式
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/Mat-T1-small/global_step_300/actor --target_dir checkpoints/Mat-T1-mini-hf
```

**看什么**（论文 Fig. 2g,h）：
- 格式奖励在前约 200 步内升到 0.9 以上。
- 语法奖励和轮数奖励逐步上升，总奖励（`critic/score/mean`）稳定上升。
- 回复长度变长，但没有撞满 3072 的上限。
- 策略熵不塌缩。

同时盯着 `logs/mcp.log`：工具报错很多，通常说明模型在编造 CIF 句柄或参数。

## 7. 部署与评测（约 4–6 小时）

模型用 `mb-verl` 环境里的 vLLM 部署。一张 80GB 卡可以同时起四个小模型；24GB 卡请两两分批。

```bash
conda activate mb-verl && source env.sh
TOOLS="--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --max-model-len 32768"
vllm serve checkpoints/Mat-T1-mini-hf --served-model-name Mat-T1-mini --port 8001 --gpu-memory-utilization 0.18 $TOOLS > logs/t1.log 2>&1 &
vllm serve models/Qwen3-1.7B        --served-model-name Qwen3-1.7B  --port 8011 --gpu-memory-utilization 0.18 $TOOLS > logs/t1b.log 2>&1 &
vllm serve $R1_MINI                  --served-model-name Mat-R1-mini --port 8002 --gpu-memory-utilization 0.25 --reasoning-parser qwen3 --max-model-len 32768 > logs/r1.log 2>&1 &
vllm serve models/Qwen3-4B           --served-model-name Qwen3-4B    --port 8012 --gpu-memory-utilization 0.25 --reasoning-parser qwen3 --max-model-len 32768 > logs/r1b.log 2>&1 &
```

评测时工具在评测进程内调用，跑在 CPU 上：

```bash
conda activate mb-tools && source env.sh
R="python -m matbrain.eval.runner --tasks data/benchmark/tasks_small.jsonl --config configs/benchmark_small.yaml --out-dir results/small --concurrency 8"
CUDA_VISIBLE_DEVICES="" $R --system Qwen3-4B-base   --mode direct     # 对应 Fig. 3a-c
CUDA_VISIBLE_DEVICES="" $R --system Mat-R1-mini     --mode direct
CUDA_VISIBLE_DEVICES="" $R --system Qwen3-1.7B-base --mode tool       # 对应 Fig. 3d-f
CUDA_VISIBLE_DEVICES="" $R --system Mat-T1-mini     --mode tool
CUDA_VISIBLE_DEVICES="" $R                          --mode decision_tree
CUDA_VISIBLE_DEVICES="" $R --system MatBrain-mini   --mode matbrain   # 对应 Fig. 3g-i
# 可选商业参照：$R --system DeepSeek-flash --mode direct
python scripts/report_benchmark.py --dir results/small > results/small/report.md
```

中断后重跑同一条命令会跳过已完成的任务。

## 8. 熵分析（可选，约 1–2 小时）

```bash
pkill -f "vllm serve"                        # 先释放第 7 步占用的显存
conda activate mb-tools && source env.sh
python scripts/make_entropy_prompts.py --tasks data/benchmark/tasks_small.jsonl --out data/entropy/r1_prompts.jsonl --n 100
python scripts/make_entropy_prompts.py --tasks data/benchmark/tasks_small.jsonl --out data/entropy/t1_prompts.jsonl --n 100 --with-tools
E="python -m matbrain.analysis.entropy generate --max-new-tokens 2048 --limit 100"
$E --model models/Qwen3-4B               --prompts data/entropy/r1_prompts.jsonl --out results/entropy/Mat-R1-base.jsonl
$E --model $R1_MINI                      --prompts data/entropy/r1_prompts.jsonl --out results/entropy/Mat-R1.jsonl
$E --model models/Qwen3-1.7B             --prompts data/entropy/t1_prompts.jsonl --out results/entropy/Mat-T1-base.jsonl
$E --model checkpoints/Mat-T1-mini-hf    --prompts data/entropy/t1_prompts.jsonl --out results/entropy/Mat-T1.jsonl
python -m matbrain.analysis.entropy plot --inputs results/entropy/*.jsonl --out-dir results/entropy/figs
```

这里只测第一轮输出的熵，不执行工具；论文 Fig. 4b,d 是含工具调用的完整轨迹，所以只比较方向，不比较数值。

## 9. 验收清单

| 检查项 | 通过的标准 | 在哪看 |
|---|---|---|
| 工具可用 | `check_tools.py` 的必需工具全部 OK（本地和 `--mcp-url` 两次） | 终端输出 |
| 泄漏审计 | 输出了五级重叠统计和干净测试集；剔除比例和论文的 16.9% 在同一量级 | `data/audit/audit_summary.md` |
| SFT | 训练和验证 loss 持续下降，没有发散 | `logging.jsonl` / tensorboard |
| RL 格式 | 格式奖励 200 步内超过 0.9 | wandb 或控制台 |
| RL 工具使用 | 语法奖励上升；工具报错率下降 | wandb、`logs/mcp.log` |
| Mat-R1-mini 对比基座 | direct 模式下多数性质指标变好（MAE 更低，B.acc 更高） | `results/small/report.md` |
| Mat-T1-mini 对比基座 | tool 模式下被拒调用（`n_rejected`）更少，完成率更高 | `results/small/predictions__*.jsonl` |
| MatBrain-mini | 至少在部分指标上不差于单个组件 | `report.md` |
| 熵（可选） | Mat-R1-mini 平均熵低于基座；Mat-T1-mini 不低于基座 | `results/entropy/figs/summary.json` |

全部通过，就可以放心租 8 卡跑论文规模（README 第 5 节）。

## 10. 常见问题

- **`check_tools.py` 里 MatGL 模型加载失败**：见第 4 步，查名称后用环境变量指定。先确认 Hugging Face（或 `HF_ENDPOINT` 镜像）能访问。
- **RL 报 prompt 超长被过滤**：工具越多提示词越长。少开几个工具，或给 `run_mat_t1_small.sh` 加 `data.max_prompt_length=12288`。
- **RL 显存不足**：追加 `actor_rollout_ref.rollout.gpu_memory_utilization=0.4`、`data.max_response_length=2048`。
- **RL 很慢**：多半是工具慢。看 `logs/mcp.log` 里 `relax_structure` 的耗时；可以把工具服务改到 GPU 上（去掉 `CUDA_VISIBLE_DEVICES=""`），前提是显存够。
- **verl 连不上 Mat-MCP**：确认 `curl http://127.0.0.1:8000/mcp` 有响应，服务启动时带了 `--flat-schemas`，而且 `MATBRAIN_ARTIFACT_DIR` 和第 6 步生成数据时的 `--artifact-dir` 是同一个目录。
- **小模型分数很低**：正常。1.7B 和 4B 的绝对分数不会接近论文，只看"训练后比训练前好"这个方向。
