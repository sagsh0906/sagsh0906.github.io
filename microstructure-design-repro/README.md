# 论文复刻：Generative design of high-fidelity microstructures using physics-aware machine learning

> Liao W., Li K., Tang B., …, Li J., Yuan R. *Nature Communications* (2026), doi:[10.1038/s41467-026-78118-3](https://doi.org/10.1038/s41467-026-78118-3)
> 官方代码：[nwpuai4msegroup/microstructures_design](https://github.com/nwpuai4msegroup/microstructures_design)（MIT）

本目录用 PyTorch / scikit-learn 从零重写了论文的整条流程：**物理感知损失的 VAE + 条件 DDPM 生成 EBSD 微观组织 → 潜空间描述子 + PCA → GBR 代理模型 → NSGA-II 多目标逆向设计 → ΔHV 追踪 → 解码生成新组织 → 晶粒统计 / 双峰分析 → 力学"虚拟验证"**。结果分两部分：

* **Part A（真实数据，精确复现 Fig. 4b–f）**：用作者公开的 25 组 IN625 实测性能、作者训练好的 VAE 所编码的 2000×128 潜变量，以及作者保存的 GBR 模型，逐项复现代理模型精度、NSGA-II 优化、ΔHV 曲线和 Pareto 候选。并额外做了稳健性检验。
* **Part B（合成数据，跑通生成模型全链路）**：原始 25 张 EBSD 大图和作者的 VAE 权重都拿不到（见 §3），所以写了一个基于物理冶金规律的 IN625 EBSD Euler 图合成器，外加一个平均场晶体塑性代理模型（替代 CPFE）。在这套数据上完整训练 VAE / DDPM，复现 Fig. 2、Fig. 3、Fig. 4、Fig. 5 和补充图的分析流程。

---

## 1. 论文方法速览

| 步骤 | 论文做法（正文 + Methods + 官方代码） |
|---|---|
| 数据 | Inconel 625，热轧压下量 0–50 % × 热处理 650–1050 °C × 1–20 h 的 25 组正交工艺；EBSD Euler 图 800×1000 px，步长 0.5 µm；拉伸得到屈服强度 / 延伸率 |
| 数据扩充 | 每张大图均匀切成 8×10 = 80 块 100×100 → 2000 块；翻转 / 旋转 ×6 → 12 000；模糊 / 仿射畸变 / 黑块 / 腐蚀 / 膨胀 ×5 → 60 000（作为"修复"训练的输入，目标为原图）；共 72 000；按原图 80/20 划分 |
| 物理感知损失 | L = L<sub>pix</sub>（MAE）+ L<sub>edge</sub>（Sobel，强调晶界）+ L<sub>SSIM</sub>（取向结构）+ L<sub>KL</sub> |
| 生成模型 | VAE（3 层卷积 / 3 层反卷积，128 维潜变量，输入缩放到 128 px）+ 以 VAE 输出为条件的 DDPM（U-Net 预测噪声，20 步 DDIM 采样）；Adam，余弦退火 1e-4→0，100 epoch，batch 50 |
| 潜空间描述子 | 每张图 80 块的潜变量逐维求均值 μ 与标准差 σ（各 128 维）→ 分别 PCA 取前 3 个主成分 → 6 维 |
| 代理模型 | GBR（RandomizedSearchCV 搜 n_estimators∈[10,500]、max_depth∈[1,20] 等），测试集 R² > 0.92 |
| 逆向设计 | NSGA-II，种群 10，500 代，变量范围 [-2,2]⁶，同时最大化 YS 与 EL；ΔHV<sub>t</sub> = HV<sub>t</sub> − HV<sub>0</sub>（Eq. 16） |
| 生成 | 最优 6 维向量 → 逆 PCA 得 μ、σ → 采样 80 个 z ~ N(μ, σ²) → VAE 解码 → DDPM 细化 |
| 验证 | CPFE（{111}〈110〉滑移、位错密度硬化、晶界强化、损伤；GND 由 ResNet-18 预测）+ 实验（1000 °C/60 % 热轧 + 850 °C/5 min，双峰晶粒组织） |

## 2. 复刻内容对照

| 论文模块 | 本仓库实现 | 文件 | 状态 |
|---|---|---|---|
| Euler 角 ↔ RGB 映射 | 从官方示例图反推出 Channel 5 立方晶系约定：R=φ1/360°、G=Φ/90°、B=φ2/90°（Φ ≤ 54.7°，与示例图 G 通道最大值 ≈155 吻合） | `msdesign/euler.py` | ✅ |
| 数据切块与三步扩充 | 与官方 `Function.py` 中的 cv2 算子一一对应（5×5 σ=5 高斯模糊、同一组仿射点对、3 个 20×20 黑块、2×2 腐蚀 / 膨胀） | `msdesign/data.py` | ✅ |
| L<sub>pix</sub>、L<sub>edge</sub>、L<sub>SSIM</sub>、KL（Eq. 1–11） | 按公式实现 | `msdesign/losses.py` | ✅ |
| VAE 结构 | 与官方 `SigmaVAE` 完全一致 | `msdesign/models.py` | ✅ |
| 条件 DDPM | 与官方结构一致（offset-cosine 连续噪声调度，12 通道输入 U-Net，20 步 DDIM）；**官方 `diffusVAE.pt` 可严格加载进本实现**，结构一致性已验证；训练损失按 Eq. 15 | `msdesign/models.py` | ✅ |
| 物理特征提取（Fig. 3） | 由黑色晶界网络分割晶粒 → 面积、等效椭圆长 / 短轴、三个 Euler 角；邻晶取向差（含立方对称）；KAM | `msdesign/features.py` | ✅ |
| 描述子 + 可逆 PCA | 同官方 notebook，并固定 PCA 符号约定（见 §5.3） | `msdesign/inverse.py` | ✅ |
| GBR + 随机搜索 | 同官方 notebook（同样的搜索空间、cv=5、n_iter=50、划分种子 75 / 98） | `msdesign/inverse.py` | ✅ |
| NSGA-II | 自写实现，算子与 pymoo 默认值一致（SBX η=15、PM η=20、二元锦标赛、拥挤度），可记录每一代 | `msdesign/nsga2.py` | ✅ |
| ΔHV（Eq. 16） | 按公式实现 | `msdesign/inverse.py` | ✅ |
| Wasserstein 代表块筛选（Supp. Note 6） | 同官方 notebook | `msdesign/inverse.py` | ✅ |
| 25 张 EBSD 原图 | **未公开** → 用物理冶金合成器生成（JMAK 再结晶、回复、晶粒长大、Σ3 退火孪晶、轧制拉长与取向梯度、1 px 黑色晶界、EBSD 噪声） | `msdesign/synthetic.py` | ⚠️ 替代 |
| CPFE 验证 | **无法在 CPU 上做全场 CPFE，补充材料中的参数也拿不到** → 平均场晶体塑性代理：逐晶粒 Schmid / Sachs 因子、Hall-Petch、KAM→GND 初始位错密度、Kocks-Mecking-Estrin 演化、异构界面 (HDI) GND 项、Taylor 等应变均匀化、Considère 判据 | `msdesign/cp_proxy.py` | ⚠️ 替代 |
| ResNet-18 GND 预测 | 用 KAM 换算 GND（ρ = 2θ/(u·b)）替代 | `msdesign/cp_proxy.py` | ⚠️ 替代 |
| 基线 WGAN / 标准 DDPM（Supp. Fig. 7） | 未实现（只做了与论文 Fig. 2 相同的"仅 L<sub>pix</sub>"消融） | — | ❌ |
| 实验合成验证 | 无法复现 | — | ❌ |

## 3. 数据可得性

| 数据 | 来源 | 本仓库 |
|---|---|---|
| 25 组工艺与 YS / EL 实测值（Supp. Table 1） | 官方仓库 `dataset.csv` | `data/official/dataset.csv` |
| 2000 块切片的 128 维潜变量（作者训练好的 VAE 编码） | 官方仓库 `z_i.csv` | `data/official/z_i.csv` |
| 10 块 100×100 EBSD 示例切片 | 官方仓库 `examples/` | `data/official/examples/` |
| 作者的 GBR 模型（sklearn 0.23 pickle） | 官方仓库 `model_save/GBR_*.pkl` | 导出为可移植的 `data/official/gbr_official_trees.npz`（`scripts/export_official_gbr.py`） |
| 作者的 DDPM 权重 `diffusVAE.pt` | 官方仓库 | 未拷贝（7.9 MB）；已验证能严格加载进 `msdesign.models.Diffusion` |
| 作者的 VAE 权重 | Google Drive（本运行环境网络策略禁止访问） | ❌ 缺失，所以 Part A 的最优潜变量无法解码成图像；`results/official/optimized_latents.npz` 已给出 80×128 潜变量，拿到 VAE 权重后可直接解码 |
| 25 张 800×1000 EBSD 原图、补充材料（CPFE 参数等） | 未公开 / nature.com 无法访问 | ❌ |

官方数据的版权与许可见 `data/official/LICENSE_official_repo`（MIT）。

## 4. 快速开始

```bash
pip install -r requirements.txt

# Part A：官方数据上的逆向设计（约 6 分钟，CPU）
python3 scripts/inverse_design_official.py            # --surrogate official|gbr|gpr

# Part B：合成数据全流程（4 核 CPU 约 3 小时）
bash run_quick.sh

# 按论文的完整设置（800×1000、100 px→128 px、72 000 对/epoch、100 epoch），需 GPU
bash run_paper_gpu.sh
```

| 脚本 | 作用 | 对应论文 |
|---|---|---|
| `scripts/inverse_design_official.py` | Part A | Fig. 4b–f |
| `scripts/make_synthetic_dataset.py` | 生成 25 张合成 Euler 图 + 代理模型标签 + 20/5 划分 | Supp. Fig. 1 / Table 1 |
| `scripts/train_vae.py --loss full/pix` | 物理感知 VAE / 仅像素损失基线 | Fig. 2a,b |
| `scripts/train_ddpm.py` | 条件 DDPM 细化器 | Fig. 1a |
| `scripts/analyze_fidelity.py` | 重建对比、六特征保真度、取向差分布、潜空间插值 / 外推 | Fig. 2c、Fig. 3、Supp. Fig. 4/6/8 |
| `scripts/inverse_design_synthetic.py` | 闭环逆向设计、解码、晶粒尺寸分布、虚拟验证、PCA 维数研究 | Fig. 4、Fig. 5、Supp. Fig. 10/14 |
| `scripts/export_official_gbr.py` | 把作者的 sklearn 0.23 pickle 导出为可移植格式（需 sklearn 1.2.x 环境） | — |

<!-- RESULTS -->
