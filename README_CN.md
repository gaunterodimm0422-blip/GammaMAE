# gmae2：GammaMAE v2 后处理 + 训练（远控机 Windows，RTX 4060）

**GATE 全部跑完、VM 停止运行后再执行**（远控机 GATE 与分析不能同时跑）。
以下 `PY` 指本机 Python 3.10+ 解释器（可写完整路径）。

## 0. 安装（一次）
```powershell
$PY = "python"   # 或完整路径，如 C:\Path\To\Python313\python.exe
& $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $PY -m pip install numpy scikit-learn pandas tabulate uproot awkward
& $PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
最后一行应输出 `True` 和 `NVIDIA GeForce RTX 4060`。

## 1. 事件列表 -> npz（GATE 包里的转换脚本）
```powershell
& $PY D:\vgate_shared\gammamae_gate_v2\tools\convert_to_npz.py --src D:\vgate_shared\output\gammamae_v2 --dst D:\gammamae_v2_npz
```
## 2. 建库（展宽、按真实时间切 1/3/10/30/100 s、混本底、测试集扰动），约 20–40 分钟
```powershell
Set-Location D:\gmae2
& $PY -m gmae2.build_dataset --events D:\gammamae_v2_npz --out D:\gammamae_v2_data
```
## 2b. 源活度扫描（约 10 分钟，选定低计数难度；不写数据集）
```powershell
& $PY -m tools.scan_activity --events D:\gammamae_v2_npz --acts 37,10,3,1,0.3
```
输出写到 `D:\gmae2\scan_activity.txt`。选好活度后用 `--activity_kbq` 重新建库：
```powershell
& $PY -m gmae2.build_dataset --events D:\gammamae_v2_npz --out D:\gammamae_v2_data --activity_kbq <选定值>
```
GATE 按 37 kBq 模拟；更低活度通过对源事件独立伯努利稀疏实现（泊松过程稀疏后仍为泊松过程，等价于活度按比例降低），本底不变。

## 3. 冒烟测试（几分钟，确认 GPU 流程无误）
```powershell
& $PY -m gmae2.run_experiments --data D:\gammamae_v2_data --out D:\gammamae_v2_smoke --quick 3000 --epochs 3 --pre_epochs 3 --ft_epochs 3 --seeds 0
```
## 3b. 学习性检查（约 5 分钟，确认网络真的在学，避免 v1 欠训练问题）
```powershell
& $PY -u -m tools.learn_check --data D:\gammamae_v2_data --n 6000 --epochs 15 *> D:\gammamae_v2_learncheck.log
```
看最后的“汇总”：cnn / transformer / gmae 的 val acc 应明显高于随机（0.10），并接近或超过 logreg。若某个 NN 停在 0.1–0.2，先别跑正式实验，把这个 log 发给我。

## 4. 正式实验（可中断，重跑同一命令会跳过已完成项）
```powershell
& $PY -m gmae2.run_experiments --data D:\gammamae_v2_data --out D:\gammamae_v2_results *> D:\gammamae_v2_results_console.log
& $PY -m gmae2.summarize --out D:\gammamae_v2_results
```
RTX 4060 上预计 8–14 小时，中途断了重跑同一命令即可续跑。
发给我：`D:\gammamae_v2_results\summary.md`、`results.csv`，以及把 `logs` 文件夹压缩成 zip（训练曲线）。

## 实验设计
- 划分：GATE 种子 0/1/2 = train/val/test，不共享任何事件；超参与早停只看 val。
- 特征：x = log1p(1000·p)，p 为归一化计数，所有谱同一尺度；逐道标准化在全部 train 谱上拟合（不含标注），所有方法共用。
- E1 同探测器分类（三探测器联合训练）+ E3 扰动：增益 ±1/±3%、分辨率 ×1.25/×1.5、本底 ×2/×4，均在事件层面施加。
- E2 留一探测器迁移：两个源探测器有标注，目标探测器只在 MSM 预训练中提供无标注谱；k=0 与每类 10 条标注。
- E4 距离迁移：只用 10 cm 标注训练，测 5/10/25 cm。
- E5 少标注：每个 (探测器, 核素, 距离, 时间) 单元只给 5 条标注 (train 与 val 同预算)，MSM 编码器仍用全部无标注 train 谱。
- 方法：泊松模板拟合（源模板+本底模板，ML-EM）、逻辑回归（val 选 C）、余弦 KNN（val 选 k）、CNN、ResNet、Transformer（与 GammaMAE 同结构、从零训练）、
  GammaMAE：gmae_msm（MSM 预训练 + 微调，主方法）、gmae_msm_probe（冻结编码器 + 逻辑回归）、gmae_dcl（MSM+DCL，消融）。
- MSM 编码器只预训练一次（每个种子），在全部 train 谱上、不用任何标注，E1/E2/E4/E5 共用；DCL 版本按各实验允许的标注分别预训练。
- 验证集消融（tools/tune_gmae.py，只用 train/val）确定：掩码率 0.5、微调 lr 1e-3（编码器 ×0.3）；DCL 在全标注下 −2.4 点、少标注下坍缩，故降为消融。
- 神经网络 3 个训练种子，报告 均值 ± 标准差；确定性方法只跑一次。
- CNN/ResNet 分类头保留能量位置（1×1 卷积降维后展平），不用全局平均池化。
- GammaMAE v2：编码器 6 层 d=256（4.8 M 参数），解码器 2 层 d=128，总 5.3 M（v1 为 28.9 M）。

## 许可证

本代码以 MIT 许可证发布，见 `LICENSE`。
