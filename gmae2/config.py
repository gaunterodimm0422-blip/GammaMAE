# -*- coding: utf-8 -*-
"""GammaMAE v2 全局设置：能量轴、探测器响应、本底、划分与测量时间。"""
import numpy as np

DETECTORS = ["NaI", "GAGG", "HPGe"]                 # detector id = index
NUCLIDES = ["Am241", "Ba133", "Co57", "Co60", "Cs137", "Eu152", "I131", "K40", "Mn54", "Na22"]
DISTANCES = [5, 10, 25]                             # cm, source to front face
SPLIT_SEED = {"train": 0, "val": 1, "test": 2}      # GATE seed replica -> split (no event shared)

# 能量轴：与 v1 相同，2048 道，0-3000 keV
N_BINS = 2048
E_MAX = 3000.0
BIN_W = E_MAX / N_BINS
E_THRESHOLD = 15.0                                  # keV, 低能阈

# 能量分辨率 FWHM(E) = sqrt(a^2 + b*E + (c*E)^2), keV
# NaI(Tl) 3"x3": 7.0% @662 keV; GAGG:Ce: 6.0% @662 keV (Yoneyama 2018 / Kamada 2011 量级)
# HPGe 同轴: 1.0 keV @122 keV, 1.9 keV @1332 keV (Knoll 2010 典型值)
RESOLUTION = {
    "NaI":  dict(a=0.0,   b=2.97,     c=0.020),
    "GAGG": dict(a=0.0,   b=2.12,     c=0.020),
    "HPGe": dict(a=0.858, b=0.002157, c=0.0),
}

def fwhm(det, e, scale=1.0):
    r = RESOLUTION[det]
    return scale * np.sqrt(r["a"] ** 2 + r["b"] * e + (r["c"] * e) ** 2)

# 环境本底：未散射主线光子注量率 PHI_LINES (cm^-2 s^-1) + 散射连续谱 KAPPA*PHI_LINES
# 各向同性场中探测器入射率 = PHI * S/4 (S 为晶体外表面积)
PHI_LINES = 1.0
KAPPA = 2.0
CRYSTAL = {"NaI": (3.81, 7.62), "GAGG": (1.27, 2.54), "HPGe": (2.5, 5.0)}   # (R, H) cm
def surface_cm2(det):
    r, h = CRYSTAL[det]
    return 2 * np.pi * r * r + 2 * np.pi * r * h

# 源活度：GATE 中每个核素按 SIM_ACTIVITY_KBQ 模拟；建库时按 ACTIVITY_KBQ 对源事件稀疏 (低计数场景)
SIM_ACTIVITY_KBQ = 37.0
ACTIVITY_KBQ = 37.0

# 测量时间与每个 job 取的不重叠窗口数上限
TIMES = [1, 3, 10, 30, 100]
MAX_WINDOWS = {1: 200, 3: 200, 10: 60, 30: 20, 100: 6}
MAX_WINDOWS_PERTURBED = {1: 50, 3: 50, 10: 50, 30: 20, 100: 6}
T_MEAS = 600.0

# 测试集扰动（在事件层面施加，物理一致）
PERTURBATIONS = {
    "gain-3%": dict(gain=-0.03), "gain-1%": dict(gain=-0.01),
    "gain+1%": dict(gain=0.01),  "gain+3%": dict(gain=0.03),
    "res x1.25": dict(res=1.25), "res x1.5": dict(res=1.5),
    "bkg x2": dict(bkg=2.0),     "bkg x4": dict(bkg=4.0),
}
