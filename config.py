"""
config.py —— 全局配置中心

所有物理常量、奖励层级、DQN 超参数集中于此，其它模块一律 `import config as C`。
在 PyCharm 里调参只需改这一个文件。

依赖：numpy / matplotlib / pillow（不需要 torch、不需要游戏引擎、不需要 ffmpeg）
"""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "dqn_model.npz"
LOG_PATH = OUT_DIR / "training_log.csv"
CURVE_PATH = OUT_DIR / "convergence_curves.png"
REPORT_JSON = OUT_DIR / "report.json"
ANIM_PATH = OUT_DIR / "interception_20.gif"
HTML_PATH = OUT_DIR / "report.html"

# ---------------------------------------------------------------- 场景几何
W = 1.0                       # 画面宽（归一化坐标）
H = 1.0                       # 画面高，y=H 为顶部，y=0 为底部
GUN_X, GUN_Y = 0.5, 0.0      # 机枪在画面最底部中央
ANGLE_MIN = 0.02              # 只防守上方 180°：角度范围 (0, pi)
ANGLE_MAX = np.pi - 0.02

# ---------------------------------------------------------------- 时间尺度
DT = 0.04                     # 一个 RL 决策步 = 0.04 s
SUBSTEPS = 6                  # 每个决策步内做 6 次物理子积分，防止子弹"穿透"目标
H_SUB = DT / SUBSTEPS         # 子步长 0.00667 s

# ---------------------------------------------------------------- 速度
V_DRONE = 0.22                # 无人机平均下降速度 v̄（即"无人机平均速度"）
SPEED_RATIO = 16.0            # 题设：子弹速度恒为无人机平均速度的 16 倍

# ---------------------------------------------------------------- 机枪
OMEGA_MAX = 0.6               # 最大角速度 rad/s（云台转速上限，是本题的主要难点）
DTHETA_MAX = OMEGA_MAX * DT   # 单步最大转角 0.024 rad ≈ 1.4°
INIT_ANGLE_JITTER = 0.35      # 初始炮口角在正上方 ±0.35 rad 内随机
ROT_LEVELS = (-1.0, -0.35, 0.0, 0.35, 1.0)   # 5 档角速度（含微调档，保证瞄准精度）
N_ACTIONS = len(ROT_LEVELS) * 2                # 5 档 × {不开火, 开火} = 10 个离散动作
FIRE_COOLDOWN_STEPS = 1       # 开火冷却（1 = 每步都可开火）
MAX_BULLETS = 80
BULLET_LIFETIME = 1.2         # 子弹存活时间 s

# ---------------------------------------------------------------- 无人机 / 命中
R_HIT = 0.065                 # 命中/毁伤判定半径（含近炸范围，约 3 倍机身半径）
DRONE_X_MARGIN = 0.06         # 水平边界
VX_MAX = 0.55 * V_DRONE       # 横向速度上限
OU_DECAY = 0.94               # OU 随机过程衰减（轨迹平滑但随机）
OU_SIGMA = 0.10 * V_DRONE
VY_JITTER = 0.18              # 竖直速度扰动
MAX_STEPS = 220               # 单回合最大步数保护

# ---------------------------------------------------------------- 奖励（严格惩罚层级）
# 量级顺序（按单回合累计最坏值估算）：
#   一级 漏防失败 ≈ 200   >>   二级 危险度 ≈ 11   >   三级 弹药 ≈ 3
R_FAIL = -200.0               # ① 飞机到达底部（最严重，一次性）
DANGER_COEF = 0.30            # ② 危险度：每步 -0.30*(1-y/H)^2，越靠近底部罚得越狠
BULLET_COST = 0.05            # ③ 经济性：每发子弹 -0.05（最轻）
R_SUCCESS = +100.0            # 成功拦截的一次性正奖励（注意：勿与命中半径 R_HIT 同名）

# ---------------------------------------------------------------- DQN
STATE_DIM = 13
HIDDEN = 64
GAMMA = 0.99
LR = 2.5e-4
LR_MIN = 5e-5                 # 余弦衰减终值，抑制训练后期策略漂移
BATCH = 64
N_DEMO_BATCH = 16             # 每个 batch 中混入的专家演示样本数
DQFD_MARGIN = 2.0             # DQfD 大间隔：Q(s,a_expert) >= Q(s,a_other) + 2
DQFD_WEIGHT = 1.0             # 演示边际损失权重（锚定强度，抑制策略漂移）
REPLAY_CAP = 120_000
DEMO_CAP = 60_000
WARMUP = 800                  # 经验池预热步数
TAU = 0.005                   # 目标网络 Polyak 软更新系数
GRAD_CLIP = 10.0
HUBER_DELTA = 1.0

# ---------------------------------------------------------------- 训练
EPISODES = 3000
EPS_START = 1.0
EPS_FINAL = 0.02              # 最终探索率 0.02 < 0.03 ✔
EPS_DECAY_FRAC = 0.55         # 按「训练回合进度」衰减，保证末期一定到 0.02
N_DEMO_EPISODES = 300         # 专家演示回合数（DQfD 式回放池播种）
DEMO_NOISE = 0.08             # 演示中混入少量随机动作，扩大状态覆盖
CURRICULUM_ON = True          # 课程学习：前期无人机稍慢，后期恢复满难度
CURRICULUM_FRAC = 0.35
CURRICULUM_START = 0.80
EVAL_EVERY = 100              # 每 100 回合做一次贪心评估
EVAL_EPISODES = 200           # 每次贪心评估回合数（用于挑最佳检查点）
CHECKPOINT_TOPK = 5           # 保留的最佳检查点数，最后再做一次锦标赛复选
CHECKPOINT_EVAL = 500         # 每个候选检查点的复选回合数

# ---------------------------------------------------------------- 交付
EVAL_EPISODES_FINAL = 500     # 最终评估回合数
DEMO_INTERCEPTIONS = 20       # 动画展示的拦截次数
SEED = 20261114

# ---------------------------------------------------------------- 动画
ANIM_FIGSIZE = (5.4, 5.4)
ANIM_DPI = 72
ANIM_FPS = 20
ANIM_STRIDE = 1               # 每个决策步 1 帧，动画接近实时速度
ANIM_HIT_HOLD = 4             # 命中后停留帧数
