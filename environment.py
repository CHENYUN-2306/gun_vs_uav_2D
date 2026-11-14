"""
environment.py —— 二维机枪拦截仿真环境（纯 NumPy，Gym 风格 API）

坐标：左下角 (0,0)，右上角 (1,1)。机枪在 (0.5, 0)，只防守上方 180°。
无人机从 y=1 顶部随机横向位置出现，向 y=0 下降，途中带 OU 随机横向扰动 +
低频阵风，形成随机轨迹。子弹从枪口以恒定速度 16*v̄ 沿当前角度直线飞行。

关键工程点：
  1) 固定步长积分 + 子步碰撞检测：子弹每子步只走 ~0.023，小于命中半径 0.055，
     避免"高速子弹一帧跨过无人机"的隧穿问题。
  2) 奖励严格按三级惩罚层级构造，见 config.py。
  3) reset()/step()/state() 与 Gym 一致，render 由 animate.py 用 matplotlib 完成，
     训练与可视化完全解耦。

API:
    env = GunDefenseEnv(seed=0, difficulty=1.0, record=False)
    s, info = env.reset()
    s2, r, done, info = env.step(action)      # action = rot_index*2 + fire
"""
from __future__ import annotations

import numpy as np

import config as C


class GunDefenseEnv:
    """单架无人机 / 单回合的拦截环境。"""

    def __init__(self, seed=None, difficulty: float = 1.0, record: bool = False):
        self.rng = np.random.default_rng(seed)
        self.difficulty = float(difficulty)
        self.record = bool(record)
        self.frames: list[dict] = []
        self._blank()

    # ------------------------------------------------------------ 速度属性
    @property
    def v_d(self) -> float:
        """当前难度下的无人机平均下降速度。"""
        return C.V_DRONE * self.difficulty

    @property
    def v_b(self) -> float:
        """子弹速度恒为无人机平均速度的 16 倍。"""
        return C.SPEED_RATIO * self.v_d

    def _blank(self):
        self.x = C.GUN_X
        self.y = C.H
        self.vx = 0.0
        self.vy = -self.v_d
        self.theta = np.pi / 2.0          # 初始指向正上方
        self.omega = 0.0
        self.cooldown = 0
        self.gust_phase = 0.0
        self.gust_freq = 2.0
        self.bullets = np.zeros((0, 5), dtype=np.float32)  # [x, y, vx, vy, life]
        self.steps = 0
        self.n_fired = 0
        self.success = False
        self.failed = False
        self.last_event = None

    # ------------------------------------------------------------ Gym API
    def reset(self):
        self._blank()
        m = C.DRONE_X_MARGIN
        self.x = float(self.rng.uniform(m, C.W - m))
        self.y = C.H
        self.vx = float(self.rng.uniform(-0.45, 0.45) * self.v_d)
        self.vy = -self.v_d * (1.0 + float(self.rng.normal(0.0, C.VY_JITTER * 0.5)))
        self.theta = float(np.clip(
            np.pi / 2.0 + self.rng.uniform(-C.INIT_ANGLE_JITTER, C.INIT_ANGLE_JITTER),
            C.ANGLE_MIN, C.ANGLE_MAX))
        self.gust_phase = float(self.rng.uniform(0, 2 * np.pi))
        self.gust_freq = float(self.rng.uniform(1.2, 3.6))
        self.frames = []
        self._snapshot(None)
        return self.state(), {"can_fire": True}

    def state(self) -> np.ndarray:
        """13 维归一化观测。角度误差、方位角、拦截时间尺度为学习提供强特征。"""
        dx = self.x - C.GUN_X
        dy = self.y - C.GUN_Y
        phi = float(np.clip(np.arctan2(dy, dx), C.ANGLE_MIN, C.ANGLE_MAX))
        dist = float(np.hypot(dx, dy))
        closing = self.v_b + max(-self.vy, 0.2 * self.v_d)
        t_int = dist / closing
        danger = float(np.clip(1.0 - self.y / C.H, 0.0, 1.0))
        s = np.array([
            np.sin(self.theta),                       # 0  炮口角 sin
            np.cos(self.theta),                       # 1  炮口角 cos
            2.0 * dx / C.W,                           # 2  目标水平相对位置
            self.y / C.H,                             # 3  目标高度
            self.vx / self.v_d,                       # 4  目标横向速度
            self.vy / self.v_d,                       # 5  目标竖直速度
            np.sin(phi),                              # 6  目标方位角 sin
            np.cos(phi),                              # 7  目标方位角 cos
            (phi - self.theta) / np.pi,               # 8  角度误差（核心特征）
            np.clip(t_int / 0.5, 0.0, 1.0),           # 9  预计弹目交会时间尺度
            danger,                                   # 10 危险度（离底部越近越大）
            1.0 if self.cooldown <= 0 else 0.0,       # 11 是否可开火
            min(len(self.bullets) / C.MAX_BULLETS, 1.0),  # 12 在场弹数
        ], dtype=np.float32)
        return s

    def bearing(self, lead: bool = False) -> float:
        """返回目标方位角；lead=True 时返回带提前量的拦截方位角。"""
        dx, dy = self.x - C.GUN_X, self.y - C.GUN_Y
        if lead:
            dist = float(np.hypot(dx, dy))
            t_int = dist / (self.v_b + max(-self.vy, 0.2 * self.v_d))
            dx += self.vx * t_int
            dy += self.vy * t_int
        return float(np.clip(np.arctan2(dy, dx), C.ANGLE_MIN, C.ANGLE_MAX))

    def step(self, action: int):
        action = int(action)
        rot_i, fire = divmod(action, 2)

        # ---- 1. 炮塔转动：角速度受限，单步最多转 DTHETA_MAX ----
        dtheta = C.ROT_LEVELS[rot_i] * C.DTHETA_MAX
        new_theta = float(np.clip(self.theta + dtheta, C.ANGLE_MIN, C.ANGLE_MAX))
        self.omega = (new_theta - self.theta) / C.DT
        self.theta = new_theta

        self.cooldown = max(0, self.cooldown - 1)

        # ---- 2. 无人机随机动力学（OU 过程 + 低频阵风 + 竖直扰动） ----
        self.vx = C.OU_DECAY * self.vx + float(
            self.rng.normal(0.0, C.OU_SIGMA * self.difficulty))
        self.gust_phase += C.DT * self.gust_freq
        self.vx += 0.05 * self.v_d * np.sin(self.gust_phase)
        vxm = C.VX_MAX * self.difficulty
        self.vx = float(np.clip(self.vx, -vxm, vxm))

        self.vy = -self.v_d + 0.92 * (self.vy + self.v_d) + float(
            self.rng.normal(0.0, C.VY_JITTER * 0.5 * self.v_d))
        self.vy = min(self.vy, -0.35 * self.v_d)     # 始终向下，意图撞击底部

        # ---- 3. 开火 ----
        n_fired = 0
        can_fire = self.cooldown <= 0 and len(self.bullets) < C.MAX_BULLETS
        if fire and can_fire:
            vx_b = self.v_b * np.cos(self.theta)
            vy_b = self.v_b * np.sin(self.theta)
            muzzle = 0.018
            shot = np.array([[
                C.GUN_X + muzzle * np.cos(self.theta),
                C.GUN_Y + muzzle * np.sin(self.theta),
                vx_b, vy_b, C.BULLET_LIFETIME,
            ]], dtype=np.float32)
            self.bullets = np.vstack([self.bullets, shot]) if len(self.bullets) else shot
            self.n_fired += 1
            n_fired = 1
            self.cooldown = C.FIRE_COOLDOWN_STEPS

        # ---- 4. 子步物理积分 + 连续碰撞检测 ----
        h = C.H_SUB
        m = C.DRONE_X_MARGIN
        hit = False
        for _ in range(C.SUBSTEPS):
            self.x += self.vx * h
            self.y += self.vy * h
            if self.x < m:
                self.x = 2 * m - self.x
                self.vx = abs(self.vx)
            elif self.x > C.W - m:
                self.x = 2 * (C.W - m) - self.x
                self.vx = -abs(self.vx)

            if len(self.bullets):
                b = self.bullets
                b[:, 0] += b[:, 2] * h
                b[:, 1] += b[:, 3] * h
                b[:, 4] -= h
                d2 = (b[:, 0] - self.x) ** 2 + (b[:, 1] - self.y) ** 2
                if np.any(d2 <= C.R_HIT ** 2):
                    hit = True
                keep = ((b[:, 0] > -0.02) & (b[:, 0] < C.W + 0.02) &
                        (b[:, 1] > -0.02) & (b[:, 1] < C.H + 0.02) &
                        (b[:, 4] > 0))
                self.bullets = b[keep]

            if hit:
                break
            if self.y <= C.GUN_Y:
                self.failed = True
                break

        self.steps += 1

        # ---- 5. 奖励：三级惩罚层级 ----
        danger = float(np.clip(1.0 - self.y / C.H, 0.0, 1.0))
        reward = -C.DANGER_COEF * danger ** 2 - C.BULLET_COST * n_fired
        done = False
        event = None
        if hit:
            reward += C.R_SUCCESS
            done, self.success, event = True, True, "hit"
        elif self.failed or self.steps >= C.MAX_STEPS:
            reward += C.R_FAIL
            done, self.success, event = True, False, "fail"

        self.last_event = event
        self._snapshot(event)
        info = {
            "success": self.success,
            "hit": hit,
            "bullets": self.n_fired,
            "steps": self.steps,
            "danger": danger,
            "can_fire": self.cooldown <= 0,
        }
        return self.state(), float(reward), done, info

    # ------------------------------------------------------------ 可视化快照
    def _snapshot(self, event):
        if not self.record:
            return
        self.frames.append({
            "x": float(self.x), "y": float(self.y),
            "vx": float(self.vx), "vy": float(self.vy),
            "theta": float(self.theta),
            "bullets": self.bullets[:, :2].copy(),
            "event": event,
            "n_fired": int(self.n_fired),
            "step": int(self.steps),
        })

    def pop_frames(self) -> list[dict]:
        f = self.frames
        self.frames = []
        return f


if __name__ == "__main__":
    # 环境冒烟测试：随机策略跑 3 回合，检查维度、终止条件、奖励层级
    env = GunDefenseEnv(seed=1, record=True)
    for ep in range(3):
        s, _ = env.reset()
        total = 0.0
        while True:
            s2, r, done, info = env.step(int(env.rng.integers(0, C.N_ACTIONS)))
            total += r
            s = s2
            if done:
                break
        print(f"ep{ep}: steps={info['steps']:3d}  success={info['success']}  "
              f"return={total:8.2f}  bullets={info['bullets']:3d}  "
              f"frames={len(env.frames)}")
        env.pop_frames()
    print("state_dim =", s.shape[0], " n_actions =", C.N_ACTIONS,
          " v_bullet/v_drone =", C.SPEED_RATIO)
