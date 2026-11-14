"""
expert.py —— 解析式"专家"策略

用途（不是替代 RL，而是 RL 的样本效率工具）：
  1) 生成少量演示轨迹，播种经验回放池（DQfD / Learning-from-Demonstrations 思路）；
  2) 作为训练早期的性能上界基线，便于判断 DQN 是否收敛。

专家做法：
  · 弹目交会时间近似 t_int ≈ d / (v_bullet + |vy|)（迎头相遇，二者相向而行）；
  · 用 t_int 预测无人机位置，得到"带提前量"的拦截方位角 phi_lead；
  · 在 5 档角速度中选一个使 |phi_lead - theta| 最小的档位；
  · 只有当转向后的残余角误差小于阈值才开火 —— 这体现"开火时机"。

最终策略仍由 DQN 从奖励中自主学习，专家数据只占每个 batch 的一小部分。
"""
from __future__ import annotations

import numpy as np

import config as C


def expert_action(env, fire_threshold: float = 0.20, noise: float = 0.0,
                  rng: np.random.Generator | None = None) -> int:
    """返回动作编号：rot_index * 2 + fire。"""
    phi_lead = env.bearing(lead=True)
    err = phi_lead - env.theta

    deltas = np.asarray(C.ROT_LEVELS, dtype=float) * C.DTHETA_MAX
    rot_i = int(np.argmin(np.abs(err - deltas)))
    residual = abs(err - deltas[rot_i])

    if noise > 0.0 and rng is not None:
        # 少量探索噪声，增加演示数据的状态覆盖
        if rng.random() < noise:
            return int(rng.integers(0, C.N_ACTIONS))

    fire = 1 if (residual < fire_threshold and env.cooldown <= 0) else 0
    return rot_i * 2 + fire


def run_expert_episodes(n_episodes: int, seed: int, difficulty: float = 1.0):
    """独立评估专家策略，返回 (成功率, 平均耗弹, 平均步数)。"""
    from environment import GunDefenseEnv

    env = GunDefenseEnv(seed=seed, difficulty=difficulty)
    rng = np.random.default_rng(seed + 777)
    wins, bullets, steps = 0, [], []
    for _ in range(n_episodes):
        env.reset()
        while True:
            _, _, done, info = env.step(expert_action(env, noise=0.0, rng=rng))
            if done:
                break
        wins += int(info["success"])
        bullets.append(info["bullets"])
        steps.append(info["steps"])
    return wins / n_episodes, float(np.mean(bullets)), float(np.mean(steps))


if __name__ == "__main__":
    sr, nb, ns = run_expert_episodes(300, seed=7)
    print(f"专家策略基线: 成功率={sr:.3%}  平均耗弹={nb:.1f}  平均步数={ns:.1f}")
