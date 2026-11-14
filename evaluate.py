"""
evaluate.py —— 训练后模型评估

两种口径都给出：
  · 部署口径 ε=0（纯贪心）：这是"拦截成功率"的正式指标；
  · 含探索口径 ε=ε_final：验证即使保留 <0.03 的探索率，性能也不塌。

同时校验题目要求的三级惩罚层级是否真的成立（按单回合累计最坏量级）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as C
from agent import DQNAgent
from environment import GunDefenseEnv


def reward_hierarchy_check() -> dict:
    """按单回合累计最坏量级验证：漏防 > 危险 > 弹药。"""
    T = C.H / (C.V_DRONE * C.DT)                       # 无人机从顶到底的决策步数
    l1 = abs(C.R_FAIL)                                  # 漏防（一次性）
    l2 = C.DANGER_COEF * T / 3.0                       # ∫(1-y/H)^2 dt 的离散近似
    l3 = C.BULLET_COST * C.MAX_BULLETS                 # 极限情况打满子弹
    l3_typical = C.BULLET_COST * 10                    # 典型 10 发
    ok = l1 > l2 > l3_typical
    return {
        "level1_miss": round(l1, 2),
        "level2_danger_worst": round(l2, 2),
        "level3_bullets_typical_10": round(l3_typical, 2),
        "level3_bullets_absolute_max": round(l3, 2),
        "ordering_valid": bool(ok and l1 > l2 > l3),
        "formula": "R = -200(漏防) - 0.30*(1-y/H)^2(每步危险) - 0.05*N_bullet(经济) + 100(命中)",
    }


def evaluate(model_path: Path = C.MODEL_PATH, n_episodes: int = C.EVAL_EPISODES_FINAL,
             seed: int = C.SEED, epsilon: float | None = None,
             greedy: bool = True) -> dict:
    agent = DQNAgent(seed=seed + 700)
    meta = agent.load(model_path)
    eps_final = float(meta.get("epsilon", C.EPS_FINAL))
    eps_used = 0.0 if greedy else (eps_final if epsilon is None else float(epsilon))

    env = GunDefenseEnv(seed=seed + 701, difficulty=1.0)
    wins, bullets, steps, returns, min_y = 0, [], [], [], []
    for _ in range(n_episodes):
        s, _ = env.reset()
        acc, lowest = 0.0, 1.0
        while True:
            a = agent.act(s, eps_used, greedy=greedy)
            s2, r, done, info = env.step(a)
            acc += r
            lowest = min(lowest, env.y)
            s = s2
            if done:
                break
        wins += int(info["success"])
        bullets.append(info["bullets"])
        steps.append(info["steps"])
        returns.append(acc)
        min_y.append(lowest)

    return {
        "episodes": n_episodes,
        "mode": "greedy (epsilon=0)" if greedy else f"epsilon-greedy (epsilon={eps_used})",
        "epsilon_used": eps_used,
        "model_final_epsilon": eps_final,
        "success_count": wins,
        "fail_count": n_episodes - wins,
        "success_rate": wins / n_episodes,
        "avg_bullets": float(np.mean(bullets)),
        "avg_steps": float(np.mean(steps)),
        "avg_return": float(np.mean(returns)),
        "lowest_drone_height": float(np.min(min_y)),
        "reward_hierarchy": reward_hierarchy_check(),
    }


if __name__ == "__main__":
    hr = reward_hierarchy_check()
    print("三级惩罚层级校验:", json.dumps(hr, ensure_ascii=False, indent=2))
    if not C.MODEL_PATH.exists():
        print("尚未找到模型，请先运行 python train.py")
    else:
        for greedy in (True, False):
            r = evaluate(greedy=greedy, n_episodes=500)
            print(f"[{r['mode']}] 成功率 {r['success_rate']:.2%} "
                  f"({r['success_count']}/{r['episodes']})  "
                  f"平均耗弹 {r['avg_bullets']:.1f}  平均步数 {r['avg_steps']:.1f}")
