"""
train.py —— 训练主程序

流程：
  1) 解析式专家生成演示轨迹，播种"演示回放池"，并对演示样本施加 DQfD 大间隔损失；
  2) 课程学习：前 35% 回合无人机略慢，之后恢复满难度（速度比恒为 16）；
  3) Double DQN 训练：ε 按回合进度从 1.0 衰减到 0.02；学习率余弦衰减 1e-3→1e-4；
  4) 每 100 回合做 200 回合纯贪心评估，保留 Top-K 检查点，训练结束后再做
     500 回合锦标赛复选，保存泛化最好的那组权重（标准 early-stopping / 模型选择）；
  5) 输出 training_log.csv / convergence_curves.png / dqn_model.npz / train_summary.json。

用法（PyCharm 直接运行本文件，或命令行）：
    python train.py                      # 完整训练
    python train.py --episodes 300      # 快速冒烟测试
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from agent import DQNAgent
from environment import GunDefenseEnv
from expert import expert_action
from viz_utils import (BLUE, GREEN, MUTED, ORANGE, PURPLE, RED, TEAL,
                       moving_average, setup_font, style_axis)

TRAIN_SUMMARY = C.OUT_DIR / "train_summary.json"


def collect_demonstrations(agent: DQNAgent, n_episodes: int, seed: int) -> float:
    """用带提前量的解析专家采集演示，写入 agent.demo（永不被普通经验覆盖）。"""
    env = GunDefenseEnv(seed=seed + 901, difficulty=1.0)
    rng = np.random.default_rng(seed + 902)
    wins = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        while True:
            a_expert = expert_action(env, fire_threshold=0.08)
            # 少量随机执行动作只为扩大状态覆盖；DQfD 标签始终用 a_expert
            a = (int(rng.integers(0, C.N_ACTIONS))
                 if rng.random() < C.DEMO_NOISE else a_expert)
            s2, r, done, info = env.step(a)
            agent.store(s, a, r, s2, done, demo=True, expert_a=a_expert)
            s = s2
            if done:
                break
        wins += int(info["success"])
    return wins / max(1, n_episodes)


def greedy_eval(agent: DQNAgent, env: GunDefenseEnv, n: int) -> dict:
    """纯贪心（ε=0）评估，用于收敛曲线与检查点选择。"""
    env.difficulty = 1.0
    wins, bullets, steps = 0, [], []
    for _ in range(n):
        s, _ = env.reset()
        while True:
            s2, _, done, info = env.step(agent.act(s, 0.0, greedy=True))
            s = s2
            if done:
                break
        wins += int(info["success"])
        bullets.append(info["bullets"])
        steps.append(info["steps"])
    return {"success": wins / n, "bullets": float(np.mean(bullets)),
            "steps": float(np.mean(steps))}


def cosine_lr(progress: float) -> float:
    """学习率余弦衰减，训练后期趋于稳定，抑制策略漂移。"""
    return C.LR_MIN + 0.5 * (C.LR - C.LR_MIN) * (1.0 + np.cos(np.pi * progress))


def train(episodes: int = C.EPISODES, seed: int = C.SEED,
          curriculum: bool | None = None, verbose: bool = True) -> dict:
    setup_font()
    curriculum = C.CURRICULUM_ON if curriculum is None else curriculum
    t0 = time.time()

    agent = DQNAgent(seed=seed)
    demo_sr = collect_demonstrations(agent, C.N_DEMO_EPISODES, seed)

    env = GunDefenseEnv(seed=seed + 100)
    eval_env = GunDefenseEnv(seed=seed + 200)
    rows, global_step = [], 0
    candidates: list[dict] = []

    if verbose:
        print(f"[配置] 训练回合={episodes}  状态={C.STATE_DIM}维  "
              f"动作={C.N_ACTIONS}  弹速/机速={C.SPEED_RATIO}x  "
              f"ε {C.EPS_START} -> {C.EPS_FINAL}  "
              f"DQfD margin={C.DQFD_MARGIN}")
        print(f"[演示] 专家演示 {C.N_DEMO_EPISODES} 回合，成功率 {demo_sr:.2%}，"
              f"演示池 {len(agent.demo)} 条")

    for ep in range(episodes):
        progress = ep / max(1, episodes - 1)
        if curriculum:
            frac = min(1.0, ep / max(1, C.CURRICULUM_FRAC * episodes))
            env.difficulty = C.CURRICULUM_START + (1 - C.CURRICULUM_START) * frac
        else:
            env.difficulty = 1.0

        eps_now = agent.epsilon(progress)
        lr_now = cosine_lr(progress)
        s, _ = env.reset()
        ep_return, losses, tds, mlosses = 0.0, [], [], []
        while True:
            a = agent.act(s, eps_now)
            s2, r, done, info = env.step(a)
            agent.store(s, a, r, s2, done)
            out = agent.learn(lr=lr_now)
            if out is not None:
                losses.append(out[0])
                tds.append(out[1])
                mlosses.append(out[2])
            ep_return += r
            global_step += 1
            s = s2
            if done:
                break

        eval_sr = np.nan
        if (ep + 1) % C.EVAL_EVERY == 0 or ep == episodes - 1:
            st = greedy_eval(agent, eval_env, C.EVAL_EPISODES)
            eval_sr = st["success"]
            candidates.append({
                "ep": ep + 1, "sr": st["success"], "bullets": st["bullets"],
                "params": agent.online.params(),
            })
            candidates.sort(key=lambda z: (-z["sr"], z["bullets"]))
            del candidates[C.CHECKPOINT_TOPK:]

        rows.append({
            "episode": ep + 1,
            "global_step": global_step,
            "return": round(ep_return, 4),
            "success": int(info["success"]),
            "steps": info["steps"],
            "bullets": info["bullets"],
            "epsilon": round(eps_now, 6),
            "lr": round(lr_now, 8),
            "loss": round(float(np.mean(losses)) if losses else 0.0, 6),
            "td_error": round(float(np.mean(tds)) if tds else 0.0, 6),
            "margin_loss": round(float(np.mean(mlosses)) if mlosses else 0.0, 6),
            "difficulty": round(env.difficulty, 4),
            "eval_success": "" if np.isnan(eval_sr) else round(eval_sr, 6),
        })

        if verbose and ((ep + 1) % 100 == 0 or ep == 0):
            sr = float(np.mean([z["success"] for z in rows[-100:]]))
            print(f"  ep {ep + 1:5d}/{episodes} | 步数 {global_step:7d} | "
                  f"ε {eps_now:.3f} | lr {lr_now:.2e} | "
                  f"近100成功率 {sr:6.2%} | "
                  f"贪心评估 {('%.1f%%' % (eval_sr * 100)) if not np.isnan(eval_sr) else '  --  '} | "
                  f"回报 {ep_return:8.2f} | {time.time() - t0:6.1f}s")

    # ---- 检查点锦标赛：用另一条随机种子流做 500 回合复选 ----
    best_score, best_params, best_stat, best_ep = None, None, None, -1
    for cand in candidates:
        agent.online.set_params(cand["params"])
        agent.target.hard_sync(agent.online)
        # 每个候选用同一批 500 回合（同一随机种子），保证公平比较
        sel_env = GunDefenseEnv(seed=seed + 300, difficulty=1.0)
        st = greedy_eval(agent, sel_env, C.CHECKPOINT_EVAL)
        score = (st["success"], -st["bullets"])
        if best_score is None or score > best_score:
            best_score, best_params, best_stat, best_ep = (
                score, cand["params"], st, cand["ep"])
        if verbose:
            print(f"  [候选 ep {cand['ep']:5d}] 评估 {cand['sr']:.2%} -> "
                  f"复选 {st['success']:.2%}, 平均耗弹 {st['bullets']:.1f}")
    agent.online.set_params(best_params)
    agent.target.hard_sync(agent.online)

    final_eps = agent.epsilon(1.0)
    final_eval = greedy_eval(agent, eval_env, 200)
    agent.save(C.MODEL_PATH, meta={
        "episodes": episodes, "global_step": global_step,
        "epsilon": final_eps, "seed": seed,
        "checkpoint_ep": best_ep,
    })

    with open(C.LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_curves(rows, checkpoint_ep=best_ep)

    summary = {
        "episodes": episodes, "global_steps": global_step,
        "selected_checkpoint_ep": best_ep,
        "final_epsilon": final_eps, "demo_success": demo_sr,
        "checkpoint_tournament": best_stat,
        "final_greedy_eval_200": final_eval,
        "elapsed_sec": round(time.time() - t0, 1),
        "reward_hierarchy": {
            "level1_miss": abs(C.R_FAIL),
            "level2_danger_worst": round(
                C.DANGER_COEF * (C.H / (C.V_DRONE * C.DT)) / 3.0, 2),
            "level3_bullets_typical": round(C.BULLET_COST * 10, 2),
        },
    }
    TRAIN_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    if verbose:
        print(f"[完成] 模型 {C.MODEL_PATH.name} | 曲线 {C.CURVE_PATH.name}")
        print(f"[指标] 复选成功率 {best_stat['success']:.2%}  "
              f"最终 ε={final_eps:.4f} (<0.03)  "
              f"平均耗弹 {best_stat['bullets']:.1f}  用时 {summary['elapsed_sec']}s")
    return summary


def plot_curves(rows: list[dict], checkpoint_ep: int | None = None):
    """6 面板收敛过程曲线；标注 best-so-far 包络与最终选中的检查点。"""
    setup_font()
    ep = np.array([z["episode"] for z in rows], float)
    ret = np.array([z["return"] for z in rows], float)
    succ = np.array([z["success"] for z in rows], float)
    eps = np.array([z["epsilon"] for z in rows], float)
    loss = np.array([z["loss"] for z in rows], float)
    margin = np.array([z["margin_loss"] for z in rows], float)
    bullets = np.array([z["bullets"] for z in rows], float)
    steps = np.array([z["steps"] for z in rows], float)
    ev_ep = np.array([float(z["episode"]) for z in rows if z["eval_success"] != ""])
    ev_sr = np.array([float(z["eval_success"]) for z in rows if z["eval_success"] != ""])
    best_so_far = np.maximum.accumulate(ev_sr) if len(ev_sr) else np.array([])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    fig.patch.set_facecolor("white")
    ax = axes.ravel()

    a = ax[0]
    a.plot(ep, ret, color=BLUE, alpha=0.16, linewidth=0.6)
    a.plot(ep, moving_average(ret, 50), color=BLUE, linewidth=2, label="滑动平均(50)")
    a.axhline(0, color=MUTED, linestyle="--", linewidth=0.8)
    style_axis(a, "① 回合累积回报 Return", "训练回合", "回报")
    a.legend(fontsize=8, frameon=False)

    a = ax[1]
    a.plot(ep, moving_average(succ, 100), color=ORANGE, linewidth=1.8,
           label="训练回合成功率(MA100)")
    a.plot(ev_ep, ev_sr, color=RED, linewidth=1.6, marker="o", markersize=2.5,
           alpha=0.75, label="贪心评估成功率(ε=0)")
    if len(ev_sr):
        a.step(ev_ep, best_so_far, where="post", color=GREEN, linewidth=2.2,
               label="历史最佳包络(best-so-far)")
    if checkpoint_ep:
        a.axvline(checkpoint_ep, color=PURPLE, linestyle=":", linewidth=1.6,
                  label=f"最终采用检查点 ep{checkpoint_ep}")
    a.axhline(1.0, color=GREEN, linestyle="--", linewidth=1.1, label="目标 100%")
    a.set_ylim(-0.03, 1.06)
    style_axis(a, "② 拦截成功率收敛", "训练回合", "成功率")
    a.legend(fontsize=8, frameon=False, loc="lower right")

    a = ax[2]
    a.plot(ep, eps, color=PURPLE, linewidth=2)
    a.axhline(0.03, color=RED, linestyle="--", linewidth=1.2, label="要求上限 0.03")
    a.set_ylim(-0.02, 1.06)
    style_axis(a, "③ 探索率 ε 衰减", "训练回合", "ε")
    a.legend(fontsize=8, frameon=False)

    a = ax[3]
    a.plot(ep, loss, color=TEAL, alpha=0.16, linewidth=0.6)
    a.plot(ep, moving_average(loss, 50), color=TEAL, linewidth=2, label="Huber TD 损失")
    a.plot(ep, moving_average(margin, 50), color=PURPLE, linewidth=1.6,
           label="DQfD 演示边际损失")
    style_axis(a, "④ 训练损失", "训练回合", "loss")
    a.legend(fontsize=8, frameon=False)

    a = ax[4]
    a.plot(ep, bullets, color=GREEN, alpha=0.16, linewidth=0.6)
    a.plot(ep, moving_average(bullets, 50), color=GREEN, linewidth=2)
    style_axis(a, "⑤ 每回合子弹消耗（经济性）", "训练回合", "发/回合")

    a = ax[5]
    a.plot(ep, steps, color=ORANGE, alpha=0.16, linewidth=0.6)
    a.plot(ep, moving_average(steps, 50), color=ORANGE, linewidth=2)
    style_axis(a, "⑥ 每回合决策步数（拦截效率）", "训练回合", "步/回合")

    fig.suptitle("Double DQN + DQfD 二维机枪拦截 · 收敛过程曲线    "
                 f"(弹速 {C.SPEED_RATIO:.0f}x机速 | gamma={C.GAMMA} | "
                 f"epsilon: {C.EPS_START:.1f} -> {C.EPS_FINAL:.2f} | "
                 f"惩罚层级: 漏防 {abs(C.R_FAIL):.0f} >> 危险 >> 弹药 {C.BULLET_COST})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(C.CURVE_PATH, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="训练 Double DQN 机枪拦截智能体")
    ap.add_argument("--episodes", type=int, default=C.EPISODES)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-curriculum", action="store_true")
    args = ap.parse_args()
    train(episodes=args.episodes, seed=args.seed,
          curriculum=False if args.no_curriculum else None)
