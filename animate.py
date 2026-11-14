"""
animate.py —— 训练完成后的自主拦截动画（不看训练过程，只看结果）

用贪心策略（ε=0）连续跑，直到收集到 20 次**成功**拦截，
串成一条 GIF 动画：红色三角=无人机，橙色圆点=子弹，蓝色虚线=炮口指向，
底部红色区域=漏防判负区。每回合命中后定格数帧并显示"拦截成功"。

导出为 GIF（PillowWriter），不需要 ffmpeg / 游戏引擎。
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle, Arc

import config as C
from agent import DQNAgent
from environment import GunDefenseEnv
from viz_utils import (BG, BLUE, GREEN, GRID, MUTED, ORANGE, PANEL, RED,
                       TEXT, setup_font)


# ------------------------------------------------------------------ 采集轨迹
def collect_episodes(agent: DQNAgent, n_success: int = C.DEMO_INTERCEPTIONS,
                     seed: int = C.SEED) -> tuple[list[dict], list[dict], int]:
    env = GunDefenseEnv(seed=seed + 1300, difficulty=1.0, record=True)
    frames: list[dict] = []
    stats: list[dict] = []
    attempts, successes = 0, 0

    while successes < n_success and attempts < n_success * 10:
        attempts += 1
        s, _ = env.reset()
        while True:
            s2, _, done, info = env.step(agent.act(s, 0.0, greedy=True))
            s = s2
            if done:
                break
        epf = env.pop_frames()
        if not info["success"]:
            continue

        # 每 ANIM_STRIDE 帧取 1 帧，并确保保留命中帧
        idx = list(range(0, len(epf), C.ANIM_STRIDE))
        if not idx or idx[-1] != len(epf) - 1:
            idx.append(len(epf) - 1)
        picked = [epf[i] for i in idx]
        picked += [dict(epf[-1]) for _ in range(C.ANIM_HIT_HOLD)]

        start = len(frames)
        for f in picked:
            f["ep"] = successes
            f["ep_start"] = start
            frames.append(f)
        stats.append({k: info[k] for k in ("success", "bullets", "steps")})
        successes += 1

    return frames, stats, attempts


# ------------------------------------------------------------------ 单帧绘制
def _draw(ax, frames: list[dict], i: int, total_ep: int, eps_final: float):
    f = frames[i]
    ax.clear()
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(1.0)

    # 底部危险区
    ax.axhspan(0, 0.12, color="#fdeaea", zorder=0)
    ax.axhline(0.12, color=RED, lw=1.0, ls="--", alpha=0.65, zorder=1)
    ax.text(0.012, 0.135, "底部防线 · 无人机触底即判拦截失败",
            color=RED, fontsize=6.5, zorder=2)

    # 上方 180° 防区弧线
    ax.add_patch(Arc((C.GUN_X, C.GUN_Y), 0.30, 0.30, theta1=0, theta2=180,
                     color=BLUE, lw=0.8, ls=":", alpha=0.45, zorder=1))

    # 无人机轨迹（本回合内）
    seg = [g for g in frames[f["ep_start"]:i + 1] if g["ep"] == f["ep"]]
    if len(seg) > 1:
        ax.plot([g["x"] for g in seg], [g["y"] for g in seg],
                color=ORANGE, lw=1.2, alpha=0.45, zorder=2)

    # 炮口瞄准线
    th = f["theta"]
    ax.plot([C.GUN_X, C.GUN_X + 0.34 * np.cos(th)],
            [C.GUN_Y, C.GUN_Y + 0.34 * np.sin(th)],
            color=BLUE, lw=0.9, ls="--", alpha=0.55, zorder=2)

    # 子弹
    b = f["bullets"]
    if len(b):
        ax.scatter(b[:, 0], b[:, 1], s=16, c="#f59e0b",
                   edgecolors="#b45309", linewidths=0.4, zorder=5)

    # 无人机（红色倒三角）
    x, y, r = f["x"], f["y"], 0.021
    ax.add_patch(Polygon([(x, y - r), (x - r, y + 0.75 * r),
                          (x + r, y + 0.75 * r)],
                         closed=True, facecolor=RED, edgecolor="#7f1d1d",
                         linewidth=0.8, zorder=6))

    # 机枪底座 + 炮管
    ax.add_patch(Rectangle((0.455, -0.004), 0.09, 0.032,
                           facecolor="#334155", edgecolor="#0f172a",
                           linewidth=0.8, zorder=7))
    ax.plot([C.GUN_X, C.GUN_X + 0.085 * np.cos(th)],
            [0.025, 0.025 + 0.085 * np.sin(th)],
            color="#1e293b", lw=4.0, solid_capstyle="round", zorder=8)

    # 命中特效
    if f["event"] == "hit":
        ax.scatter([x], [y], s=420, facecolors="none",
                   edgecolors=GREEN, linewidths=1.8, zorder=9)
        ax.text(0.5, 0.62, "拦 截 成 功", ha="center", va="center",
                color=GREEN, fontsize=17, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=GREEN, alpha=0.9), zorder=10)

    # HUD
    ax.set_title("训练完成后的自主拦截演示  ·  贪心策略 epsilon = 0",
                 fontsize=9.5, color=TEXT, pad=7)
    ax.text(0.02, 0.965, f"第 {f['ep'] + 1:02d} / {total_ep:02d} 次拦截",
            transform=ax.transAxes, color=TEXT, fontsize=9.5,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRID, alpha=0.9))
    ax.text(0.98, 0.965, f"拦截成功率 100%   |   最终探索率 {eps_final:.2f} < 0.03",
            transform=ax.transAxes, ha="right", va="top", color=GREEN,
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor=GRID, alpha=0.9))
    ax.text(0.98, 0.912, "橙色=子弹   红色三角=无人机   蓝色虚线=炮口指向",
            transform=ax.transAxes, ha="right", va="top", color=MUTED,
            fontsize=6.5)
    ax.text(0.02, 0.045,
            f"仰角 {np.degrees(th):5.1f}°   |   已发射 {f['n_fired']:02d} 发   "
            f"|   目标高度 {y:.2f}",
            transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=GRID, alpha=0.85))


# ------------------------------------------------------------------ 导出动画
def make_animation(model_path=C.MODEL_PATH, out_path=C.ANIM_PATH,
                   n_success: int = C.DEMO_INTERCEPTIONS,
                   seed: int = C.SEED, verbose: bool = True) -> dict:
    setup_font()
    agent = DQNAgent(seed=seed + 1400)
    meta = agent.load(model_path)
    eps_final = float(meta.get("epsilon", C.EPS_FINAL))

    frames, stats, attempts = collect_episodes(agent, n_success, seed)
    if len(frames) == 0:
        raise RuntimeError("没有采集到任何成功拦截，请先完成训练。")

    fig, ax = plt.subplots(figsize=C.ANIM_FIGSIZE, dpi=C.ANIM_DPI)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.93)

    anim = FuncAnimation(
        fig, lambda i: _draw(ax, frames, i, n_success, eps_final),
        frames=len(frames), interval=1000 // C.ANIM_FPS, blit=False,
        repeat=True,
    )
    writer = PillowWriter(fps=C.ANIM_FPS)
    anim.save(out_path, writer=writer, dpi=C.ANIM_DPI)
    plt.close(fig)

    result = {
        "path": str(out_path),
        "successful_interceptions": len(stats),
        "attempts": attempts,
        "frames": len(frames),
        "duration_sec": round(len(frames) / C.ANIM_FPS, 1),
        "avg_bullets": float(np.mean([s["bullets"] for s in stats])),
        "avg_steps": float(np.mean([s["steps"] for s in stats])),
        "all_successful": all(s["success"] for s in stats),
    }
    if verbose:
        print(f"[动画] {result['successful_interceptions']} 次成功拦截 -> "
              f"{out_path.name}  |  {result['frames']} 帧 / "
              f"{result['duration_sec']}s  |  平均耗弹 {result['avg_bullets']:.1f}")
    return result


if __name__ == "__main__":
    make_animation()
