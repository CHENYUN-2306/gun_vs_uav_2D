"""
main.py —— 一键入口：训练 → 评估 → 20 次拦截动画 → 汇总报告

    python main.py                 # 全套流程（约几分钟）
    python main.py --skip-train    # 复用已有模型，只重新评估 + 生成动画/报告
    python main.py --episodes 300  # 快速验证链路

产物统一放在 outputs/：
    dqn_model.npz            训练好的 Double DQN 权重
    training_log.csv         每回合日志
    convergence_curves.png   6 面板收敛过程曲线
    interception_20.gif      训练后 20 次成功拦截动画
    report.json              指标汇总
    report.html              一页式可视化报告（曲线 + 动画 + 指标）
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config as C


def write_html_report(report: dict):
    def card(label, value, note=""):
        return (f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="note">{note}</div></div>')

    ev = report["evaluation_greedy"]
    ev2 = report["evaluation_with_epsilon"]
    an = report["animation"]
    hr = ev["reward_hierarchy"]
    tr = report["training"]

    cards = "".join([
        card("拦截成功率", f'{ev["success_rate"]:.2%}',
             f'贪心策略 ε=0，{ev["episodes"]} 回合，失败 {ev["fail_count"]} 次'),
        card("最终探索率", f'{tr["final_epsilon"]:.3f}',
             "要求 < 0.03；由 1.0 线性衰减"),
        card("含探索成功率", f'{ev2["success_rate"]:.2%}',
             f'ε={ev2["epsilon_used"]:.3f}，{ev2["episodes"]} 回合'),
        card("动画拦截次数", f'{an["successful_interceptions"]} 次',
             f'{an["frames"]} 帧 / {an["duration_sec"]}s，全部成功={an["all_successful"]}'),
        card("平均耗弹", f'{ev["avg_bullets"]:.1f} 发',
             f'平均 {ev["avg_steps"]:.1f} 步完成拦截'),
        card("弹速 / 机速", f'{C.SPEED_RATIO:.0f} ×',
             f'v_bullet={C.SPEED_RATIO * C.V_DRONE:.2f}, v_drone={C.V_DRONE:.2f}'),
    ])

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in {
            "状态维度 / 动作数": f'{C.STATE_DIM} / {C.N_ACTIONS}（5 档角速度 × 开火/不开火）',
            "网络": f'{C.STATE_DIM}-{C.HIDDEN}-{C.HIDDEN}-{C.N_ACTIONS} MLP（纯 NumPy）',
            "算法": "Double DQN + 经验回放 + 目标网络软更新 + 演示回放池 + 课程学习",
            "折扣 γ / 学习率": f'{C.GAMMA} / {C.LR}',
            "训练回合 / 全局步数": f'{tr["episodes"]} / {tr["global_steps"]}',
            "最终采用检查点": f'第 {tr.get("selected_checkpoint_ep", "-")} 回合'
                          '（Top-K 检查点 + 500 回合锦标赛复选）',
            "① 漏防惩罚": f'{hr["level1_miss"]}（一次性，最严重）',
            "② 危险度惩罚": f'-{C.DANGER_COEF}×(1-y/H)² 每步，累计最坏 ≈ {hr["level2_danger_worst"]}',
            "③ 弹药惩罚": f'-{C.BULLET_COST}/发，典型 ≈ {hr["level3_bullets_typical_10"]}，'
                        f'极限 ≈ {hr["level3_bullets_absolute_max"]}',
            "层级校验": "通过 ✔ 漏防 > 危险 > 弹药" if hr["ordering_valid"] else "未通过",
            "物理积分": f'DT={C.DT}s，每步 {C.SUBSTEPS} 次子步碰撞检测，命中半径 {C.R_HIT}',
        }.items())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>强化学习二维机枪拦截 · 训练结果报告</title>
<style>
  :root {{ --bg:#ffffff; --panel:#f6f8fc; --line:#dde3ed; --text:#1f2937;
           --muted:#6b7280; --blue:#2563eb; --green:#059669; --red:#dc2626; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px 32px 48px; background:var(--bg); color:var(--text);
         font-family:"Microsoft YaHei","Noto Sans SC","Segoe UI",sans-serif; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  h2 {{ font-size:15px; margin:26px 0 12px; padding-left:10px;
        border-left:4px solid var(--blue); }}
  .sub {{ color:var(--muted); font-size:12.5px; margin-bottom:20px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:14px 16px; }}
  .label {{ font-size:11.5px; color:var(--muted); }}
  .value {{ font-size:24px; font-weight:700; margin:6px 0 3px; color:var(--blue); }}
  .note {{ font-size:11px; color:var(--muted); }}
  .grid {{ display:grid; grid-template-columns:1.25fr 1fr; gap:18px; align-items:start; }}
  @media (max-width:980px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .box {{ background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:14px; }}
  img {{ width:100%; display:block; border-radius:6px; background:#fff; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  td {{ padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }}
  td:first-child {{ color:var(--muted); width:34%; }}
  .ok {{ color:var(--green); font-weight:600; }}
</style></head><body>
<h1>基于 Double DQN 的二维平面机枪自主拦截 · 结果报告</h1>
<div class="sub">纯 NumPy 实现 · 仅依赖 numpy / matplotlib / pillow · 训练与可视化解耦 ·
生成时间随 main.py 运行更新</div>
<div class="cards">{cards}</div>

<h2>收敛过程曲线</h2>
<div class="box"><img src="convergence_curves.png" alt="收敛过程曲线"></div>

<h2>训练完成后的拦截动画（20 次成功拦截）</h2>
<div class="grid">
  <div class="box"><img src="interception_20.gif" alt="20 次拦截动画"></div>
  <div class="box">
    <table>{rows}</table>
    <p class="sub" style="margin-top:14px">
      动画只展示训练完成后的贪心策略，不含训练过程可视化；
      每次拦截均从无人机在顶部随机出现开始，到子弹命中结束。
    </p>
  </div>
</div>
</body></html>"""
    C.HTML_PATH.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="一键训练 + 评估 + 动画")
    ap.add_argument("--skip-train", action="store_true", help="复用已有模型")
    ap.add_argument("--episodes", type=int, default=C.EPISODES)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    if args.skip_train:
        if not C.MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到模型 {C.MODEL_PATH}，请先运行 python train.py")
        p = C.OUT_DIR / "train_summary.json"
        train_summary = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
            "episodes": 0, "global_steps": 0, "final_epsilon": C.EPS_FINAL}
    else:
        from train import train
        train_summary = train(episodes=args.episodes, seed=args.seed)

    from evaluate import evaluate
    from animate import make_animation
    from train import plot_curves

    # 用最终日志重绘一次曲线，保证 best-so-far 包络 / 检查点标注与当前模型一致
    if C.LOG_PATH.exists():
        with open(C.LOG_PATH, "r", encoding="utf-8-sig") as f:
            log_rows = list(csv.DictReader(f))
        with np.load(C.MODEL_PATH) as z:
            ckpt = int(z["m_checkpoint_ep"]) if "m_checkpoint_ep" in z.files else None
        plot_curves(log_rows, checkpoint_ep=ckpt)
        train_summary["selected_checkpoint_ep"] = ckpt

    greedy = evaluate(greedy=True, n_episodes=C.EVAL_EPISODES_FINAL, seed=args.seed)
    with_eps = evaluate(greedy=False, n_episodes=C.EVAL_EPISODES_FINAL, seed=args.seed)
    anim = make_animation(seed=args.seed)

    report = {
        "training": train_summary,
        "evaluation_greedy": greedy,
        "evaluation_with_epsilon": with_eps,
        "animation": anim,
        "requirements": {
            "success_rate_100pct": greedy["success_rate"] >= 1.0,
            "epsilon_below_0_03": train_summary["final_epsilon"] < 0.03,
            "twenty_interceptions": anim["successful_interceptions"] >= C.DEMO_INTERCEPTIONS,
        },
    }
    C.REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    (C.OUT_DIR / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(report)

    print("\n" + "=" * 68)
    print("最终交付校验")
    print("=" * 68)
    print(f"  拦截成功率(ε=0, {greedy['episodes']} 回合): {greedy['success_rate']:.2%} "
          f"[{greedy['success_count']}/{greedy['episodes']}]  -> "
          f"{'达标' if greedy['success_rate'] >= 1.0 else '未达标'}")
    print(f"  含探索成功率(ε={with_eps['epsilon_used']:.3f}): "
          f"{with_eps['success_rate']:.2%}")
    print(f"  最终探索率: {train_summary['final_epsilon']:.4f}  -> "
          f"{'达标 (<0.03)' if train_summary['final_epsilon'] < 0.03 else '未达标'}")
    print(f"  惩罚层级: 漏防 {greedy['reward_hierarchy']['level1_miss']} > "
          f"危险 {greedy['reward_hierarchy']['level2_danger_worst']} > "
          f"弹药 {greedy['reward_hierarchy']['level3_bullets_absolute_max']}  -> "
          f"{'达标' if greedy['reward_hierarchy']['ordering_valid'] else '未达标'}")
    print(f"  动画: {anim['successful_interceptions']} 次成功拦截, "
          f"{anim['frames']} 帧 -> {Path(anim['path']).name}")
    print(f"  报告: {C.HTML_PATH.name} / {C.REPORT_JSON.name} / {C.CURVE_PATH.name}")


if __name__ == "__main__":
    main()


