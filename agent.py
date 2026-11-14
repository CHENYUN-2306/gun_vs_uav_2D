"""
agent.py —— 纯 NumPy 实现的 Double DQN + DQfD 专家边际损失（不依赖 PyTorch）

包含：
  · QNetwork : 3 层 MLP，手写前向 / 反向传播 + Huber 损失 + Adam + 全局梯度裁剪
  · ReplayBuffer : 环形经验池（普通经验 / 专家演示两个独立池）
  · DQNAgent : ε-greedy、Double DQN 目标、目标网络 Polyak 软更新、模型存取

稳定性设计（关键）：
  DQfD(Hester et al.) 的大间隔分类损失：对演示样本要求
      Q(s, a_expert) >= Q(s, a_other) + margin
  它把策略锚定在已验证 100% 成功的解析专家附近，避免长训练后期
  出现的"策略漂移 / Q 值过估计 / 临近底部反而不开火"等退化现象。
  TD 损失仍负责学习真实的回报，ε-greedy 仍负责探索 —— 本质是 RL，
  专家数据只是辅助的正则项与回放池播种。

为什么手写网络？题目要求在 PyCharm 里即可运行，不引入深度学习框架，
只依赖 numpy / matplotlib / pillow。网络规模很小（13-64-64-10），
NumPy 的前反向传播比框架调用更快，且梯度完全可检（见 gradient_check）。
"""
from __future__ import annotations

import numpy as np

import config as C


# ----------------------------------------------------------------- 网络
class QNetwork:
    """q(s,a) 的函数逼近器：13 -> 64(ReLU) -> 64(ReLU) -> 10。"""

    KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")

    def __init__(self, in_dim=C.STATE_DIM, hidden=C.HIDDEN,
                 out_dim=C.N_ACTIONS, seed: int = 0):
        rng = np.random.default_rng(seed)

        def he(fan_in, fan_out):
            return rng.normal(0.0, np.sqrt(2.0 / fan_in),
                              size=(fan_in, fan_out)).astype(np.float32)

        self.p = {
            "W1": he(in_dim, hidden), "b1": np.zeros(hidden, np.float32),
            "W2": he(hidden, hidden), "b2": np.zeros(hidden, np.float32),
            "W3": he(hidden, out_dim), "b3": np.zeros(out_dim, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}   # Adam 一阶矩
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}   # Adam 二阶矩
        self.t = 0

    # ---------------------------------------------------------- 前向
    def forward(self, x: np.ndarray, cache: dict | None = None) -> np.ndarray:
        p = self.p
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0.0)
        q = a2 @ p["W3"] + p["b3"]
        if cache is not None:
            cache.update(x=x, z1=z1, a1=a1, z2=z2, a2=a2, q=q)
        return q

    def q_values(self, s: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(s, dtype=np.float32)[None, :])[0]

    # ---------------------------------------------------------- 反向（给定 dQ）
    def _grads_from_dq(self, cache: dict, dq: np.ndarray) -> dict:
        """从输出层梯度 dQ 反传到全部参数，并做全局范数裁剪。"""
        p = self.p
        dW3 = cache["a2"].T @ dq
        db3 = dq.sum(axis=0)
        da2 = dq @ p["W3"].T
        dz2 = da2 * (cache["z2"] > 0)
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2,
                 "b2": db2, "W3": dW3, "b3": db3}
        norm = float(np.sqrt(sum(np.sum(v * v) for v in grads.values())))
        scale = min(1.0, C.GRAD_CLIP / (norm + 1e-8))
        if scale < 1.0:
            grads = {k: v * scale for k, v in grads.items()}
        return grads

    def _backward(self, cache: dict, actions: np.ndarray,
                  targets: np.ndarray) -> tuple[dict, float, float]:
        """仅 TD(Huber) 损失的梯度，供 gradient_check 数值校验使用。"""
        q = cache["q"]
        B = q.shape[0]
        idx = np.arange(B)
        d = q[idx, actions] - targets
        absd = np.abs(d)
        delta = C.HUBER_DELTA
        loss = float(np.where(absd <= delta,
                              0.5 * d * d,
                              delta * (absd - 0.5 * delta)).mean())
        g = np.where(absd <= delta, d, delta * np.sign(d)) / B
        dq = np.zeros_like(q)
        dq[idx, actions] = g.astype(np.float32)
        return self._grads_from_dq(cache, dq), loss, float(np.mean(absd))

    # ---------------------------------------------------------- 参数更新
    def _adam(self, grads: dict, lr: float):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1 ** self.t
        bc2 = 1.0 - beta2 ** self.t
        for k in self.KEYS:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g * g)
            mh = self.m[k] / bc1
            vh = self.v[k] / bc2
            self.p[k] = (self.p[k] - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def update(self, s, a, y, demo_mask=None, expert_actions=None,
               lr: float = C.LR) -> tuple[float, float, float]:
        """一次梯度步 = TD(Huber) 损失 + 演示样本上的 DQfD 大间隔损失。"""
        cache: dict = {}
        q = self.forward(s, cache)
        B = q.shape[0]
        idx = np.arange(B)

        d = q[idx, a] - y
        absd = np.abs(d)
        delta = C.HUBER_DELTA
        loss = float(np.where(absd <= delta,
                              0.5 * d * d,
                              delta * (absd - 0.5 * delta)).mean())
        dq = np.zeros_like(q)
        dq[idx, a] = (np.where(absd <= delta, d, delta * np.sign(d)) / B).astype(np.float32)

        margin_loss = 0.0
        if demo_mask is not None and np.any(demo_mask):
            dm = np.where(demo_mask)[0]                  # 演示样本下标
            # 用「真实专家动作」做标签，而不是演示中可能带噪声的实际执行动作
            ae = (a[dm] if expert_actions is None
                  else np.asarray(expert_actions, dtype=np.int64))
            m = np.full((len(dm), C.N_ACTIONS), C.DQFD_MARGIN, np.float32)
            rows = np.arange(len(dm))
            m[rows, ae] = 0.0
            a_star = np.argmax(q[dm] + m, axis=1)
            qd = q[dm]
            diff = (qd[rows, a_star] + m[rows, a_star]) - qd[rows, ae]
            margin_loss = float(np.maximum(diff, 0.0).mean())

            # hinge 次梯度：间隔已满足（a_star == a_expert）的样本梯度为 0。
            # 若对这些样本仍施加单向推力，Q 值会无界膨胀并引发后期策略漂移。
            viol = a_star != ae
            coef = np.float32(C.DQFD_WEIGHT / len(dm))
            if np.any(viol):
                dv, av, sv = dm[viol], ae[viol], a_star[viol]
                dq[dv, av] -= coef
                np.add.at(dq, (dv, sv), coef)

        grads = self._grads_from_dq(cache, dq)
        self._adam(grads, lr)
        return loss, float(np.mean(absd)), margin_loss

    # ---------------------------------------------------------- 目标网络
    def polyak_from(self, other: "QNetwork", tau: float = C.TAU):
        for k in self.KEYS:
            self.p[k] = ((1 - tau) * self.p[k] + tau * other.p[k]).astype(np.float32)

    def hard_sync(self, other: "QNetwork"):
        for k in self.KEYS:
            self.p[k] = other.p[k].copy()

    def params(self) -> dict:
        return {k: v.copy() for k, v in self.p.items()}

    def set_params(self, params: dict):
        self.p = {k: np.asarray(v, np.float32).copy() for k, v in params.items()}


# ----------------------------------------------------------------- 经验池
class ReplayBuffer:
    """环形数组经验池，避免 Python 对象开销。"""

    def __init__(self, capacity: int, state_dim: int = C.STATE_DIM):
        self.cap = int(capacity)
        self.s = np.zeros((self.cap, state_dim), np.float32)
        self.a = np.zeros(self.cap, np.int64)
        self.r = np.zeros(self.cap, np.float32)
        self.s2 = np.zeros((self.cap, state_dim), np.float32)
        self.done = np.zeros(self.cap, np.float32)
        self.ea = np.zeros(self.cap, np.int64)        # 专家动作标签（DQfD 用）
        self.n = 0
        self.ptr = 0

    def add(self, s, a, r, s2, done, expert_a=None):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.done[i] = float(done)
        self.ea[i] = a if expert_a is None else expert_a
        self.ptr = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, size=n)
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s2[idx], self.done[idx], self.ea[idx])

    def __len__(self):
        return self.n


# ----------------------------------------------------------------- 智能体
class DQNAgent:
    def __init__(self, seed: int = C.SEED):
        self.rng = np.random.default_rng(seed + 31)
        self.online = QNetwork(seed=seed)
        self.target = QNetwork(seed=seed + 1)
        self.target.hard_sync(self.online)
        self.replay = ReplayBuffer(C.REPLAY_CAP)
        self.demo = ReplayBuffer(C.DEMO_CAP)
        self.n_learn = 0

    # ---- 探索率：按训练回合进度线性衰减到 0.02（< 0.03） ----
    @staticmethod
    def epsilon(progress: float) -> float:
        """progress = 当前回合 / 总回合数，保证训练末期一定到达终值。"""
        frac = min(1.0, max(0.0, progress) / C.EPS_DECAY_FRAC)
        return float(C.EPS_FINAL + (C.EPS_START - C.EPS_FINAL) * (1 - frac))

    def act(self, s: np.ndarray, eps: float = 0.0, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < eps:
            return int(self.rng.integers(0, C.N_ACTIONS))
        return int(np.argmax(self.online.q_values(s)))

    def store(self, s, a, r, s2, done, demo: bool = False, expert_a=None):
        (self.demo if demo else self.replay).add(s, a, r, s2, done,
                                                expert_a=expert_a)

    def learn(self, lr: float = C.LR) -> tuple[float, float, float] | None:
        if len(self.replay) < C.WARMUP:
            return None
        n_demo = C.N_DEMO_BATCH if len(self.demo) >= C.N_DEMO_BATCH else 0
        n_replay = C.BATCH - n_demo
        rb = self.replay.sample(n_replay, self.rng)
        if n_demo:
            db = self.demo.sample(n_demo, self.rng)
            s = np.concatenate([rb[0], db[0]])
            a = np.concatenate([rb[1], db[1]])
            r = np.concatenate([rb[2], db[2]])
            s2 = np.concatenate([rb[3], db[3]])
            done = np.concatenate([rb[4], db[4]])
            ea = np.concatenate([rb[5], db[5]])
            demo_mask = np.zeros(C.BATCH, dtype=bool)
            demo_mask[n_replay:] = True
            expert_actions = ea[demo_mask]
        else:
            s, a, r, s2, done = rb[:5]
            demo_mask, expert_actions = None, None

        # Double DQN：在线网选动作，目标网评估，抑制 Q 值过估计
        next_q_online = self.online.forward(s2)
        best_a = np.argmax(next_q_online, axis=1)
        next_q_target = self.target.forward(s2)
        y = np.asarray(r + C.GAMMA * (1.0 - done) *
                       next_q_target[np.arange(len(a)), best_a], np.float32)

        loss, td, mloss = self.online.update(
            s, a, y, demo_mask=demo_mask,
            expert_actions=expert_actions, lr=lr)
        self.target.polyak_from(self.online)
        self.n_learn += 1
        return loss, td, mloss

    # ---- 模型存取 ----
    def save(self, path, meta: dict | None = None):
        d = {f"p_{k}": self.online.p[k] for k in QNetwork.KEYS}
        d.update({f"m_{k}": np.asarray(meta[k]) for k in (meta or {})})
        np.savez_compressed(path, **d)

    def load(self, path) -> dict:
        z = np.load(path)
        params = {k: z[f"p_{k}"] for k in QNetwork.KEYS}
        self.online.set_params(params)
        self.target.hard_sync(self.online)
        return {k[2:]: z[k].item() for k in z.files if k.startswith("m_")}


# ----------------------------------------------------------------- 梯度自检
def gradient_check(seed: int = 0, tol: float = 2e-4) -> float:
    """数值梯度 vs 解析梯度，验证手写反向传播正确性。"""
    rng = np.random.default_rng(seed)
    net = QNetwork(in_dim=5, hidden=7, out_dim=3, seed=seed)
    x = rng.normal(0, 1, (6, 5)).astype(np.float32)
    a = rng.integers(0, 3, size=6)
    y = rng.normal(0, 1, size=6).astype(np.float32)
    cache: dict = {}
    net.forward(x, cache)
    grads, _, _ = net._backward(cache, a, y)

    def loss_of():
        c: dict = {}
        q = net.forward(x, c)
        d = q[np.arange(6), a] - y
        return float(np.where(np.abs(d) <= 1, 0.5 * d * d,
                              np.abs(d) - 0.5).mean())

    worst = 0.0
    eps = 1e-3
    for key in QNetwork.KEYS:
        flat = net.p[key].ravel()
        gflat = grads[key].ravel()
        for j in range(min(12, flat.size)):
            old = flat[j]
            flat[j] = old + eps
            lp = loss_of()
            flat[j] = old - eps
            lm = loss_of()
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            den = max(1.0, abs(num), abs(gflat[j]))
            worst = max(worst, abs(num - gflat[j]) / den)
    assert worst < tol, f"梯度校验失败: 最大相对误差 {worst:.2e}"
    return worst


if __name__ == "__main__":
    w = gradient_check()
    print(f"手写反向传播梯度校验通过，最大相对误差 = {w:.2e}")
    ag = DQNAgent(seed=0)
    print("初始 ε =", ag.epsilon(0.0), " 终值 ε =", ag.epsilon(1.0),
          " 动作数 =", C.N_ACTIONS, " 状态维度 =", C.STATE_DIM)
