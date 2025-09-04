#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from ..utils.config import ScoringConfig


def _l2n(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def _aggregate(sim_mat: np.ndarray, mode: str = "max", topk: int = 3) -> np.ndarray:
    if sim_mat.size == 0:
        return np.full((sim_mat.shape[0] if sim_mat.ndim else 0,), -1.0, dtype=np.float32)

    if mode == "max":
        return sim_mat.max(axis=1)
    elif mode == "mean":
        return sim_mat.mean(axis=1)
    elif mode == "mean_topk":
        k = min(topk, sim_mat.shape[1])
        if k <= 0:
            return np.full((sim_mat.shape[0],), -1.0, dtype=np.float32)
        part = np.partition(sim_mat, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    else:
        return sim_mat.max(axis=1)


def _to_matrix(X: np.ndarray, D: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X.reshape(0, (D if D is not None else 0))
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if D is not None and X.shape[1] != D:
        raise ValueError(f"Dim mismatch: expected D={D}, got {X.shape}")
    return X


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    if A.ndim == 1: A = A.reshape(1, -1)
    elif A.ndim > 2: A = A.reshape(A.shape[0], -1)
    if B.ndim == 1: B = B.reshape(1, -1)
    elif B.ndim > 2: B = B.reshape(B.shape[0], -1)

    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dim mismatch in _cosine_matrix: A{A.shape} vs B{B.shape}")

    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

    sims = A @ B.T
    sims = np.clip(sims, -1.0, 1.0).astype(np.float32)
    return sims


def _l2n_torch(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _cos(A: np.ndarray, B: np.ndarray):
    if A.size == 0 or B.size == 0:
        return np.full((A.shape[0], B.shape[0]), -1.0, dtype=np.float32)

    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)

    a_norm = np.linalg.norm(A, axis=1, keepdims=True)
    b_norm = np.linalg.norm(B, axis=1, keepdims=True).T 

    a_zero = (a_norm < 1e-6)
    b_zero = (b_norm < 1e-6)

    a_norm_safe = np.where(a_zero, 1.0, a_norm)
    b_norm_safe = np.where(b_zero, 1.0, b_norm)

    S = (A @ B.T) / (a_norm_safe * b_norm_safe) 
    S = np.clip(S, -1.0, 1.0)
    S = np.nan_to_num(S, nan=-1.0, posinf=-1.0, neginf=-1.0)

    if a_zero.any():
        S[a_zero[:, 0], :] = -1.0
    if b_zero.any():
        S[:, b_zero[0, :]] = -1.0
    return S.astype(np.float32, copy=False)

def _agg_scores(scores: np.ndarray, aggregation: str, topk: int = 3) -> np.ndarray:
    if aggregation == 'mean':
        return np.mean(scores, axis=1)
    elif aggregation == 'max':
        return np.max(scores, axis=1)
    elif aggregation == 'topk':
        k = min(topk, scores.shape[1])
        # take top-k mean
        return np.mean(np.partition(scores, -k, axis=1)[:, -k:], axis=1)
    else:
        raise ValueError(f'Unknown aggregation: {aggregation}')


def _safe_get_int(d: Dict[str, Any], key: str, default: int) -> int:
    v = (d or {}).get(key, default)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _safe_get_float(d: Dict[str, Any], key: str, default: float) -> float:
    v = (d or {}).get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _safe_get_bool(d: Dict[str, Any], key: str, default: bool) -> bool:
    v = (d or {}).get(key, default)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    # Handle common string cases
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ("1", "true", "yes", "y", "on"): return True
        if lv in ("0", "false", "no", "n", "off"): return False
        # fallback
        return default
    try:
        return bool(v)
    except Exception:
        return default

class ScoreCalculator:
    """
    Главные отличия от старой логики:
    1) работаем с сырыми косинусами [-1,1], не маппим в [0,1] до решения;
    2) для решения используем neg_for_decision=max(neg_raw,0) — отрицательные
       косинусы негативов не улучшают diff, а просто нулируются;
    3) консенсус-порог, переданный как 0..1 (как раньше), переводим во внутренний
       косинусный порог: thr_cos = 2*thr01 - 1.
    """

    def __init__(self, params: Dict[str, Any] | None = None, config: ScoringConfig | None = None):
        """Инициализирует калькулятор скоров.
        
        Args:
            params: Словарь параметров для конфигурации
            config: Готовая конфигурация скоринга
        """

        # приоритет: config > params > defaults (совместимость с вашим CLI)
        if config is not None:
            self.config = config
        else:
            if params is None:
                params = {}
            self.config = ScoringConfig(
                min_pos_score=_safe_get_float(params, 'min_positive_score', _safe_get_float(params, 'min_pos_score', 0.62)),      # в КОСИНУСАХ
                decision_threshold=_safe_get_float(params, 'decision_threshold', 0.06),  # тоже в косинусах
                class_separation=_safe_get_float(params, 'class_separation', 0.04),     # разница с 2-м лучшим
                neg_cap=_safe_get_float(params, 'neg_cap', 0.90),                  # ограничение сверху для neg_for_decision
                topk=_safe_get_int(params, 'topk', 5),
                consensus_k=_safe_get_int(params, 'consensus_k', 0),
                consensus_thr=_safe_get_float(params, 'consensus_thr', 0.45),       # ПРИНИМАЕМ 0..1, ниже переведём в косинус
                adaptive_ratio=_safe_get_float(params, 'adaptive_ratio', 0.85),     # оставлено для совместимости (ViT не нужен)
                adaptive_diff_floor=_safe_get_float(params, 'adaptive_diff_floor', 0.04),
                adaptive_trigger_pos_range=_safe_get_float(params, 'adaptive_trigger_pos_range', 0.20),
                adaptive_trigger_neg_range=_safe_get_float(params, 'adaptive_trigger_neg_range', 0.20),
                margin=_safe_get_float(params, 'score_margin', 0.00),               # дополнительный сдвиг порога diff
                ratio=_safe_get_float(params, 'score_ratio', 1.01),                 # p/(neg+eps) — оставлено
                confidence=_safe_get_float(params, 'score_confidence', 0.50),       # min визуальной уверенности (на вывод)
                allow_unknown=_safe_get_bool(params, 'allow_unknown', True),
                verbose=_safe_get_bool(params, 'verbose', True)
            )

        # alias для краткости
        self.min_pos_score = self.config.min_pos_score
        self.decision_threshold = self.config.decision_threshold
        self.class_separation = self.config.class_separation
        self.neg_cap = self.config.neg_cap
        self.topk = self.config.topk
        self.consensus_k = self.config.consensus_k
        self.consensus_thr_01 = self.config.consensus_thr  # 0..1 на входе
        self.adaptive_ratio = self.config.adaptive_ratio
        self.adaptive_diff_floor = self.config.adaptive_diff_floor
        self.adaptive_trigger_pos_range = self.config.adaptive_trigger_pos_range
        self.adaptive_trigger_neg_range = self.config.adaptive_trigger_neg_range
        self.margin = self.config.margin
        self.ratio = self.config.ratio
        self.confidence_min = self.config.confidence
        self.allow_unknown = self.config.allow_unknown
        self.verbose = self.config.verbose

        # переведём consensus_thr в косинус (-1..1)
        self.consensus_thr_cos = float(2 * self.consensus_thr_01 - 1.0)

        # ▼ Новый параметр: режим агрегации positive-скоров
        #   'max'       — как в тестовом скрипте (рекомендуется)
        #   'mean_topk' — среднее по top-k (k = self.topk)
        #   'mean'      — среднее по всем positive
        # Убираем зависимость от неопределённого 'detector'
        self.pos_agg = str((((params or {}).get('positive_aggregation')) or ((params or {}).get('pos_agg')) or 'max')).lower()
        if self.verbose:
            print(f"   ⚙️ POS_AGG_MODE = {self.pos_agg}")

    # ----- агрегаторы

    def _aggregate_positive(self, sims_pos: np.ndarray) -> np.ndarray:
        """
        Агрегируем сырые косинусы [-1,1] по positive-примерам.
        По умолчанию — 'max' (как в тестовом dinov3_similarity.py).
        """
        if sims_pos.size == 0 or sims_pos.shape[1] == 0:
            return np.zeros((sims_pos.shape[0] if sims_pos.ndim > 0 else 0,), dtype=np.float32)

        mode = self.pos_agg
        if mode in ("max", "top1"):
            pos = sims_pos.max(axis=1)
        elif mode in ("mean_topk", "topk"):
            k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
            if k > 1:
                part = np.partition(sims_pos, -k, axis=1)[:, -k:]
                pos = part.mean(axis=1)
            else:
                pos = sims_pos.max(axis=1)
        elif mode == "mean":
            pos = sims_pos.mean(axis=1)
        else:
            # fallback: старая формула (0.7*max + 0.3*mean_topk)
            maxv = sims_pos.max(axis=1)
            k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
            if k > 1:
                part = np.partition(sims_pos, -k, axis=1)[:, -k:]
                topk_mean = part.mean(axis=1)
            else:
                topk_mean = maxv
            pos = 0.7 * maxv + 0.3 * topk_mean

        return np.clip(pos.astype(np.float32), -1.0, 1.0)

    def _aggregate_negative(self, sims_neg: np.ndarray) -> np.ndarray:
        """Берём максимум по негативам (сырые косинусы, [-1,1])."""
        if sims_neg.size == 0 or sims_neg.shape[1] == 0:
            return np.zeros((sims_neg.shape[0] if sims_neg.ndim > 0 else 0,), dtype=np.float32)
        neg_raw = np.max(sims_neg, axis=1)
        neg_raw = np.clip(neg_raw, -1.0, 1.0).astype(np.float32)
        return neg_raw

    def _consensus_count(self, sims_pos_cls: np.ndarray) -> np.ndarray:
        """Сколько positive-примеров у класса дают косинус >= consensus_thr_cos."""
        if sims_pos_cls.size == 0:
            return np.zeros((0,), dtype=np.int32)
        return (sims_pos_cls >= self.consensus_thr_cos).sum(axis=1).astype(np.int32)

    # ----- основной мультикласс-скоринг

    def _aggregate_pos(self, sims_vec: np.ndarray):
        """Агрегация positive-скоров для ОДНОЙ маски по одному классу."""
        if sims_vec.size == 0:
            return -1.0
        s = np.sort(sims_vec)[::-1]
        if self.pos_agg == "max":
            return float(s[0])
        if self.pos_agg == "mean":
            return float(s.mean())
        if self.pos_agg == "mean_topk" and self.topk and self.topk > 0:
            k = min(self.topk, s.shape[0])
            return float(s[:k].mean())
        return float(s[0])

    def score_multiclass(self, mask_vecs: np.ndarray, q_pos: dict, q_neg: np.ndarray, online_negatives=None):
        """
        mask_vecs: (M,D)
        q_pos: {class: (K,D)}
        q_neg: (N,D) — может быть (0,D)

        Возвращает список решений по маскам.
        """
        M = mask_vecs.shape[0]
        decisions = []
        if M == 0 or not q_pos:
            return decisions

        # предсчёт по negative
        neg_scores = None
        if q_neg.size > 0:
            neg_scores = _cos(mask_vecs, q_neg).max(axis=1)  # (M,)
        else:
            # если нет отрицательных — отрицательный «скор» будем считать 0
            # (или можно -1.0 — выбирай стратегию)
            neg_scores = np.zeros((M,), dtype=np.float32)

        for i in range(M):
            mv = mask_vecs[i:i+1]  # (1,D)
            best_cls = None
            best_pos = -1.0

            for cls, Q in q_pos.items():
                sims = _cos(mv, Q)[0]    # (K,)
                pos = self._aggregate_pos(sims)
                if pos > best_pos:
                    best_pos = pos
                    best_cls = cls

            neg = float(neg_scores[i])
            diff = best_pos - neg
            accepted = (best_pos >= self.min_pos_score) and (diff >= self.decision_threshold)

            decisions.append({
                "mask_index": i,
                "class": best_cls,
                "pos": float(best_pos),
                "neg": float(neg),
                "diff": float(diff),
                "accepted": bool(accepted),
                "confidence": float(best_pos),  # или diff, если предпочитаешь
            })
        
        # Добавляем debug информацию
        debug = {
            "num_masks": M,
            "num_classes": len(q_pos),
            "has_negatives": q_neg.size > 0 if q_neg is not None else False
        }
        return decisions, debug

    # ----- бинарный режим (оставлен для совместимости/APIs)

    def score_and_decide(self, mask_vecs: np.ndarray, q_pos: np.ndarray, q_neg: np.ndarray
                         ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        mask_vecs = _to_matrix(mask_vecs)
        q_pos = _to_matrix(q_pos, mask_vecs.shape[1])
        q_neg = _to_matrix(q_neg, mask_vecs.shape[1])

        sims_pos = _cosine_matrix(mask_vecs, q_pos)  # [-1,1]
        pos_scores = self._aggregate_positive(sims_pos)  # [-1,1]

        if q_neg.shape[0] > 0:
            sims_neg = _cosine_matrix(mask_vecs, q_neg)   # [-1,1]
            neg_raw = self._aggregate_negative(sims_neg)  # [-1,1]
        else:
            neg_raw = np.zeros(mask_vecs.shape[0], dtype=np.float32)

        neg_for_decision = np.clip(np.maximum(neg_raw, 0.0), 0.0,
                                   (self.neg_cap if self.neg_cap is not None else 1.0)).astype(np.float32)

        accepted = []
        eps = 1e-8
        for i, (p, n) in enumerate(zip(pos_scores, neg_for_decision)):
            diff = p - n - self.margin
            ratio_ok = (p / (n + eps)) >= float(self.ratio) if n > 0 else (p >= self.min_pos_score)
            if (p >= self.min_pos_score) and ((diff >= self.decision_threshold) or ratio_ok) and (p > n):
                accepted.append(i)

        return accepted, pos_scores, neg_for_decision


def score_multiclass(
    mask_vecs: np.ndarray,                # [M, D], L2-норм. если нет — нормируем внутри
    q_pos: dict[str, np.ndarray],         # {cls: [N_c, D]} (уже L2-норм.)
    q_neg: np.ndarray | None,             # [K, D] или (0, D)
    *,
    pos_agg: str = "max",
    topk: int = 3,
    min_pos_score: float = 0.70,
    decision_threshold: float = 0.10,
    clamp_neg_to_zero: bool = True,
    verbose: bool = True,
):
    """
    Считает per-mask скор по каждому классу, выбирает лучший класс и решает принимать/отклонять.

    Возвращает:
      decisions: list[dict] с полями:
         {'class': str|None, 'pos': float, 'neg_raw': float, 'neg': float,
          'diff': float, 'accepted': bool}
      debug: dict с диагностикой (средние pos/neg и т.п.)
    """
    if mask_vecs.ndim == 1:
        mask_vecs = mask_vecs[None, :]
    mask_vecs = mask_vecs.astype(np.float32, copy=False)
    mask_vecs = _l2n(mask_vecs, axis=1)
    M, D = mask_vecs.shape

    # подготовим негативы
    if q_neg is None or q_neg.size == 0:
        # онлайн-негативы = позитивы других классов
        neg_pool = []
        for cls, Q in q_pos.items():
            if Q is not None and Q.size:
                neg_pool.append(Q)
        if neg_pool:
            q_neg_eff = _l2n(np.concatenate(neg_pool, axis=0), axis=1)
        else:
            q_neg_eff = np.zeros((0, D), dtype=np.float32)
        used_online_negs = True
    else:
        q_neg_eff = q_neg.astype(np.float32, copy=False)
        if q_neg_eff.ndim == 1:
            q_neg_eff = q_neg_eff[None, :]
        q_neg_eff = _l2n(q_neg_eff, axis=1)
        used_online_negs = False

    # считаем pos по классам
    per_class_pos = {}       # cls -> [M]
    per_class_best = {}      # cls -> {'scores':[M], 'details':...}
    all_pos_vals = []

    for cls, Q in q_pos.items():         # ВАЖНО: .items() → (cls, ndarray)
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]
        if Q.size == 0:
            if verbose:
                print(f"   📊 Класс '{cls}': 0 примеров")
            per_class_pos[cls] = np.full((M,), -1.0, dtype=np.float32)
            continue

        Q = _l2n(Q, axis=1)
        # косинусы: [M, N_c] = [M,D] @ [D,N_c]
        sim = mask_vecs @ Q.T
        pos_cls = _aggregate(sim, mode=pos_agg, topk=topk)
        per_class_pos[cls] = pos_cls
        all_pos_vals.extend(pos_cls.tolist())

        if verbose:
            # покажем все значения по маскам в стиле логов
            flat = sim.reshape(-1)
            cls_list = [f"{v:+.3f}" if v < 0 else f"{v:.3f}" for v in pos_cls]
            print(f"   📊 Класс '{cls}': {Q.shape[0]} примеров → скоры={cls_list[:8] + (['...'] if len(cls_list)>8 else [])}")

    # если совсем нет валидных классов
    if not per_class_pos:
        if verbose:
            print("   ❌ Нет валидных классов с эмбеддингами - возвращаем пустой результат")
        return [], {'pos_avg': 0.0, 'neg_avg': 0.0}

    # соберём per-mask лучший класс и pos
    best_cls = []
    best_pos = []
    for m in range(M):
        cls_scores = {cls: per_class_pos[cls][m] for cls in per_class_pos.keys()}
        cls = max(cls_scores, key=cls_scores.get)
        best_cls.append(cls)
        best_pos.append(cls_scores[cls])
        if verbose:
            # печать как в логах
            vals = ", ".join([f"{c}={cls_scores[c]:.3f}" for c in [cls]])
            print(f"     Маска {m}: [{vals}] → выбран {cls} ({cls_scores[cls]:.3f})")

    best_pos = np.array(best_pos, dtype=np.float32)

    # считаем «негативный» скор: max по всем нег-запросам
    if q_neg_eff.size > 0:
        neg_sim = mask_vecs @ q_neg_eff.T           # [M, K]
        neg_raw = neg_sim.max(axis=1)               # [M]
    else:
        neg_raw = np.zeros((M,), dtype=np.float32)  # если не из чего — нули

    neg = np.maximum(neg_raw, 0.0) if clamp_neg_to_zero else neg_raw
    diff = best_pos - neg

    pos_avg = float(np.mean(best_pos)) if len(best_pos) else 0.0
    neg_avg = float(np.mean(neg_raw)) if len(neg_raw) else 0.0
    if verbose:
        print(f"📊 Скоры мультикласса: pos_avg={pos_avg:.3f}, neg_avg={neg_avg:.3f}")

    # решение: принять маску?
    decisions = []
    for m in range(M):
        accepted = (best_pos[m] >= min_pos_score) and (diff[m] >= decision_threshold)
        decisions.append({
            'class': best_cls[m],
            'pos': float(best_pos[m]),
            'neg_raw': float(neg_raw[m]),
            'neg': float(neg[m]),
            'diff': float(diff[m]),
            'accepted': bool(accepted),
        })
        if verbose:
            sep = best_pos[m]  # «class separation» в твоих логах по сути сам pos
            cons = "1/1"       # консенсус теперь считается вне этой функции
            print(f"   Маска {m}: класс={best_cls[m]}, pos={best_pos[m]:.3f}, "
                  f"neg_raw={neg_raw[m]:+.3f}, neg={neg[m]:.3f}, "
                  f"diff={diff[m]:+.3f}, sep=+{sep:.3f}, cons={cons} → принято={accepted}")

    debug = {
        'pos_avg': pos_avg,
        'neg_avg': neg_avg,
        'used_online_negs': used_online_negs,
        'best_pos': best_pos,
        'neg_raw': neg_raw,
        'diff': diff,
    }
    return decisions, debug