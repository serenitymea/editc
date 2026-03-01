from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND = "numpy"
_torch   = None
_cupy   = None
_device  = None

def _init_backend() -> None:
    global _BACKEND, _torch, _cupy, _device

    try:
        import torch
        if torch.cuda.is_available():
            _torch   = torch
            _device  = torch.device("cuda:0")
            _BACKEND = "torch"
            cap = torch.cuda.get_device_properties(0)
            logger.info(
                "[gpu_kernels] Backend: PyTorch CUDA — %s (%.1f GB)",
                cap.name, cap.total_memory / 1e9,
            )
            return
        else:
            logger.info("[gpu_kernels] PyTorch available but no CUDA device.")
    except ImportError:
        pass

    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        _cupy    = cp
        _BACKEND = "cupy"
        dev = cp.cuda.Device(0)
        logger.info("[gpu_kernels] Backend: CuPy — device %d", dev.id)
        return
    except Exception:
        pass

    logger.info("[gpu_kernels] No GPU backend found — using NumPy CPU fallback.")
    _BACKEND = "numpy"


_init_backend()


def get_backend() -> str:
    return _BACKEND

def _to_gpu_f32(arr: np.ndarray):

    a = np.ascontiguousarray(arr, dtype=np.float32)
    if _BACKEND == "torch":
        return _torch.from_numpy(a).pin_memory().to(_device, non_blocking=True)
    elif _BACKEND == "cupy":
        return _cupy.asarray(a)
    return a


def _to_numpy(t) -> np.ndarray:
    if _BACKEND == "torch":
        return t.cpu().numpy()
    elif _BACKEND == "cupy":
        return _cupy.asnumpy(t)
    return np.asarray(t, dtype=np.float32)

def gpu_aggregate_segments(
    matrix: np.ndarray,       # [N, K] float32
    starts: np.ndarray,       # [B]    int32
    ends:   np.ndarray,       # [B]    int32
) -> np.ndarray:              # [B, K] float32

    B  = len(starts)
    N, K = matrix.shape

    if B == 0:
        return np.zeros((0, K), dtype=np.float32)

    lengths   = ends - starts          # [B]
    max_len   = int(lengths.max())

    if _BACKEND == "torch":
        return _torch_aggregate_segments(matrix, starts, ends, lengths, max_len, B, N, K)
    elif _BACKEND == "cupy":
        return _cupy_aggregate_segments(matrix, starts, ends, lengths, max_len, B, N, K)
    else:
        return _numpy_aggregate_segments(matrix, starts, ends, B, K)


def _torch_aggregate_segments(matrix, starts, ends, lengths, max_len, B, N, K):
    mat_g  = _to_gpu_f32(matrix)                          # [N, K]
    lens_g = _torch.from_numpy(lengths.astype(np.int32)).to(_device)  # [B]

    row_idx = _torch.zeros((B, max_len), dtype=_torch.long, device=_device)
    mask    = _torch.zeros((B, max_len), dtype=_torch.bool, device=_device)

    for b in range(B):

        l = int(lengths[b])
        if l > 0:
            row_idx[b, :l] = _torch.arange(starts[b], ends[b], device=_device)
            mask[b, :l]    = True

    gathered = mat_g[row_idx]                              # [B, max_len, K]
    gathered[~mask] = 0.0                                  # zero-pad

    sums   = gathered.sum(dim=1)                          # [B, K]
    counts = lens_g.float().clamp(min=1).unsqueeze(1)     # [B, 1]
    means  = sums / counts                                # [B, K]

    return _to_numpy(means)


def _cupy_aggregate_segments(matrix, starts, ends, lengths, max_len, B, N, K):
    mat_g = _to_gpu_f32(matrix)   # [N, K]

    row_idx = _cupy.zeros((B, max_len), dtype=_cupy.int32)
    mask    = _cupy.zeros((B, max_len), dtype=_cupy.bool_)

    for b in range(B):
        l = int(lengths[b])
        if l > 0:
            row_idx[b, :l] = _cupy.arange(starts[b], ends[b])
            mask[b, :l]    = True

    gathered = mat_g[row_idx]     # [B, max_len, K]
    gathered[~mask] = 0.0

    counts = _cupy.maximum(lengths.astype(_cupy.float32), 1)[:, None]
    means  = gathered.sum(axis=1) / counts

    return _to_numpy(means)


def _numpy_aggregate_segments(matrix, starts, ends, B, K):
    out = np.empty((B, K), dtype=np.float32)
    for b in range(B):
        seg = matrix[starts[b]:ends[b]]
        out[b] = seg.mean(axis=0) if len(seg) > 0 else matrix[starts[b]]
    return out

def gpu_diversity_penalties(
    feature_vector: np.ndarray,   # [K]
    clip_matrix:    np.ndarray,   # [E, K]
) -> np.ndarray:                  # [E] penalties

    if len(clip_matrix) == 0:
        return np.ones(0, dtype=np.float32)

    if _BACKEND == "torch":
        return _torch_diversity(feature_vector, clip_matrix)
    elif _BACKEND == "cupy":
        return _cupy_diversity(feature_vector, clip_matrix)
    else:
        return _numpy_diversity(feature_vector, clip_matrix)


def _torch_diversity(fv, cm):
    import torch.nn.functional as F  # type: ignore
    fv_g  = _to_gpu_f32(fv)          # [K]
    cm_g  = _to_gpu_f32(cm)          # [E, K]

    # Expand fv → [E, K]
    fv_exp = fv_g.unsqueeze(0).expand(cm_g.shape[0], -1)   # [E, K]
    sims   = F.cosine_similarity(cm_g, fv_exp, dim=1)       # [E]
    factors = 1.0 - 0.3 * sims.clamp(min=0.0)
    return _to_numpy(factors)


def _cupy_diversity(fv, cm):
    fv_g = _to_gpu_f32(fv)   # [K]
    cm_g = _to_gpu_f32(cm)   # [E, K]

    fv_norm = _cupy.linalg.norm(fv_g)
    cm_norms = _cupy.linalg.norm(cm_g, axis=1)              # [E]

    dots = cm_g @ fv_g                                       # [E]
    norm_prod = fv_norm * cm_norms                           # [E]

    sims = _cupy.where(norm_prod > 0, dots / norm_prod, _cupy.zeros_like(dots))
    factors = 1.0 - 0.3 * _cupy.maximum(0.0, sims)
    return _to_numpy(factors)


def _numpy_diversity(fv, cm):
    fv = fv.astype(np.float32)
    cm = cm.astype(np.float32)
    fv_norm  = np.linalg.norm(fv)
    cm_norms = np.linalg.norm(cm, axis=1)
    norm_prod = fv_norm * cm_norms
    sims = np.where(norm_prod > 0, (cm @ fv) / np.maximum(norm_prod, 1e-12), 0.0)
    return (1.0 - 0.3 * np.maximum(0.0, sims)).astype(np.float32)


def gpu_batch_motion_features(
    motion_scores: np.ndarray,   # [N] float32
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if _BACKEND == "torch":
        return _torch_motion_features(motion_scores, window_size)
    elif _BACKEND == "cupy":
        return _cupy_motion_features(motion_scores, window_size)
    else:
        return _numpy_motion_features(motion_scores, window_size)


def _torch_motion_features(ms, win):
    N  = len(ms)
    ms_g = _to_gpu_f32(ms)  # [N]

    pad = win
    ms_padded = _torch.nn.functional.pad(ms_g.unsqueeze(0).unsqueeze(0),
                                          (pad, pad), mode="reflect").squeeze()  # [N + 2*win]

    kernel = 2 * win + 1
    windows = ms_padded.unfold(0, kernel, 1)[:N]   # [N, kernel]

    means  = windows.mean(dim=1)                    # [N]
    maxs   = windows.max(dim=1).values              # [N]
    stds   = windows.std(dim=1)                     # [N]

    p90s   = _torch.quantile(windows, 0.90, dim=1)  # [N]

    ratios = maxs / (means + 1e-6)                  # [N]

    return (
        _to_numpy(means),
        _to_numpy(maxs),
        _to_numpy(p90s),
        _to_numpy(stds),
        _to_numpy(ratios),
    )


def _cupy_motion_features(ms, win):
    N   = len(ms)
    ms_g = _to_gpu_f32(ms)

    pad = win

    left  = _cupy.flip(ms_g[1 : pad + 1])
    right = _cupy.flip(ms_g[N - pad - 1 : N - 1])
    ms_padded = _cupy.concatenate([left, ms_g, right])  # [N + 2*win]

    kernel = 2 * win + 1

    shape   = (N, kernel)
    strides = (ms_padded.strides[0], ms_padded.strides[0])
    windows = _cupy.lib.stride_tricks.as_strided(ms_padded, shape=shape, strides=strides)

    windows = windows.copy()   # [N, kernel]

    means  = windows.mean(axis=1)
    maxs   = windows.max(axis=1)
    stds   = windows.std(axis=1)

    p90s   = _cupy.percentile(windows, 90, axis=1)
    ratios = maxs / (means + 1e-6)

    return (
        _to_numpy(means), _to_numpy(maxs),
        _to_numpy(p90s), _to_numpy(stds),
        _to_numpy(ratios),
    )


def _numpy_motion_features(ms, win):
    N = len(ms)
    pad = win
    ms_padded = np.pad(ms, pad, mode="reflect")  # [N + 2*win]
    kernel = 2 * win + 1
    shape   = (N, kernel)
    strides = (ms_padded.strides[0], ms_padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(ms_padded, shape=shape, strides=strides).copy()
    means  = windows.mean(axis=1)
    maxs   = windows.max(axis=1)
    stds   = windows.std(axis=1)
    p90s   = np.percentile(windows, 90, axis=1).astype(np.float32)
    ratios = maxs / (means + 1e-6)
    return means.astype(np.float32), maxs.astype(np.float32), p90s, stds.astype(np.float32), ratios.astype(np.float32)

def gpu_batch_temporal_context(
    motion_scores:      np.ndarray,   # [N]
    window:             int,
    audio_rms_contrast: np.ndarray,   # [N]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    momentum      = (last - first) / window_len  для каждого скользящего окна
    buildup       = (momentum + rms_contrast) * 0.5
    derivative    = max(0, ms[i] - ms[i-1])

    GPU: всё через tensor ops без python-цикла.
    """
    if _BACKEND == "torch":
        return _torch_temporal_context(motion_scores, window, audio_rms_contrast)
    elif _BACKEND == "cupy":
        return _cupy_temporal_context(motion_scores, window, audio_rms_contrast)
    else:
        return _numpy_temporal_context(motion_scores, window, audio_rms_contrast)


def _torch_temporal_context(ms, win, rms_contrast):
    N    = len(ms)
    ms_g = _to_gpu_f32(ms)
    rms_g = _to_gpu_f32(rms_contrast)

    pad     = win
    ms_pad  = _torch.nn.functional.pad(ms_g.unsqueeze(0).unsqueeze(0),
                                        (pad, pad), mode="reflect").squeeze()
    kernel  = 2 * win + 1
    windows = ms_pad.unfold(0, kernel, 1)[:N]  # [N, kernel]

    # momentum = (last - first) / kernel
    momentums = (windows[:, -1] - windows[:, 0]) / kernel   # [N]

    buildups  = (momentums + rms_g) * 0.5                   # [N]

    # derivative: diff[i] = ms[i] - ms[i-1], clamp negative
    ms_shifted = _torch.nn.functional.pad(ms_g[:-1].unsqueeze(0),
                                           (1, 0), mode="constant", value=0).squeeze()
    derivatives = _torch.clamp(ms_g - ms_shifted, min=0.0)  # [N]

    return _to_numpy(momentums), _to_numpy(buildups), _to_numpy(derivatives)


def _cupy_temporal_context(ms, win, rms_contrast):
    N   = len(ms)
    ms_g = _to_gpu_f32(ms)
    rms_g = _to_gpu_f32(rms_contrast)

    left  = _cupy.flip(ms_g[1 : win + 1])
    right = _cupy.flip(ms_g[N - win - 1 : N - 1])
    ms_pad = _cupy.concatenate([left, ms_g, right])

    kernel  = 2 * win + 1
    shape   = (N, kernel)
    strides = (ms_pad.strides[0], ms_pad.strides[0])
    windows = _cupy.lib.stride_tricks.as_strided(ms_pad, shape=shape, strides=strides).copy()

    momentums   = (windows[:, -1] - windows[:, 0]) / kernel
    buildups    = (momentums + rms_g) * 0.5
    shifted     = _cupy.concatenate([_cupy.zeros(1, dtype=_cupy.float32), ms_g[:-1]])
    derivatives = _cupy.maximum(0.0, ms_g - shifted)

    return _to_numpy(momentums), _to_numpy(buildups), _to_numpy(derivatives)


def _numpy_temporal_context(ms, win, rms_contrast):
    N  = len(ms)
    ms_pad = np.pad(ms, win, mode="reflect")
    kernel = 2 * win + 1
    shape   = (N, kernel)
    strides = (ms_pad.strides[0], ms_pad.strides[0])
    windows = np.lib.stride_tricks.as_strided(ms_pad, shape=shape, strides=strides).copy()
    momentums   = ((windows[:, -1] - windows[:, 0]) / kernel).astype(np.float32)
    buildups    = ((momentums + rms_contrast) * 0.5).astype(np.float32)
    shifted     = np.concatenate([[0.0], ms[:-1]])
    derivatives = np.maximum(0.0, ms - shifted).astype(np.float32)
    return momentums, buildups, derivatives

def gpu_batch_heuristic_scores(
    candidates:   np.ndarray,    # [C, K] float32
    weights_vec:  np.ndarray,    # [14] float32
    feat_indices: dict,
    blur_col:     int,
) -> np.ndarray:                 # [C] float32  scores in [0, 1]

    K = candidates.shape[1]
    C = len(candidates)

    def safe(col):
        if col >= 0 and col < K:
            return candidates[:, col].astype(np.float32)
        return np.zeros(C, dtype=np.float32)

    p90        = safe(feat_indices.get("motion_p90", -1))
    peak_ratio = safe(feat_indices.get("motion_peak_ratio", -1))
    beat       = safe(feat_indices.get("beat_alignment_score", -1))
    bass       = safe(feat_indices.get("bass_energy", -1))
    blur_inv   = 1.0 - safe(blur_col)
    buildup    = safe(feat_indices.get("combined_buildup", -1))
    rms_peak   = safe(feat_indices.get("rms_peak", -1))
    edge       = safe(feat_indices.get("edge_density", -1))
    col_var    = safe(feat_indices.get("color_variance", -1))
    m_std      = safe(feat_indices.get("motion_temporal_std",
                                       feat_indices.get("motion_std", -1)))
    p95_ratio  = safe(feat_indices.get("motion_p95_mean_ratio",
                                       feat_indices.get("motion_peak_ratio", -1)))
    rms_std    = safe(feat_indices.get("audio_rms_std", -1))
    edge_grad  = safe(feat_indices.get("edge_contrast_grad", -1))
    contrast   = safe(feat_indices.get("contrast_mean", -1))

    # Build [C, 14] feature matrix for linear dot
    feat_mat = np.stack([
        p90,
        peak_ratio  * 0.12,
        beat,
        bass,
        blur_inv,
        buildup,
        rms_peak,
        edge,
        col_var,
        np.zeros(C, dtype=np.float32),   # slot 9: m_std — penalty only
        p95_ratio   * 0.12,
        np.zeros(C, dtype=np.float32),   # slot 11: rms_std — penalty only
        edge_grad,
        contrast,
    ], axis=1).astype(np.float32)        # [C, 14]

    if _BACKEND == "torch":
        return _torch_nonlinear_scores(feat_mat, weights_vec, p90, peak_ratio,
                                        m_std, rms_std, rms_peak)
    elif _BACKEND == "cupy":
        return _cupy_nonlinear_scores(feat_mat, weights_vec, p90, peak_ratio,
                                       m_std, rms_std, rms_peak)
    else:
        return _numpy_nonlinear_scores(feat_mat, weights_vec, p90, peak_ratio,
                                        m_std, rms_std, rms_peak)


def _torch_nonlinear_scores(feat_mat, weights_vec, p90, peak_ratio,
                              m_std, rms_std, rms_peak):
    fm_g         = _to_gpu_f32(feat_mat)
    wv_g         = _to_gpu_f32(weights_vec.astype(np.float32))
    p90_g        = _to_gpu_f32(p90)
    pr_g         = _to_gpu_f32(peak_ratio)
    mstd_g       = _to_gpu_f32(m_std)
    rstd_g       = _to_gpu_f32(rms_std)
    rpk_g        = _to_gpu_f32(rms_peak)

    base = fm_g @ wv_g
    peak = (p90_g * pr_g.clamp(min=0.0)).clamp(min=0.0).pow(0.55)
    flat = _torch.exp(-mstd_g * 6.0) * _torch.exp(-rstd_g * 4.0)
    sat  = (rpk_g > 0.6).float() * _torch.exp(-rstd_g * 3.0)

    score = _torch.tanh(base * 2.2)
    score = score * (1.0 - 0.55 * flat)
    score = score * (1.0 - 0.30 * sat)
    score = score * (1.0 + 0.40 * peak)
    score = score.clamp(0.0, 1.0).pow(1.3)
    return _to_numpy(score)


def _cupy_nonlinear_scores(feat_mat, weights_vec, p90, peak_ratio,
                             m_std, rms_std, rms_peak):
    cp = _cupy
    fm_g   = _to_gpu_f32(feat_mat)
    wv_g   = _to_gpu_f32(weights_vec.astype(np.float32))
    p90_g  = _to_gpu_f32(p90)
    pr_g   = _to_gpu_f32(peak_ratio)
    mstd_g = _to_gpu_f32(m_std)
    rstd_g = _to_gpu_f32(rms_std)
    rpk_g  = _to_gpu_f32(rms_peak)

    base = fm_g @ wv_g
    peak = cp.maximum(0.0, p90_g * cp.maximum(0.0, pr_g)) ** 0.55
    flat = cp.exp(-mstd_g * 6.0) * cp.exp(-rstd_g * 4.0)
    sat  = (rpk_g > 0.6).astype(cp.float32) * cp.exp(-rstd_g * 3.0)

    score = cp.tanh(base * 2.2)
    score = score * (1.0 - 0.55 * flat)
    score = score * (1.0 - 0.30 * sat)
    score = score * (1.0 + 0.40 * peak)
    score = cp.clip(score, 0.0, 1.0) ** 1.3
    return _to_numpy(score)


def _numpy_nonlinear_scores(feat_mat, weights_vec, p90, peak_ratio,
                              m_std, rms_std, rms_peak):
    base = feat_mat @ weights_vec.astype(np.float32)
    peak = np.maximum(0.0, p90 * np.maximum(0.0, peak_ratio)) ** 0.55
    flat = np.exp(-m_std * 6.0) * np.exp(-rms_std * 4.0)
    sat  = (rms_peak > 0.6).astype(np.float32) * np.exp(-rms_std * 3.0)

    score = np.tanh(base * 2.2)
    score = score * (1.0 - 0.55 * flat)
    score = score * (1.0 - 0.30 * sat)
    score = score * (1.0 + 0.40 * peak)
    score = np.clip(score, 0.0, 1.0) ** 1.3
    return score.astype(np.float32)


def gpu_batch_overlap_penalties(
    cand_starts:  np.ndarray,   # [C] int32
    cand_ends:    np.ndarray,   # [C] int32
    clip_starts:  np.ndarray,   # [E] int32
    clip_ends:    np.ndarray,   # [E] int32
    clip_frames:  int,
    decay_ratio:  float,
) -> np.ndarray:                # [C] float32

    if len(clip_starts) == 0:
        return np.ones(len(cand_starts), dtype=np.float32)

    if _BACKEND == "torch":
        return _torch_batch_overlap(cand_starts, cand_ends, clip_starts, clip_ends, clip_frames, decay_ratio)
    elif _BACKEND == "cupy":
        return _cupy_batch_overlap(cand_starts, cand_ends, clip_starts, clip_ends, clip_frames, decay_ratio)
    else:
        return _numpy_batch_overlap(cand_starts, cand_ends, clip_starts, clip_ends, clip_frames, decay_ratio)


def _compute_overlap_matrix(cs, ce, es, ee, clip_frames, decay_ratio, lib, exp_fn):
    inv_decay = 1.0 / (decay_ratio * clip_frames)

    # Broadcast: [C, 1] vs [1, E]
    cs2 = cs[:, None].astype(float)
    ce2 = ce[:, None].astype(float)
    es2 = es[None, :].astype(float)
    ee2 = ee[None, :].astype(float)

    # overlap[c, e] = max(0, min(ce[c], ee[e]) - max(cs[c], es[e]))
    min_ends  = lib.minimum(ce2, ee2)
    max_starts = lib.maximum(cs2, es2)
    overlaps  = lib.maximum(0.0, min_ends - max_starts)   # [C, E]

    # penalty[c, e] = exp(-overlap * inv_decay)
    pen_mat = exp_fn(-overlaps * inv_decay)                 # [C, E]

    # product по E axis
    return pen_mat.prod(axis=-1)                            # [C]


def _torch_batch_overlap(cs, ce, es, ee, clip_frames, decay_ratio):
    cs_g = _torch.from_numpy(cs.astype(np.float32)).to(_device)
    ce_g = _torch.from_numpy(ce.astype(np.float32)).to(_device)
    es_g = _torch.from_numpy(es.astype(np.float32)).to(_device)
    ee_g = _torch.from_numpy(ee.astype(np.float32)).to(_device)

    inv_decay = 1.0 / (decay_ratio * clip_frames)

    min_ends   = _torch.minimum(ce_g[:, None], ee_g[None, :])
    max_starts = _torch.maximum(cs_g[:, None], es_g[None, :])
    overlaps   = _torch.clamp(min_ends - max_starts, min=0.0)
    pen_mat    = _torch.exp(-overlaps * inv_decay)
    penalties  = pen_mat.prod(dim=1)

    return _to_numpy(penalties)


def _cupy_batch_overlap(cs, ce, es, ee, clip_frames, decay_ratio):
    cs_g = _cupy.asarray(cs, dtype=_cupy.float32)
    ce_g = _cupy.asarray(ce, dtype=_cupy.float32)
    es_g = _cupy.asarray(es, dtype=_cupy.float32)
    ee_g = _cupy.asarray(ee, dtype=_cupy.float32)

    inv_decay  = 1.0 / (decay_ratio * clip_frames)
    min_ends   = _cupy.minimum(ce_g[:, None], ee_g[None, :])
    max_starts = _cupy.maximum(cs_g[:, None], es_g[None, :])
    overlaps   = _cupy.maximum(0.0, min_ends - max_starts)
    pen_mat    = _cupy.exp(-overlaps * inv_decay)
    penalties  = pen_mat.prod(axis=1)

    return _to_numpy(penalties)


def _numpy_batch_overlap(cs, ce, es, ee, clip_frames, decay_ratio):
    cs = cs.astype(np.float32)
    ce = ce.astype(np.float32)
    es = es.astype(np.float32)
    ee = ee.astype(np.float32)
    inv_decay  = 1.0 / (decay_ratio * clip_frames)
    min_ends   = np.minimum(ce[:, None], ee[None, :])
    max_starts = np.maximum(cs[:, None], es[None, :])
    overlaps   = np.maximum(0.0, min_ends - max_starts)
    pen_mat    = np.exp(-overlaps * inv_decay)
    return pen_mat.prod(axis=1)