import numpy as np
import torch


def ema_smooth_global_rot_per_obj_id_adaptive(
    mhr_dict,
    num_frames,
    frame_obj_ids,
    key_name="global_rot",
    alpha_strong=0.1,
    alpha_weak=0.3,
    motion_low=0.05,
    motion_high=0.30,
    empty_thresh=1e-6,
):
    if key_name not in mhr_dict:
        return mhr_dict

    rot = mhr_dict[key_name]
    device = rot.device
    B, D = rot.shape
    if D != 3:
        return mhr_dict

    any_key = next(iter(mhr_dict.keys()))
    B_check = mhr_dict[any_key].shape[0]
    assert B_check == B
    assert B % num_frames == 0
    num_humans = B // num_frames

    assert len(frame_obj_ids) == num_frames

    rot_np = rot.detach().cpu().numpy()

    # stronger adaptive hyper-params
    alpha_ultra      = 0.03
    alpha_strong_eff = 0.07
    alpha_weak_eff   = 0.18

    very_low_motion = 0.02
    low_motion      = 0.07
    high_motion     = 0.50  # <<<<<< updated (was 0.25)

    for obj_id in range(1, num_humans + 1):
        slot = obj_id - 1

        valid_indices = []
        for t in range(num_frames):
            if obj_id in frame_obj_ids[t]:
                valid_indices.append(t * num_humans + slot)

        if len(valid_indices) == 0:
            continue

        track = rot_np[valid_indices, :]
        if np.linalg.norm(track) < empty_thresh:
            continue

        T_valid = track.shape[0]
        if T_valid <= 1:
            continue

        diff = np.linalg.norm(track[1:] - track[:-1], axis=1)
        motion_level = float(np.median(diff))

        if motion_level < very_low_motion:
            alpha = alpha_ultra
        elif motion_level < low_motion:
            alpha = alpha_strong_eff
        elif motion_level < high_motion:
            alpha = alpha_weak_eff
        else:
            continue  # skip (still too dynamic)

        smooth = np.zeros_like(track, dtype=np.float32)
        smooth[0] = track[0]
        for i in range(1, T_valid):
            smooth[i] = alpha * track[i] + (1.0 - alpha) * smooth[i - 1]

        if not np.isfinite(smooth).all():
            continue

        rot_np[valid_indices, :] = smooth

    mhr_dict[key_name] = torch.from_numpy(rot_np).to(device)
    return mhr_dict


def kalman_smooth_constant_velocity_safe(Y, q_pos=1e-4, q_vel=1e-6, r_obs=1e-2):
    """
    Robust constant-velocity Kalman smoothing on (T, D).

    Y: (T, D) numpy array of valid observations for a single obj_id.
       Missing frames are handled outside this function.
    """
    Y = np.asarray(Y, dtype=np.float32)
    T, D = Y.shape
    if T == 0:
        return Y.copy()

    # Remove NaN / inf from input
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    q_pos = float(max(q_pos, 0.0))
    q_vel = float(max(q_vel, 0.0))
    r_obs = float(max(r_obs, 1e-12))

    # Initial state: first observation as position, zero velocity
    x = Y[0].copy()                   # (D,)
    v = np.zeros(D, dtype=np.float32) # (D,)

    Pxx = np.ones(D, dtype=np.float32)
    Pxv = np.zeros(D, dtype=np.float32)
    Pvv = np.ones(D, dtype=np.float32)

    X = np.zeros_like(Y)
    X[0] = x

    eps = 1e-8
    max_val = 1e6

    for t in range(1, T):
        # ---- Prediction ----
        x_pred = x + v
        v_pred = v

        Pxx_pred = Pxx + 2 * Pxv + Pvv + q_pos
        Pxv_pred = Pxv + Pvv
        Pvv_pred = Pvv + q_vel

        Pxx_pred = np.clip(Pxx_pred, -max_val, max_val)
        Pxv_pred = np.clip(Pxv_pred, -max_val, max_val)
        Pvv_pred = np.clip(Pvv_pred, -max_val, max_val)

        # ---- Update ----
        y = Y[t]
        S = Pxx_pred + r_obs
        S = np.where(np.abs(S) < eps, eps, S)

        K_pos = Pxx_pred / S
        K_vel = Pxv_pred / S

        innovation = y - x_pred

        x = x_pred + K_pos * innovation
        v = v_pred + K_vel * innovation

        Pxx = (1.0 - K_pos) * Pxx_pred
        Pxv = (1.0 - K_pos) * Pxv_pred
        Pvv = Pvv_pred - K_vel * Pxv_pred

        # Clamp to avoid NaNs / inf
        x = np.nan_to_num(x, nan=0.0, posinf=max_val, neginf=-max_val)
        v = np.nan_to_num(v, nan=0.0, posinf=max_val, neginf=-max_val)
        Pxx = np.nan_to_num(Pxx, nan=1.0, posinf=max_val, neginf=0.0)
        Pxv = np.nan_to_num(Pxv, nan=0.0, posinf=max_val, neginf=-max_val)
        Pvv = np.nan_to_num(Pvv, nan=1.0, posinf=max_val, neginf=0.0)

        X[t] = x

    X = np.nan_to_num(X, nan=0.0, posinf=max_val, neginf=-max_val)
    return X


def adaptive_strong_smoothing(
    track_valid,
    strong_q_pos=1e-7,
    strong_q_vel=1e-8,
    strong_r_obs=10.0,
    motion_low=0.15,
    motion_high=0.50,
    noise_raw_scale=0.05,
    min_stable_len=2,
):
    """
    Very aggressive adaptive smoothing used by kalman_smooth_mhr_params_per_obj_id_adaptive.

    - Always compute a VERY strong Kalman track 'heavy'.
    - Default behavior is already strongly biased to 'heavy'.
    - If motion pattern looks like stable -> burst -> stable,
      the middle burst segment is forced fully to 'heavy'.
    """
    track_valid = np.asarray(track_valid, dtype=np.float32)
    T, D = track_valid.shape
    if T <= 1:
        return track_valid.copy()

    # 1) Very strong Kalman (heavy)
    heavy = kalman_smooth_constant_velocity_safe(
        track_valid,
        q_pos=strong_q_pos,
        q_vel=strong_q_vel,
        r_obs=strong_r_obs,
    )

    # 2) Motion magnitude on raw track
    diff_raw = np.linalg.norm(track_valid[1:] - track_valid[:-1], axis=1)  # (T-1,)
    motion_raw = np.concatenate(([diff_raw[0]], diff_raw))                 # (T,)

    # 3) Base w_raw from motion_raw (strong bias toward heavy)
    denom = max(motion_high - motion_low, 1e-8)
    w_raw = (motion_raw - motion_low) / denom
    w_raw = np.clip(w_raw, 0.0, 1.0)      # (T,)
    # Make it even closer to 0 for low/medium motion
    w_raw = w_raw ** 2
    # Cap max raw weight to 0.3 -> at least 70% heavy even for large motion
    w_raw = np.minimum(w_raw, 0.3)
    w_raw = w_raw[:, None]                # (T, 1)

    # ---------- Very permissive "stable -> burst -> stable" detection ----------
    low_th = motion_low
    high_th = motion_high

    high_mask = motion_raw > high_th
    if high_mask.any():
        t_start = int(high_mask.argmax())
        t_end = int(len(motion_raw) - 1 - high_mask[::-1].argmax())
        if t_end > t_start:
            prefix = motion_raw[:t_start]
            suffix = motion_raw[t_end+1:]
            mid    = motion_raw[t_start:t_end+1]

            if len(prefix) >= min_stable_len and len(suffix) >= min_stable_len:
                prefix_stable_ratio = (prefix < low_th).mean() if len(prefix) > 0 else 0.0
                suffix_stable_ratio = (suffix < low_th).mean() if len(suffix) > 0 else 0.0
                mid_high_ratio      = (mid    > high_th).mean() if len(mid)    > 0 else 0.0

                # thresholds are relaxed so occlusion-like pattern is easier to trigger
                if prefix_stable_ratio > 0.5 and suffix_stable_ratio > 0.5 and mid_high_ratio > 0.5:
                    # Occlusion-like middle segment: almost fully trust heavy
                    w_raw[t_start:t_end+1, :] *= noise_raw_scale  # e.g. 0.05 → 95% heavy

    # 4) Final blend (already heavily biased to heavy)
    out = w_raw * track_valid + (1.0 - w_raw) * heavy
    return out


def kalman_smooth_mhr_params_per_obj_id_adaptive(
    mhr_dict,
    num_frames,
    frame_obj_ids,
    keys_to_smooth=None,
    kalman_cfg=None,   # kept for API compatibility, not strictly used here
    empty_thresh=1e-6,
):
    """
    Per-obj_id aggressive adaptive smoothing with occlusion awareness.

    - Very strong Kalman for body_pose / hand.
    - Strong default bias toward heavy even without occlusion.
    - Occlusion-like patterns (stable -> burst -> stable) are more easily
      detected and middle segments are almost fully pulled to heavy.
    """
    if keys_to_smooth is None:
        keys_to_smooth = ["body_pose", "hand"]

    if kalman_cfg is None:
        kalman_cfg = {}

    assert len(frame_obj_ids) == num_frames, "frame_obj_ids length must equal num_frames"

    new_mhr = {}

    any_key = next(iter(mhr_dict.keys()))
    B = mhr_dict[any_key].shape[0]
    assert B % num_frames == 0, "B must be divisible by num_frames"
    num_humans = B // num_frames

    for k, v in mhr_dict.items():
        if k in keys_to_smooth:
            device = v.device
            B, D = v.shape
            v_np = v.detach().cpu().numpy()  # (B, D)

            # Aggressive hyper-params per key
            if k == "body_pose":
                strong_q_pos, strong_q_vel, strong_r_obs = 1e-7, 1e-8, 10.0
                motion_low, motion_high = 0.15, 0.50
                noise_raw_scale = 0.05
            elif k == "hand":
                strong_q_pos, strong_q_vel, strong_r_obs = 1e-7, 1e-8, 10.0
                motion_low, motion_high = 0.20, 0.60
                noise_raw_scale = 0.05
            else:
                strong_q_pos, strong_q_vel, strong_r_obs = 1e-7, 1e-8, 10.0
                motion_low, motion_high = 0.15, 0.55
                noise_raw_scale = 0.05

            # Loop over obj_id = 1..num_humans
            for obj_id in range(1, num_humans + 1):
                slot = obj_id - 1  # slot index in [0, num_humans)

                valid_indices = []
                for t in range(num_frames):
                    if obj_id in frame_obj_ids[t]:
                        idx = t * num_humans + slot
                        valid_indices.append(idx)

                if len(valid_indices) == 0:
                    continue

                track_valid = v_np[valid_indices, :]  # (T_valid, D)

                if np.linalg.norm(track_valid) < empty_thresh:
                    continue

                smoothed_valid = adaptive_strong_smoothing(
                    track_valid,
                    strong_q_pos=strong_q_pos,
                    strong_q_vel=strong_q_vel,
                    strong_r_obs=strong_r_obs,
                    motion_low=motion_low,
                    motion_high=motion_high,
                    noise_raw_scale=noise_raw_scale,
                    min_stable_len=2,
                )

                if not np.isfinite(smoothed_valid).all():
                    continue

                v_np[valid_indices, :] = smoothed_valid

            new_mhr[k] = torch.from_numpy(v_np).to(device)
        else:
            new_mhr[k] = v

    return new_mhr


def local_window_smooth(Y, window=9, weights=None):
    """
    Strong local smoothing over a temporal window.

    Args:
        Y:       np.ndarray, shape (T, D)
        window:  odd int, temporal window size (e.g., 7 or 9)
                 for frame t, we average over [t-half, t+half]
        weights: optional np.ndarray, shape (T,)
                 per-frame reliability/visibility in [0, 1].
                 If provided, use weighted average inside the window.

    Returns:
        Smoothed Y of shape (T, D)
    """
    Y = np.asarray(Y, dtype=np.float32)
    T, D = Y.shape
    out = np.zeros_like(Y)
    half = window // 2

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        w = np.clip(w, 0.0, 1.0)
    else:
        w = None

    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)  # [s, e)

        if w is None:
            out[t] = Y[s:e].mean(axis=0)
        else:
            ww = w[s:e]
            ww_sum = ww.sum()
            if ww_sum < 1e-6:
                # if all weights ~0, fall back to simple mean
                out[t] = Y[s:e].mean(axis=0)
            else:
                ww_norm = ww / ww_sum
                out[t] = (Y[s:e] * ww_norm[:, None]).sum(axis=0)

    return out


def smooth_scale_shape_local(mhr, num_frames, window=9,
                             vis_scale=None, vis_shape=None):
    """
    Apply strong local window smoothing on 'scale' and 'shape' for multi-human case.

    Args:
        mhr:         dict with 'scale' and 'shape' tensors of shape (B, D)
        num_frames:  int, T
        window:      odd int, temporal window size
        vis_scale:   optional (B,) or (T,) visibility/confidence for scale
        vis_shape:   optional (B,) or (T,) visibility/confidence for shape

    Returns:
        new_scale, new_shape: tensors with the same shape as input
    """
    scale = mhr["scale"]
    shape = mhr["shape"]
    device = scale.device

    B, D_scale = scale.shape
    _, D_shape = shape.shape
    assert B % num_frames == 0, "B must be divisible by num_frames"
    num_humans = B // num_frames

    scale_np = scale.detach().cpu().numpy().reshape(num_frames, num_humans, D_scale)
    shape_np = shape.detach().cpu().numpy().reshape(num_frames, num_humans, D_shape)

    # Optional visibility weights per frame (shared across humans)
    if vis_scale is not None:
        vs = np.asarray(vis_scale, dtype=np.float32).reshape(num_frames)
    else:
        vs = None

    if vis_shape is not None:
        vh = np.asarray(vis_shape, dtype=np.float32).reshape(num_frames)
    else:
        vh = None

    for h in range(num_humans):
        scale_np[:, h, :] = local_window_smooth(scale_np[:, h, :], window=window, weights=vs)
        shape_np[:, h, :] = local_window_smooth(shape_np[:, h, :], window=window, weights=vh)

    scale_smooth = torch.from_numpy(scale_np.reshape(B, D_scale)).to(device)
    shape_smooth = torch.from_numpy(shape_np.reshape(B, D_shape)).to(device)
    return scale_smooth, shape_smooth
