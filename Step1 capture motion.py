"""
capture.py v3 — MediaPipe 0.10+ Tasks API capture pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES from v2:

  BUG 1 — Handedness mirror (CRITICAL):
    MediaPipe HandLandmarker reports handedness from the CAMERA's point of
    view, not the subject's. 'Left' in MediaPipe = subject's RIGHT hand.
    v2 assigned 'Left' → lhand_arr, which was wrong.
    v3 FIX: 'Left' → rhand_arr, 'Right' → lhand_arr.

  BUG 2 — Pose landmarks were image-space, not world-space (CRITICAL):
    v2 used pose_res.pose_landmarks (normalized 0–1 image coords, X-axis
    mirrored). This caused left-right arm swap in the avatar.
    v3 FIX: Uses pose_res.pose_world_landmarks (metric, subject-space,
    non-mirrored). X=subject's right, Y=UP, Z=toward camera.

  BUG 3 — World landmark visibility field:
    pose_world_landmarks landmarks have a .visibility attribute just like
    pose_landmarks. The array shape stays (33, 4).

Usage:
  python capture.py --video path/to/video.mp4 --output motion_data.json
  python capture.py --video path/to/video.mp4 --output motion_data.json --no-smooth
  python capture.py --video path/to/video.mp4 --output motion_data.json --hand-smooth-cutoff 0.6
"""
import cv2
import json
import numpy as np
import argparse
import os
import urllib.request
import numpy as np


# ── One Euro Filter ───────────────────────────────────────────────
class OneEuroFilter:
    """Per-coordinate 1€ filter: low jitter at rest, low lag when moving."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff) if cutoff > 0 else 0
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

    def __call__(self, x, t):
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(dt, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def smooth_landmarks_array(arr_list, fps, min_cutoff=1.2, beta=0.007):
    """Smooth a list of (N, 4) landmark arrays over time."""
    if not arr_list or len(arr_list) == 0:
        return arr_list
    n_landmarks = arr_list[0].shape[0]
    n_coords = 4  # x, y, z, visibility
    filters = [
        [OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
         for _ in range(n_coords)]
        for _ in range(n_landmarks)
    ]
    out = []
    for fi, arr in enumerate(arr_list):
        t = fi / max(fps, 1.0)
        smoothed = np.empty_like(arr)
        for i in range(n_landmarks):
            for j in range(n_coords):
                smoothed[i, j] = filters[i][j](float(arr[i, j]), t)
        out.append(smoothed)
    return out


# ── Model download ────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# Verified working URLs (versioned, not 'latest' which 404s)
MODELS = {
    'pose': (
        'pose_landmarker_full.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_full/float16/1/pose_landmarker_full.task'
    ),
    # lite model — proven to work; full variant URL returns 404 from Google CDN
    'hand': (
        'hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/1/hand_landmarker.task'
    ),
    'face': (
        'face_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
        'face_landmarker/float16/1/face_landmarker.task'
    ),
}


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for key, (fname, url) in MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {key} model → {fname} ...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  OK {fname} downloaded")
            except Exception as e:
                # Remove partial file if download failed
                if os.path.exists(path):
                    os.remove(path)
                raise RuntimeError(
                    f"Failed to download {key} model from:\n  {url}\n"
                    f"Error: {e}\n"
                    f"Please manually download the model and place it at:\n  {path}"
                ) from e
    return {k: os.path.join(MODEL_DIR, v[0]) for k, v in MODELS.items()}


def run_capture(video_path: str, output_json: str, args=None):
    if args is None:
        class DefaultArgs:
            smooth = True
            hand_smooth_cutoff = 0.8
            pose_smooth_cutoff = 1.2
            face_smooth_cutoff = 1.0
        args = DefaultArgs()

    model_paths = ensure_models()

    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base

    # ── Landmarker config ─────────────────────────────────────────
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(
            model_asset_path=model_paths['pose']
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_base.BaseOptions(
            model_asset_path=model_paths['hand']
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_base.BaseOptions(
            model_asset_path=model_paths['face']
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video : {video_path}")
    print(f"FPS   : {fps:.2f}  |  Frames: {n_frames}")

    results_list = []

    # Track last valid hand data for hold-through
    last_valid_lhand = None
    last_valid_rhand = None
    lhand_lost_frames = 0
    rhand_lost_frames = 0
    MAX_HOLD_FRAMES = 5  # hold last hand data for up to 5 frames

    with (mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lmk,
          mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lmk,
          mp_vision.FaceLandmarker.create_from_options(face_opts) as face_lmk):

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_res = pose_lmk.detect(mp_img)
            hand_res = hand_lmk.detect(mp_img)
            face_res = face_lmk.detect(mp_img)

            # ── Pose: USE WORLD LANDMARKS (metric, subject-space) ─
            # BUG 2 FIX: pose_world_landmarks gives true 3D coords
            # in subject space (X=subject's right, Y=UP, Z=toward cam).
            # pose_landmarks (image-space) had X mirrored and Y=DOWN
            # which caused arms to be swapped left↔right in the avatar.
            pose_arr = None
            if pose_res.pose_world_landmarks:
                lms = pose_res.pose_world_landmarks[0]
                pose_arr = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                    dtype=np.float32
                )

            # ── Hands ─────────────────────────────────────────────
            # BUG 1 FIX: MediaPipe reports handedness from CAMERA perspective.
            # 'Left' in MediaPipe = subject's RIGHT hand (mirror of camera view).
            # Swap the assignment so pkl['left_hand'] = anatomical left hand.
            lhand_arr = rhand_arr = None
            if hand_res.hand_world_landmarks:
                for i, hand_lms in enumerate(hand_res.hand_world_landmarks):
                    hw = np.array(
                        [[lm.x, lm.y, lm.z, 1.0] for lm in hand_lms],
                        dtype=np.float32
                    )
                    handedness = hand_res.handedness[i][0].category_name
                    # CORRECTED assignment (opposite of what MediaPipe reports):
                    if handedness == 'Left':
                        rhand_arr = hw   # MP 'Left' = subject's RIGHT hand
                    else:
                        lhand_arr = hw   # MP 'Right' = subject's LEFT hand

            # Hold-through: if hand was just lost, use last valid data
            if lhand_arr is not None:
                last_valid_lhand = lhand_arr.copy()
                lhand_lost_frames = 0
            elif last_valid_lhand is not None and lhand_lost_frames < MAX_HOLD_FRAMES:
                lhand_arr = last_valid_lhand
                lhand_lost_frames += 1

            if rhand_arr is not None:
                last_valid_rhand = rhand_arr.copy()
                rhand_lost_frames = 0
            elif last_valid_rhand is not None and rhand_lost_frames < MAX_HOLD_FRAMES:
                rhand_arr = last_valid_rhand
                rhand_lost_frames += 1

            # ── Face ──────────────────────────────────────────────
            face_arr = None
            face_blendshapes = {}
            if face_res.face_landmarks:
                lms = face_res.face_landmarks[0]
                face_arr = np.array(
                    [[lm.x, lm.y, lm.z, 1.0] for lm in lms],
                    dtype=np.float32
                )
            if face_res.face_blendshapes:
                for bs in face_res.face_blendshapes[0]:
                    face_blendshapes[bs.category_name] = float(bs.score)

            results_list.append({
                'frame':            frame_idx,
                'pose':             pose_arr,
                'left_hand':        lhand_arr,
                'right_hand':       rhand_arr,
                'face':             face_arr,
                'face_blendshapes': face_blendshapes,
            })

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{n_frames} frames ...")

    cap.release()
    print(f"\n✅ Captured {frame_idx} frames")

    # ── Temporal smoothing ────────────────────────────────────────
    if frame_idx > 0 and getattr(args, 'smooth', True):
        print("Applying temporal smoothing...")

        hand_cutoff = getattr(args, 'hand_smooth_cutoff', 0.8)
        pose_cutoff = getattr(args, 'pose_smooth_cutoff', 1.2)
        face_cutoff = getattr(args, 'face_smooth_cutoff', 1.0)

        # Pose smoothing
        pose_filled = []
        for fd in results_list:
            p = fd.get('pose')
            pose_filled.append(
                p if p is not None
                else (pose_filled[-1] if pose_filled else None)
            )
        if pose_filled and pose_filled[0] is not None:
            pose_smoothed = smooth_landmarks_array(
                pose_filled, fps, min_cutoff=pose_cutoff
            )
            for i, fd in enumerate(results_list):
                if fd.get('pose') is not None:
                    fd['pose'] = pose_smoothed[i]

        # Hands + face: tighter smoothing to preserve fast finger motion
        for key, cutoff in [
            ('left_hand', hand_cutoff),
            ('right_hand', hand_cutoff),
            ('face', face_cutoff),
        ]:
            filled = []
            for fd in results_list:
                v = fd.get(key)
                filled.append(
                    v if v is not None
                    else (filled[-1] if filled else None)
                )
            if filled and filled[0] is not None:
                smoothed = smooth_landmarks_array(
                    filled, fps, min_cutoff=cutoff
                )
                for i, fd in enumerate(results_list):
                    if fd.get(key) is not None:
                        fd[key] = smoothed[i]

        print(f"  ✅ Smoothed: pose(cutoff={pose_cutoff}), "
              f"hands(cutoff={hand_cutoff}), face(cutoff={face_cutoff})")

    data = {'fps': fps, 'frames': results_list}
    def to_serializable(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, dict): return {k: to_serializable(val) for k, val in v.items()}
        if isinstance(v, list): return [to_serializable(x) for x in v]
        return v
    
    data_serializable = {'fps': fps, 'frames': to_serializable(results_list)}
    with open(output_json, 'w') as f:
        json.dump(data_serializable, f)
    print(f"✅ Saved → {output_json}")
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MediaPipe v3 capture pipeline — saves landmarks to pkl'
    )
    parser.add_argument('--video',  default="", help='Input video file')
    parser.add_argument('--output', default="/Users/andreamercy/Desktop/new model /motion_data.json", help='Output .json path')
    parser.add_argument('--no-smooth', action='store_true', default=False,
                        help='Disable temporal smoothing')
    parser.add_argument('--hand-smooth-cutoff', type=float, default=0.8,
                        help='One-Euro min_cutoff for hand landmarks (default 0.8)')
    parser.add_argument('--pose-smooth-cutoff', type=float, default=1.2,
                        help='One-Euro min_cutoff for pose landmarks (default 1.2)')
    parser.add_argument('--face-smooth-cutoff', type=float, default=1.0,
                        help='One-Euro min_cutoff for face landmarks (default 1.0)')
    parser.add_argument('--hand-model', choices=['lite', 'full'], default='lite',
                        help='Hand model variant: lite (default) or full')
    args = parser.parse_args()

    # Override hand model entry based on user selection.
    # Note: 'full' variant is unavailable on Google CDN; falls back to lite URL.
    if args.hand_model == 'full':
        MODELS['hand'] = (
            'hand_landmarker_full.task',
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker_full/float16/1/hand_landmarker_full.task'
        )
    else:
        MODELS['hand'] = (
            'hand_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task'
        )

    args.smooth = not args.no_smooth

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    run_capture(args.video, args.output, args)