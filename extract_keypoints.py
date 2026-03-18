"""
Sign Language Motion Extractor
--------------------------------
Extracts body, hand, and face keypoints from a video using MediaPipe Holistic.
Outputs a JSON file ready to use in a web/mobile app.

Install dependencies:
    pip install mediapipe opencv-python

Usage:
    python extract_keypoints.py --input your_video.mp4 --output keypoints.json
"""

import cv2
import json
import argparse
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def landmarks_to_list(landmarks, keys=None):
    """Convert a MediaPipe landmark object to a plain list of dicts."""
    if landmarks is None:
        return None
    result = []
    for i, lm in enumerate(landmarks.landmark):
        entry = {"x": round(lm.x, 5), "y": round(lm.y, 5), "z": round(lm.z, 5)}
        if hasattr(lm, "visibility"):
            entry["v"] = round(lm.visibility, 3)
        if keys:
            entry["name"] = keys[i]
        result.append(entry)
    return result


# Named keys for body pose landmarks (33 points)
POSE_KEYS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Named keys for hand landmarks (21 points each)
HAND_KEYS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def extract(video_path: str, output_path: str, include_face: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames")

    frames = []
    frame_idx = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,           # 0=fast, 1=balanced, 2=accurate
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,  # set True if you need detailed face mesh
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe needs RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            frame_data = {
                "frame": frame_idx,
                "time_s": round(frame_idx / fps, 4),
                "pose": landmarks_to_list(results.pose_landmarks, POSE_KEYS),
                "left_hand": landmarks_to_list(results.left_hand_landmarks, HAND_KEYS),
                "right_hand": landmarks_to_list(results.right_hand_landmarks, HAND_KEYS),
            }

            if include_face:
                frame_data["face"] = landmarks_to_list(results.face_landmarks)

            frames.append(frame_data)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    output = {
        "meta": {
            "source": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "duration_s": round(frame_idx / fps, 3),
            "includes_face": include_face,
            "landmark_sets": {
                "pose": "33 body keypoints (x, y, z, visibility)",
                "left_hand": "21 left hand keypoints (x, y, z)",
                "right_hand": "21 right hand keypoints (x, y, z)",
            },
        },
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))  # compact JSON

    size_kb = round(__import__("os").path.getsize(output_path) / 1024, 1)
    print(f"\nDone! Saved {frame_idx} frames → {output_path} ({size_kb} KB)")
    print("Load this JSON in your web/mobile app to animate a skeleton.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sign language keypoints from video")
    parser.add_argument("--input",  required=True, help="Path to input video file")
    parser.add_argument("--output", default="keypoints.json", help="Output JSON file path")
    parser.add_argument("--face",   action="store_true", help="Include face mesh landmarks (468 pts)")
    args = parser.parse_args()

    extract(args.input, args.output, include_face=args.face)
