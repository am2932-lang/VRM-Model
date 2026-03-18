# Motion Capture → RPM Avatar → FBX Export
## Full Pipeline Guide

---

## What This Does

Records or processes a **video of one person** → extracts their body motion
using AI (MediaPipe) → applies it to your Ready Player Me avatar in Blender
→ exports an **animated .FBX file** ready for Unity, Unreal, or any 3D tool.

```
📹 Video / Webcam
      ↓
  step1_capture_motion.py   (Python + MediaPipe)
      ↓
  motion_data.json          (keyframe angles)
      ↓
  step2_apply_and_export.py (inside Blender)
      ↓
  avatar_animated.fbx  ✅
```

---

## Files

| File | Purpose |
|---|---|
| `step1_capture_motion.py` | Captures pose from webcam or video file |
| `step1b_hands.py` | Enhanced version: also tracks individual fingers |
| `step2_apply_and_export.py` | Blender script — applies motion + exports FBX |

---

## STEP 0 — Install Requirements

```bash
pip install mediapipe opencv-python numpy
```

Blender: download from https://blender.org (3.x or 4.x)

---

## STEP 1 — Capture Motion

### Option A: Use webcam (live recording)
```bash
python step1_capture_motion.py --show
```
Press **Q** to stop recording.

### Option B: Use a video file
```bash
python step1_capture_motion.py --video myvideo.mp4
```

### Option C: With full finger tracking
```bash
python step1b_hands.py --video myvideo.mp4 --show
```

### All options
```
--video   path to video file (omit = webcam)
--fps     keyframes per second to output (default: 24)
--output  output JSON filename (default: motion_data.json)
--show    display a preview window while processing
```

### Output
Creates `motion_data.json` — a list of bone rotation keyframes.

---

## STEP 2 — Apply to Avatar in Blender

1. **Open Blender**

2. **Import your avatar**
   - File → Import → FBX  (if you have a .fbx from RPM)
   - File → Import → glTF 2.0  (if you have a .glb from RPM)
   - Download avatars free at: https://readyplayer.me

3. **Open the Scripting workspace**
   - Click the "Scripting" tab at the top of Blender

4. **Open step2_apply_and_export.py**
   - Click "Open" in the text editor panel
   - Select `step2_apply_and_export.py`

5. **Set the path to your JSON**
   - Near the top of the script, edit:
     ```python
     MOTION_JSON_PATH = "motion_data.json"   # ← full path if needed
     FBX_EXPORT_PATH  = "avatar_animated.fbx"
     ```

6. **Run the script**
   - Click ▶ **Run Script** (or press Alt+P)

7. **Done!**
   - `avatar_animated.fbx` appears next to your .blend file
   - Press **Spacebar** in Blender to preview the animation

---

## Tips for Best Results

### Video setup
- **Good lighting** — face and arms clearly visible
- **Plain background** — helps MediaPipe track you better
- **Wear fitted clothing** — loose clothing confuses joint detection
- **Stay in frame** — both arms should be visible at all times
- **720p or higher** — better resolution = more accurate tracking
- **Distance** — stand 1.5–3 metres from camera (full upper body visible)

### Rotation tuning
If the avatar moves too much or too little, edit in `step2_apply_and_export.py`:
```python
ROTATION_SCALE = 1.0   # increase for bigger movements, decrease for subtler
```

### Smoothing jitter
If the animation looks shaky:
```python
SMOOTH_FCURVES = True
SMOOTH_PASSES  = 4     # more passes = smoother (try 2–6)
```

Or in Blender's Graph Editor: select all → Key → Smooth Keys

---

## Bone Names (Ready Player Me / Mixamo)

The scripts automatically map to these RPM armature bone names:

| Body Part | Bone Name |
|---|---|
| Left upper arm | LeftArm |
| Left forearm | LeftForeArm |
| Left hand | LeftHand |
| Right upper arm | RightArm |
| Right forearm | RightForeArm |
| Right hand | RightHand |
| Head | Head |
| Neck | Neck |
| Spine | Spine |
| Left index finger | LeftHandIndex1/2/3 |
| Right thumb | RightHandThumb1/2 |
| … etc | … |

---

## Exporting to Unity / Unreal

After getting `avatar_animated.fbx`:

**Unity:**
1. Drag FBX into Assets panel
2. Click it → Inspector → Rig → Animation Type: **Humanoid** → Apply
3. Drag onto scene

**Unreal Engine:**
1. Import → select FBX → Skeletal Mesh
2. Enable "Import Animations"
3. Use in Animation Blueprint

---

## Limitations

- MediaPipe Pose tracks **upper body** (arms, head, spine) but not legs/hips in detail
- Finger tracking in `step1_capture_motion.py` is **estimated** from hand openness
- For accurate per-finger data, use `step1b_hands.py`
- Fast movements may cause jitter — use `SMOOTH_PASSES = 4` to reduce
- Works best with a single person clearly visible in frame

---

## Troubleshooting

**"No armature found"**
→ Import your avatar into Blender before running the script

**"Bone not found: LeftArm"**  
→ Your avatar uses different bone names. Check `available_bones` printed in Blender console
→ Edit BONE_MAP in `step2_apply_and_export.py` to match your rig

**Avatar barely moves**
→ Increase `ROTATION_SCALE` to 1.5 or 2.0

**Animation is very shaky**
→ Set `SMOOTH_PASSES = 4` or more, or use a slower `--fps` value

**MediaPipe not detecting pose**
→ Try better lighting, move camera further back, wear contrasting clothing
