#!/usr/bin/env python3

import sys
try:
    import bpy
    import mathutils
except ImportError:
    print("\n❌ ERROR: This script MUST be run from INSIDE Blender!")
    print("   1. Open Blender.")
    print("   2. Open the 'Scripting' workspace tab.")
    print("   3. Open 'Step2 apply and export.py' from that editor.")
    print("   4. Click the 'Run Script' button (▶️).\n")
    sys.exit(1)

"""
blender_animate.py v19 (ACCURATE HAND & FINGER SOLVER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES from v18:

  BUG 1 — Finger flex written to WRONG axis (fingers completely static):
    v18 wrote flex to Z-axis: Euler(0, spread, z_sign*flex).
    FULL_CONSTRAINTS limits Z to [0°, 0°] for ALL finger joints.
    This clamped flex to zero → fingers never moved.
    v19 FIX: flex → X-axis, spread → Z-axis.
    Mixamo/RPM rigs use LOCAL X as the curl axis for finger bones.
    FULL_CONSTRAINTS confirms: X is [-20°,90°] for MCP, [0°,110°] for PIP.

  BUG 2 — Flex sign always positive (curl direction lost):
    acos() always returns a positive value. When fingers extend,
    the avatar still showed curl because sign was never checked.
    v19 FIX: cross(v_in, v_out) · palm_normal determines curl direction.
    Positive = curling toward palm, negative = hyperextending.

  BUG 3 — Palm normal inconsistently flipped between hands:
    v18 manually negated palm_normal for left hand after computing it,
    but the curl_sign cross-product also uses palm_normal, creating
    double-negation that broke left-hand finger direction.
    v19 FIX: Palm normal computed consistently for both hands from
    cross(wrist→index_MCP, wrist→pinky_MCP). No manual flip.
    The x_sign/z_sign per-side convention handles axis orientation.

  BUG 4 — mode_set(OBJECT) called between finger keyframe inserts:
    v18 called bpy.ops.object.mode_set(mode='OBJECT') inside the per-side
    hand loop, after only one side's fingers were written. For the second
    side, mode_set(POSE) was called again but keyframe_insert was sometimes
    called in OBJECT mode → silent failures.
    v19 FIX: Enter POSE mode ONCE before the hand/finger loop.
    Exit to OBJECT mode ONCE after BOTH sides are complete.

  BUG 5 — Hand orientation target collapses when hand faces camera:
    When person holds hand toward camera, MP Z ≈ 0, hand_dir ≈ (x, 0, z).
    The Y-component (depth) can be near-zero, making the target barely
    move from the wrist → DAMPED_TRACK produces no rotation.
    v19 FIX: Minimum forward offset of 0.4×hbl ensures the target is
    always far enough from the wrist to create meaningful rotation.

  BUG 6 — Per-frame scale (arm reach accuracy):
    v18 computed scale once from frame 0. If the person was not in a
    neutral pose in frame 0, all subsequent frames had wrong arm length.
    v19 FIX: Shoulder width re-measured every frame → per-frame scale.

  BUG 7 — h_arr partial data guard missing:
    If MediaPipe returns a hand with fewer than 21 landmarks (partial
    detection), indexing h_arr[20] would throw IndexError.
    v19 FIX: Guard `if h_arr is None or len(h_arr) < 21: continue`.

APPROACH:
  METHOD A (Spine / Neck / Head): Direct Euler rotation from landmark angles.
  METHOD B (Arms / Hands): DAMPED_TRACK with Empty targets.
  METHOD C (Fingers): Direct signed Euler — X=flex (curl), Z=spread (splay).
"""

import os, sys, json, math
from mathutils import Vector, Euler, Matrix

pkg_dir = os.path.expanduser("~/.blender_packages")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

import bpy

# ── Argument parsing ──────────────────────────────────────────────
import argparse
MOTION_JSON_PATH = "/Users/andreamercy/Desktop/new model /motion_data.json"
FBX_EXPORT_PATH = "/Users/andreamercy/Desktop/new model /avatar_animated.fbx"

class Args:
    avatar = '/Users/andreamercy/Desktop/avatarformodel.fbx'
    json = MOTION_JSON_PATH
    output = FBX_EXPORT_PATH
    fps = 25.0
    motion_strength = 1.0
    min_visibility = 0.0

args = Args()

# ── Landmark indices ──────────────────────────────────────────────
# Pose (33 landmarks, world-space from PoseLandmarker world_landmarks)
# Coordinate system: X=subject-right, Y=UP, Z=toward-camera
P_NOSE = 0
P_LSH = 11; P_RSH = 12
P_LEL = 13; P_REL = 14
P_LWR = 15; P_RWR = 16
P_LHP = 23; P_RHP = 24

# Face (478 landmarks, normalized image-space from FaceLandmarker)
F_NOSE = 1; F_LEAR = 234; F_REAR = 454
F_LET = 159; F_LEB = 145; F_RET = 386; F_REB = 374
F_LEI = 133; F_LEO = 33;  F_REI = 362; F_REO = 263
F_LT = 0; F_LB = 17; F_FORE = 10; F_CHIN = 152

# Hand finger chains: (bone_key, parent_lm_idx, this_lm_idx, child_lm_idx)
# MediaPipe hand landmark indices:
#   0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
H_PAIRS = {
    'l': [
        [('lth1', 0, 1, 2), ('lth2', 1, 2, 3), ('lth3', 2, 3, 4)],
        [('li1',  0, 5, 6), ('li2',  5, 6, 7), ('li3',  6, 7, 8)],
        [('lm1',  0, 9,10), ('lm2',  9,10,11), ('lm3', 10,11,12)],
        [('lr1',  0,13,14), ('lr2', 13,14,15), ('lr3', 14,15,16)],
        [('lp1',  0,17,18), ('lp2', 17,18,19), ('lp3', 18,19,20)],
    ],
    'r': [
        [('rth1', 0, 1, 2), ('rth2', 1, 2, 3), ('rth3', 2, 3, 4)],
        [('ri1',  0, 5, 6), ('ri2',  5, 6, 7), ('ri3',  6, 7, 8)],
        [('rm1',  0, 9,10), ('rm2',  9,10,11), ('rm3', 10,11,12)],
        [('rr1',  0,13,14), ('rr2', 13,14,15), ('rr3', 14,15,16)],
        [('rp1',  0,17,18), ('rp2', 17,18,19), ('rp3', 18,19,20)],
    ],
}

# ── Bone name lookup ──────────────────────────────────────────────
# Supports 5 naming conventions: Mixamo, Mixamo-prefixed, UE4, UE4-lower, Blender
BONE_NAMES = {
    'spine':  ['Spine',       'mixamorig:Spine'],
    'spine1': ['Spine1',      'mixamorig:Spine1'],
    'spine2': ['Spine2',      'mixamorig:Spine2'],
    'neck':   ['Neck',        'mixamorig:Neck'],
    'head':   ['Head',        'mixamorig:Head'],
    'lua':    ['LeftArm',     'mixamorig:LeftArm'],
    'rua':    ['RightArm',    'mixamorig:RightArm'],
    'lla':    ['LeftForeArm', 'mixamorig:LeftForeArm'],
    'rla':    ['RightForeArm','mixamorig:RightForeArm'],
    'lhand':  ['LeftHand',    'mixamorig:LeftHand',  'Hand_L', 'hand_l', 'f_hand.L'],
    'rhand':  ['RightHand',   'mixamorig:RightHand', 'Hand_R', 'hand_r', 'f_hand.R'],
}

for p in ['l', 'r']:
    side_ue   = 'L' if p == 'l' else 'R'
    side_full = 'LeftHand' if p == 'l' else 'RightHand'
    bl_side   = 'L' if p == 'l' else 'R'
    for f_key, full, bl_prefix in zip(
        ['th', 'i', 'm', 'r', 'p'],
        ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'],
        ['thumb', 'index', 'middle', 'ring', 'pinky'],
    ):
        for i in [1, 2, 3]:
            BONE_NAMES[f'{p}{f_key}{i}'] = [
                f'{side_full}{full}{i}',
                f'mixamorig:{side_full}{full}{i}',
                f'{full}_{i:02d}_{side_ue}',
                f'{full.lower()}_{i:02d}_{side_ue.lower()}',
                f'f_{bl_prefix}.{i:02d}.{bl_side}',
            ]


# ── Full biomechanical constraint table ──────────────────────────
# (use_x, min_x°, max_x°, use_y, min_y°, max_y°, use_z, min_z°, max_z°)
# X = FLEX axis for fingers (confirmed by Mixamo rig anatomy)
# Z = SPREAD axis for MCP joints only
FULL_CONSTRAINTS = {
    'Hips':            (True,-30, 30,  True,-20, 20,  True,-15, 15),
    'Spine':           (True,-20, 25,  True,-15, 15,  True,-15, 15),
    'Spine1':          (True,-15, 20,  True,-20, 20,  True,-15, 15),
    'Spine2':          (True,-10, 20,  True,-20, 20,  True,-15, 15),
    'Neck':            (True,-20, 30,  True,-50, 50,  True,-30, 30),
    'Head':            (True,-15, 20,  True,-30, 30,  True,-20, 20),
    'Jaw':             (True,  0, 25,  True,  0,  0,  True,  0,  0),
    'LeftShoulder':    (True,-15, 30,  True, -5,  5,  True,-20, 30),
    'LeftArm':         (True,-50,180,  True,-90, 90,  True,-30,150),
    'LeftForeArm':     (True,  0,  0,  True,-80, 80,  True,  0,145),
    'LeftHand':        (True,-70, 70,  True,-30, 30,  True,-20, 35),
    # Fingers — X is flex, Z is spread (only MCP/joint1 uses spread)
    'LeftHandIndex1':  (True,-20, 90,  True,  0,  0,  True,-20, 20),
    'LeftHandIndex2':  (True,  0,110,  True,  0,  0,  True,  0,  0),
    'LeftHandIndex3':  (True,  0, 90,  True,  0,  0,  True,  0,  0),
    'LeftHandMiddle1': (True,-20, 90,  True,  0,  0,  True,-20, 20),
    'LeftHandMiddle2': (True,  0,110,  True,  0,  0,  True,  0,  0),
    'LeftHandMiddle3': (True,  0, 90,  True,  0,  0,  True,  0,  0),
    'LeftHandRing1':   (True,-20, 90,  True,  0,  0,  True,-20, 20),
    'LeftHandRing2':   (True,  0,110,  True,  0,  0,  True,  0,  0),
    'LeftHandRing3':   (True,  0, 90,  True,  0,  0,  True,  0,  0),
    'LeftHandPinky1':  (True,-20, 90,  True,  0,  0,  True,-20, 20),
    'LeftHandPinky2':  (True,  0,110,  True,  0,  0,  True,  0,  0),
    'LeftHandPinky3':  (True,  0, 90,  True,  0,  0,  True,  0,  0),
    'LeftHandThumb1':  (True,-30, 40,  True,-50, 50,  True,-20, 20),
    'LeftHandThumb2':  (True,-10, 60,  True,-15, 15,  True,  0,  0),
    'LeftHandThumb3':  (True,  0, 80,  True,  0,  0,  True,  0,  0),
    'LeftUpLeg':       (True,-30,120,  True,-45, 45,  True,-30, 45),
    'LeftLeg':         (True,  0,  0,  True,  0,  0,  True,  0,150),
    'LeftFoot':        (True,-45, 35,  True,-25, 20,  True,-10, 10),
}

# Auto-mirror Left → Right (Z-axis sign-flip for right side)
for _ln in list(FULL_CONSTRAINTS.keys()):
    if _ln.startswith('Left'):
        _rn = _ln.replace('Left', 'Right')
        if _rn not in FULL_CONSTRAINTS:
            ux,mnx,mxx, uy,mny,mxy, uz,mnz,mxz = FULL_CONSTRAINTS[_ln]
            FULL_CONSTRAINTS[_rn] = (ux,mnx,mxx, uy,mny,mxy, uz,-mxz,-mnz)


# ── Bone lookup helpers ───────────────────────────────────────────

def get_bone_by_name(arm, name):
    for candidate in [name, f'mixamorig:{name}']:
        if candidate in arm.pose.bones:
            return arm.pose.bones[candidate]
    for aliases in BONE_NAMES.values():
        if name in aliases:
            for alias in aliases:
                if alias in arm.pose.bones:
                    return arm.pose.bones[alias]
    return None


def get_bone(arm, key):
    for n in BONE_NAMES.get(key, []):
        if n in arm.pose.bones:
            return arm.pose.bones[n]
    return None


def bone_world_head(arm, bone):
    return arm.matrix_world @ bone.head


def bone_world_tail(arm, bone):
    return arm.matrix_world @ bone.tail


def bone_length_world(arm, bone):
    return (bone_world_tail(arm, bone) - bone_world_head(arm, bone)).length


def make_empty(name):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    em = bpy.context.active_object
    em.name = name
    em.empty_display_size = 0.02
    return em


# ══════════════════════════════════════════════════════════════════
# COORDINATE CONVERSIONS
# ══════════════════════════════════════════════════════════════════
#
# Pose world landmarks (from pose_world_landmarks):
#   MP: X=subject-right, Y=UP, Z=toward-camera
#   BL: X=right,         Y=fwd, Z=up
#   → BL = (MP.x, -MP.z, MP.y)   [Y and Z swap, Z negated]
#
# Hand / Face landmarks (image-space, normalized 0-1):
#   MP: X=image-right, Y=DOWN, Z=toward-camera (relative depth)
#   BL: X=right,       Y=fwd,  Z=up
#   → BL = (MP.x, -MP.z, -MP.y)  [flip Y→Z, negate both non-X]
#   Used ONLY for direction vectors, never absolute positions.
#

def mp2bl_pose(arr, idx):
    """Pose world landmark → Blender space (metric coords)."""
    v = arr[idx]
    # World: X=subj-right, Y=UP, Z=toward-cam
    # Blender: X=right, Y=forward(into screen), Z=up
    return Vector((float(v[0]), -float(v[2]), float(v[1])))


def mp2bl_hand(arr, idx, aspect=1.0):
    """Hand world landmark → Blender space (metric coords).
    No aspect ratio correction needed since world landmarks are in true 3D metric space.
    MediaPipe World Hands => X: image-right, Y: image-down, Z: forward/depth.
    Blender => X: right, Y: forward, Z: up
    """
    v = arr[idx]
    return Vector((float(v[0]), -float(v[2]), -float(v[1])))


def mp2bl_face(arr, idx, aspect=16/9):
    """Face image-space landmark → Blender space (direction use only).
    Same 16:9 aspect correction as mp2bl_hand.
    """
    v = arr[idx]
    return Vector((float(v[0]) * aspect, -float(v[2]), -float(v[1])))


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def clamp01(v):
    return max(0.0, min(1.0, v))


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MediaPipe → Avatar v19 [ACCURATE HAND & FINGER SOLVER]")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    import json
    with open(args.json, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    fps = float(data.get('fps', args.fps))
    n = len(frames)
    print(f"Frames: {n} @ {fps} fps")

    # ── Import avatar ─────────────────────────────────────────────
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    if args.avatar.endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=args.avatar)
    else:
        bpy.ops.import_scene.gltf(filepath=args.avatar)

    arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm:
        sys.exit("ERROR: No armature found in avatar file")

    shape_meshes = [
        o for o in bpy.data.objects
        if o.type == 'MESH' and o.data.shape_keys
        and len(o.data.shape_keys.key_blocks) > 1
    ]

    sc = bpy.data.scenes['Scene']
    sc.render.fps = int(round(fps))
    sc.frame_start = 1
    sc.frame_end = n
    arm.animation_data_clear()

    # ── Bone name diagnostics ─────────────────────────────────────
    finger_kw = ('index', 'middle', 'ring', 'pinky', 'thumb',
                 'Index', 'Middle', 'Ring', 'Pinky', 'Thumb',
                 '_01_', '_02_', '_03_', 'Hand', 'hand')
    finger_bones_found = [b.name for b in arm.pose.bones
                          if any(kw in b.name for kw in finger_kw)]
    print(f"Rig has {len(arm.pose.bones)} bones total. "
          f"Finger-related: {len(finger_bones_found)}")
    for bn in sorted(finger_bones_found)[:30]:
        print(f"  BONE: {bn}")

    # ── Set ALL bones to XYZ euler mode up front ──────────────────
    # CRITICAL: Must do this before adding constraints and before
    # the animation loop. Bones default to QUATERNION; writing
    # rotation_euler while in QUATERNION mode is silently ignored.
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    for b in arm.pose.bones:
        b.rotation_mode = 'XYZ'
    bpy.ops.object.mode_set(mode='OBJECT')

    # ── Create tracking empties for METHOD B ──────────────────────
    bpy.context.view_layer.objects.active = None
    T = {}
    for k in ['lel', 'rel', 'lwr', 'rwr',
               'lhand_tgt', 'rhand_tgt', 'lhand_up', 'rhand_up']:
        T[k] = make_empty(f'T_{k}')

    # ── Setup DAMPED_TRACK constraints on arm/hand bones ──────────
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')

    def add_damped_track(bone_key, empty, track_axis='TRACK_Y'):
        b = get_bone(arm, bone_key)
        if not b or not empty:
            return
        for c in list(b.constraints):
            b.constraints.remove(c)
        c = b.constraints.new('DAMPED_TRACK')
        c.target = empty
        c.track_axis = track_axis

    add_damped_track('lua', T['lel'])
    add_damped_track('rua', T['rel'])
    add_damped_track('lla', T['lwr'])
    add_damped_track('rla', T['rwr'])
    add_damped_track('lhand', T['lhand_tgt'])
    add_damped_track('rhand', T['rhand_tgt'])

    def add_locked_track(bone_key, empty):
        b = get_bone(arm, bone_key)
        if not b or not empty:
            return
        c = b.constraints.new('LOCKED_TRACK')
        c.target = empty
        c.track_axis = 'TRACK_Z'
        c.lock_axis = 'LOCK_Y'

    add_locked_track('lhand', T['lhand_up'])
    add_locked_track('rhand', T['rhand_up'])

    # ── Full biomechanical constraints ────────────────────────────
    applied = 0
    for bone_real_name, limits in FULL_CONSTRAINTS.items():
        b = get_bone_by_name(arm, bone_real_name)
        if not b:
            continue
        for existing_c in list(b.constraints):
            if existing_c.type == 'LIMIT_ROTATION':
                b.constraints.remove(existing_c)
        ux, mnx, mxx, uy, mny, mxy, uz, mnz, mxz = limits
        c = b.constraints.new('LIMIT_ROTATION')
        c.owner_space = 'LOCAL'
        c.use_limit_x = ux
        c.min_x = math.radians(mnx); c.max_x = math.radians(mxx)
        c.use_limit_y = uy
        c.min_y = math.radians(mny); c.max_y = math.radians(mxy)
        c.use_limit_z = uz
        c.min_z = math.radians(mnz); c.max_z = math.radians(mxz)
        applied += 1
    print(f"Applied biomechanical constraints to {applied} bones")

    # Clear constraints on spine/head/neck (METHOD A = direct Euler)
    for k in ['spine', 'spine1', 'spine2', 'neck', 'head']:
        b = get_bone(arm, k)
        if b:
            for c in list(b.constraints):
                b.constraints.remove(c)

    bpy.ops.object.mode_set(mode='OBJECT')

    # ── Finger bone resolution diagnostic ────────────────────────
    print("\n── Finger bone resolution check ──")
    missing_finger_keys = []
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    for side in ['l', 'r']:
        for chain in H_PAIRS[side]:
            for bk, *_ in chain:
                fb = get_bone(arm, bk)
                if fb:
                    print(f"  OK   {bk:8s} → '{fb.name}'  rot_mode={fb.rotation_mode}")
                else:
                    tried = BONE_NAMES.get(bk, ['?'])
                    print(f"  MISS {bk:8s} → NOT FOUND (tried: {tried[:2]}...)")
                    missing_finger_keys.append(bk)
    bpy.ops.object.mode_set(mode='OBJECT')
    if missing_finger_keys:
        print(f"\n  ⚠️  {len(missing_finger_keys)} finger keys unresolved.")
        print(f"  All rig bone names: {[b.name for b in arm.pose.bones]}")
    else:
        print(f"  ✅ All 30 finger bone keys resolved.")
    print("──────────────────────────────────\n")

    # ── Scale calibration (avatar shoulder width) ─────────────────
    first_p = next(
        (fd['pose'] for fd in frames if fd.get('pose') is not None), None
    )
    global_scale = 1.0
    avatar_shoulder_width = 0.25  # fallback
    if first_p is not None:
        mp_lsh = mp2bl_pose(first_p, P_LSH)
        mp_rsh = mp2bl_pose(first_p, P_RSH)
        mp_sw = (mp_lsh - mp_rsh).length
        b_lua = get_bone(arm, 'lua')
        b_rua = get_bone(arm, 'rua')
        if b_lua and b_rua and mp_sw > 1e-6:
            avatar_lsh = bone_world_head(arm, b_lua)
            avatar_rsh = bone_world_head(arm, b_rua)
            avatar_shoulder_width = (avatar_lsh - avatar_rsh).length
            global_scale = avatar_shoulder_width / mp_sw
    print(f"Global scale: {global_scale:.4f}  "
          f"(avatar shoulder: {avatar_shoulder_width:.4f}m)")

    # ── Measure avatar hand bone lengths ──────────────────────────
    hand_bone_len = {}
    for side_key in ['lhand', 'rhand']:
        hb = get_bone(arm, side_key)
        hand_bone_len[side_key] = bone_length_world(arm, hb) if hb else 0.05
    print(f"Hand bone lengths: L={hand_bone_len['lhand']:.4f}m, "
          f"R={hand_bone_len['rhand']:.4f}m")

    # ══════════════════════════════════════════════════════════════
    # ANIMATION LOOP
    # ══════════════════════════════════════════════════════════════
    ms = args.motion_strength
    total_finger_keys = 0
    print("Animating...")

    for fd in frames:
        fnum = fd['frame'] + 1
        sc.frame_set(fnum)

        p  = fd.get('pose')
        lh = fd.get('left_hand')
        rh = fd.get('right_hand')
        f  = fd.get('face')

        # ── Visibility gate ───────────────────────────────────────
        use_pose = p is not None
        if (use_pose and args.min_visibility > 0
                and len(p) > 0 and len(p[0]) >= 4):
            avg_vis = sum(float(p[i][3]) for i in range(len(p))) / len(p)
            if avg_vis < args.min_visibility:
                use_pose = False

        # ── Per-frame scale (BUG 6 FIX) ──────────────────────────
        # Recompute scale every frame so arm reach stays proportional
        # even if person moves closer/farther from camera.
        scale = global_scale
        if use_pose:
            mp_sw = (mp2bl_pose(p, P_LSH) - mp2bl_pose(p, P_RSH)).length
            if mp_sw > 1e-4:
                scale = avatar_shoulder_width / mp_sw

        # ══════════════════════════════════════════════════════════
        # METHOD A: Spine / Neck / Head (direct Euler)
        # ══════════════════════════════════════════════════════════
        if use_pose:
            sh_mid = (mp2bl_pose(p, P_LSH) + mp2bl_pose(p, P_RSH)) * 0.5
            hp_mid = (mp2bl_pose(p, P_LHP) + mp2bl_pose(p, P_RHP)) * 0.5

            spine_vec = sh_mid - hp_mid
            spine_dir = spine_vec.normalized() if spine_vec.length > 1e-6 \
                        else Vector((0, 0, 1))

            # pitch: forward/back lean. +Y in BL = forward, Z = up.
            # Positive pitch = leaning forward → should be positive angle.
            # (was -spine_dir.y which inverted the lean direction)
            pitch = math.atan2(spine_dir.y, spine_dir.z)
            pitch = clamp(pitch, math.radians(-30), math.radians(30))
            # roll: left/right tilt. -X because leaning to the left (−X) = roll+.
            roll  = math.atan2(-spine_dir.x, spine_dir.z)
            roll  = clamp(roll,  math.radians(-30), math.radians(30))

            bpy.context.view_layer.objects.active = arm
            bpy.ops.object.mode_set(mode='POSE')

            for sk, factor in [('spine', 0.25), ('spine1', 0.35), ('spine2', 0.40)]:
                b = get_bone(arm, sk)
                if b:
                    b.rotation_euler = Euler(
                        (pitch * factor * ms, 0, roll * factor * ms), 'XYZ')
                    b.keyframe_insert('rotation_euler', frame=fnum)

            # ── Head / Neck ───────────────────────────────────────
            hb = get_bone(arm, 'head')
            nb = get_bone(arm, 'neck')
            if f is not None and len(f) > max(F_FORE, F_CHIN, F_LEAR, F_REAR):
                chin      = mp2bl_face(f, F_CHIN)
                forehead  = mp2bl_face(f, F_FORE)
                # BUG 3 FIX (face ear handedness):
                # Face landmarks are in image-space (camera perspective, NOT
                # subject-space). F_LEAR=234 is image-LEFT = subject's RIGHT ear.
                # F_REAR=454 is image-RIGHT = subject's LEFT ear.
                # Swapping here so left_ear/right_ear refer to SUBJECT anatomy.
                left_ear  = mp2bl_face(f, F_REAR)   # image-right = subject's left
                right_ear = mp2bl_face(f, F_LEAR)   # image-left  = subject's right

                head_up = (forehead - chin)
                head_up = head_up.normalized() if head_up.length > 1e-6 \
                          else Vector((0, 0, 1))

                head_right = (left_ear - right_ear)
                head_right = head_right.normalized() if head_right.length > 1e-6 \
                             else Vector((1, 0, 0))

                head_fwd = head_up.cross(head_right).normalized()

                head_pitch = math.atan2(head_up.y, head_up.z)
                head_pitch = clamp(head_pitch, math.radians(-40), math.radians(40))

                # head_yaw: rotation about Z in BL XY-plane.
                # head_fwd points toward -Y when looking straight at camera.
                # Turning left → head_fwd.x goes negative → yaw should be negative.
                # atan2(-x, y) gives correct sign relative to -Y forward axis.
                # (was atan2(x, -y) which inverted the direction)
                head_yaw = math.atan2(-head_fwd.x, head_fwd.y)
                head_yaw = clamp(head_yaw, math.radians(-70), math.radians(70))

                if nb:
                    nb.rotation_euler = Euler(
                        (head_pitch * 0.3 * ms, 0.0, head_yaw * 0.3 * ms), 'XYZ')
                    nb.keyframe_insert('rotation_euler', frame=fnum)
                if hb:
                    hb.rotation_euler = Euler(
                        (head_pitch * 0.7 * ms, 0.0, head_yaw * 0.7 * ms), 'XYZ')
                    hb.keyframe_insert('rotation_euler', frame=fnum)

            bpy.ops.object.mode_set(mode='OBJECT')

            # ══════════════════════════════════════════════════════
            # METHOD B: Arms (DAMPED_TRACK)
            # ══════════════════════════════════════════════════════

            # Upper arms: shoulder → elbow
            b_lua = get_bone(arm, 'lua')
            b_rua = get_bone(arm, 'rua')
            if b_lua and b_rua:
                seg_lel = mp2bl_pose(p, P_LEL) - mp2bl_pose(p, P_LSH)
                seg_rel = mp2bl_pose(p, P_REL) - mp2bl_pose(p, P_RSH)
                T['lel'].location = bone_world_head(arm, b_lua) + scale * seg_lel
                T['rel'].location = bone_world_head(arm, b_rua) + scale * seg_rel
                T['lel'].keyframe_insert('location', frame=fnum)
                T['rel'].keyframe_insert('location', frame=fnum)

            # Force depsgraph so forearm bone heads are up to date
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            arm_eval  = arm.evaluated_get(depsgraph)

            # Forearms: elbow → wrist
            b_lla = get_bone(arm, 'lla')
            b_rla = get_bone(arm, 'rla')
            if b_lla and b_rla:
                seg_lwr = mp2bl_pose(p, P_LWR) - mp2bl_pose(p, P_LEL)
                seg_rwr = mp2bl_pose(p, P_RWR) - mp2bl_pose(p, P_REL)

                lla_eval = arm_eval.pose.bones.get(b_lla.name)
                rla_eval = arm_eval.pose.bones.get(b_rla.name)

                if lla_eval and rla_eval:
                    T['lwr'].location = (arm_eval.matrix_world @ lla_eval.head
                                         + scale * seg_lwr)
                    T['rwr'].location = (arm_eval.matrix_world @ rla_eval.head
                                         + scale * seg_rwr)
                else:
                    T['lwr'].location = bone_world_head(arm, b_lla) + scale * seg_lwr
                    T['rwr'].location = bone_world_head(arm, b_rla) + scale * seg_rwr

                T['lwr'].keyframe_insert('location', frame=fnum)
                T['rwr'].keyframe_insert('location', frame=fnum)

        # ══════════════════════════════════════════════════════════
        # METHOD B (hands) + METHOD C (fingers)
        # CRITICAL: Enter POSE mode ONCE here, exit ONCE at the end.
        # Never call mode_set inside the per-side loop.
        # ══════════════════════════════════════════════════════════

        # Refresh depsgraph so hand bone positions reflect arm solve
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        arm_eval  = arm.evaluated_get(depsgraph)

        # Enter POSE mode once for all hand + finger work this frame
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')

        for side, h_arr, hand_key, tgt_key in [
            ('l', lh, 'lhand', 'lhand_tgt'),
            ('r', rh, 'rhand', 'rhand_tgt'),
        ]:
            # BUG 7 FIX: Guard against partial/missing hand data
            if h_arr is None or len(h_arr) < 21:
                continue

            hnd_b = get_bone(arm, hand_key)
            if not hnd_b:
                continue

            hnd_eval = arm_eval.pose.bones.get(hnd_b.name)
            hand_world_pos = (arm_eval.matrix_world @ hnd_eval.head
                              if hnd_eval
                              else bone_world_head(arm, hnd_b))

            hbl = hand_bone_len[hand_key]

            # ── Hand orientation targets ──────────────────────────
            wrist_mp     = mp2bl_hand(h_arr, 0)
            index_mcp_mp = mp2bl_hand(h_arr, 5)
            pinky_mcp_mp = mp2bl_hand(h_arr, 17)
            mid_mcp_mp   = mp2bl_hand(h_arr, 9)

            # Forward direction: wrist → middle MCP
            hand_dir = mid_mcp_mp - wrist_mp
            if hand_dir.length > 1e-6:
                hand_dir = hand_dir.normalized()
            else:
                hand_dir = Vector((0, 1, 0))

            # BUG 3 FIX: Palm normal computed consistently for both hands.
            # cross(wrist→index_MCP, wrist→pinky_MCP) points out the back
            # of the hand (dorsal). No manual flip needed — the signed
            # curl computation handles orientation correctly per-side.
            v1 = index_mcp_mp - wrist_mp
            v2 = pinky_mcp_mp - wrist_mp
            palm_normal = v1.cross(v2)
            if palm_normal.length > 1e-6:
                palm_normal = palm_normal.normalized()
            else:
                palm_normal = Vector((0, 0, 1))

            # BUG 5 FIX: Minimum offset so target never overlaps wrist
            # even when hand is facing directly toward camera (MP Z ≈ 0).
            hand_tgt_dist = max(
                abs(hand_dir.dot(Vector((0, 1, 0)))), 0.4
            ) * hbl
            T[tgt_key].location = hand_world_pos + hand_dir * scale * hand_tgt_dist
            T[tgt_key].keyframe_insert('location', frame=fnum)

            up_key = 'lhand_up' if side == 'l' else 'rhand_up'
            T[up_key].location = hand_world_pos + palm_normal * hbl
            T[up_key].keyframe_insert('location', frame=fnum)

            # ── FINGER SOLVER ─────────────────────────────────────
            # X = flex/curl (Mixamo rig confirms X is active axis).
            # Z = spread (abduction/adduction at MCP only).
            # Flex sign from cross(v_in, v_out)·palm_normal.
            x_sign = 1.0                        # same for both sides
            z_sign = 1.0 if side == 'l' else -1.0

            # Reference direction for spread: wrist → MIDDLE MCP (landmark 9).
            # Middle finger is the anatomical centre of the hand, so measuring
            # spread from it gives symmetric splay (index + pinky equidistant).
            # (was wrist→index MCP which made index always read spread=0)
            ref_dir = (mp2bl_hand(h_arr, 9) - mp2bl_hand(h_arr, 0))
            if ref_dir.length > 1e-6:
                ref_dir = ref_dir.normalized()
            else:
                ref_dir = Vector((1, 0, 0))

            for chain_idx, chain in enumerate(H_PAIRS[side]):
                for joint_rank, (bk, i0, i1, i2) in enumerate(chain):
                    fb = get_bone(arm, bk)
                    if not fb:
                        if fnum == 1:
                            print(f"  MISS finger bone: {bk} "
                                  f"(tried: {BONE_NAMES.get(bk, ['?'])[:2]}...)")
                        continue

                    fb.rotation_mode = 'XYZ'

                    p0 = mp2bl_hand(h_arr, i0)
                    p1 = mp2bl_hand(h_arr, i1)
                    p2 = mp2bl_hand(h_arr, i2)

                    v_in  = p1 - p0
                    v_out = p2 - p1
                    if v_in.length  < 1e-6: v_in  = Vector((0, 1, 0))
                    if v_out.length < 1e-6: v_out = Vector((0, 1, 0))
                    v_in  = v_in.normalized()
                    v_out = v_out.normalized()

                    # Signed flex angle
                    dot_flex       = clamp(v_in.dot(v_out), -1.0, 1.0)
                    flex_magnitude = math.acos(dot_flex)
                    curl_cross     = v_in.cross(v_out)
                    curl_sign      = 1.0 if curl_cross.dot(palm_normal) > 0 else -1.0
                    flex_rad       = curl_sign * flex_magnitude

                    # Anatomical clamp per joint
                    if joint_rank == 0:    # MCP
                        flex_rad = clamp(flex_rad, math.radians(-20), math.radians(90))
                    elif joint_rank == 1:  # PIP
                        flex_rad = clamp(flex_rad, math.radians(0), math.radians(110))
                    else:                  # DIP
                        flex_rad = clamp(flex_rad, math.radians(0), math.radians(90))

                    # Spread (MCP only)
                    spread_rad = 0.0
                    if joint_rank == 0:
                        v_proj = v_in - palm_normal * v_in.dot(palm_normal)
                        if v_proj.length > 1e-6:
                            v_proj = v_proj.normalized()
                            cs = ref_dir.cross(v_proj)
                            ss = 1.0 if cs.dot(palm_normal) > 0 else -1.0
                            spread_rad = ss * math.acos(clamp(ref_dir.dot(v_proj), -1.0, 1.0))
                        spread_rad = clamp(spread_rad, math.radians(-20), math.radians(20))

                    fb.rotation_euler = Euler(
                        (x_sign * flex_rad,    # X = flex/curl
                         0.0,
                         z_sign * spread_rad), # Z = spread
                        'XYZ'
                    )
                    fb.keyframe_insert('rotation_euler', frame=fnum)
                    total_finger_keys += 1





        # Exit POSE mode ONCE after all hand+finger work for this frame
        bpy.ops.object.mode_set(mode='OBJECT')

        # ══════════════════════════════════════════════════════════
        # FACE BLENDSHAPES
        # ══════════════════════════════════════════════════════════
        if f is not None:
            nbs = fd.get('face_blendshapes', {})
            shapes = dict(nbs) if nbs else {}

            if not shapes:
                def ds(i, j):
                    return (mp2bl_face(f, i) - mp2bl_face(f, j)).length
                fh = ds(F_FORE, F_CHIN) + 1e-6
                shapes = {
                    'jawOpen':       clamp01(ds(F_LT, F_LB) / (fh * 0.15)),
                    'eyeBlinkLeft':  clamp01(1 - ds(F_LET, F_LEB)
                                             / (ds(F_LEI, F_LEO) + 1e-6) * 6),
                    'eyeBlinkRight': clamp01(1 - ds(F_RET, F_REB)
                                             / (ds(F_REI, F_REO) + 1e-6) * 6),
                }

            for m in shape_meshes:
                for sk in m.data.shape_keys.key_blocks:
                    if sk.name == 'Basis':
                        continue
                    val = shapes.get(sk.name)
                    if val is None:
                        val = next(
                            (v for k, v in shapes.items()
                             if k.lower() == sk.name.lower()), None)
                    if val is not None:
                        sk.value = float(val)
                        sk.keyframe_insert('value', frame=fnum)

        if fnum % 100 == 1:
            print(f"  Frame {fnum}/{n}  |  finger keys so far: {total_finger_keys}")

    print(f"\nTotal finger keyframes written: {total_finger_keys}")
    expected = 30 * n  # 30 finger bones × n frames
    if total_finger_keys < expected * 0.5:
        print(f"  ⚠️  Expected ~{expected}, got {total_finger_keys}. "
              f"Check MISS lines above and add aliases to BONE_NAMES.")
    else:
        print(f"  ✅ Finger keyframes look complete.")

    # ══════════════════════════════════════════════════════════════
    # BAKE: DAMPED_TRACK visual → FK keyframes
    # ══════════════════════════════════════════════════════════════
    print("\nBaking constraints → FK keyframes...")
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.nla.bake(
        frame_start=1, frame_end=n,
        visual_keying=True,
        clear_constraints=True,
        bake_types={'POSE'}
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    if arm.animation_data and arm.animation_data.action:
        act = arm.animation_data.action
        track = arm.animation_data.nla_tracks.new()
        track.strips.new(act.name, 1, act)
        arm.animation_data.action = None

    # ══════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════
    out = os.path.abspath(args.output)
    glb = out.replace('.fbx', '.glb')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if o.type in ['ARMATURE', 'MESH'] and not o.name.startswith('T_'):
            o.select_set(True)

    print(f"Exporting GLB → {glb}")
    bpy.ops.export_scene.gltf(
        filepath=glb, export_format='GLB',
        export_animations=True,
        export_animation_mode='ACTIONS',
        export_nla_strips=True,
    )

    print(f"Exporting FBX → {out}")
    bpy.ops.export_scene.fbx(
        filepath=out, use_selection=True,
        add_leaf_bones=False, bake_anim=True,
        bake_anim_use_nla_strips=True,
        bake_anim_force_startend_keying=True,
    )

    print(f"✅ Done!  FBX={os.path.getsize(out)/1e6:.1f}MB  "
          f"GLB={os.path.getsize(glb)/1e6:.1f}MB")


if __name__ == '__main__':
    main()
