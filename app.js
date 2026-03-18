import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import * as Kalidokit from 'kalidokit';
import { MediaPipeHolisticSystem } from './utils/holistic.js';

// ---- Three.js Setup ----
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35.0, window.innerWidth / window.innerHeight, 0.1, 20.0);
camera.position.set(0.0, 1.4, 1.8);

const light = new THREE.DirectionalLight(0xffffff, 1.5);
light.position.set(1.0, 1.0, 1.0).normalize();
scene.add(light);
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

// ---- VRM Avatar Setup ----
let currentVrm = null;
// Stores the T-pose bone rest directions for Direct IK
let restDirs = {};
const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

// Capture the T-pose rest world direction of each bone we'll animate
function captureRestDirections(vrm) {
  restDirs = {};
  const bones = [
    'leftUpperArm', 'leftLowerArm',
    'rightUpperArm', 'rightLowerArm',
    'chest', 'spine', 'neck'
  ];
  bones.forEach(name => {
    const bone = vrm.humanoid.getNormalizedBoneNode(name);
    if (bone) {
      // Store the bone's local -Z axis in world space (the bone "pointing" direction)
      const dir = new THREE.Vector3(0, 0, -1);
      dir.applyQuaternion(bone.getWorldQuaternion(new THREE.Quaternion()));
      restDirs[name] = dir.clone().normalize();
    }
  });
}

function loadVRM(url) {
  gltfLoader.load(
    url,
    (gltf) => {
      const vrm = gltf.userData.vrm;
      if (currentVrm) {
        scene.remove(currentVrm.scene);
        currentVrm.dispose();
      }
      currentVrm = vrm;
      scene.add(vrm.scene);
      vrm.scene.rotation.y = 0;
      captureRestDirections(vrm);
      console.log('Avatar loaded successfully');
      document.getElementById('status-text').innerText = "Avatar Loaded!";
    },
    (progress) => {
      document.getElementById('status-text').innerText = `Loading... ${Math.round(100.0 * progress.loaded / progress.total)}%`;
    },
    (error) => {
      console.error(error);
      document.getElementById('status-text').innerText = "Error loading avatar";
    }
  );
}

loadVRM('/avatar/default.vrm');

document.getElementById('avatar-upload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  loadVRM(url);
});

// ---- Animation Loop ----
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const deltaTime = clock.getDelta();
  if (currentVrm) {
    currentVrm.update(deltaTime);
  }
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================
// HYBRID MOTION SOLVER
// - Body/Arms: Direct Vector IK (most accurate)
// - Face: Kalidokit Face.solve
// - Hands: Kalidokit Hand.solve with correct VRM bone map
// ============================================================

// Convert MediaPipe 3D world landmark to normalized Three.js Vector3
// MediaPipe world: X = right, Y = up, Z = toward camera
const mpToVec = (lm) => new THREE.Vector3(lm.x, lm.y, -lm.z);

// Smoothly apply a direction vector to a bone using quaternion rotation
// restDir: the bone's rest-pose world direction (captured from T-pose)
// targetDir: the desired world direction (computed from landmarks)
const applyBoneDirection = (boneName, targetDir, lerpAmount = 0.4) => {
  if (!currentVrm || !restDirs[boneName]) return;
  const bone = currentVrm.humanoid.getNormalizedBoneNode(boneName);
  if (!bone) return;

  const rest = restDirs[boneName];
  const target = targetDir.clone().normalize();

  // Quaternion that rotates from the rest direction to the target direction
  const rotQuat = new THREE.Quaternion().setFromUnitVectors(rest, target);
  bone.quaternion.slerp(rotQuat, lerpAmount);
};

// Simple Euler rotation helper (for face/spine where direct Euler is cleaner)
const rigRotation = (boneName, rotation, dampener = 1, lerpAmount = 0.3) => {
  if (!currentVrm || !rotation) return;
  const boneNode = currentVrm.humanoid.getNormalizedBoneNode(boneName);
  if (!boneNode) return;
  const euler = new THREE.Euler(
    rotation.x * dampener,
    rotation.y * dampener,
    rotation.z * dampener
  );
  boneNode.quaternion.slerp(new THREE.Quaternion().setFromEuler(euler), lerpAmount);
};

// ============================================================
// MAIN TRACKING CALLBACK
// ============================================================
function onResults(results) {
  if (!currentVrm) return;

  const videoElement = document.getElementById('video-player');
  if (!videoElement || videoElement.videoWidth === 0) return;

  const sizeObj = { width: videoElement.videoWidth, height: videoElement.videoHeight };

  // Resolve 3D world pose (minified in CDN build)
  const pose3D = results.poseWorldLandmarks || results.ea || results.za;
  const pose2D = results.poseLandmarks;

  // Debug overlay
  let dbg = 'Tracking:';
  if (pose2D) dbg += ' Pose';
  if (pose3D) dbg += ' 3D';
  if (results.faceLandmarks) dbg += ' Face';
  if (results.leftHandLandmarks) dbg += ' LHand';
  if (results.rightHandLandmarks) dbg += ' RHand';
  document.getElementById('status-text').innerText = dbg;

  // ── 1. BODY / ARMS — Direct Vector IK ──────────────────────────
  if (pose3D && pose2D) {
    try {
      const lm = pose3D; // World 3D landmarks

      // Convert key landmarks to Three.js vectors
      const lShoulder = mpToVec(lm[11]);
      const rShoulder = mpToVec(lm[12]);
      const lElbow    = mpToVec(lm[13]);
      const rElbow    = mpToVec(lm[14]);
      const lWrist    = mpToVec(lm[15]);
      const rWrist    = mpToVec(lm[16]);
      const lHip      = mpToVec(lm[23]);
      const rHip      = mpToVec(lm[24]);

      // --- Upper Arms: Direction from Shoulder to Elbow ---
      applyBoneDirection('leftUpperArm',  lElbow.clone().sub(lShoulder));
      applyBoneDirection('rightUpperArm', rElbow.clone().sub(rShoulder));

      // --- Lower Arms: Direction from Elbow to Wrist ---
      applyBoneDirection('leftLowerArm',  lWrist.clone().sub(lElbow));
      applyBoneDirection('rightLowerArm', rWrist.clone().sub(rElbow));

      // --- Spine: Direction from Hip midpoint to Shoulder midpoint ---
      const hipMid = lHip.clone().add(rHip).multiplyScalar(0.5);
      const shoulderMid = lShoulder.clone().add(rShoulder).multiplyScalar(0.5);
      const spineDir = shoulderMid.clone().sub(hipMid).normalize();

      // Spine lean: compute pitch (forward/back) and roll (side tilt) from spine direction
      const spinePitch = Math.atan2(spineDir.z, spineDir.y);
      const spineRoll  = Math.atan2(-spineDir.x, spineDir.y);

      const spineRot = { x: spinePitch, y: 0, z: spineRoll };
      rigRotation('spine', spineRot, 0.45, 0.3);
      rigRotation('chest', spineRot, 0.25, 0.3);

    } catch(e) {
      console.warn('Pose IK error:', e);
    }
  }

  // ── 2. FACE & HEAD — Kalidokit (well-tested solver) ────────────
  if (results.faceLandmarks) {
    try {
      const faceRig = Kalidokit.Face.solve(results.faceLandmarks, {
        runtime: 'mediapipe',
        video: sizeObj
      });
      if (faceRig) {
        rigRotation('neck', faceRig.head, 0.7, 0.3);

        if (currentVrm.expressionManager) {
          const m = currentVrm.expressionManager;
          m.setValue('blinkLeft',  faceRig.eye.l);
          m.setValue('blinkRight', faceRig.eye.r);
          m.setValue('aa', faceRig.mouth.shape.A);
          m.setValue('ee', faceRig.mouth.shape.E);
          m.setValue('ih', faceRig.mouth.shape.I);
          m.setValue('oh', faceRig.mouth.shape.O);
          m.setValue('ou', faceRig.mouth.shape.U);
        }
      }
    } catch(e) { console.warn('Face solve error:', e); }
  }

  // ── 3. HANDS & FINGERS — Kalidokit + correct VRM bone map ──────
  const rigHand = (handFit, side) => {
    if (!handFit) return;
    // MediaPipe "Left" from camera = subject's right hand (mirror effect)
    const vrmSide = side === 'Left' ? 'right' : 'left';
    const kSide = side;

    const rigFinger = (kaliKey, vrmBoneName) => {
      const rot = handFit[kaliKey];
      if (!rot) return;
      const boneNode = currentVrm.humanoid.getNormalizedBoneNode(vrmBoneName);
      if (!boneNode) return;
      const euler = new THREE.Euler(rot.x, rot.y, rot.z);
      boneNode.quaternion.slerp(new THREE.Quaternion().setFromEuler(euler), 0.3);
    };

    // Wrist
    const wristRot = handFit[`${kSide}Wrist`];
    if (wristRot && isFinite(wristRot.x) && isFinite(wristRot.y) && isFinite(wristRot.z)) {
      rigFinger(`${kSide}Wrist`, `${vrmSide}Hand`);
    }

    // Thumb: VRM uses Metacarpal, Proximal, Distal
    rigFinger(`${kSide}ThumbProximal`,     `${vrmSide}ThumbMetacarpal`);
    rigFinger(`${kSide}ThumbIntermediate`, `${vrmSide}ThumbProximal`);
    rigFinger(`${kSide}ThumbDistal`,       `${vrmSide}ThumbDistal`);

    // Index
    rigFinger(`${kSide}IndexProximal`,     `${vrmSide}IndexProximal`);
    rigFinger(`${kSide}IndexIntermediate`, `${vrmSide}IndexIntermediate`);
    rigFinger(`${kSide}IndexDistal`,       `${vrmSide}IndexDistal`);

    // Middle
    rigFinger(`${kSide}MiddleProximal`,     `${vrmSide}MiddleProximal`);
    rigFinger(`${kSide}MiddleIntermediate`, `${vrmSide}MiddleIntermediate`);
    rigFinger(`${kSide}MiddleDistal`,       `${vrmSide}MiddleDistal`);

    // Ring
    rigFinger(`${kSide}RingProximal`,     `${vrmSide}RingProximal`);
    rigFinger(`${kSide}RingIntermediate`, `${vrmSide}RingIntermediate`);
    rigFinger(`${kSide}RingDistal`,       `${vrmSide}RingDistal`);

    // Little/Pinky
    rigFinger(`${kSide}LittleProximal`,     `${vrmSide}LittleProximal`);
    rigFinger(`${kSide}LittleIntermediate`, `${vrmSide}LittleIntermediate`);
    rigFinger(`${kSide}LittleDistal`,       `${vrmSide}LittleDistal`);
  };

  if (results.leftHandLandmarks) {
    try {
      const fit = Kalidokit.Hand.solve(results.leftHandLandmarks, 'Left');
      rigHand(fit, 'Left');
    } catch(e) { console.warn('Left hand error:', e); }
  }
  if (results.rightHandLandmarks) {
    try {
      const fit = Kalidokit.Hand.solve(results.rightHandLandmarks, 'Right');
      rigHand(fit, 'Right');
    } catch(e) { console.warn('Right hand error:', e); }
  }
}

// ---- MediaPipe Setup ----
const videoElement = document.getElementById('video-player');

document.getElementById('video-upload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  videoElement.src = url;
  videoElement.play();
  if (!mpSystem.isRunning) {
    mpSystem.isRunning = true;
    mpSystem.startVideoProcessing();
  }
});

const mpSystem = new MediaPipeHolisticSystem(videoElement, onResults);