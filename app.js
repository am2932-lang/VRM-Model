import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import * as Kalidokit from 'kalidokit';
import { MediaPipeTasksSystem } from './utils/tasks.js';

// ── Three.js Scene ────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35.0, window.innerWidth / window.innerHeight, 0.1, 20.0);
camera.position.set(0.0, 1.4, 1.8);

scene.add(new THREE.DirectionalLight(0xffffff, 1.5).position.set(1, 1, 1).normalize().parent);
scene.add(new THREE.AmbientLight(0xffffff, 0.5));

// ── VRM Avatar ────────────────────────────────────────────────
let currentVrm = null;
const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

function loadVRM(url) {
  gltfLoader.load(
    url,
    (gltf) => {
      const vrm = gltf.userData.vrm;
      if (currentVrm) { scene.remove(currentVrm.scene); currentVrm.dispose(); }
      currentVrm = vrm;
      scene.add(vrm.scene);
      vrm.scene.rotation.y = 0;
      document.getElementById('status-text').innerText = 'Avatar loaded!';
    },
    (p) => { document.getElementById('status-text').innerText = `Loading ${Math.round(100 * p.loaded / p.total)}%`; },
    (e) => { console.error(e); document.getElementById('status-text').innerText = 'Error loading avatar'; }
  );
}
loadVRM('/avatar/default.vrm');
document.getElementById('avatar-upload').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
  loadVRM(URL.createObjectURL(f));
});

// ── Render Loop ───────────────────────────────────────────────
const clock = new THREE.Clock();
(function animate() {
  requestAnimationFrame(animate);
  if (currentVrm) currentVrm.update(clock.getDelta());
  renderer.render(scene, camera);
})();
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Bone Rotation Helper ──────────────────────────────────────
// Applies Kalidokit-style Euler rotation to a VRM normalized bone.
// No axis remapping — Kalidokit Pose.solve() already outputs VRM-compatible Eulers.
const rigRotation = (boneName, rotation, dampener = 1, lerpAmount = 0.3) => {
  if (!currentVrm || !rotation) return;
  const bone = currentVrm.humanoid.getNormalizedBoneNode(boneName);
  if (!bone) return;
  const q = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(rotation.x * dampener, rotation.y * dampener, rotation.z * dampener)
  );
  bone.quaternion.slerp(q, lerpAmount);
};

// ── Pose Result Handler ───────────────────────────────────────
// Tasks API returns: { landmarks: [[{x,y,z,visibility},...]], worldLandmarks: [[{x,y,z},...]] }
function onPoseResults(result) {
  if (!currentVrm) return;
  if (!result.landmarks?.length || !result.worldLandmarks?.length) return;

  const videoEl = document.getElementById('video-player');
  if (!videoEl || !videoEl.videoWidth) return;

  const pose2D = result.landmarks[0];     // normalized image coords
  const pose3D = result.worldLandmarks[0]; // metric 3D world coords

  const sizeObj = { width: videoEl.videoWidth, height: videoEl.videoHeight };

  document.getElementById('status-text').innerText = `Tracking: Pose (${pose2D.length} pts)`;

  try {
    // Kalidokit.Pose.solve() accepts a flat array of landmark objects
    const poseRig = Kalidokit.Pose.solve(pose3D, pose2D, {
      runtime: 'mediapipe',
      video: sizeObj
    });

    if (poseRig) {
      // Torso
      rigRotation('spine', poseRig.Spine, 0.45, 0.3);
      rigRotation('chest', poseRig.Spine, 0.25, 0.3);

      // Arms — direct output, no axis inversion
      rigRotation('rightUpperArm', poseRig.RightUpperArm, 1, 0.3);
      rigRotation('rightLowerArm', poseRig.RightLowerArm, 1, 0.3);
      rigRotation('leftUpperArm',  poseRig.LeftUpperArm,  1, 0.3);
      rigRotation('leftLowerArm',  poseRig.LeftLowerArm,  1, 0.3);

      // Legs
      rigRotation('leftUpperLeg',  poseRig.LeftUpperLeg,  1, 0.3);
      rigRotation('leftLowerLeg',  poseRig.LeftLowerLeg,  1, 0.3);
      rigRotation('rightUpperLeg', poseRig.RightUpperLeg, 1, 0.3);
      rigRotation('rightLowerLeg', poseRig.RightLowerLeg, 1, 0.3);
    }
  } catch(e) { console.warn('Pose solve error:', e); }
}

// ── Hand Result Handler ───────────────────────────────────────
// Tasks API returns: { landmarks: [[...], [...]], handedness: [{categoryName: 'Left'|'Right'},...] }
function onHandResults(result) {
  if (!currentVrm) return;
  if (!result.landmarks?.length) return;

  result.landmarks.forEach((landmarks, idx) => {
    const handedness = result.handedness[idx]?.[0]?.categoryName;
    if (!handedness || landmarks.length < 21) return;

    try {
      // Kalidokit hand solver
      const handFit = Kalidokit.Hand.solve(landmarks, handedness);
      if (!handFit) return;

      // Tasks API returns true handedness (no mirroring for video files)
      // "Left" = subject's left hand, "Right" = subject's right
      const vrmSide = handedness === 'Left' ? 'left' : 'right';
      const kSide = handedness;

      // Helper to apply a finger bone rotation
      const rigFinger = (kaliKey, vrmBoneName) => {
        const rot = handFit[kaliKey];
        if (!rot) return;
        const bone = currentVrm.humanoid.getNormalizedBoneNode(vrmBoneName);
        if (!bone) return;
        bone.quaternion.slerp(
          new THREE.Quaternion().setFromEuler(new THREE.Euler(rot.x, rot.y, rot.z)), 0.3
        );
      };

      // Wrist
      const wr = handFit[`${kSide}Wrist`];
      if (wr && isFinite(wr.x) && isFinite(wr.y) && isFinite(wr.z)) {
        rigFinger(`${kSide}Wrist`, `${vrmSide}Hand`);
      }

      // Thumb: Kalidokit Proximal→VRM Metacarpal, Intermediate→Proximal, Distal→Distal
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

    } catch(e) { console.warn('Hand solve error:', e); }
  });
}

// ── MediaPipe Tasks Setup ─────────────────────────────────────
const videoEl = document.getElementById('video-player');

const mpSystem = new MediaPipeTasksSystem(videoEl, onPoseResults, onHandResults);

document.getElementById('video-upload').addEventListener('change', (e) => {
  const file = e.target.files[0]; if (!file) return;
  videoEl.src = URL.createObjectURL(file);
  videoEl.play();
  if (!mpSystem.isRunning) {
    mpSystem.isRunning = true;
    mpSystem.startVideoProcessing();
  }
});