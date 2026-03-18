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
const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

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
      vrm.scene.rotation.y = Math.PI; // Let's check if 180 makes it back or front. Wait, I'll remove it.
      // Actually, standard VRMs face +Z. Camera is at +Z looking at -Z. So they face away. 
      // Rotating by Math.PI makes them face -Z (towards camera).
      // Wait, in the screenshot, I added `Math.PI` and it faced AWAY. So it must need `0` rotation!
      vrm.scene.rotation.y = 0; 

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

// Load default avatar
loadVRM('/avatar/default.vrm');

// Avatar Upload Logic
document.getElementById('avatar-upload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  loadVRM(url);
});

// ---- Animation Tracking Loop ----
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

// ---- Kalidokit Solvers ----

// Helper to smoothly apply rotations to VRM bones
const rigRotation = (boneName, rotation, dampener = 1, lerpAmount = 0.3) => {
    if (!currentVrm || !rotation) return;
    const boneNode = currentVrm.humanoid.getNormalizedBoneNode(boneName);
    if (!boneNode) return;
    
    const euler = new THREE.Euler(
        rotation.x * dampener,
        rotation.y * dampener,
        rotation.z * dampener
    );
    const targetQuat = new THREE.Quaternion().setFromEuler(euler);
    boneNode.quaternion.slerp(targetQuat, lerpAmount);
};

// Callback from MediaPipe Holistic
function onResults(results) {
  if (!currentVrm) return;
  
  const videoElement = document.getElementById('video-player');
  // Wait until video has loaded dimensions before trying to solve poses
  if (!videoElement || videoElement.videoWidth === 0) return;

  const sizeObj = { width: videoElement.videoWidth, height: videoElement.videoHeight };

  // Debug: see what Mediapipe provides visually
  if (!window.hasLoggedKeys) {
      const keysText = "Keys available: " + Object.keys(results).join(', ');
      document.getElementById('status-text').innerHTML = keysText + "<br>";
      window.hasLoggedKeys = true;
  }

  // 1. Pose (Arms, Spine, etc.)
  // Kalidokit requires both 2D and 3D pose landmarks for full body tracking
  const pose3D = results.poseWorldLandmarks || results.ea || results.za;

  // Update on-screen debug visually
  let debugStr = document.getElementById('status-text').innerHTML;
  if (!debugStr.includes("Tracking:")) debugStr += "Tracking: ";
  if (results.poseLandmarks) debugStr += "Pose ";
  if (pose3D) debugStr += "Pose3D ";
  if (results.faceLandmarks) debugStr += "Face ";
  if (results.leftHandLandmarks) debugStr += "LHand ";
  if (results.rightHandLandmarks) debugStr += "RHand ";
  document.getElementById('status-text').innerText = debugStr;

  if (results.poseLandmarks && pose3D) {
     try {
         const poseRig = Kalidokit.Pose.solve(pose3D, results.poseLandmarks, {
             runtime: 'mediapipe',
             video: sizeObj
         });
         
         if (poseRig) {
             // 3. Pose (Arms & Torso) - Adjusted to fix inverted/backward bending
             rigRotation('chest', poseRig.Spine, 0.25, 0.3);
             rigRotation('spine', poseRig.Spine, 0.45, 0.3);
             
             // The arm rotations from Kalidokit often assume a different forward-facing axis than VRMs natively use 
             // (usually Y-up Z-forward vs Y-up Z-backward). We must invert X and Z to fix the "backward bend".
             const fixArm = (rot) => ({ x: -rot.x, y: rot.y, z: -rot.z });
             
             rigRotation('rightUpperArm', fixArm(poseRig.RightUpperArm), 1, 0.3);
             rigRotation('rightLowerArm', fixArm(poseRig.RightLowerArm), 1, 0.3);
             rigRotation('leftUpperArm', fixArm(poseRig.LeftUpperArm), 1, 0.3);
             rigRotation('leftLowerArm', fixArm(poseRig.LeftLowerArm), 1, 0.3);
             
             rigRotation('leftUpperLeg', poseRig.LeftUpperLeg, 1, 0.3);
             rigRotation('leftLowerLeg', poseRig.LeftLowerLeg, 1, 0.3);
             rigRotation('rightUpperLeg', poseRig.RightUpperLeg, 1, 0.3);
             rigRotation('rightLowerLeg', poseRig.RightLowerLeg, 1, 0.3);
         }
     } catch (e) {
         document.getElementById('status-text').innerText = "Pose Error: " + e.message;
         console.warn("Pose solve error:", e);
     }
  } else if (results.poseLandmarks && !pose3D) {
      document.getElementById('status-text').innerText += " (Missing 3D Pose data!)";
  }
  
  // 2. Face & Head
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
                 m.setValue('blinkLeft', faceRig.eye.l);
                 m.setValue('blinkRight', faceRig.eye.r);
                 
                 m.setValue('aa', faceRig.mouth.shape.A);
                 m.setValue('ee', faceRig.mouth.shape.E);
                 m.setValue('ih', faceRig.mouth.shape.I);
                 m.setValue('oh', faceRig.mouth.shape.O);
                 m.setValue('ou', faceRig.mouth.shape.U);
             }
         }
     } catch (e) {
         console.warn("Face solve error:", e);
     }
  }

  // 3. Hands & Fingers
  const handleHand = (landmarks, handedness) => {
    if (!landmarks) return;
    
    // We are no longer mirroring the webcam, so Kalidokit should use the exact handedness MediaPipe reports
    const actualHandedness = handedness;
    
    try {
        const handFit = Kalidokit.Hand.solve(landmarks, actualHandedness);
        if (!handFit) return;
        
        const vrmPrefix = actualHandedness === "Left" ? "left" : "right";
        const kaliPrefix = actualHandedness;
        
        const fingers = ["Thumb", "Index", "Middle", "Ring", "Little"];
        const segments = ["Proximal", "Intermediate", "Distal"];
        fingers.forEach(finger => {
          segments.forEach(segment => {
            const rot = handFit[`${kaliPrefix}${finger}${segment}`];
            if (rot) {
                rigRotation(`${vrmPrefix}${finger}${segment}`, rot, 1, 0.3);
            }
          });
        });
        
        const wristRot = handFit[`${kaliPrefix}Wrist`];
        if (wristRot) {
            rigRotation(`${vrmPrefix}Hand`, { x: wristRot.x, y: wristRot.y, z: wristRot.z }, 1, 0.3);
        }
    } catch (e) {
        console.warn("Hand solve error:", e);
    }
  };

  handleHand(results.leftHandLandmarks, "Left");
  handleHand(results.rightHandLandmarks, "Right");
}

// ---- MediaPipe Setup ----
const videoElement = document.getElementById('video-player');

// Attach video upload logic
document.getElementById('video-upload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  videoElement.src = url;
  videoElement.play(); // Auto-play the video to start tracking
  
  // Start engine if it's the first time
  if (!mpSystem.isRunning) {
      mpSystem.isRunning = true;
      mpSystem.startVideoProcessing();
  }
});

const mpSystem = new MediaPipeHolisticSystem(videoElement, onResults);