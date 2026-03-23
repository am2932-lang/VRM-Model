import { PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export class MediaPipeTasksSystem {
  constructor(videoElement, onPoseResult, onHandResult) {
    this.videoElement = videoElement;
    this.onPoseResult = onPoseResult;
    this.onHandResult = onHandResult;
    this.poseLandmarker = null;
    this.handLandmarker = null;
    this.isRunning = false;
    this._init().catch(e => {
      console.error('MediaPipe Tasks init failed:', e);
      document.getElementById('status-text').innerText = 'Error loading AI models: ' + e.message;
    });
  }

  async _init() {
    document.getElementById('status-text').innerText = 'Loading AI models...';

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );

    // BlazePose-GHUM Full — highest accuracy pose model
    this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.6,
      minPosePresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
      outputSegmentationMasks: false,
    });

    // Dedicated HandLandmarker — more accurate than Holistic hands
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    document.getElementById('status-text').innerText = 'Models ready — upload a video!';
    console.log('MediaPipe Tasks API ready.');
  }

  startVideoProcessing() {
    let lastVideoTime = -1;

    const processFrame = () => {
      const vid = this.videoElement;

      if (vid.paused || vid.ended || !vid.videoWidth || !this.poseLandmarker) {
        requestAnimationFrame(processFrame);
        return;
      }

      if (vid.currentTime !== lastVideoTime) {
        lastVideoTime = vid.currentTime;
        const ts = performance.now();

        try {
          const poseResult = this.poseLandmarker.detectForVideo(vid, ts);
          this.onPoseResult(poseResult);
        } catch (e) { console.warn('Pose detect error:', e); }

        try {
          const handResult = this.handLandmarker.detectForVideo(vid, ts);
          this.onHandResult(handResult);
        } catch (e) { console.warn('Hand detect error:', e); }
      }

      requestAnimationFrame(processFrame);
    };

    processFrame();
  }
}
