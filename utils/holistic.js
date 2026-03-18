export class MediaPipeHolisticSystem {
  constructor(videoElement, onResultsCallback) {
    this.videoElement = videoElement;
    this.onResultsCallback = onResultsCallback;

    // Initialize MediaPipe Holistic via window globals (loaded from CDN)
    this.holistic = new window.Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      }
    });

    this.holistic.setOptions({
      modelComplexity: 2, // 2 is the most accurate (but heaviest) model
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    this.holistic.onResults(this.onResultsCallback);
  }

  startVideoProcessing() {
    let lastVideoTime = -1;
    
    const processVideo = async () => {
      if (this.videoElement.paused || this.videoElement.ended || !this.videoElement.videoWidth) {
         // Re-queue the loop immediately without processing if the video isn't ready
         requestAnimationFrame(processVideo);
         return;
      }
      
      if (this.videoElement.currentTime !== lastVideoTime) {
         lastVideoTime = this.videoElement.currentTime;
         try {
             await this.holistic.send({ image: this.videoElement });
         } catch (e) {
             console.warn("Holistic Send Error:", e);
         }
      }
      requestAnimationFrame(processVideo);
    };
    
    // Kick off loop
    processVideo();
  }
}
