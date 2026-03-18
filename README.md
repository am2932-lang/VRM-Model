# Full-Body Video VRM Tracker

A real-time, browser-based motion capture system that extracts full-body motion (pose, face, and hands) from a video file and applies it to a 3D VRM avatar.

This project completely runs entirely in the browser using JavaScript, removing the need for heavy Python scripts or 3D software like Blender.

## Tech Stack
* **[Three.js](https://threejs.org/)**: Renders the 3D scene and the avatar.
* **[@pixiv/three-vrm](https://github.com/pixiv/three-vrm)**: Loads and manages `.vrm` avatar models.
* **[MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)**: Provides state-of-the-art AI tracking to extract 3D landmarks for the body, face, and hands from standard video.
* **[Kalidokit](https://github.com/yeemachine/kalidokit)**: A kinematic solver that calculates the exact Euler angles and Blendshapes needed to rotate the VRM bones based on the MediaPipe landmarks.
* **[Vite](https://vitejs.dev/)**: A fast local development server.

## Features
* **Video Upload**: Upload any standard `.mp4` video.
* **Avatar Upload**: Swap out the default avatar by uploading any valid `.vrm` file.
* **Full-Body Tracking**: Tracks the spine, arms, legs, individual fingers, and facial expressions (blinking, speaking).
* **Maximum Accuracy**: Configured to use MediaPipe's `Model Complexity 2` for the most accurate joint estimation from pre-recorded videos.

## How to Run

1. **Install Dependencies:**
   Ensure you have Node.js installed, then run:
   ```bash
   npm install
   ```

2. **Start the Development Server:**
   ```bash
   npx vite
   ```

3. **Open the App:**
   Navigate into your browser to `http://localhost:5173`.
   
4. **Usage:**
   - Upload a `.vrm` model via the **Avatar Controls** section.
   - Upload an `.mp4` video via the **Video Target** section.
   - The avatar will automatically begin tracking the human in the video.

## Architecture & Code Structure
* `index.html`: The UI layout, video player, and 3D canvas.
* `app.js`: Contains the Three.js rendering loop, VRM loading logic, and the Kalidokit solvers mapping data to bones.
* `utils/holistic.js`: Handles the MediaPipe AI instance, frame-by-frame video processing, and landmark extraction.
