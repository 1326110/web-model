const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let maskModel;
let faceDetector;
const labels = ["mask_incorrectly", "no_mask", "with_mask"];
let isProcessing = false;

// 1. Initialize System
async function init() {
    try {
        statusText.innerText = "Requesting Camera...";
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;

        await new Promise((resolve) => (video.onloadedmetadata = resolve));
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        statusText.innerText = "Loading AI Models...";
        
        // Parallel loading for maximum speed
        [maskModel, faceDetector] = await Promise.all([
            tf.loadLayersModel('model/model.json'),
            blazeface.load()
        ]);

        // GPU Warmup
        tf.tidy(() => maskModel.predict(tf.zeros([1, 224, 224, 3])));

        statusText.innerText = "System Live - Monitoring";
        detectFrame();
    } catch (err) {
        statusText.innerText = "Critical Error: " + err.message;
        console.error(err);
    }
}

// 2. Detection Loop
async function detectFrame() {
    if (isProcessing) {
        requestAnimationFrame(detectFrame);
        return;
    }

    // Draw the current camera frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (faceDetector && maskModel) {
        isProcessing = true;

        // Stage 1: Locate Faces
        const faces = await faceDetector.estimateFaces(video, false);

        if (faces.length > 0) {
            for (const face of faces) {
                // Ensure whole numbers for WebGL stability
                const x1 = Math.floor(face.topLeft[0]);
                const y1 = Math.floor(face.topLeft[1]);
                const x2 = Math.floor(face.bottomRight[0]);
                const y2 = Math.floor(face.bottomRight[1]);

                const width = Math.min(x2 - x1, video.videoWidth - x1);
                const height = Math.min(y2 - y1, video.videoHeight - y1);

                if (width > 0 && height > 0) {
                    // Stage 2: Predict Mask Status
                    const [prediction, probabilities] = tf.tidy(() => {
                        const tensor = tf.browser.fromPixels(video)
                            .slice([y1, x1, 0], [height, width, 3])
                            .resizeBilinear([224, 224])
                            .div(127.5).sub(1) // Normalize [-1, 1]
                            .expandDims(0);
                        
                        const result = maskModel.predict(tensor);
                        return [
                            result.argMax(-1).dataSync()[0],
                            result.dataSync()
                        ];
                    });

                    drawUI(x1, y1, width, height, prediction, probabilities);
                }
            }
        }
        isProcessing = false;
    }
    requestAnimationFrame(detectFrame);
}

// 3. UI Drawing Function
function drawUI(x, y, w, h, labelIdx, probs) {
    const label = labels[labelIdx];
    const confidence = (probs[labelIdx] * 100).toFixed(0);
    
    let color = "#FFD700"; // Yellow
    if (label === "with_mask") color = "#00FF00"; // Green
    if (label === "no_mask") color = "#FF0000";   // Red

    // Draw Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    // Draw Tag
    ctx.fillStyle = color;
    const text = `${label.replace("_", " ").toUpperCase()} ${confidence}%`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x, y - 25, textWidth + 10, 25);

    // Draw Text
    ctx.fillStyle = "black";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(text, x + 5, y - 7);
}

init();
