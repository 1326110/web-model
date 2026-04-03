// ==============================
// GLOBAL VARIABLES
// ==============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let model;
let faceModel;
// Alphabetical order: mask_incorrectly (0), no_mask (1), with_mask (2)
let labels = ["mask_incorrectly", "no_mask", "with_mask"]; 
let frameCount = 0;

// ==============================
// SETUP CAMERA
// ==============================
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
            width: { ideal: 640 }, 
            height: { ideal: 480 },
            facingMode: "user" 
        },
        audio: false
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

// ==============================
// LOAD MODELS
// ==============================
async function loadModels() {
    try {
        // Use WebGL for speed. If it fails, TF.js falls back to CPU automatically.
        await tf.ready(); 
        statusText.innerText = "Loading AI models...";

        // 1. Load Mask Classifier (Path: model/model.json)
        model = await tf.loadLayersModel('model/model.json');
        console.log("Mask Model Loaded");

        // 2. Load Face Detector (BlazeFace)
        faceModel = await blazeface.load();
        console.log("Face Detector Loaded");

        statusText.innerText = "Models loaded. Starting camera...";

    } catch (err) {
        console.error("Critical Load Error:", err);
        statusText.innerText = "Error: " + err.message;
    }
}

// ==============================
// DRAWING BOXES
// ==============================
function drawBox(x1, y1, x2, y2, text, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = color;
    ctx.font = "bold 16px Arial";
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x1, y1 - 22, textWidth + 10, 22);
    
    ctx.fillStyle = "black";
    ctx.fillText(text, x1 + 5, y1 - 6);
}

// ==============================
// MAIN DETECTION LOOP
// ==============================
async function detect() {
    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Run AI every 2nd frame for performance boost
    if (frameCount % 2 === 0 && faceModel && model) {
        const predictions = await faceModel.estimateFaces(video, false);

        if (predictions.length > 0) {
            predictions.forEach(pred => {
                // Get raw coordinates
                const rawX1 = pred.topLeft[0];
                const rawY1 = pred.topLeft[1];
                const rawX2 = pred.bottomRight[0];
                const rawY2 = pred.bottomRight[1];

                tf.tidy(() => {
                    // --- PREVENT SHADER ERROR: FORCE INTEGERS ---
                    const x = Math.max(0, Math.floor(rawX1));
                    const y = Math.max(0, Math.floor(rawY1));
                    let w = Math.floor(rawX2 - rawX1);
                    let h = Math.floor(rawY2 - rawY1);

                    // Safety boundaries
                    if (x + w > video.videoWidth) w = video.videoWidth - x;
                    if (y + h > video.videoHeight) h = video.videoHeight - y;

                    // 1. Crop face from video
                    let faceTensor = tf.browser.fromPixels(video)
                        .slice([y, x, 0], [h, w, 3]);

                    // 2. Resize to 224x224 (Matches MobileNetV2 training)
                    faceTensor = tf.image.resizeBilinear(faceTensor, [224, 224]);

                    // 3. Preprocess: Scales 0..255 to -1..1
                    const offset = tf.scalar(127.5);
                    const normalized = faceTensor.sub(offset).div(offset).expandDims(0);

                    // 4. Predict Mask Status
                    const prediction = model.predict(normalized);
                    const probabilities = prediction.dataSync();
                    const labelIndex = prediction.argMax(-1).dataSync()[0];

                    // 5. Result Formatting
                    const labelText = labels[labelIndex];
                    const confidence = (probabilities[labelIndex] * 100).toFixed(1);
                    const displayText = `${labelText.replace("_", " ").toUpperCase()} ${confidence}%`;

                    // Color Coding
                    let color = "#FFD700"; // Gold/Yellow for incorrect
                    if (labelText === "with_mask") color = "#00FF00"; // Green
                    if (labelText === "no_mask") color = "#FF0000";   // Red

                    drawBox(x, y, x + w, y + h, displayText, color);
                });
            });
        }
    }

    frameCount++;
    requestAnimationFrame(detect);
}

// ==============================
// INIT APP
// ==============================
(async () => {
    await loadModels();
    await setupCamera();

    // Match canvas to camera video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    statusText.innerText = "System Live";
    detect();
})();
