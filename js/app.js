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

    if (frameCount % 2 === 0 && faceModel && model) {
        const predictions = await faceModel.estimateFaces(video, false);

        if (predictions.length > 0) {
            predictions.forEach(pred => {
                // 1. Extract raw coordinates
                const rawX1 = pred.topLeft[0];
                const rawY1 = pred.topLeft[1];
                const rawX2 = pred.bottomRight[0];
                const rawY2 = pred.bottomRight[1];

                tf.tidy(() => {
                    // 2. FORCE conversion to pure whole integers
                    // Math.trunc is used to remove any decimal possibility 
                    const x = Math.trunc(Math.max(0, rawX1));
                    const y = Math.trunc(Math.max(0, rawY1));
                    
                    // 3. Calculate width/height as whole integers
                    let w = Math.trunc(rawX2 - rawX1);
                    let h = Math.trunc(rawY2 - rawY1);

                    // 4. Final safety check: ensure slice stays inside video boundaries
                    if (x + w > video.videoWidth) w = Math.trunc(video.videoWidth - x);
                    if (y + h > video.videoHeight) h = Math.trunc(video.videoHeight - y);

                    // 5. CROP: The integer values now prevent the Shader Error
                    let faceTensor = tf.browser.fromPixels(video)
                        .slice([y, x, 0], [h, w, 3]);

                    // 6. RESIZE & NORMALIZE
                    faceTensor = tf.image.resizeBilinear(faceTensor, [224, 224]);
                    const offset = tf.scalar(127.5);
                    const normalized = faceTensor.sub(offset).div(offset).expandDims(0);

                    // 7. PREDICT
                    const prediction = model.predict(normalized);
                    const probabilities = prediction.dataSync();
                    const labelIndex = prediction.argMax(-1).dataSync()[0];

                    // 8. UI MAPPING
                    const labelText = labels[labelIndex];
                    const confidence = (probabilities[labelIndex] * 100).toFixed(1);
                    const displayText = `${labelText.replace("_", " ").toUpperCase()} ${confidence}%`;

                    let color = "yellow"; 
                    if (labelText === "with_mask") color = "#00FF00"; 
                    if (labelText === "no_mask") color = "#FF0000";   

                    drawBox(x, y, x + w, y + h, displayText, color);
                });
            });
        }
    }

    frameCount++;
    requestAnimationFrame(detect);
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
