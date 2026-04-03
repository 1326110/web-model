// ==============================
// GLOBAL VARIABLES
// ==============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let model;
let faceModel;
// Labels: 0: incorrect, 1: no mask, 2: with mask
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
        await tf.ready(); 
        statusText.innerText = "Loading AI models...";

        // 1. Load Mask Classifier (Ensures path is correct)
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
    ctx.font = "bold 16px sans-serif";
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x1, y1 - 22, textWidth + 10, 22);
    
    ctx.fillStyle = "black";
    ctx.fillText(text, x1 + 5, y1 - 6);
}

// ==============================
// MAIN DETECTION LOOP
// ==============================
async function detect() {
    // 1. Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. Run AI every 2nd frame for performance
    if (frameCount % 2 === 0 && faceModel && model) {
        const predictions = await faceModel.estimateFaces(video, false);

        if (predictions.length > 0) {
            predictions.forEach(pred => {
                // Raw coordinates from detector
                const x1 = pred.topLeft[0];
                const y1 = pred.topLeft[1];
                const x2 = pred.bottomRight[0];
                const y2 = pred.bottomRight[1];

                tf.tidy(() => {
                    // --- PREVENT SHADER ERROR: FORCE INTEGERS ---
                    const startX = Math.max(0, Math.floor(x1));
                    const startY = Math.max(0, Math.floor(y1));
                    
                    // Force dimensions to be whole numbers
                    const rawWidth = Math.floor(x2 - x1);
                    const rawHeight = Math.floor(y2 - y1);

                    // Ensure slice doesn't go off the edge of the video
                    const sliceW = Math.min(rawWidth, video.videoWidth - startX);
                    const sliceH = Math.min(rawHeight, video.videoHeight - startY);

                    // 1. Crop face from pixels
                    let faceTensor = tf.browser.fromPixels(video)
                        .slice([startY, startX, 0], [sliceH, sliceW, 3]);

                    // 2. Resize to 224x224 (Model Input Size)
                    faceTensor = tf.image.resizeBilinear(faceTensor, [224, 224]);

                    // 3. Normalize (-1 to 1 for MobileNetV2)
                    const offset = tf.scalar(127.5);
                    const normalized = faceTensor.sub(offset).div(offset).expandDims(0);

                    // 4. Run Prediction
                    const prediction = model.predict(normalized);
                    const probabilities = prediction.dataSync();
                    const labelIndex = prediction.argMax(-1).dataSync()[0];

                    // 5. Formatting UI Results
                    const labelText = labels[labelIndex];
                    const confidence = (probabilities[labelIndex] * 100).toFixed(1);
                    const displayText = `${labelText.replace("_", " ").toUpperCase()} ${confidence}%`;

                    let color = "yellow"; // mask_incorrectly
                    if (labelText === "with_mask") color = "#00FF00"; // Green
                    if (labelText === "no_mask") color = "#FF0000";   // Red

                    drawBox(startX, startY, startX + sliceW, startY + sliceH, displayText, color);
                }); // End tf.tidy
            }); // End forEach
        } // End if predictions
    } // End if frameCount

    frameCount++;
    requestAnimationFrame(detect);
}

// ==============================
// INITIALIZE APP
// ==============================
(async () => {
    await loadModels();
    await setupCamera();

    // Match canvas size to video stream
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    statusText.innerText = "System Live";
    detect();
})();
