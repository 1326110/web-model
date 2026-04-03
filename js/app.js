// ==============================
// GLOBAL VARIABLES
// ==============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let model;
let faceModel;
// We will hardcode labels for direct use, but load labels.json as a fallback
let labels = ["mask_incorrectly", "no_mask", "with_mask"]; 
let frameCount = 0;

// ==============================
// SETUP CAMERA
// ==============================
async function setupCamera() {
    // Optimization for mobile: Use 'user' for selfie, 'environment' for back camera
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
/* async function loadModels() {
    try {
        await tf.setBackend("webgl");
        await tf.ready();

        statusText.innerText = "Loading AI models...";

        // 1. Load your trained MobileNetV2 Mask Classifier
        // Path matches your project structure: model/model.json
        model = await tf.loadLayersModel('model/model.json');

        // 2. Load Face Detector (BlazeFace)
        faceModel = await blazeface.load();

        statusText.innerText = "Models loaded. Starting camera...";

    } catch (err) {
        console.error("Model Load Error:", err);
        statusText.innerText = "Error loading model! Check console.";
    }
}*/
async function loadModels() {
    try {
        await tf.ready(); // Ensure TF.js is fully initialized
        statusText.innerText = "Loading AI models...";

        // Load mask classifier
        model = await tf.loadLayersModel('model/model.json');
        console.log("Mask Model Loaded");

        // Load face detector
        faceModel = await blazeface.load();
        console.log("Face Detector Loaded");

        if (!faceModel) {
            throw new Error("BlazeFace failed to initialize");
        }

        statusText.innerText = "Models loaded. Starting camera...";

    } catch (err) {
        console.error("Critical Load Error:", err);
        statusText.innerText = "Error: " + err.message;
    }
}

// ==============================
// DRAWING & FORMATTING
// ==============================
function drawBox(x1, y1, x2, y2, text, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = color;
    ctx.font = "bold 18px sans-serif";
    // Draw background for text
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
    
    ctx.fillStyle = "black";
    ctx.fillText(text, x1 + 5, y1 - 7);
}

// ==============================
// MAIN DETECTION LOOP
// ==============================
async function detect() {
    // Draw the current video frame to the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Run detection every 2nd frame for better mobile performance
    if (frameCount % 2 === 0) {
        const predictions = await faceModel.estimateFaces(video, false);

        if (predictions.length > 0) {
            predictions.forEach(pred => {
                const x1 = pred.topLeft[0];
                const y1 = pred.topLeft[1];
                const x2 = pred.bottomRight[0];
                const y2 = pred.bottomRight[1];
                const width = x2 - x1;
                const height = y2 - y1;


                
                tf.tidy(() => {
    // 1. Force all coordinates to be Integers using Math.floor or Math.round
    const startY = Math.max(0, Math.floor(y1));
    const startX = Math.max(0, Math.floor(x1));
    
    // 2. Ensure dimensions are also whole numbers
    const sliceH = Math.min(video.videoHeight - startY, Math.floor(height));
    const sliceW = Math.min(video.videoWidth - startX, Math.floor(width));

    // 3. Crop using the cleaned integers
    let face = tf.browser.fromPixels(video)
        .slice([startY, startX, 0], [sliceH, sliceW, 3]);

    // Resize and normalize stay the same
    face = tf.image.resizeBilinear(face, [224, 224]);
    const offset = tf.scalar(127.5);
    const normalized = face.sub(offset).div(offset).expandDims(0);

    const prediction = model.predict(normalized);
    // ... rest of your code
});

                    // 2. Resize to 224x224 (Matches your MobileNetV2 training)
                    face = tf.image.resizeBilinear(face, [224, 224]);

                    // 3. MobileNetV2 Preprocessing: (pixel / 127.5) - 1
                    // This is critical for accuracy!
                    const offset = tf.scalar(127.5);
                    const normalized = face.sub(offset).div(offset).expandDims(0);

                    // 4. Predict
                    const prediction = model.predict(normalized);
                    const probabilities = prediction.dataSync();
                    const labelIndex = prediction.argMax(-1).dataSync()[0];

                    // 5. Map to your specific labels
                    const labelText = labels[labelIndex];
                    const confidence = (probabilities[labelIndex] * 100).toFixed(1);
                    
                    // Display formatting
                    const displayText = `${labelText.replace("_", " ").toUpperCase()} ${confidence}%`;

                    // Color logic based on your 3 labels
                    let color = "yellow"; // mask_incorrectly
                    if (labelText === "with_mask") color = "#00FF00"; // Green
                    if (labelText === "no_mask") color = "#FF0000";   // Red

                    drawBox(x1, y1, x2, y2, displayText, color);
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

    // Set canvas dimensions to match the actual video stream
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    statusText.innerText = "Scanning...";
    detect();
})();
