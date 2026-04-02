// ==============================
// GLOBAL VARIABLES
// ==============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let model;
let faceModel;
let labels = {};
let frameCount = 0;


// ==============================
// SETUP CAMERA
// ==============================
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
    });

    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(video);
    });
}


// ==============================
// LOAD MODELS + LABELS
// ==============================
async function loadModels() {
    try {
        // Set backend for speed
        await tf.setBackend("webgl");
        await tf.ready();

        statusText.innerText = "Loading AI model...";

        // Load mask classifier
        model = await tf.loadLayersModel('https://1326110.github.io/web-model/model/model.json');//tf.loadLayersModel("model/model.json");

        // Load face detector
        faceModel = await blazeface.load();

        // Load labels
        const res = await fetch("labels.json");
        labels = await res.json();

        statusText.innerText = "Model loaded. Starting camera...";

    } catch (err) {
        console.error(err);
        statusText.innerText = "Error loading model!";
    }
}


// ==============================
// DRAW BOX
// ==============================
function drawBox(x1, y1, x2, y2, text, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = color;
    ctx.font = "16px Arial";
    ctx.fillText(text, x1, y1 - 5);
}


// ==============================
// FORMAT LABEL
// ==============================
function formatLabel(label) {
    return label.replace("_", " ").toUpperCase();
}


// ==============================
// MAIN DETECTION LOOP
// ==============================
async function detect() {

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Run face detection every 2 frames (performance boost)
    if (frameCount % 2 === 0) {

        const predictions = await faceModel.estimateFaces(video, false);

        // Wrap in tf.tidy to avoid memory leaks
        tf.tidy(() => {

            predictions.forEach(pred => {

                // Get bounding box
                const x1 = Math.floor(pred.topLeft[0]);
                const y1 = Math.floor(pred.topLeft[1]);
                const x2 = Math.floor(pred.bottomRight[0]);
                const y2 = Math.floor(pred.bottomRight[1]);

                // Crop face from video
                const faceTensor = tf.browser.fromPixels(video)
                    .slice([y1, x1, 0], [y2 - y1, x2 - x1, 3])
                    .resizeBilinear([128, 128])
                    .div(255.0)
                    .expandDims(0);

                // Predict
                const prediction = model.predict(faceTensor);
                const labelIndex = prediction.argMax(-1).dataSync()[0];

                // Get label from JSON
                const labelText = labels[labelIndex];
                const displayText = formatLabel(labelText);

                // Color mapping
                const colorMap = {
                    "masked": "lime",
                    "no_mask": "red",
                    "incorrect_mask": "yellow"
                };

                const color = colorMap[labelText] || "white";

                drawBox(x1, y1, x2, y2, displayText, color);

            });

        });
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

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    statusText.innerText = "Running...";

    detect();

})();
