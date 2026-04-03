// ==============================
// 1. GLOBALS & CONFIG
// ==============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let maskModel;     // The custom model (Expert)
let faceDetector;  // BlazeFace (Scout)

// Indices: 0: mask_incorrectly, 1: no_mask, 2: with_mask
const labels = ["mask_incorrectly", "no_mask", "with_mask"];
let isProcessing = false;

// ==============================
// 2. INITIALIZATION
// ==============================
async function init() {
    try {
        statusText.innerText = "Starting Camera...";
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;

        // Wait for video to be ready to get dimensions
        await new Promise((resolve) => (video.onloadedmetadata = resolve));
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        statusText.innerText = "Loading AI Models...";
        
        // Load both models at the same time
        [maskModel, faceDetector] = await Promise.all([
            tf.loadLayersModel('model/model.json'),
            blazeface.load()
        ]);

        // GPU Warmup: Prime the engine so the first frame isn't laggy
        tf.tidy(() => maskModel.predict(tf.zeros([1, 224, 224, 3])));

        statusText.innerText = "System Live";
        detectFrame();
    } catch (err) {
        statusText.innerText = "Error: " + err.message;
        console.error(err);
    }
}

// ==============================
// 3. DETECTION LOOP
// ==============================
async function detectFrame() {
    if (isProcessing) {
        requestAnimationFrame(detectFrame);
        return;
    }

    // Always draw the background video first
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (faceDetector && maskModel) {
        isProcessing = true;

        // STEP 1: Find faces (The Scout)
        const faces = await faceDetector.estimateFaces(video, false);

        if (faces.length > 0) {
            for (const face of faces) {
                // Force Integers to prevent WebGL/Fragment Shader errors
                const x1 = Math.floor(face.topLeft[0]);
                const y1 = Math.floor(face.topLeft[1]);
                const x2 = Math.floor(face.bottomRight[0]);
                const y2 = Math.floor(face.bottomRight[1]);

                const width = Math.min(x2 - x1, video.videoWidth - x1);
                const height = Math.min(y2 - y1, video.videoHeight - y1);

                if (width > 0 && height > 0) {
                    // STEP 2: Predict Mask (The Expert)
                    // tf.tidy handles memory cleanup automatically
                    const [prediction, probabilities] = tf.tidy(() => {
                        const tensor = tf.browser.fromPixels(video)
                            .slice([y1, x1, 0], [height, width, 3]) // Crop face
                            .resizeBilinear([224, 224])             // Standardize size
                            .div(127.5).sub(1)                      // Normalize (-1 to 1)
                            .expandDims(0);                         // Create batch
                        
                        const result = maskModel.predict(tensor);
                        return [
                            result.argMax(-1).dataSync()[0], // Index of highest probability
                            result.dataSync()                // The raw percentage list
                        ];
                    });

                    // STEP 3: Draw results to screen
                    drawUI(x1, y1, width, height, prediction, probabilities);
                }
            }
        }
        isProcessing = false;
    }

    // Keep the loop running
    requestAnimationFrame(detectFrame);
}

// ==============================
// 4. UI RENDERING
// ==============================
function drawUI(x, y, w, h, labelIdx, probs) {
    const label = labels[labelIdx];
    const confidence = (probs[labelIdx] * 100).toFixed(1);
    
    // Set color based on classification
    let color = "#FFD700"; // Yellow (Incorrect)
    if (label === "with_mask") color = "#00FF00"; // Green
    if (label === "no_mask") color = "#FF0000";   // Red

    // 1. Draw the box
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    // 2. Draw the label background
    ctx.fillStyle = color;
    const text = `${label.toUpperCase()} ${confidence}%`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x, y - 25, textWidth + 10, 25);

    // 3. Draw the text
    ctx.fillStyle = "#000000";
    ctx.font = "bold 16px Arial";
    ctx.fillText(text, x + 5, y - 7);
}

// Kick off the script
init();
