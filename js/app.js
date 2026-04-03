// ==============================
// GLOBAL VARIABLES
// ==============================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let model;
let faceModel;
const labels = ["mask_incorrectly", "no_mask", "with_mask"]; 
let frameCount = 0;
let isDetecting = false;

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
// LOAD MODELS & WARMUP
// ==============================
async function loadModels() {
    try {
        await tf.ready(); 
        // Optimization: Set flag for high-performance power preference
        if (tf.getBackend() === 'webgl') {
            const gl = canvas.getContext('webgl');
            gl?.getExtension('WEBGL_debug_renderer_info');
        }

        statusText.innerText = "Loading AI models...";
        [model, faceModel] = await Promise.all([
            tf.loadLayersModel('model/model.json'),
            blazeface.load()
        ]);

        // WARMUP: Run a fake tensor through the model to prime the GPU
        tf.tidy(() => {
            model.predict(tf.zeros([1, 224, 224, 3]));
        });

        statusText.innerText = "System Live";
    } catch (err) {
        console.error("Load Error:", err);
        statusText.innerText = "Error: " + err.message;
    }
}

// ==============================
// OPTIMIZED DRAWING
// ==============================
function drawBox(x1, y1, x2, y2, text, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = color;
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x1, y1 - 22, textWidth + 10, 22);
    
    ctx.fillStyle = "black";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText(text, x1 + 5, y1 - 6);
}

// ==============================
// MAIN DETECTION LOOP
// ==============================
async function detect() {
    if (isDetecting) {
        requestAnimationFrame(detect);
        return;
    }

    // 1. Static frame update
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. Throttle detection (Every 2nd or 3rd frame)
    if (frameCount % 2 === 0 && faceModel && model) {
        isDetecting = true;
        
        const predictions = await faceModel.estimateFaces(video, false);

        if (predictions.length > 0) {
            for (const pred of predictions) {
                const x1 = Math.floor(pred.topLeft[0]);
                const y1 = Math.floor(pred.topLeft[1]);
                const x2 = Math.floor(pred.bottomRight[0]);
                const y2 = Math.floor(pred.bottomRight[1]);

                // Calculate safe dimensions
                const sliceW = Math.min(x2 - x1, video.videoWidth - x1);
                const sliceH = Math.min(y2 - y1, video.videoHeight - y1);

                if (sliceW > 0 && sliceH > 0) {
                    const result = tf.tidy(() => {
                        const img = tf.browser.fromPixels(video);
                        const face = img.slice([y1, x1, 0], [sliceH, sliceW, 3])
                                        .resizeBilinear([224, 224])
                                        .reshape([1, 224, 224, 3]);
                        
                        // Normalization: (img / 127.5) - 1
                        const normalized = face.toFloat().sub(127.5).div(127.5);
                        return model.predict(normalized);
                    });

                    // Async retrieval to prevent UI blocking
                    const probs = await result.data();
                    const labelIdx = result.argMax(-1).dataSync()[0];
                    result.dispose(); // Manual cleanup

                    const labelText = labels[labelIdx];
                    const conf = (probs[labelIdx] * 100).toFixed(1);
                    
                    let color = "#FFD700"; 
                    if (labelText === "with_mask") color = "#00FF00"; 
                    if (labelText === "no_mask") color = "#FF0000";   

                    drawBox(x1, y1, x1 + sliceW, y1 + sliceH, `${labelText.toUpperCase()} ${conf}%`, color);
                }
            }
        }
        isDetecting = false;
    }

    frameCount++;
    requestAnimationFrame(detect);
}

// ==============================
// INIT
// ==============================
(async () => {
    await loadModels();
    await setupCamera();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detect();
})();
