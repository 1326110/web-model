const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let maskModel;
let faceDetector;
const labels = ["mask_incorrectly", "no_mask", "with_mask"];
let isProcessing = false;

async function setupApp() {
    try {
        statusText.innerText = "Requesting Camera...";
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;

        // Ensure canvas matches video size once stream starts
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        statusText.innerText = "Loading AI Models...";
        
        // Path is relative to index.html
        [maskModel, faceDetector] = await Promise.all([
            tf.loadLayersModel('model/model.json'),
            blazeface.load()
        ]);

        // Warm up GPU
        tf.tidy(() => maskModel.predict(tf.zeros([1, 224, 224, 3])));

        statusText.innerText = "System Live";
        runDetection();
    } catch (err) {
        statusText.innerText = "Error: " + err.message;
        console.error(err);
    }
}

async function runDetection() {
    // 1. Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (faceDetector && maskModel && !isProcessing) {
        isProcessing = true;

        // Detect face locations
        const faces = await faceDetector.estimateFaces(video, false);

        if (faces.length > 0) {
            for (const face of faces) {
                // Integer-safe coordinates for WebGL stability
                const startX = Math.floor(face.topLeft[0]);
                const startY = Math.floor(face.topLeft[1]);
                const endX = Math.floor(face.bottomRight[0]);
                const endY = Math.floor(face.bottomRight[1]);
                
                const w = endX - startX;
                const h = endY - startY;

                if (w > 0 && h > 0) {
                    // AI Prediction logic
                    const [labelIdx, probs] = tf.tidy(() => {
                        const img = tf.browser.fromPixels(video)
                            .slice([startY, startX, 0], [h, w, 3])
                            .resizeBilinear([224, 224])
                            .div(127.5).sub(1)
                            .expandDims(0);
                        
                        const out = maskModel.predict(img);
                        return [out.argMax(-1).dataSync()[0], out.dataSync()];
                    });

                    renderBox(startX, startY, w, h, labelIdx, probs);
                }
            }
        }
        isProcessing = false;
    }
    requestAnimationFrame(runDetection);
}

function renderBox(x, y, w, h, idx, probs) {
    const label = labels[idx];
    const conf = (probs[idx] * 100).toFixed(0);
    
    let color = "#FFD700"; // Yellow
    if (label === "with_mask") color = "#00FF00"; // Green
    if (label === "no_mask") color = "#FF0000";   // Red

    // Draw Rectangle
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    // Draw Label Label
    ctx.fillStyle = color;
    const txt = `${label.toUpperCase()} ${conf}%`;
    ctx.fillRect(x, y - 25, ctx.measureText(txt).width + 10, 25);

    ctx.fillStyle = "black";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(txt, x + 5, y - 7);
}

setupApp();
