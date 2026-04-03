const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let model, faceModel;
const labels = ["mask_incorrectly", "no_mask", "with_mask"];

async function init() {
    try {
        // 1. Setup Camera
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        
        await new Promise(r => video.onloadedmetadata = r);
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // 2. Load Models
        statusText.innerText = "Loading Models...";
        [model, faceModel] = await Promise.all([
            tf.loadLayersModel('model/model.json'),
            blazeface.load()
        ]);

        statusText.innerText = "System Active";
        detect();
    } catch (e) {
        statusText.innerText = "Error: " + e.message;
        console.error(e);
    }
}

async function detect() {
    // Clear canvas and draw fresh video frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Perform detection
    const predictions = await faceModel.estimateFaces(video, false);

    if (predictions.length > 0) {
        for (const pred of predictions) {
            const startX = Math.floor(pred.topLeft[0]);
            const startY = Math.floor(pred.topLeft[1]);
            const endX = Math.floor(pred.bottomRight[0]);
            const endY = Math.floor(pred.bottomRight[1]);
            const width = endX - startX;
            const height = endY - startY;

            // --- AI PREDICTION BLOCK ---
            const result = tf.tidy(() => {
                const img = tf.browser.fromPixels(video);
                const face = img.slice([startY, startX, 0], [height, width, 3])
                                .resizeBilinear([224, 224])
                                .div(127.5).sub(1) // Normalization
                                .expandDims(0);
                return model.predict(face);
            });

            const probs = await result.data();
            const labelIdx = result.argMax(-1).dataSync()[0];
            result.dispose();

            // --- DRAWING BLOCK ---
            const label = labels[labelIdx];
            const color = label === "with_mask" ? "#00FF00" : (label === "no_mask" ? "#FF0000" : "#FFFF00");
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(startX, startY, width, height);

            ctx.fillStyle = color;
            ctx.font = "bold 18px Arial";
            const txt = `${label.toUpperCase()} ${(probs[labelIdx] * 100).toFixed(0)}%`;
            ctx.fillRect(startX, startY - 25, ctx.measureText(txt).width + 10, 25);
            
            ctx.fillStyle = "black";
            ctx.fillText(txt, startX + 5, startY - 7);
        }
    }

    requestAnimationFrame(detect);
}

init();
