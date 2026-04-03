const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");

let maskModel;
let faceDetector;
const labels = ["mask_incorrectly", "no_mask", "with_mask"];

async function startApp() {
    try {
        // 1. Setup Camera
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;

        // Ensure canvas matches video pixels
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        // 2. Load Models
        statusText.innerText = "Loading AI Models...";
        
        // This path looks inside your 'model' folder
        [maskModel, faceDetector] = await Promise.all([
            tf.loadLayersModel('model/model.json'),
            blazeface.load()
        ]);

        statusText.innerText = "Scanning for Faces...";
        detect();
    } catch (err) {
        statusText.innerText = "Error: " + err.message;
        console.error(err);
    }
}

async function detect() {
    // Clear the canvas every frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Step 1: Find Face
    const predictions = await faceDetector.estimateFaces(video, false);

    if (predictions.length > 0) {
        for (const pred of predictions) {
            const [x1, y1] = pred.topLeft;
            const [x2, y2] = pred.bottomRight;
            const width = x2 - x1;
            const height = y2 - y1;

            // Step 2: Crop Face and Predict Mask
            const [labelIdx, probs] = tf.tidy(() => {
                const img = tf.browser.fromPixels(video)
                    .slice([Math.floor(y1), Math.floor(x1), 0], [Math.floor(height), Math.floor(width), 3])
                    .resizeBilinear([224, 224])
                    .div(127.5).sub(1) // Normalization
                    .expandDims(0);
                
                const output = maskModel.predict(img);
                return [output.argMax(-1).dataSync()[0], output.dataSync()];
            });

            // Step 3: Draw Result
            const label = labels[labelIdx];
            const confidence = (probs[labelIdx] * 100).toFixed(0);
            
            // Color Logic
            let color = "#FFD700"; // Yellow
            if (label === "with_mask") color = "#00FF00"; // Green
            if (label === "no_mask") color = "#FF0000";   // Red

            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(x1, y1, width, height);

            ctx.fillStyle = color;
            ctx.font = "bold 16px sans-serif";
            ctx.fillText(`${label.toUpperCase()} ${confidence}%`, x1, y1 > 20 ? y1 - 5 : 20);
        }
    }

    requestAnimationFrame(detect);
}

startApp();
