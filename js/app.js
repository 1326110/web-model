// Hardcoded labels in alphabetical order (matches Sklearn's LabelEncoder)
const TARGET_LABELS = ["mask_incorrectly", "no_mask", "with_mask"]; 
let model;

async function setupWebcam() {
    const video = document.getElementById('webcam');
    // 'facingMode: environment' uses the back camera; 'user' uses the selfie camera
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" }, 
        audio: false 
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(video);
    });
}

async function runInference() {
    // 1. Load the model from your folder
    // Ensure 'tfjs_model' folder is in the same directory as this file
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Model successfully loaded!");

    const video = await setupWebcam();
    
    // Create an infinite loop for real-time detection
    while (true) {
        const predictionData = tf.tidy(() => {
            // 2. Capture the frame and resize to 224x224 (your IMG_SIZE)
            let img = tf.browser.fromPixels(video)
                .resizeNearestNeighbor([224, 224])
                .toFloat();
            
            // 3. Apply MobileNetV2 preprocessing: (pixel - 127.5) / 127.5
            // This scales 0-255 pixels to a range of [-1, 1]
            const offset = tf.scalar(127.5);
            const normalized = img.sub(offset).div(offset);
            
            // Add a batch dimension: [224, 224, 3] -> [1, 224, 224, 3]
            const batched = normalized.expandDims(0);
            
            return model.predict(batched).dataSync();
        });

        // 4. Find the index with the highest probability
        const maxProbability = Math.max(...predictionData);
        const predictionIndex = predictionData.indexOf(maxProbability);
        
        // 5. Update the UI with the label name
        const resultText = document.getElementById('prediction');
        const confidence = (maxProbability * 100).toFixed(1);
        
        resultText.innerText = `${TARGET_LABELS[predictionIndex]} (${confidence}%)`;

        // Change color based on safety
        if (predictionIndex === 2) resultText.style.color = "green"; // with_mask
        else resultText.style.color = "red"; // no_mask or incorrect

        // Give the browser a millisecond to breathe
        await tf.nextFrame();
    }
}

runInference();
