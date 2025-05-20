let model;

fetch('tfjs_model/model.json')
    .then(response => response.json())
    .then(data => console.log("Model JSON Loaded:", data))
    .catch(err => console.error("Failed to fetch model.json:", err));
    window.addEventListener("DOMContentLoaded", async () => {
        try {
            model = await tf.loadGraphModel('tfjs_model/model.json'); // Load as a Graph Model
            console.log("✅ Model loaded successfully!");
        } catch (err) {
            console.error("❌ Failed to load model:", err);
            alert("Error loading model. Ensure 'model.json' and weight files exist.");
        }
    
        document.getElementById("upload").addEventListener("change", previewImage);
    });

function previewImage(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        const imgElement = document.getElementById("preview");
        imgElement.src = e.target.result;
        document.getElementById("result").innerText = "";
    };

    reader.readAsDataURL(file);
}

async function predict() {
    const imageElement = document.getElementById("preview");

    if (!imageElement.src || imageElement.src === window.location.href) {
        alert("Please upload an image first.");
        return;
    }

    if (!model) {
        alert("Model not loaded yet.");
        return;
    }

    document.getElementById("loading").style.display = "block";
    document.getElementById("result").innerText = "";

    try {
        const tensor = tf.browser
            .fromPixels(imageElement)
            .resizeNearestNeighbor([100, 100]) // Match input size
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims(); // Add batch dimension

        const prediction = await model.predict(tensor).data();
        console.log("📊 Prediction:", prediction);

        const classes = ["Belum Matang", "Matang", "Terlalu Matang"];
        const maxIndex = prediction.indexOf(Math.max(...prediction));
        const confidence = (prediction[maxIndex] * 100).toFixed(2);

        document.getElementById("result").innerText = `Result: ${classes[maxIndex]} (${confidence}%)`;
    } catch (err) {
        console.error("❌ Error during prediction:", err);
        alert("An error occurred. Check the console for details.");
    } finally {
        document.getElementById("loading").style.display = "none";
    }
}