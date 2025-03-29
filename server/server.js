const express = require("express");
const cors = require("cors");
const { PythonShell } = require("python-shell");

const app = express();
const PORT = 5000;

// Middleware
app.use(express.json());
app.use(cors());

// API Endpoint for Predictions
app.post("/predict", async (req, res) => {
    console.log(`received /predict, req: `);
    console.log(req.body);
    try {
        const inputData = req.body; // Receive JSON input

        // Call the Python script for prediction
        let options = {
            mode: "json",
            pythonOptions: ["-u"],
            scriptPath: "./", // Path to Python script
            args: [JSON.stringify(inputData)], // Send data as argument
        };

        console.log("running predict.py");
        PythonShell.run("predict.py", options)
            .then(function (results) {
                console.log("running predict.py");

                // Send the Python model's response
                res.json({ prediction: results[0] });
            })
            .catch((err) => {
                console.log(err);
                res.status(500).json({ error: "Error in prediction" });
            });
    } catch (error) {
        res.status(500).json({ error: "Error in prediction" });
    }
});

// Start Server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
