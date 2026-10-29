import React, { useState, useCallback } from "react";
import ImageUpload from "./components/ImageUpload";
import PredictionResult from "./components/PredictionResult";
import LoadingSpinner from "./components/LoadingSpinner";
import { useFlowerModel } from "./hooks/useFlowerModel";
import type { Prediction } from "./types";
import "./App.css";

type AppState = "idle" | "uploaded" | "predicting" | "result" | "error";

function App() {
  const { model, labels, isLoadingModel, loadError, predict } = useFlowerModel();
  const [state, setState] = useState<AppState>("idle");
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = useCallback(
    async (dataUrl: string) => {
      setImageDataUrl(dataUrl);
      setError(null);
      setState("predicting");

      try {
        const result = await predict(dataUrl);
        setPrediction(result);
        setState("result");
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Something went wrong during prediction."
        );
        setState("error");
      }
    },
    [predict]
  );

  const handleReset = () => {
    setImageDataUrl(null);
    setPrediction(null);
    setError(null);
    setState("idle");
  };

  const isModelReady = !isLoadingModel && model && labels;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Flower Classifier</h1>
        <p>Upload a photo of a flower and find out what type it is!</p>
      </header>

      <main className="app-main">
        {!isModelReady && isLoadingModel ? (
          <LoadingSpinner message="Loading AI model..." />
        ) : loadError ? (
          <div className="error-banner">
            Failed to load model: {loadError}
            <br />
            Please make sure the model files are in <code>public/model/</code>
          </div>
        ) : state === "idle" ? (
          <div className="upload-prompt">
            <ImageUpload
              onImageUpload={handleImageUpload}
              disabled={!isModelReady}
            />
          </div>
        ) : state === "predicting" ? (
          <div className="predicting-view">
            {imageDataUrl && (
              <img
                src={imageDataUrl}
                alt="Upload preview"
                className="preview-image"
              />
            )}
            <LoadingSpinner message="Analyzing your flower..." />
          </div>
        ) : (
          <div className="result-view">
            {imageDataUrl && prediction && (
              <>
                <img
                  src={imageDataUrl}
                  alt="Upload preview"
                  className="result-image"
                />
                <PredictionResult
                  prediction={prediction}
                  onReset={handleReset}
                />
              </>
            )}
          </div>
        )}

        {state === "error" && error && (
          <div className="error-banner">{error}</div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Powered by TensorFlow.js CNN model |
          Flower types: Daisy, Dandelion, Rose, Sunflower, Tulip
        </p>
      </footer>
    </div>
  );
}

export default App;
