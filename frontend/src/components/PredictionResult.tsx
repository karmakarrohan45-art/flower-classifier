import React from "react";
import "./PredictionResult.css";
import type { Prediction } from "../types";

interface PredictionResultProps {
  prediction: Prediction;
  onReset: () => void;
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="metadata-row">
      <span className="metadata-label">{label}</span>
      <span className="metadata-value">{value}</span>
    </div>
  );
}

const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  onReset,
}) => {
  const confidence = prediction.confidence;
  const topClass = prediction.topClass;
  const allPredictions = prediction.allClasses || [];
  const meta = prediction.metadata;
  const isFlower = prediction.isFlower;

  return (
    <div className="prediction-result">
      <div className="prediction-header">
        <h2>Prediction Result</h2>
      </div>

      {!isFlower && (
        <div className="warning-banner">
          This doesn't look like a flower. The model can only identify
          daisies, dandelions, roses, sunflowers, and tulips.
          <br />
          Try uploading an actual flower photo for better results.
        </div>
      )}

      <div className="prediction-main">
        <div className="predicted-name">{topClass}</div>
        <div className="confidence-bar-container">
          <div
            className="confidence-bar-fill"
            style={{ width: `${Math.min(confidence, 100)}%` }}
          />
          <span className="confidence-text">
            {confidence.toFixed(1)}% confidence
          </span>
        </div>
      </div>

      {allPredictions.length > 1 && (
        <div className="all-predictions">
          <h3>Other guesses:</h3>
          {allPredictions.slice(1, 4).map((p) => (
            <div key={p.className} className="prediction-row">
              <span>{p.className}</span>
              <span>{p.confidence.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {meta && (
        <div className="metadata-section">
          <h3>About {meta.name}</h3>
          {meta.scientific_name && (
            <MetadataRow label="Scientific Name" value={meta.scientific_name} />
          )}
          <MetadataRow label="Height" value={meta["height (cm)"] + " cm"} />
          <MetadataRow label="Origin" value={meta["place of origin"]} />
          <MetadataRow label="Color" value={meta.color} />
          <MetadataRow label="Climate" value={meta.climate} />
          <MetadataRow
            label="Uses"
            value={
              [
                meta["cut flowers"] === "1" ? "Cut flower" : null,
                meta.perfumes === "1" ? "Perfume" : null,
                meta.medicine === "1" ? "Medicine" : null,
              ]
                .filter(Boolean)
                .join(", ") || "None"
            }
          />
          <MetadataRow
            label="Petals"
            value={meta["average number of petals"]}
          />
          <MetadataRow label="Longevity" value={meta["longevity (years)"] + " years"} />
        </div>
      )}

      <button className="reset-button" onClick={onReset}>
        Upload Another Image
      </button>
    </div>
  );
};

export default PredictionResult;
