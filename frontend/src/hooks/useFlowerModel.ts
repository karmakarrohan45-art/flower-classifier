import { useState, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import type { Prediction, FlowerMetadata } from "../types";

const MODEL_URL = "/model/model.json";
const LABELS_URL = "/model/labels.json";
const FLOWERS_URL = "/flowers.json";
const IMAGE_SIZE = 180;
const METADATA_KEY_MAP: Record<string, string> = {
  daisy: "daisy",
  dandelion: "dandelion",
  roses: "rose",
  sunflowers: "sunflower",
  tulips: "tulip",
};
const CONFIDENCE_THRESHOLD = 0.65;

async function loadHtmlImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = dataUrl;
  });
}

function preprocessImage(img: HTMLImageElement): tf.Tensor {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(img);
    const channels = tensor.shape[2];
    let rgb;
    if (channels === 4) {
      rgb = tensor.slice([0, 0, 0], [-1, -1, 3]);
    } else if (channels === 1) {
      const gray = tensor.slice([0, 0, 0], [-1, -1, 1]);
      rgb = tf.concat([gray, gray, gray], 2);
    } else {
      rgb = tensor.slice([0, 0, 0], [-1, -1, Math.min(channels, 3)]);
    }
    const resized = tf.image.resizeBilinear(rgb, [IMAGE_SIZE, IMAGE_SIZE]);
    const expanded = resized.expandDims(0);
    const normalized = expanded.toFloat().div(255.0);
    return normalized.clone();
  });
}

export function useFlowerModel() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [labels, setLabels] = useState<string[] | null>(null);
  const [metadata, setMetadata] = useState<Record<string, FlowerMetadata>>({});
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.setBackend("webgl");
        await tf.ready();
      } catch (e) {
        console.warn("WebGL backend failed, falling back to CPU:", e);
        await tf.setBackend("cpu");
        await tf.ready();
      }

      try {
        const [loadedModel, labelRes, metaRes] = await Promise.all([
          tf.loadGraphModel(MODEL_URL),
          fetch(LABELS_URL),
          fetch(FLOWERS_URL),
        ]);
        setModel(loadedModel);

        const labelData = await labelRes.json();
        setLabels(labelData.labels);

        const metaData = await metaRes.json();
        setMetadata(metaData);

        setIsLoadingModel(false);
      } catch (err) {
        console.error("Failed to load model:", err);
        setLoadError(
          err instanceof Error ? err.message : "Failed to load model"
        );
        setIsLoadingModel(false);
      }
    };

    loadModel();
  }, []);

  const predict = useCallback(
    async (dataUrl: string): Promise<Prediction> => {
      if (!model || !labels) {
        throw new Error("Model not loaded");
      }

      // Wait for the uploaded image to finish loading in the browser
      const img = await loadHtmlImage(dataUrl);

      // Preprocess on the GPU (resize + normalize)
      const inputTensor = preprocessImage(img);

      let output: tf.Tensor | tf.Tensor[];
      try {
        output = await model.executeAsync({ x: inputTensor });
      } catch (inferErr) {
        console.warn("executeAsync failed, trying predict:", inferErr);
        output = model.predict(inputTensor) as tf.Tensor | tf.Tensor[];
      }
      const predictionTensor = Array.isArray(output) ? output[0] : output;

      const predictionData = Array.from(await predictionTensor.data());

      // Sort results by confidence (descending)
      const sortedIndices = predictionData
        .map((val, idx) => ({ idx, val }))
        .sort((a, b) => b.val - a.val);

      const topIdx = sortedIndices[0].idx;
      const topClassName = labels[topIdx];
      const topConfidence = predictionData[topIdx] * 100;

      const allClasses = sortedIndices.map((item) => ({
        className: labels[item.idx],
        confidence: item.val * 100,
      }));

      // Look up flower metadata from flowers.csv (converted to JSON)
      const metaKey = METADATA_KEY_MAP[topClassName] || topClassName;
      const flowerMeta = metadata[metaKey];

      const isFlower = topConfidence / 100 >= CONFIDENCE_THRESHOLD;

      // Clean up tensors
      inputTensor.dispose();
      predictionTensor.dispose();

      return {
        topClass: flowerMeta?.name || topClassName,
        confidence: topConfidence,
        allClasses,
        metadata: flowerMeta,
        isFlower,
      };
    },
    [model, labels, metadata]
  );

  return { model, labels, metadata, isLoadingModel, loadError, predict };
}
