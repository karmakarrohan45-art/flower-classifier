import { useState, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import type { Prediction, FlowerMetadata } from "../types";

const MODEL_URL = "/model/model.json?v=6";
const LABELS_URL = "/model/labels.json";
const FLOWERS_URL = "/flowers.json";
const IMAGE_SIZE = 180;
const METADATA_KEY_MAP: Record<string, string> = {
  daisy: "daisy",
  dandelion: "dandelion",
  roses: "rose",
  sunflowers: "sunflower",
  tulips: "tulip",
  not_flower: "not_flower",
};
const CONFIDENCE_THRESHOLD = 0.85;
const MARGIN_THRESHOLD = 0.20;
const NOT_FLOWER_REJECT_THRESHOLD = 0.15;

async function loadHtmlImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error("Failed to load image for prediction"));
    img.src = dataUrl;
  });
}

function preprocessImage(img: HTMLImageElement): tf.Tensor {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(img);
    const channels = tensor.shape[2];
    let rgb: tf.Tensor3D;
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
    tensor.dispose();
    rgb.dispose();
    resized.dispose();
    expanded.dispose();
    return normalized.clone();
  });
}

const PERSON_REJECT_THRESHOLD = 0.45;

export function useFlowerModel() {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [detector, setDetector] = useState<cocoSsd.ObjectDetection | null>(null);
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
        const [loadedModel, loadedDetector, labelRes, metaRes] = await Promise.all([
          tf.loadGraphModel(MODEL_URL),
          cocoSsd.load(),
          fetch(LABELS_URL),
          fetch(FLOWERS_URL),
        ]);
        setModel(loadedModel);
        setDetector(loadedDetector);

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

      const img = await loadHtmlImage(dataUrl);

      if (detector) {
        try {
          const detections = await detector.detect(img);
          const person = detections.find(
            (d) => d.class === "person" && d.score >= PERSON_REJECT_THRESHOLD
          );
          if (person) {
            return {
              topClass: "Not a Flower",
              confidence: person.score * 100,
              allClasses: labels.map((name) => ({
                className: name,
                confidence: name === "not_flower" ? person.score * 100 : 0,
              })),
              metadata: metadata["not_flower"],
              isFlower: false,
            };
          }
        } catch (e) {
          console.warn("Person detection failed, continuing:", e);
        }
      }

      const inputTensor = preprocessImage(img);

      const output = await model.executeAsync({ x: inputTensor });
      const predictionTensor = Array.isArray(output) ? output[0] : output;

      const predictionData = Array.from(await predictionTensor.data());

      const sortedIndices = predictionData
        .map((val, idx) => ({ idx, val }))
        .sort((a, b) => b.val - a.val);

      const topIdx = sortedIndices[0].idx;
      const topClassName = labels[topIdx];
      const topConfidence = predictionData[topIdx];
      const secondConfidence = sortedIndices[1]?.val || 0;
      const margin = topConfidence - secondConfidence;
      const notFlowerIdx = labels.indexOf("not_flower");
      const notFlowerConfidence = notFlowerIdx >= 0 ? predictionData[notFlowerIdx] : 0;
      const isFlower =
        topClassName !== "not_flower" &&
        notFlowerConfidence < NOT_FLOWER_REJECT_THRESHOLD &&
        topConfidence >= CONFIDENCE_THRESHOLD &&
        margin >= MARGIN_THRESHOLD;

      const allClasses = sortedIndices.map((item) => ({
        className: labels[item.idx],
        confidence: item.val * 100,
      }));

      const metaKey = METADATA_KEY_MAP[topClassName] || topClassName;
      const flowerMeta = metadata[metaKey];

      inputTensor.dispose();
      predictionTensor.dispose();

      return {
        topClass: flowerMeta?.name || topClassName,
        confidence: topConfidence * 100,
        allClasses,
        metadata: flowerMeta,
        isFlower,
      };
    },
    [model, detector, labels, metadata]
  );

  return { model, labels, metadata, isLoadingModel, loadError, predict };
}
