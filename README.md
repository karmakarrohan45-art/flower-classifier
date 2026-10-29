# Flower Classification System

A beginner-friendly web application that identifies flower types from images using a
Convolutional Neural Network (CNN). The model runs entirely in the browser with
TensorFlow.js — no backend server or external API required.

## Project Structure

```
flower/
├── flowers.csv                 # Tabular metadata for 72 flower types
├── requirements.txt            # Python dependencies
├── train_model.py              # Train + evaluate + convert model (one command)
├── evaluate_and_convert.py     # Evaluate + convert an existing model
├── annotate.py                 # Label correction utility
├── model/
│   ├── flower_classifier.keras  # Trained Keras model
│   ├── labels.json              # Class label map
│   ├── confusion_matrix.png     # Evaluation confusion matrix
│   └── classification_report.txt
├── frontend/                    # React + TypeScript web app
│   ├── public/
│   │   ├── index.html
│   │   ├── flowers.json         # Flower metadata (from flowers.csv)
│   │   └── model/                 # TensorFlow.js model (generated)
│   └── src/
│       ├── App.tsx              # Main component
│       ├── App.css
│       ├── index.tsx
│       ├── index.css
│       ├── types.ts
│       ├── hooks/
│       │   └── useFlowerModel.ts  # Model loading + inference logic
│       └── components/
│           ├── ImageUpload.tsx    # Drag-and-drop image upload
│           ├── ImageUpload.css
│           ├── PredictionResult.tsx  # Prediction display
│           ├── PredictionResult.css
│           ├── LoadingSpinner.tsx
│           └── LoadingSpinner.css
├── README.md
└── AGENTS.md
```

## What It Does

1. **Dataset**: Uses TensorFlow Datasets' `tf_flowers` — 3,637 images across 5 classes
   (daisy, dandelion, roses, sunflowers, tulips).
2. **Model**: A CNN with 3 convolutional layers + 2 dense layers, trained for 10 epochs.
3. **Evaluation**: Reports test accuracy, confusion matrix, and per-class precision/recall/F1.
4. **Web App**: A React frontend that loads the TF.js model in-browser and classifies
   uploaded flower photos with confidence percentages and metadata from `flowers.csv`.

## Setup

### Prerequisites
- Python 3.11 or later
- Node.js 20+ (for the frontend)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Windows note**: If you get `ImportError: Could not find the DLL(s) 'msvcp140.dll'`,
> ensure `C:\Windows\System32` is on your PATH:
> ```cmd
> set PATH=%SystemRoot%\System32;%PATH%
> ```

### 2. Train the model (or use the pre-trained model)

If the pre-trained model is not included, run:

```bash
python train_model.py
```

This will:
1. Download the tf_flowers dataset
2. Preprocess and split into train/val/test
3. Build and train the CNN
4. Evaluate on the test set
5. Save the model to `model/flower_classifier.keras`
6. Convert to TensorFlow.js format in `frontend/public/model/`

### 3. Run the web app

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

### 4. (Optional) Label correction tool

```bash
python annotate.py --limit 50
```

Displays images and lets you correct mislabelled ones. Corrections are saved to
`corrected_labels.csv`.

## Using the Web App

1. Wait for the AI model to load (a few seconds on first visit).
2. Drag and drop a flower photo or click **Choose a photo**.
3. The app shows your uploaded image and a prediction with confidence %.
4. See the predicted flower name with metadata (origin, colors, uses, etc.)
5. Click **Upload Another Image** to classify a different photo.

## Supported Flowers

| tf_flowers class | Common name   |
|------------------|---------------|
| daisy            | Daisy         |
| dandelion        | Dandelion     |
| roses            | Rose          |
| sunflowers       | Sunflower     |
| tulips           | Tulip         |
