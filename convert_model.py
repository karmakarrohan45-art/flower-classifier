import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import shutil

# Load the model
model = tf.keras.models.load_model('model/flower_classifier.h5')
print("Model loaded")

# Save as SavedModel format using tf.saved_model
save_model_dir = 'model/saved_model'
if os.path.exists(save_model_dir):
    shutil.rmtree(save_model_dir)

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 180, 180, 3], dtype=tf.float32)])
def serving_fn(x):
    return {"predictions": model(x)}

tf.saved_model.save(model, save_model_dir, signatures={"serving_default": serving_fn})
print(f"SavedModel saved -> {save_model_dir}")

# Convert to TF.js
from tensorflowjs.converters import converter
args = [
    "--input_format", "tensorflow_saved_model",
    "--output_format", "tfjs_graph_model",
    save_model_dir,
    "frontend/public/model",
]
converter.convert(args)
print("TF.js model saved -> frontend/public/model")

shutil.copy('model/labels.json', 'frontend/public/model/labels.json')
print("Labels copied -> frontend/public/model/labels.json")
print("Done!")
