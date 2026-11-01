export PATH="/c/Windows/System32:$PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /c/Users/User/Desktop/flower
rm -f model/*.h5 model/*.keras model/*.json model/*.png model/*.txt model/best_model*
rm -rf model/saved_model frontend/public/model/*
"/c/Program Files/Python314/python.exe" train_model.py > training.log 2>&1
