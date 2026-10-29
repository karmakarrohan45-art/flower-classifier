export PATH="/c/Windows/System32:$PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /c/Users/User/Desktop/flower
rm -f model/* frontend/public/model/* 2>/dev/null
"/c/Program Files/Python314/python.exe" train_model.py > training.log 2>&1
echo "EXIT_CODE=$?" >> training.log
