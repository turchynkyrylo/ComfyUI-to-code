To run this code:
```
git clone https://github.com/turchynkyrylo/ComfyUI-to-code.git
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
- MAKE SURE THAT YOU HAVE CUDA PYTORCH

Download ```flux1-dev-fp8.safetensors``` from https://huggingface.co/lllyasviel/flux1_dev/blob/main/flux1-dev-fp8.safetensors to the folder ```ComfyUI-to-code/models/checkpoints```

To run workflow:
```
python workflow_api.py
```
