@echo off
cd /d "%~dp0"
echo Installing PyTorch with CUDA support (about 2.5 GB)...
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -c "import torch;print('CUDA available:',torch.cuda.is_available())"
pause
