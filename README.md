# neural-video-compress
Video compression using neural networks.

# Environment setup
The project uses the following:
- Python 3.6.8
- Pytorch (https://pytorch.org/get-started/locally/#anaconda)

The file pip_freeze.txt keeps the list of dependencies.
When adding new libraries it's best to regenerate this file in order to keep trak of dependencies versions.
I order to generate a new pip_freeze file please use the command:  
pip freeze > pip_freeze.txt

In order to install pytorch please issue the next commands:  
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-win_amd64.whl  
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-win_amd64.whl  