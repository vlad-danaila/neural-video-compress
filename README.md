# neural-video-compress
Video compression using neural networks.

# Environment setup
The project uses the following:
- Miniconda python distribution (https://docs.conda.io/en/latest/miniconda.html)
- Pytorch (https://pytorch.org/get-started/locally/#anaconda)

The file conda_dependencies.txt keeps the list of conda dependencies.
It can be used to create a similar environment using this command:
conda create --name neural-video-compress --file conda_dependencies.txt

Whenever changing the dependencies please update the conda_dependencies.txt file acordingly by using the command:
conda list --export > conda_dependencies.txt