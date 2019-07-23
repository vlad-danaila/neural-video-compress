# neural-video-compress
Video compression using neural networks.

# Environment setup
The project uses the following:
- Miniconda python distribution (https://docs.conda.io/en/latest/miniconda.html)
- Pytorch (https://pytorch.org/get-started/locally/#anaconda)

The file conda_dependencies.txt keeps the list of conda dependencies.
It can be used to create a similar environment using the next steps:

First we add the channel (repository) from where conda can download the pytorch dependencies with the command:
conda config --append channels pytorch 

We can list our current channels with:
conda config --get channels

Next, we create the conda environment:
conda create --name neural-video-compress --file conda_dependencies.txt

Whenever changing the dependencies please update the conda_dependencies.txt file acordingly by using the command:
conda list --export > conda_dependencies.txt

You can enable the created conda environment with:
conda activate neural-video-compress

If you get an error while issuing the command above, in the GIT bash on Windows, please issue the command:
git init bash

In case you would like to list all conda environments please use:
conda info --envs

