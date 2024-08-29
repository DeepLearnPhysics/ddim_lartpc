# Denoising Diffusion Implicit Models (DDIM) for LArTPC Image Generation

This repository contains the code for training and sampling from a Denoising Diffusion Implicit Model (DDIM) for LArTPC Image Generation.

DDIMs are a non-Markovian, deterministic version of DDPMs (Diffusion Probabilistic Models) that can produce samples much quicker than DDPMs. DDPMs have been shown to be directly related to score-based models, so you should expect similar performance to the score-based models in the paper [Score-based Diffusion Models for Generating Liquid Argon Time Projection Chamber Images](https://arxiv.org/abs/2307.13687).

DDIM paper: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

## Running the Experiments

### Download the LArTPC Dataset

```
python download_lartpc_dataset.py --exp {PROJECT_PATH}
```

where
- `PROJECT_PATH` is the path to the experiment directory, where you will create a subdirectory for this experiment. All files will be saved in this directory.

This will download the dataset of 64x64 cropped LArTPC images used in [Score-based Diffusion Models for Generating Liquid Argon Time Projection Chamber Images](https://arxiv.org/abs/2307.13687), which is found on [Zenodo](https://zenodo.org/record/8300355) and was derived from the [PILArNet dataset](https://osf.io/bu4fp/). See the paper for more details on this dataset.

### Train a model
Training is exactly the same as DDPM with the following:

```python
python main.py --config lartpc.yml --exp {PROJECT_PATH} --doc {RUN_NAME}
```
where
- `RUN_NAME` is the name of the run.

### Sampling from the model

#### Sampling from the generalized model

This will create 8 sample images from the model.

```python
python main.py --config lartpc.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --timesteps {STEPS} --eta {ETA}
```
where 
- `ETA` controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM).
- `STEPS` controls how many timesteps used in the process. The model is trained on 1000 steps, so 1000 is a good starting point.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.

#### Sampling from the sequence of images that lead to the sample
Use `--sequence` option instead.

The above two cases contain some hard-coded lines specific to producing the image, so modify them according to your needs.


## References and Acknowledgements


This implementation is based on:

- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (the DDIM repo), 
- [https://arxiv.org/abs/2307.13687](https://arxiv.org/abs/2307.13687) (the score-based diffusion for LArTPC images paper).