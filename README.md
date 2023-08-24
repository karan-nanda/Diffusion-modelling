# Diffusion-modelling
# Image Generation with DDPM and DDIM

This repository contains code for generating images using the **Differentiable Diffusion Probabilistic Models (DDPM)** and **Differentiable Diffusion Implicit Models (DDIM)** techniques. These methods allow for the synthesis of high-quality images by modeling the diffusion process of noise over time. Both DDPM and DDIM have been implemented using PyTorch.


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Comparative Analysis](#comparative-analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This repository presents the implementation of DDPM and DDIM, which are powerful methods for generating images through the controlled diffusion of noise. These methods can produce visually appealing results and find applications in various domains, including image synthesis, style transfer, and more.

## Installation
1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your-username/diffusion-image-generation.git
   ```
2. Navigate to the repository's directory:
   ```
   cd diffusion-image-generation
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Follow these steps to utilize DDPM and DDIM for image generation:

1. Import the required modules and functions:
   ```python
   import torch
   from ddpm_model import ContextUnet  # Import the appropriate model
   from ddpm_utilities import *
   ```

2. Initialize the model with the desired parameters:
   ```python
   nn_model = ContextUnet(in_channels, n_feat, n_cfeat, height).to(device)
   ```

3. Load pre-trained model weights (if available) and set the model to evaluation mode:
   ```python
   nn_model.load_state_dict(torch.load(f"{save_dir}/model_31.pth", map_location=device))
   nn_model.eval()
   ```

4. Utilize the sampling functions to generate images:
   ```python
   samples, intermediate = sample_ddpm(n_sample, save_rate=20)  # or use sample_ddim
   ```

5. Visualize the generated samples using the provided plotting functions:
   ```python
   animation = plot_sample(intermediate, n_sample, num_cols, save_dir, "animation_name", None, save=False)
   ```

## Model Architecture
The DDPM and DDIM models are based on the `ContextUnet` architecture, which includes various components for predicting and manipulating noise throughout the image diffusion process. The architecture involves embedding time steps and context labels to enhance the synthesis process.

## Training
The models can be trained using custom datasets. The training process involves perturbing the data, predicting noise, and optimizing the model to minimize the mean squared error between the predicted and true noise.

## Results
Generated images using DDPM and DDIM exhibit impressive quality and coherence.

## Comparative Analysis
Comparative analysis between DDPM and DDIM indicates differences in sampling speeds and quality of generated images. Both methods offer unique advantages, with DDIM offering faster sampling at the cost of some visual quality.

## Contributing
Contributions to this repository are welcome. Feel free to open issues and submit pull requests to enhance the functionality and usability of the code.

## License
This project is licensed under the [Apache 2.0 License](LICENSE).

---
*Disclaimer: This is a fictional README file generated for demonstration purposes based on the provided code snippet.*
