# DreamGaussian: Efficient 3D Content Generation Framework

YOUTUBE VIDEO--WHOLE CODE WALKTHROUGH  

https://youtu.be/EcAQYZYakO0?si=OIi10gU8i6b039U1



Welcome to **DreamGaussian** – a novel framework that achieves both efficiency and quality in 3D content generation by leveraging generative 3D Gaussian Splatting and Score Distillation Sampling (SDS). This project demonstrates how, starting from a single-view image or a text prompt, we generate a high-quality, textured 3D mesh in roughly 2 minutes—approximately 10× faster than existing approaches.

---

## Table of Contents

- [Overview](#overview)
- [Abstract & Key Innovations](#abstract--key-innovations)
- [Project Pipeline](#project-pipeline)
  - [Stage 1: Generative Gaussian Splatting](#stage-1-generative-gaussian-splatting)
  - [Stage 2: Mesh Extraction & Texture Refinement](#stage-2-mesh-extraction--texture-refinement)
- [Mathematical Foundations](#mathematical-foundations)
  - [Score Distillation Sampling (SDS)](#score-distillation-sampling-sds)
  - [3D Gaussian Representation](#3d-gaussian-representation)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Code Walk-through](#code-walk-through)
- [Visual Explanation](#visual-explanation)
- [Challenges & Considerations](#challenges--considerations)
- [Accessing the Project](#accessing-the-project)
- [Conclusion](#conclusion)

---

## Overview

**DreamGaussian** is designed to overcome the slow per-sample optimization of traditional 3D generation methods that utilize Score Distillation Sampling (SDS). By representing the 3D scene as a collection of Gaussians and progressively densifying them during optimization, our framework rapidly captures the underlying 3D structure. Then, an efficient mesh extraction and UV-space texture refinement pipeline converts these Gaussians into high-quality, textured polygonal meshes.

---

## Abstract & Key Innovations

Traditional 3D generation via SDS has demonstrated promising results by leveraging powerful 2D diffusion models. However, these methods suffer from slow per-sample optimization, making them impractical for many real-world applications.

**DreamGaussian** addresses these issues with the following innovations:

- **Generative 3D Gaussian Splatting:**  
  Instead of using Neural Radiance Fields with occupancy pruning, we represent the 3D scene with small, parameterized Gaussians. Each Gaussian is defined by:
  $$
  \Theta_i = \{ \mathbf{x}_i, \mathbf{s}_i, \mathbf{q}_i, \alpha_i, \mathbf{c}_i \}
  $$
  where:
  - \(\mathbf{x}_i \in \mathbb{R}^3\) is the center position,
  - \(\mathbf{s}_i \in \mathbb{R}^3\) is the scale,
  - \(\mathbf{q}_i \in \mathbb{R}^4\) is the rotation quaternion,
  - \(\alpha_i\) is the opacity, and
  - \(\mathbf{c}_i \in \mathbb{R}^3\) is the color.

- **Progressive Densification:**  
  Instead of employing occupancy pruning, our framework progressively densifies the Gaussians during optimization. This leads to significantly faster convergence on generative tasks.

- **Mesh Extraction and UV Texture Refinement:**  
  We introduce an efficient algorithm to convert the optimized 3D Gaussians into a polygonal mesh through local density queries and the Marching Cubes algorithm. Finally, a UV-space refinement stage uses a pixel-wise MSE loss to enhance texture quality—producing high-quality, ready-to-use 3D assets.

**Advantages:**
- High-quality textured meshes are generated in roughly 2 minutes.
- Overall acceleration of approximately 10× over prior methods.
- A practical and robust pipeline suited for downstream applications.

---

## Project Pipeline

### Stage 1: Generative Gaussian Splatting

1. **Initialization:**  
   We initialize a set of Gaussians randomly within a sphere. Each Gaussian is parameterized as:
   $$
   \Theta_i = \{ \mathbf{x}_i, \mathbf{s}_i, \mathbf{q}_i, \alpha_i, \mathbf{c}_i \}
   $$
2. **Optimization via SDS:**  
   The pretrained 2D diffusion model provides guidance by computing the SDS loss. The loss is defined as:
   $$
   \nabla_{\Theta} L_{SDS} = \mathbb{E}_{t,p,\epsilon} \left[ w(t) \left( \epsilon_{\phi}(I_{p}^{RGB}; t, e) - \epsilon \right) \frac{\partial I_{p}^{RGB}}{\partial \Theta} \right]
   $$
   where:
   - \(w(t)\) is a weighting function,
   - \(\epsilon_{\phi}\) is the noise prediction,
   - \(I_{p}^{RGB}\) is the rendered image from a random camera pose \(p\),
   - \(e\) is the input embedding (text or image), and
   - \(\epsilon\) is random Gaussian noise.
3. **Progressive Densification:**  
   Additional Gaussians are periodically added during training to capture finer details.

### Stage 2: Mesh Extraction & Texture Refinement

1. **Mesh Extraction:**  
   We compute a dense grid of densities using the Gaussians:
   $$
   d(\mathbf{x}) = \sum_{i} \alpha_i \exp\left(-\frac{1}{2} (\mathbf{x} - \mathbf{x}_i)^T \Sigma_i^{-1} (\mathbf{x} - \mathbf{x}_i)\right)
   $$
   Here, \(\Sigma_i\) is the covariance matrix formed using the scale and rotation of each Gaussian. This grid is processed using the Marching Cubes algorithm to extract a polygonal mesh. Post-processing (decimation, remeshing) refines the mesh further.

2. **UV Texture Refinement:**  
   The mesh is unwrapped using UV mapping. An initial texture is created by back-projecting colors from rendered views. This texture is then refined using a pixel-wise MSE loss:
   $$
   L_{MSE} = \| I_{p}^{fine} - I_{p}^{coarse} \|_2^2
   $$
   This minimizes texture artifacts and enhances details.

---

## Mathematical Foundations

### Score Distillation Sampling (SDS)

The SDS loss is pivotal for guiding the optimization of our Gaussian parameters. It is formulated as:
$$
\nabla_{\Theta} L_{SDS} = \mathbb{E}_{t,p,\epsilon} \left[ w(t) \left( \epsilon_{\phi}(I_{p}^{RGB}; t, e) - \epsilon \right) \frac{\partial I_{p}^{RGB}}{\partial \Theta} \right]
$$

- \(w(t)\): weighting function based on DDPM principles.
- \(\epsilon_{\phi}\): noise prediction from the pretrained 2D diffusion model.
- \(I_{p}^{RGB}\): rendered image from camera pose \(p\).
- \(e\): embedding (text prompt or reference image).

### 3D Gaussian Representation

Each Gaussian is represented as:
$$
\Theta_i = \{ \mathbf{x}_i, \mathbf{s}_i, \mathbf{q}_i, \alpha_i, \mathbf{c}_i \}
$$
These parameters define the geometry and appearance of the 3D model.

---

## Installation and Setup

Run the following commands in your terminal or notebook cell to set up the repository:

```bash
# Remove any existing directory and clone the repository
%rm -r dreamgaussian
!git clone https://github.com/dreamgaussian/dreamgaussian
%cd dreamgaussian

# Install necessary dependencies
!pip install -q einops plyfile dearpygui huggingface_hub diffusers accelerate transformers xatlas trimesh PyMCubes pymeshlab rembg[gpu,cli] omegaconf ninja

# Install additional libraries for differentiable rasterization and nearest neighbor search
!pip install -q git+https://github.com/NVlabs/nvdiffrast
!pip install -q git+https://github.com/ashawkey/kiuikit
%mkdir -p data
!git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
%pip install git+https://github.com/camenduru/simple-knn
%cd data/
%rm * # remove all old files!
from google.colab import files
uploaded = files.upload()
IMAGE = os.path.basename(next(iter(uploaded)))  # filename
%cd ..

from IPython.display import Image, display
display(Image(f'data/{IMAGE}', width=256, height=256))
# Set elevation (example: -20 degrees)
Elevation = -20

# Stage 1: Optimize 3D Gaussians
%run main.py --config configs/image.yaml input=data/{IMAGE_PROCESSED} save_path={NAME} elevation={Elevation} force_cuda_rast=True
%run -m kiui.render logs/{NAME}_mesh.obj --save_video {NAME}.mp4 --wogui --force_cuda_rast

from IPython.display import HTML
from base64 import b64encode

def show_video(video_path, video_width=450):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

show_video(f'{NAME}.mp4')
