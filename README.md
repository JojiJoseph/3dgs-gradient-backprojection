# Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting

This repository contains the code for the **SIGGRAPH Asia 2025** paper **Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting**.



Project page: https://jojijoseph.github.io/3dgs-backprojection

[Paper](https://dl.acm.org/doi/10.1145/3757377.3763926)


## Setup

Please install the dependencies listed in `requirements.txt` via `pip install -r requirements.txt`. Download `lseg_minimal_e200.ckpt` from https://mitprod-my.sharepoint.com/:u:/g/personal/jkrishna_mit_edu/EVlP4Ggf3OlMgDACHNVYuIYBZ4JNi5nJCQA1kXM-_nrB3w?e=XnPT39 and place it in the `./checkpoints` folder. 

Other than that, it's a self-contained repo. Please feel free to raise an issue if you face any problems while running the code.

## Demo



https://github.com/user-attachments/assets/1aecd2d1-8e16-499e-98ce-a1667be5114d

Left: Original rendering, Mid: Extraction, Right: Deletion

Sample data (garden) can be found [here](https://drive.google.com/file/d/1cEPby9zWgG40dJ4eRiHu15Jdg7FgvTdG/view?usp=sharing). Please create a folder named `data` on root folder and extract the contents of zip file to that folder.

**Backprojection**

To backproject the features run 

```bash
python backproject.py --help
```

**Segmentation**

Once backprojection is completed, run the following to see the segmentation results.

```bash
python segment.py --help
```


Trained Mip-NeRF 360 Gaussian splat models (using [gsplat](https://github.com/nerfstudio-project/gsplat) with data factor = 4) can be found [here](https://drive.google.com/file/d/1ZCTgAE6vZOeUBdR3qPXdSPY01QQBHxeO/view?usp=sharing). Extract them to `data` folder.


**Application - Click and Segment**



https://github.com/user-attachments/assets/3f1c797f-db29-416f-8917-9be7885231b5



```bash
python click_and_segment.py
```

Click left button to select positive visual prompts and middle button to select negative visual prompts. `ctrl+lbutton` and `ctrl+mbutton` to remove selected prompts.

**Application - Editing with LLM**

```bash
python viewer_with_llm.py --checkpoint data/garden/ckpts/ckpt_29999_rank0.pt --data-dir data/garden  --lseg-checkpoint results/garden/features_lseg.pt
```

Press ` to start prompting. At present it supports only a single query at a time. Queries can be of changing view, segment and change color.



https://github.com/user-attachments/assets/126583ab-1f6f-4cc3-ab60-21453a7f3f5a



## Acknowledgements

A big thanks to the following tools/libraries, which were instrumental in this project:

- [gsplat](https://github.com/nerfstudio-project/gsplat): 3DGS rasterizer.
- [LSeg](https://github.com/isl-org/lang-seg) and [LSeg Minimal](https://github.com/krrish94/lseg-minimal) : To generate features to be backprojected.


## Citation
If you find this paper or the code helpful for your work, please consider citing our paper,
```
@inproceedings{joseph2024gradientweightedfeaturebackprojection,
author = {Joseph, Joji and Amrutur, Bharadwaj and Bhatnagar, Shalabh},
title = {Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting},
year = {2025},
isbn = {9798400721373},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3757377.3763926},
doi = {10.1145/3757377.3763926},
abstract = {We propose a training-free method for feature field rendering in 3D Gaussian Splatting, enabling fast and scalable embedding of high-dimensional features into 3D scenes. Unlike training-based feature distillation methods, which are computationally expensive and often yield feature embeddings that poorly reflect the rendered semantics, our approach back-projects 2D features onto pre-trained 3D Gaussians using influence weights derived from the rendering equation. This projection produces a queryable 3D feature field, validated on tasks including 2D and 3D segmentation, affordance transfer, and identity encoding, spanning queries using language, pixel, and synthetic embeddings. These capabilities, in turn, enable downstream applications in augmented and virtual reality, interactive scene editing, and robotics. Across different tasks, our method achieves performance comparable to or better than training-based approaches, while significantly reducing computational cost. The project page is at https://jojijoseph.github.io/3dgs-backprojection.},
booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
articleno = {178},
numpages = {12},
keywords = {3DGS, Feature Field Distillation, Feature Lifting},
location = {
},
series = {SA Conference Papers '25}
}
```
