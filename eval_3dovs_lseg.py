# This script is used for evaluating 3DOVS clip text

import os
import glob
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Union

import cv2
import numpy as np
import open_clip
import torch
import tyro
import yaml
import pprint
import matplotlib

from lseg import LSegNet
import open_clip

matplotlib.use("TkAgg")  # To avoid conflict with opencv

from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
    FeatureRenderer,
    to_builtin,
)

@dataclass
class Args:
    data_dir: str = "./data/3DOVS/bed"  # Path to the COLMAP data directory.
    checkpoint: str = (
        "./results/3DOVS/bed/ckpts/ckpt_29999_rank0.pt"  # Path to the checkpoint file, can be generated from the original 3DGS repo.
    )
    results_dir: str = "./results/3DOVS_eval/bed"  # Output directory for results.
    format: Literal["inria", "gsplat"] = (
        "gsplat"  # Checkpoint format: "inria" (original) or "gsplat".
    )
    data_factor: int = 4  # Downscale resolution by this factor.
    tag: str = None  # Optional tag for this evaluation run.
    prune: bool = True  # Whether to prune the 3DGS using gradients.
    feature_path: str = "./results/3DOVS_feats/bed/features_lseg_bed.pt"


def get_iou(gt_mask, mask):
    intersection = gt_mask & mask
    union = gt_mask | mask
    iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
    return iou

def main(args: Args):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    net = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
        # Load pre-trained weights
    net.load_state_dict(
        torch.load("./checkpoints/lseg_minimal_e200.ckpt")
    )
    net.eval()

    
    
    clip_text_encoder = net.clip_pretrained.encode_text

    del net

    # Already made inside __main__, but keeping in case any change in code outside the calling of this function
    os.makedirs(args.results_dir, exist_ok=True)

    splats = load_checkpoint(
        args.checkpoint, args.data_dir, format=args.format, data_factor=args.data_factor
    )

    if args.prune:
        splats_optimized = prune_by_gradients(splats)
        test_proper_pruning(splats, splats_optimized)
        splats = splats_optimized

    # Load features
    features = torch.load(args.feature_path, weights_only=False)

    # Extract Gaussian Parameters
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    width = K[0, 2] * 2
    height = K[1, 2] * 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = FeatureRenderer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=None,
        Ks=None,
        width=width,
        height=height,
    )

    classes_path = os.path.join(args.data_dir, "segmentations", "classes.txt")
    classes = []
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    classes = sorted(classes)

    net = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
    # Load pre-trained weights
    net.load_state_dict(
        torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device)
    )
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text

    # pos_prompt_length = len(prompt.split(";"))

    # prompts = prompt.split(";") + neg_prompt.split(";")

    tokens = open_clip.tokenize(classes)#+["other"])
    text_embeddings = clip_text_encoder(tokens.cuda()).float()

    # text_embeddings = clip_model.encode_text(open_clip.tokenize(classes+["things","stuff","texture","object"]).cuda()).float()
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    features = torch.nn.functional.normalize(features, dim=-1)
    last_dir = sorted(dir_name for dir_name in os.listdir(os.path.join(args.data_dir, "segmentations")) if os.path.isdir(os.path.join(args.data_dir, "segmentations",dir_name)))[-1]
    
    ious = []
    accs = []
    for image in splats["colmap_project"].images.values():
        image_stem = image.name.split(".")[0]
        flag = False
        # Hack: To fix the bug related to 3DOVS colmap dataset
        if os.path.basename(os.path.normpath(args.data_dir))  == "table":
            flag = image_stem == "IMG_20230412_214716"
        if image_stem == last_dir or flag:
            # Hack: To fix the bug related to 3DOVS colmap dataset
            if flag:
                image_stem = "30"
            viewmat = get_viewmat_from_colmap_image(image)
            out, _, _ = renderer.render(viewmats=viewmat[None], Ks=K[None],sh_degree=3)
            out = out[0].cpu().numpy()
            # cv2.imshow("Render", out[...,::-1])
            # cv2.waitKey(0)
            scores = features @ text_embeddings.T
            for idx, class_ in enumerate(classes):
                mask3d = scores.argmax(dim=1) == idx
                colors_temp = colors_dc.clone()
                colors_temp[~mask3d] = 0
                colors_temp[mask3d] = 1
                out, _, _ = renderer.render(viewmats=viewmat[None], Ks=K[None], colors=colors_temp[...,0],sh_degree=None)
                mask = cv2.medianBlur((out[0].cpu().numpy()[...,0] > 0.5).astype(np.uint8), 5)
                # cv2.imshow("Mask", mask*255)
                # cv2.waitKey(0)
                gt_mask_path = os.path.join(args.data_dir, "segmentations", image_stem, f"{class_}.png")
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                iou = get_iou(gt_mask>0, mask)
                accs.append(((gt_mask>0)==mask).sum()/gt_mask.size)
                ious.append(iou)
                # combined = cv2.hconcat([gt_mask, (mask*255)])
                # cv2.imshow("Combined", combined)
                # cv2.waitKey(0)

    print(ious)
    print(accs)
    print("mIoU", mIoU := np.mean(ious))
    print("mAcc", acc := np.mean(accs))

    # Save the results
    results_path = os.path.join(
        args.results_dir, f"3dovs_lseg_evaluation_2_{args.tag}.json"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"mIoU": mIoU, "mAcc": acc}, f)


if __name__ == "__main__":

    cfg = tyro.cli(Args)

    print("Configuration:")
    pprint.pprint(to_builtin(cfg))
    print("\n")

    # Save config
    os.makedirs(cfg.results_dir, exist_ok=True)
    tag = cfg.tag if cfg.tag is not None else time.strftime("%Y%m%d_%H%M%S")
    cfg_out_path = os.path.join(cfg.results_dir, f"3dovs_evaluation_{tag}.yaml")
    with open(cfg_out_path, "w") as f:
        yaml.dump(to_builtin(cfg), f)

    # Run Evaluation
    main(cfg)
