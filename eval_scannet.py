# The code to evaluate scannet
import csv

from natsort import natsorted
from label_mapping import read_label_mapping
from scannet_constants import COCOMAP_CLASS_LABELS, COLORMAP

label_to_cocomap = read_label_mapping("scannetv2-labels.modified.tsv","id","cocomapid")
labelset = list(COCOMAP_CLASS_LABELS)
labelset = ["other"] + labelset

from typing import Literal
import tyro
import os
import torch
import cv2
import imageio  # To generate gifs
from gsplat import rasterization
import numpy as np
import matplotlib
from sklearn.decomposition import PCA

matplotlib.use("TkAgg")

from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
    torch_to_cv,
    load_checkpoint_scannet,
    get_frames_scannet
)


def render_pca(
    splats,
    features,
    output_path,
    pca_on_gaussians=True,
    scale=1.0,
    feedback=True,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("PCA", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    aux_dir = output_path + ".images"
    os.makedirs(aux_dir, exist_ok=True)

    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features.detach().cpu().numpy())
    feats_min = np.min(features_pca, axis=(0, 1))
    feats_max = np.max(features_pca, axis=(0, 1))
    features_pca = (features_pca - feats_min) / (feats_max - feats_min)
    features_pca = torch.tensor(features_pca).float().cuda()
    if pca_on_gaussians:
        for image in sorted(
            splats["colmap_project"].images.values(), key=lambda x: x.name
        ):
            viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features_pca,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            frame = torch_to_cv(features_rendered)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
            if feedback:
                cv2.imshow("PCA", frame[..., ::-1])
                cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
                cv2.waitKey(1)
    else:
        for image in sorted(
            splats["colmap_project"].images.values(), key=lambda x: x.name
        ):
            viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            h, w, c = features_rendered.shape
            features_rendered = (
                features_rendered.reshape(h * w, c).detach().cpu().numpy()
            )
            features_rendered = pca.transform(features_rendered)
            features_rendered = features_rendered.reshape(h, w, 3)
            features_rendered = (features_rendered - feats_min) / (
                feats_max - feats_min
            )
            frame = (features_rendered * 255).astype(np.uint8)
            frames.append(frame[..., ::-1])
            if feedback:
                cv2.imshow("PCA", frame)
                cv2.imwrite(f"{aux_dir}/{image.name}", frame)
                cv2.waitKey(1)
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()

def render_pca_scannet(splats,
    features,
    output_path,
    pca_on_gaussians=True,
    scale=1.0,
    feedback=True,
):
    
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("PCA", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    aux_dir = output_path + ".images"
    os.makedirs(aux_dir, exist_ok=True)

    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features.detach().cpu().numpy())
    feats_min = np.min(features_pca, axis=(0, 1))
    feats_max = np.max(features_pca, axis=(0, 1))
    features_pca = (features_pca - feats_min) / (feats_max - feats_min)
    features_pca = torch.tensor(features_pca).float().cuda()
    data_dir = splats["data_dir"]
    if pca_on_gaussians:
        for frame in get_frames_scannet(scannet_dir=data_dir):
            viewmat = frame["viewmat"]
            image_name = frame["image_name"]
            # viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features_pca,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            frame = torch_to_cv(features_rendered)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
            if feedback:
                cv2.imshow("PCA", frame[..., ::-1])
                cv2.imwrite(f"{aux_dir}/{image_name}", frame[..., ::-1])
                cv2.waitKey(1)
    else:
        for frame in get_frames_scannet(scannet_dir=data_dir):
            viewmat = frame["viewmat"]
            # viewmat = get_viewmat_from_colmap_image(image)
            # viewmat = get_viewmat_from_colmap_image(image)
            image_name = frame["image_name"]
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            h, w, c = features_rendered.shape
            features_rendered = (
                features_rendered.reshape(h * w, c).detach().cpu().numpy()
            )
            features_rendered = pca.transform(features_rendered)
            features_rendered = features_rendered.reshape(h, w, 3)
            features_rendered = (features_rendered - feats_min) / (
                feats_max - feats_min
            )
            frame = (features_rendered * 255).astype(np.uint8)
            frames.append(frame[..., ::-1])
            if feedback:
                cv2.imshow("PCA", frame)
                cv2.imwrite(f"{aux_dir}/{image_name}", frame)
                cv2.waitKey(1)
    # imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()

def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    feature_checkpoint: str = "./results/garden/features_dino.pt",  # path to features, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    format: Literal[
        "inria", "gsplat", "ply"
    ] = "gsplat",  # Original or gsplat for checkpoints
    data_factor: int = 4,
    show_visual_feedback: bool = True,
    tag: str = None,
    prune: bool = True
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint_scannet(
        checkpoint, data_dir, format=format, data_factor=data_factor
    )
    #if False and prune:
    # splats_optimized, mask = prune_by_gradients(splats)
    # test_proper_pruning(splats, splats_optimized)
    # splats = splats_optimized

    features = torch.load(feature_checkpoint)
    # features = features[mask]
    if False:
        mask = torch.argmax(features, dim=1) == 47

        splats["means"] = splats["means"][mask]
        splats["features_dc"] = splats["features_dc"][mask]
        splats["features_rest"] = splats["features_rest"][mask]
        splats["scaling"] = splats["scaling"][mask]
        splats["rotation"] = splats["rotation"][mask]
        splats["opacity"] = splats["opacity"][mask]
        features = features[mask]

    if tag is None:
        tag = os.path.basename(feature_checkpoint).split(".")[0]

    if False:
        render_pca_scannet(
            splats,
            features,
            f"{results_dir}/pca_gaussians_{tag}.gif",
            pca_on_gaussians=True,
            scale=1.0,
            feedback=show_visual_feedback,
        )

        render_pca_scannet(
            splats,
            features,
            f"{results_dir}/pca_renderings_{tag}.gif",
            pca_on_gaussians=False,
            feedback=show_visual_feedback,
        )

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]

    features_normalized = torch.nn.functional.normalize(features, dim=1)

    rm_mask = features_normalized.sum(dim=1) == 0
    means = means[~rm_mask]
    colors_dc = colors_dc[~rm_mask]
    colors_rest = colors_rest[~rm_mask]
    colors = colors[~rm_mask]
    opacities = opacities[~rm_mask]
    scales = scales[~rm_mask]
    quats = quats[~rm_mask]
    features_normalized = features_normalized[~rm_mask]
    # K = K[~rm_mask]

    from lseg import LSegNet

    net = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
    
    device = torch.device("cuda")
    # Load pre-trained weights
    net.load_state_dict(
        torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location="cuda")
    )
    net.eval()
    net.to("cuda")

    # self.device = device
    # self.net = net
    # self.dim = 512
    
    clip_text_encoder = net.clip_pretrained.encode_text

    del net

    prompts = labelset
    import clip
    text_embeddings = clip_text_encoder(clip.tokenize(prompts).to(device)).float()
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)

    scores = features_normalized @ text_embeddings.T

    # mask_3d = scores[:,0] > torch.max(scores[:,1:], dim=1).values
    # assert mask_3d.sum() != 0

    # print(scores.min(), scores.max())
    # exit()
    # colors = colors_dc.clone()*1/np.sqrt(4*np.pi) + 0.5
    # colors[:,0,0] = (colors_dc[:,0,0]  * (1/np.sqrt(4*np.pi)) + 0.5) * scores[:,0]
    # print(scores.shape, colors.shape)
    # exit()

    scannet_frames = get_frames_scannet(scannet_dir=data_dir)
    scannet_frames = natsorted(scannet_frames, key=lambda x: x["image_name"])
    scannet_frames = scannet_frames[::10]
    # Take every 8th frame
    scannet_eval_frames = scannet_frames[::8]

    scene_name = os.path.basename(data_dir.rstrip("/"))

    def get_miou(gt_label_mask_cocomap, label_mask):
        # convert to cocomap
        # Map gt_label_mask to cocomap using label_to_cocomap
        
        iou = 0
        cnt = 0
        unique_labels = np.unique(gt_label_mask_cocomap)
        for label in unique_labels:
            # if label == 0:
            #     continue
            intersection = np.sum((gt_label_mask_cocomap == label) & (label_mask == label))
            union = np.sum((gt_label_mask_cocomap == label) | (label_mask == label))
            iou += intersection / union if union > 0 else 0
            cnt += 1
        return iou / cnt if cnt > 0 else 0
    

    def get_acc(gt_label_mask_cocomap, label_mask):
        acc = 0
        height, width = gt_label_mask_cocomap.shape
        acc = np.sum(gt_label_mask_cocomap == label_mask) / height / width
        return acc

    ious = []
    accs = []
    width, height = None, None

    for frame in scannet_eval_frames:
        viewmat = frame["viewmat"]
        # viewmat = get_viewmat_from_colmap_image(image)
        # viewmat = get_viewmat_from_colmap_image(image)
        image_name = frame["image_name"]
        if width is None:
            img = cv2.imread(os.path.join(data_dir, "output","color", image_name))
            height, width = img.shape[:2]
        gt_mask = cv2.imread(os.path.join(data_dir, f"{scene_name}_2d-label-filt","label-filt", image_name.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED)
        gt_label_mask_cocomap = np.vectorize(lambda x: label_to_cocomap.get(int(x), 0))(gt_mask)
        # colors_rendered, alphas, meta = rasterization(
        #     means[mask_3d],
        #     quats[mask_3d],
        #     scales [mask_3d]* 1,
        #     opacities[mask_3d],
        #     colors[mask_3d,0],
        #     viewmats=viewmat[None],
        #     Ks=K[None],
        #     width=K[0, 2] * 2,
        #     height=K[1, 2] * 2,
        #     # sh_degree=3,
        # )

        features_rendered, alphas, meta = rasterization(
            means,
            quats,
            scales* 1,
            opacities,
            features_normalized,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            # sh_degree=3,
        )
        features_rendered = features_rendered[0]
        features_rendered = torch.nn.functional.normalize(features_rendered, dim=2)
        scores = features_rendered @ text_embeddings.T # H, W, C
        class_ids = scores.argmax(dim=2) # H, W

        iou = get_miou(gt_label_mask_cocomap, class_ids.cpu().numpy())
        ious.append(iou)

        accs.append(get_acc(gt_label_mask_cocomap, class_ids.cpu().numpy()))
        print(class_ids.shape)
        global COLORMAP
        COLORMAP = np.array(COLORMAP)
        # Fill class ids with color map
        color_mask = COLORMAP[class_ids.cpu().numpy()] / 255.0  # shape (H, W, 3)
        
        print(color_mask.shape)
        # mask2d = scores[:,:,0] >= torch.max(scores[:,:,1:], dim=2).values


        # colors_rendered = colors_rendered[0].detach().cpu().numpy()

        
        
        # cv2.imshow("Output", colors_rendered[...,::-1])

        colors_rendered, alphas, meta = rasterization(
            means,
            quats,
            scales * 1,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=3,
        )

        # colors_rendered[0][mask2d] = 0
        colors_rendered = colors_rendered[0].detach().cpu().numpy()

        cv2.imshow("Output", colors_rendered[...,::-1])

        # print(colors_rendered.shape)

        # mask2d = mask2d[...,None].detach().cpu().numpy() * np.array([[1.0,0.0,0.0]])

        # colors_rendered = colors_rendered * 0.5 + mask2d

        output = colors_rendered * 0.5 + color_mask * 0.5

        cv2.imshow("Output with mask", output[...,::-1])

        cv2.waitKey(100)
        print(ious)
        print(np.mean(ious))
        print(accs)
        print(np.mean(accs))
        # features_rendered = features_rendered[0]
        # h, w, c = features_rendered.shape
        # features_rendered = (
        #     features_rendered.reshape(h * w, c).detach().cpu().numpy()
        # )
        # features_rendered = pca.transform(features_rendered)
        # features_rendered = features_rendered.reshape(h, w, 3)
        # features_rendered = (features_rendered - feats_min) / (
        #     feats_max - feats_min
        # )
        # frame = (features_rendered * 255).astype(np.uint8)
        # frames.append(frame[..., ::-1])
        # if feedback:
        #     cv2.imshow("PCA", frame)
        #     cv2.imwrite(f"{aux_dir}/{image_name}", frame)
        #     cv2.waitKey(1)
    print("mIoU", np.mean(ious))
    print("mAcc", np.mean(accs))


if __name__ == "__main__":
    tyro.cli(main)
