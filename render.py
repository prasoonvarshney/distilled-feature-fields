import os
import time
from pathlib import Path

import imageio
#import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from einops import rearrange
from models.rendering import render
from models.networks import NGP
from opt import get_opts
from tqdm import tqdm
# from train import NeRFSystem, depth2img
from utils import load_ckpt

import clip
import yaml
import os
from clip_utils import CLIPEditor
#os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch/"
os.environ["TORCH_HOME"] = "/tmp/torch/"

if __name__ == "__main__":
    hparams = get_opts()

    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, feature_out_dim=hparams.feature_dim).cuda()
    load_ckpt(model, hparams.ckpt_path)

    if hparams.edit_config is not None:
        with open(hparams.edit_config, 'r') as f:
            edit_config = yaml.safe_load(f)
            print(edit_config)

        # setup query
        model.positive_ids = edit_config["query"]["positive_ids"]
        model.score_threshold = edit_config["query"]["score_threshold"]
        if edit_config["query"]["query_type"] == "text":
            clip_editor = CLIPEditor()
            model.query_features = clip_editor.encode_text([t.replace("_", " ") for t in edit_config["query"]["texts"]])
            """
            clip_model, _ = clip.load("ViT-B/32", device="cuda", download_root="/tmp/")
            tokenized_texts = clip.tokenize(edit_config["query"]["texts"]).cuda()
            with torch.no_grad():
                text_features = clip_model.encode_text(tokenized_texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            model.query_features = text_features
            """
        else:
            raise NotImplementedError

        # setup editing
        edit_dict = {}
        for op in edit_config["edit"]["operations"]:
            if op["edit_type"] == "extraction":
                edit_dict["extraction"] = True
            elif op["edit_type"] == "deletion":
                edit_dict["deletion"] = True
            elif op["edit_type"] == "color_func":
                edit_dict["color_func"] = eval(op["func_str"])
            else:
                pass
    else:
        model.query_features = None
        edit_dict = {}

    # dataset
    llff_view = True  # True for NeRFy datasets. apple, banana need 360 view (i.e. False)

    split="test_traj_fixed"
    fps=15
    if llff_view:
        split="llff"
        fps=15

    dataset = dataset_dict[hparams.dataset_name](
        hparams.root_dir,
        split=split,
        # split="test_traj_fixed",
        downsample=hparams.downsample,
    )

    edit_description = ""
    if hparams.edit_config is not None:
        edit_description = (
            f"""__edittype_{edit_config["edit"]["operations"][0]["edit_type"]}""" +
            f"""__query_{"-".join(edit_config["query"]["texts"])}""" + 
            f"""__target_{"-".join([str(x) for x in edit_config["query"]["positive_ids"]])}"""
        )

    output_directory = f"./renders/default/{hparams.exp_name}"
    if hparams.freenerf_mask:
        output_directory = f"./renders/freenerf/{hparams.exp_name}"
    os.makedirs(output_directory, exist_ok=True)

    output_file_name = f"cliptext_{hparams.clipnerf_text}{edit_description}.gif"

    # start
    directions = dataset.directions.cuda()
    images = []
    for img_idx in tqdm(range(len(dataset))):
        poses = dataset[img_idx]["pose"].cuda()
        rays_o, rays_d = get_rays(directions, poses)
        results = render(model, rays_o, rays_d, **{"test_time": True, "T_threshold": 1e-2, "exp_step_factor": 1 / 256.0,
                                                   "edit_dict": edit_dict})

        w, h = dataset.img_wh
        image = results["rgb"].reshape(h, w, 3)
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        images.append(image)
    imageio.mimsave(os.path.join(output_directory, output_file_name), images, fps=fps)  # TODO: cv2
