import os
import pickle
import re
import torch
import pandas as pd
from tqdm import tqdm

from dassl.data.datasets import DATASET_REGISTRY, Datum, Datum_video, DatasetBase
from dassl.utils import mkdir_if_missing

from movinets import MoViNet
from movinets.config import _C

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class EGTEA(DatasetBase):

    dataset_dir = "egtea"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir) 
        self.video_dir = os.path.join(self.dataset_dir, "cropped_clips") 
        self.split_fewshot_dir = os.path.join(cfg.SHOT_DIR, "egtea", "split_fewshot") 
        self.label_dir = os.path.join(self.dataset_dir, "action_annotation") 
        mkdir_if_missing(self.split_fewshot_dir)

        self.train_seeds = ["train_split1.txt", "train_split2.txt", "train_split3.txt"]
        self.test_seeds = ["test_split1.txt", "test_split2.txt", "test_split3.txt"]

        train = self.read_data(mode="train", \
                            lab2cname = pd.read_csv(os.path.join(self.label_dir, "raw_annotations/cls_label_index.csv"), sep=";", index_col="# Action ID"), \
                            text_file = self.train_seeds[cfg.SEED-1])
        val = self.read_data(num_crop=1, \
                            mode="validation", \
                            lab2cname = pd.read_csv(os.path.join(self.label_dir, "raw_annotations/cls_label_index.csv"), sep=";", index_col="# Action ID"), \
                            num_clip=cfg.TEST.CLIP, \
                            text_file = self.test_seeds[cfg.SEED-1])
        test = self.read_data(mode="validation", \
                            num_crop=3, \
                            lab2cname = pd.read_csv(os.path.join(self.label_dir, "raw_annotations/cls_label_index.csv"), sep=";", index_col="# Action ID"), \
                            num_clip=cfg.TEST.CLIP, \
                            text_file = self.test_seeds[cfg.SEED-1])

        # Few-shot
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, lab2cname, text_file, num_clip=10, num_crop=3, mode="train"):
        data_list = os.path.join(self.label_dir, text_file)
        items = []

        with open(data_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                filename, label = line.split()[:2]

                filename_split = filename.split("-")
                file_folder = "-".join(filename.split("-")[:3])
                file_prefix = "-".join(filename.split("-")[:5])
                
                label = int(label)-1
                cname = lab2cname.loc[label][" Action Label"]

                impath = os.path.join(self.video_dir, file_folder, filename+".mp4")

                if mode=="train": # Random Crop
                    item = Datum_video(impath=impath, label=label, classname=cname, clip=-1, crop=-1)
                    items.append(item)
                elif mode=="validation": # Center Crop
                    item = Datum_video(impath=impath, label=label, classname=cname, clip=num_clip//2, crop=1)
                    items.append(item)
                elif mode=="test":
                    for clip_idx in range(num_clip):    # 10 clips (0 ~ 9)
                        for crop_idx in range(num_crop):    # 3 crops for Testset (0 ~ 2)
                            item = Datum_video(impath=impath, label=label, classname=cname, clip=clip_idx, crop=crop_idx)
                            items.append(item)
        
        return items
