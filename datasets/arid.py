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
class ARID(DatasetBase):

    dataset_dir = "arid/ARID_v1_5_211015"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir) 
        self.video_dir = os.path.join(self.dataset_dir, "clips_v1.5") 
        self.split_fewshot_dir = os.path.join(cfg.SHOT_DIR, "arid", "split_fewshot") 
        self.label_dir = os.path.join(self.dataset_dir, "list_cvt") 
        mkdir_if_missing(self.split_fewshot_dir)

        self.train_seeds = ["split_0/split0_train.txt", "split_1/split1_train.txt", "split_1/split1_train.txt"]
        self.val_seeds = ["split_0/split0_others.txt", "split_1/split1_other.txt", "split_1/split1_other.txt"]
        self.test_seeds = ["split_0/split0_test.txt", "split_1/split1_test.txt", "split_1/split1_test.txt"]

        train = self.read_data(mode="train", \
                            text_file = self.train_seeds[cfg.SEED-1])
        val = self.read_data(num_crop=1, \
                            mode="validation", \
                            num_clip=cfg.TEST.CLIP, \
                            text_file = self.val_seeds[cfg.SEED-1])
        test = self.read_data(mode="validation", \
                            num_crop=3, \
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
    
    def read_data(self, text_file, num_clip=10, num_crop=3, mode="train"):
        data_list = os.path.join(self.label_dir, text_file)
        items = []

        with open(data_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                label, filepath = line.strip().split("\t")[-2:]
                label = int(label)
                cname = filepath.split("/")[0]
                impath = os.path.join(self.video_dir, filepath)

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
