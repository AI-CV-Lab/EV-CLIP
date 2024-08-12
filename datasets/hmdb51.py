import os
import pickle
import re
import torch
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, Datum_video, DatasetBase
from dassl.utils import mkdir_if_missing

from movinets import MoViNet
from movinets.config import _C

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class HMDB51(DatasetBase):

    """
        Label Mapping Reference: https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hmdb51/label_map.txt
    """
    dataset_dir = "hmdb51"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir) 
        self.avi_dir = os.path.join(self.dataset_dir, "video_data") 
        self.split_path = os.path.join(self.dataset_dir, "test_train_splits") 
        self.split_fewshot_dir = os.path.join(cfg.SHOT_DIR, "hmdb51", "split_fewshot") 
        self.text2class = pd.read_csv(os.path.join(self.dataset_dir, "labels", "hmdb51_labels.csv")) 
        mkdir_if_missing(self.split_fewshot_dir)

        num_frames = cfg.INPUT.WHOLE_FRAMES
        if num_frames == 1:
            train = self.read_data(seed=cfg.SEED, mode="train")
            val = self.read_data(seed=cfg.SEED, num_clip=cfg.TEST.CLIP, num_crop=1, mode="validation")
            test = self.read_data(seed=cfg.SEED, num_clip=cfg.TEST.CLIP, num_crop=1, mode="validation")
        elif num_frames > 1:
            train = self.read_data(seed=cfg.SEED, mode="train")
            val = self.read_data(seed=cfg.SEED, num_clip=cfg.TEST.CLIP, num_crop=1, mode="validation")
            test = self.read_data(seed=cfg.SEED, num_clip=cfg.TEST.CLIP, num_crop=3, mode="validation")

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
    
    def read_data(self, seed, num_clip=10, num_crop=3, mode="train"):
        """
            code: TRAIN - 1
                   TEST - 2
        """
        items = [] 
        splits = [file for file in os.listdir(self.split_path) if str(seed) in file]

        for split in splits:
            with open(os.path.join(self.split_path, split)) as f:
                lines = f.readlines()
                for line in lines:
                    avi, tag, _ = line.split(" ")

                    # Train set
                    if (mode=="train") and (tag == "1"):
                        impath = os.path.join(self.avi_dir, split.split("_test")[0], avi)
                        label_text = split.split("_test")[0]
                        label = self.text2class[self.text2class.name==label_text]["id"].values[0]
                        
                        item = Datum_video(impath=impath, label=label, classname=label_text, clip=-1, crop=-1)
                        items.append(item)
                    
                    # Validation set    
                    elif (mode=="validation") and (tag == "2"):
                        impath = os.path.join(self.avi_dir, split.split("_test")[0], avi)
                        label_text = split.split("_test")[0]
                        label = self.text2class[self.text2class.name==label_text]["id"].values[0]
                        
                        for crop_idx in range(num_crop):    # 3 crops for Testset (0 ~ 2)
                            item = Datum_video(impath=impath, label=label, classname=label_text, clip=num_clip//2, crop=crop_idx)
                            items.append(item)
                        
                    # Test set
                    elif (mode=="test") and (tag == "2"):
                        impath = os.path.join(self.avi_dir, split.split("_test")[0], avi)
                        label_text = split.split("_test")[0]
                        label = self.text2class[self.text2class.name==label_text]["id"].values[0]
                        for clip_idx in range(num_clip):    # 10 clips (0 ~ 9)
                            for crop_idx in range(num_crop):    # 3 crops for Testset (0 ~ 2)
                                item = Datum_video(impath=impath, label=label, classname=label_text, clip=clip_idx, crop=crop_idx)
                                items.append(item) 
        return items
