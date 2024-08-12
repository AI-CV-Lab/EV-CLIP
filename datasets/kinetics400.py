import os
import pickle
import re
import torch
import cv2
import numpy as np
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, Datum_video, DatasetBase
from dassl.utils import mkdir_if_missing

from movinets import MoViNet
from movinets.config import _C
from tqdm import tqdm

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class Kinetics400(DatasetBase):

    dataset_dir = "kinetics-dataset/k400"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir) 
        self.split_fewshot_dir = os.path.join(cfg.SHOT_DIR, "kinetics400/split_fewshot")
        self.label_dir = os.path.join(self.dataset_dir, "annotations") 
        self.text2class = {v:k for k, v in pd.read_csv(os.path.join(self.dataset_dir, "annotations", "kinetics_400_labels.csv"), index_col="id").to_dict()["name"].items()}
        mkdir_if_missing(self.split_fewshot_dir)

        train = self.read_data(mode="train")
        val = self.read_data(num_clip=cfg.TEST.CLIP, num_crop=1, mode="val")
        test = self.read_data(num_clip=cfg.TEST.CLIP, num_crop=1, mode="test")

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
    
    def read_data(self, num_clip=10, num_crop=3, mode="train"):

        items = []
        items_app = items.append

        # No append

        data_path = os.path.join(self.dataset_dir, "{}".format(mode))
        # youtube_id
        id_info = {v:k for k, v in pd.read_csv(os.path.join(self.label_dir, "{}.csv".format(mode))).youtube_id.to_dict().items()}   
        ids = set(id_info.keys())
        # label
        data_info = pd.read_csv(os.path.join(self.label_dir, "{}.csv".format(mode)))
        label_info = data_info.label.to_dict()

        data_files = set(data_info.youtube_id + "_" \
                        + data_info.time_start.astype(np.str).str.pad(width=6, side="left", fillchar="0") + "_" \
                        + data_info.time_end.astype(np.str).str.pad(width=6, side="left", fillchar="0") + ".mp4")

        if mode=="train":

            # replacement folder
            replacement_path = os.path.join(self.dataset_dir, "replacement", "replacement_for_corrupted_k400")
            replaced_file = set(os.listdir(replacement_path))

            # train folder
            train_file = set(os.listdir(data_path))

            for file in tqdm(data_files):
                # If the file is replaced
                if file in replaced_file:
                    impath = os.path.join(replacement_path, file)
                    if not self.is_corrupted(impath):
                        label_text = label_info[id_info[file[:11]]]
                        label = self.text2class[label_text]

                        item = Datum_video(impath=impath, label=label, classname=label_text, clip=-1, crop=-1)
                        items_app(item)

                elif file in train_file:
                    impath = os.path.join(data_path, file)
                    if not self.is_corrupted(impath):
                        label_text = label_info[id_info[file[:11]]]
                        label = self.text2class[label_text]

                        item = Datum_video(impath=impath, label=label, classname=label_text, clip=-1, crop=-1)
                        items_app(item)
                
        elif mode=="val":
            
            # val folder
            val_file = set(os.listdir(data_path))

            for file in tqdm(data_files):
                if file in val_file:
                    impath = os.path.join(data_path, file)
                    if not self.is_corrupted(impath):
                        label_text = label_info[id_info[file[:11]]]
                        label = self.text2class[label_text]

                        item = Datum_video(impath=impath, label=label, classname=label_text, clip=num_clip//2, crop=1)
                        items_app(item)
            
        elif mode=="test":

            # test folder
            test_file = set(os.listdir(data_path))

            for file in tqdm(data_files):
                if file in test_file:
                    impath = os.path.join(data_path, file)
                    if not self.is_corrupted(impath):
                        label_text = label_info[id_info[file[:11]]]
                        label = self.text2class[label_text]

                        # for clip_idx in range(num_clip):    # 10 clips (0 ~ 9)
                        #     for crop_idx in range(num_crop):    # 3 crops for Testset (0 ~ 2)
                        #         item = Datum_video(impath=impath, label=label, classname=label_text, clip=clip_idx, crop=crop_idx)
                        #         items_app(item)

                        item = Datum_video(impath=impath, label=label, classname=label_text, clip=num_clip//2, crop=1)
                        items_app(item)
                
        return items
    def is_corrupted(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): # corrupted
            return True
        for i in range(8):
            run, _ = cap.read()
            if not run:
                return True

        cap.release()   # not corrupted
        return False