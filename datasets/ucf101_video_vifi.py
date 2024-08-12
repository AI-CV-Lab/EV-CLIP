import os
import pickle
import re

from dassl.data.datasets import DATASET_REGISTRY, Datum, Datum_video, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class UCF101_video_ViFi(DatasetBase):

    dataset_dir = "ucf101_video"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        # root="/workspace"
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.video_dir = os.path.join(self.dataset_dir, "UCF-101-video")
        self.split_path = os.path.join(cfg.SHOT_DIR, "ucf101_video_splits")
        self.split_fewshot_dir = os.path.join(cfg.SHOT_DIR, "ucf101_video_splits/split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        with open(os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt"), "r") as f:
            lines = f.readlines()
        lab2cname = {(int(line.split()[0])-1):line.split()[1] for line in lines}

        num_shots = cfg.DATASET.NUM_SHOTS

        train = self.read_data(lab2cname=lab2cname, \
                               seed=cfg.SEED, \
                               num_shots=num_shots, \
                               mode="train")
        val = self.read_data(lab2cname=lab2cname,  \
                             seed=cfg.SEED, \
                             num_shots=num_shots, \
                             num_clip=cfg.TEST.CLIP, num_crop=1, mode="validation")
        test = self.read_data(lab2cname=lab2cname,  \
                              seed=cfg.SEED, \
                              num_shots=num_shots, \
                              num_clip=cfg.TEST.CLIP, num_crop=1, mode="validation")
        
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

    def read_data(self, lab2cname, num_shots, seed, num_clip=10, num_crop=3, mode="train"):
        
        if mode=="train":
            text_file = os.path.join(self.split_path, "train1_few_shot_{}.txt".format(num_shots))
        elif mode in ["validation", "test"]:
            text_file = os.path.join(self.split_path, "val{}.txt".format(seed))
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                avi, label = line.split()
                label = int(label)
                label_text = lab2cname[label]

                elements = re.findall("[A-Z][^A-Z]*", label_text)
                renamed_label_text = "_".join(elements)

                impath = os.path.join(self.video_dir, label_text, avi)

                if mode=="train": # Random Crop
                    item = Datum_video(impath=impath, label=label, classname=renamed_label_text, clip=-1, crop=-1)
                    items.append(item)
                elif mode=="validation": # Center Crop
                    item = Datum_video(impath=impath, label=label, classname=renamed_label_text, clip=num_clip//2, crop=1)
                    items.append(item)
                elif mode=="test":
                    for clip_idx in range(num_clip):    # 10 clips (0 ~ 9)
                        for crop_idx in range(num_crop):    # 3 crops for Testset (0 ~ 2)
                            item = Datum_video(impath=impath, label=label, classname=renamed_label_text, clip=clip_idx, crop=crop_idx)
                            items.append(item)
        
        return items
