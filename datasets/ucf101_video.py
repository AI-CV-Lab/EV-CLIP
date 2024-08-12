import os
import pickle
import re

from dassl.data.datasets import DATASET_REGISTRY, Datum, Datum_video, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class UCF101_video(DatasetBase):

    dataset_dir = "ucf101_video"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.video_dir = os.path.join(self.dataset_dir, "UCF-101-video")
        self.split_path = os.path.join(self.dataset_dir, "ucfTrainTestlist", "split_zhou_UCF101.json")
        self.split_fewshot_dir = os.path.join(cfg.SHOT_DIR, "ucf101_video", "split_fewshot")
        
        self.train_seeds = ["trainlist01.txt", "trainlist02.txt", "trainlist03.txt"]
        self.test_seeds = ["testlist01.txt", "testlist02.txt", "testlist03.txt"]
        mkdir_if_missing(self.split_fewshot_dir)

        cname2lab = {}
        filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                label, classname = line.strip().split(" ")
                label = int(label) - 1  # conver to 0-based index
                cname2lab[classname] = label


        train = self.read_data(cname2lab=cname2lab, \
                                text_file=os.path.join("ucfTrainTestlist", \
                                self.train_seeds[cfg.SEED-1]), mode="train")
        val = self.read_data(cname2lab=cname2lab,  \
                                text_file=os.path.join("ucfTrainTestlist", \
                                self.test_seeds[cfg.SEED-1]), \
                                num_clip=cfg.TEST.CLIP, num_crop=3, mode="validation")
        test = self.read_data(cname2lab=cname2lab,  \
                                text_file=os.path.join("ucfTrainTestlist", \
                                self.test_seeds[cfg.SEED-1]), \
                                num_clip=cfg.TEST.CLIP, num_crop=3, mode="validation")
        # test = self.read_data(cname2lab=cname2lab,  \
        #                         text_file=os.path.join("ucfTrainTestlist", \
        #                         self.test_seeds[cfg.SEED-1]), \
        #                         num_clip=cfg.TEST.CLIP, num_crop=3, mode="test")

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

    def read_data(self, cname2lab, text_file, num_clip=10, num_crop=3, mode="train"):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                # BasketballDunk -> Basketball_Dunk
                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                impath = os.path.join(self.video_dir, action, filename)

                if mode=="train": # Random Crop
                    item = Datum_video(impath=impath, label=label, classname=renamed_action, clip=-1, crop=-1)
                    items.append(item)
                elif mode=="validation": # Center Crop
                    item = Datum_video(impath=impath, label=label, classname=renamed_action, clip=num_clip//2, crop=1)
                    items.append(item)
                elif mode=="test":
                    for clip_idx in range(num_clip):    # 10 clips (0 ~ 9)
                        for crop_idx in range(num_crop):    # 3 crops for Testset (0 ~ 2)
                            item = Datum_video(impath=impath, label=label, classname=renamed_action, clip=clip_idx, crop=crop_idx)
                            items.append(item)
        
        return items
