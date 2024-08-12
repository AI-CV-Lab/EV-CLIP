import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F
import os
import warnings

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform

from torchvision.io import read_video, read_image
import random


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        """
            data type / task type에 따른 DataWrapper Option 추가
        """
        if cfg.INPUT.TASK == "image":
            dataset_wrapper = DatasetWrapper
        elif cfg.INPUT.TASK == "video":
            dataset_wrapper = DatasetWrapper_Video

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        """
            data type / task type에 따른 Transform Option 추가
        
        """
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg                      # Config
        self.data_source = data_source      # 
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

class DatasetWrapper_Video(TorchDataset):
    """
        Dataset Wrapper for Video tensors
        Modified Functions:
            - Since the type of the data is tensor, it uses the torch.nn.functional library to resize the data shape.
            - It can change the number of the frames. It controls the resolution of the number of the frames, instead of cutting the frames off.
            - If the input dataset is 'spinMNIST', it permutes the positions of the dimensions.
    """

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training 
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1 # We don't use Data Augmentation
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        self.cnt_skip = 0
        
        self.train_resize = False
        self.test_resize = False
        self.normalize = False
        self.permutation = False
        self.randomhrflip = False
        self.video_timestamp = False
        self.clipcrop = False

        if self.cfg.INPUT.INTERPOLATION in ['nearest','linear','bilinear','bicubic','trilinear','area','nearest-exact']:
            self.train_resize = True
            self.test_resize = True
        
        if "normalize_video" in cfg.INPUT.TRANSFORMS:
            self.normalize = True

        if self.cfg.DATASET.NAME in ["HMDB51", "HMDB51_ViFi"]: # T H W C -> C T H W
            self.permutation = True
            self.clipcrop = True
            self.permute_dim = (3, 0, 1, 2)
        
        elif self.cfg.DATASET.NAME in ["UCF101_video", "UCF101_video_ViFi"]: # T H W C -> C T H W
            self.permutation = True
            self.clipcrop = True
            self.permute_dim = (3, 0, 1, 2)

        elif self.cfg.DATASET.NAME in ["Kinetics400"]: # T H W C -> C T H W
            self.permutation = True
            self.clipcrop = True
            self.permute_dim = (3, 0, 1, 2)
        
        elif self.cfg.DATASET.NAME in ["EGTEA"]: # T H W C -> C T H W
            self.permutation = True
            self.clipcrop = True
            self.permute_dim = (3, 0, 1, 2)
        
        elif self.cfg.DATASET.NAME in ["ARID"]: # T H W C -> C T H W
            self.permutation = True
            self.clipcrop = True
            self.permute_dim = (3, 0, 1, 2)

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):

        item = self.data_source[idx]
        output = {
            "label": item.label,    # The corresponding label of the data
            "domain": item.domain,  # The domain type of the datas
            "impath": item.impath,  # The path of the video tensor file
            "index": idx            # The index of the data
        }

        # Load a video tensor
        if item.impath.strip().split(".")[-1]=="avi" or item.impath.strip().split(".")[-1]=="mp4": # .avi / .mp4 type
            video, _, fps = read_video(item.impath.strip(), pts_unit='sec')
            video = video.float()
        elif item.impath.split(".")[-1]=="pt": # .pt type
            video = torch.load(item.impath.strip()).float()
        else: # Frames(Images) in Directory
            video = torch.concat([read_image(os.path.join(item.impath.strip(), frame)).unsqueeze(0) \
                    for frame in sorted(os.listdir(item.impath.strip()))[item.timestamp["start"]:item.timestamp["end"]]])
            video = video.float()
        
        # If the data is in the list, it permutes the dimension positions.
        if self.permutation:
            video = video.permute(self.permute_dim)
        
        if self.video_timestamp:
            video = video[:, int(item.timestamp["start"]*float(fps['video_fps'])):int(item.timestamp["end"]*float(fps['video_fps']))-1]

        # Resize the frame size
        if (self.train_resize) and (self.is_train):  # Resize for Train set
            video = F.interpolate(video, size=self.cfg.INPUT.EXPAND_TRAIN, mode=self.cfg.INPUT.INTERPOLATION)

        if (self.test_resize) and (not self.is_train):   # Resize for Test and Validation set
            video = F.interpolate(video, size=self.cfg.INPUT.EXPAND_TEST, mode=self.cfg.INPUT.INTERPOLATION)
        video = torch.clamp(video, min=0, max=255) # Prevent the Value Overshooting

        # Normalize & Scaling (into 0 ~ 1)
        if self.normalize:
            video = self._normalize_tensor(video)

        # Clip & Crop the Video
        if self.clipcrop:
            video = self._clip_crop(video, whole_frames=self.cfg.INPUT.WHOLE_FRAMES, sample_frames=self.cfg.INPUT.FRAMES, \
                                    size=self.cfg.INPUT.SIZE, spatial_idx=item.crop, temporal_idx=item.clip)

        # Random Horizontal Flip (For train set)
        if self.is_train and self.randomhrflip:
            video = self._random_horizontal_flip(video)

        # Save the data to output dictionary
        output["img"] = video
        if self.return_img0:
            output["img0"] = self.to_tensor(video)  # without any augmentation
        
        return output
    
    def _normalize_tensor(self, tensor):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize. [C x T x H x W]
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        # Scaling
        if torch.max(tensor) > 1:
            tensor = tensor.div(255)

        # Normalization
        shape = (-1,) + (1,) * (tensor.dim() - 1)
        mean = torch.as_tensor(self.cfg.INPUT.PIXEL_MEAN).reshape(shape)
        std = torch.as_tensor(self.cfg.INPUT.PIXEL_STD).reshape(shape)
        return (tensor - mean) / std
    
    def _random_horizontal_flip(self, tensor, p=0.5):
        if random.random() < p:
            return tensor.flip(dims=(-1,))
        return tensor
    
    def _random_crop(self, tensor, size=(200, 200)):
        h, w = tensor.shape[-2:]
        th, tw = size
        if w==tw and h==th:
            i, j = 0, 0
        else:
            i = random.randint(0, h-th)
            j = random.randint(0, w-tw)

        return tensor[:, :, i:i+th, j:j+tw]
    
    def _clip_crop(self, tensor, whole_frames=64, sample_frames=32, size=(224, 224), spatial_idx=-1, temporal_idx=-1):
        """
            Spatial Sampling
                - spatial_idx: -1 => Random Sampling    
                - spatial_idx: 0 => Left
                - spatial_idx: 1 => Mid
                - spatial_idx: 2 => Right

            Temporal Sampling
                - temporal_idx: -1 => Random Sampling 
                - temporal_idx: else => Sequential Sampling
        """

        assert spatial_idx in [-1, 0, 1, 2], f"Spatial idx should be in [-1, 0, 1, 2], but got {spatial_idx}."
        assert temporal_idx in list(range(-1, self.cfg.TEST.CLIP)), f"Temporal idx should be in -1 ~ {self.cfg.TEST.CLIP}, but got {temporal_idx}."

        t, h, w = tensor.shape[-3:]
        th, tw = size

        ttw = whole_frames
        tts = sample_frames

        # assert t >= ttw, f"Clip size is over original Temporal size!! Original Temporal Size: {t} / Clip Size: {ttw}"
        assert w >= tw, f"Width size is over original size!! Original Size: {w} / Crop Width Size: {tw}"
        assert h >= th, f"Height size is over original size!! Original Size: {h} / Crop Height Size: {th}"
        if t < ttw:
            ttw = t

        if h==th:
            i = 0
        if w==tw:
            j = 0
        if t==ttw:
            k = 0

        # Spatial Cropping
        if spatial_idx==-1:
            i = random.randint(0, h-th)
            j = random.randint(0, w-tw)
        else:
            i = int((h-th)/2)
            j = int(((w-tw)/2)*(spatial_idx))
            
        if ttw == 1:
            k = t // 2
            tts = 1
        elif self.cfg.TEST.CLIP==1:
            k = (t-ttw)//2
        else:
            # Temporal Clipping
            if temporal_idx==-1:
                k = random.randint(0, t-ttw)
            else:
                temporal_step = max(1.0*(t-ttw)/(self.cfg.TEST.CLIP-1), 0)
                k = int(temporal_step*temporal_idx)

        # Temporal Sampling
        denominator = ttw // tts if ttw // tts !=0 else 1
        k_indices = [idx+k for idx in range(0, ttw, denominator)][:tts]

        return tensor[:, k_indices, i:i+th, j:j+tw]
    