# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– imagenet/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [ImageNet](#imagenet)
- [UCF101](#ucf101)

The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### ImageNet
- Create a folder named `imagenet/` under `$DATA`.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`. The directory structure should look like
```
imagenet/
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```
- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `$DATA/imagenet/images`.
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).


### UCF101
- Create a folder named `ucf101/` under `$DATA`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```