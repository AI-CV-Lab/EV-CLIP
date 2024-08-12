## EVo-CLIP: External Visual-only Prompt Tuning for Video Adaptaion in VLMs

### Overview

---

Description will be provided.

### Setup

---

### 1. Install the conda environment

1. **evoclip.yaml**
    
    ```python
    # Create a new environment
    conda create -n evoclip python=3.10.14
    
    # Activate the environment
    conda activate evoclip
    ```
    
2. **Install requirements**
    
    ```python
    # Install Pytorch
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    
    # overlay the requirements.yaml
    conda env update -f requirements.yaml
    
    # pip install the requirements.txt
    pip install -r evoclip.txt
    ```
    
3. **CLIP**
    
    ```python
    pip install git+https://github.com/openai/CLIP.git
    ```
    
4. **Movinets**
    
    ```python
    pip install git+https://github.com/Atze00/MoViNet-pytorch.git
    ```
    
5. **OpenCV**
    
    ```python
    pip install opencv-python
    ```
    

### Train

---

```bash
# Train the model.
sh ./scripts/evoclip/evo-true_coop-false2.sh 0 arid omniS_vit_b16 8 4 768 Both mean_pool
```

**Configs**

1. **script file**
    
    Before you start training, you should set the “***DATA***”, “***SHOT_DIR***” first. These indicates the dataset location and location to save shot lists, respectively.
    
    - evo-true_coop-false1.sh : Train EVo-CLIP with a single seed.
    - evo-true_coop-false2.sh : Train EVo-CLIP with two seeds.
    - evo-true_coop-false.sh : Train EVo-CLIP with three seeds.
2. **GPU**
    
    Set your GPU(Cuda) number to use for training.
    
3. **Dataset**
    
    Available Dataset List: ucf101_vifi, arid, egtea
    
    - ucf101_vifi : Train for the first split of UCF101 dataset, using identical shot samples with [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) and [EZ-CLIP](https://github.com/Shahzadnit/EZ-CLIP).
    - arid: Train for the two splits of ARID dataset, which features dark scene conditions.
    - egtea: Train for the three splits of EGTEA Gaze+ dataset, which features egocentric viewpoints.
4. **Config File**
    
    Select a config file in “~/EVo-CLIP/configs/trainers/***dataset/filename.yaml***”.
    
    In the command line, set the name as “***filename***”.
    
5. **The number of Frames**
    
    ex) 2, 4, 8, 16, …
    
6. **The number of Shots**
    
    ex) 2, 4, 8, 16, …
    
7. **The channel dimension size extracted from the video model.**
    - Omni-Tiny , Omni-Small: 768
    - Omni-Base: 1024
8. **Prompt Choice**
    - Mask : only the mask prompt
    - Context : only the context prompt
    - Both : both prompts
9. **Temporal Aggregation**
    - “***mean_pool***” is the only and defualt setting.

### Evaluation

---

```bash
# Evaluate the trained model.
sh ./scripts/evoclip/eval.sh 0 arid omniS_vit_b16 8 4
```

1. **script file**
    
    Before you start the evaluation, you should set the iteration number for each dataset in the file.
    
    - eval.sh
2. **GPU**
    
    Set your GPU(Cuda) number to use for training.
    
3. **Dataset**
    
    Available Dataset List: ucf101_vifi, arid, egtea
    
    - ucf101_vifi : Train for the first split of UCF101 dataset, using identical shot samples with [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) and [EZ-CLIP](https://github.com/Shahzadnit/EZ-CLIP).
    - arid: Train for the two splits of ARID dataset, which features dark scene conditions.
    - egtea: Train for the three splits of EGTEA Gaze+ dataset, which features egocentric viewpoints.
4. **Config File**
    
    Select a config file in “~/EVo-CLIP/configs/trainers/***dataset/filename.yaml***”.
    
    Don’t forget to set the name as “***filename***”.
    
5. **The number of Frames**
    
    ex) 2, 4, 8, 16, …
    
6. **The number of Shots**
    
    ex) 2, 4, 8, 16, …
    

### Acknowledgements

---

This code is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch).