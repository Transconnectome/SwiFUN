<div align="center">    
 
# Predicting task-related brain activity from resting-state brain dynamics with fMRI Transformer

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.7+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

</div>


## 📌&nbsp;&nbsp;Introduction
This project is a continuing effort after [SwiFT](https://arxiv.org/abs/2307.05916) and the official code repo for 'Predicting task-related brain activity from resting-state brain dynamics with fMRI Transformer.' Feel free to ask the authors any questions regarding this project. 

**Contact**
- First author
  - Junbeom Kwon: kjb961013@snu.ac.kr
- Corresponding author
  - Professor Jiook Cha: connectome@snu.ac.kr


> Effective usage of this repository requires learning a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai). Knowledge of some experiment logging frameworks like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) is also recommended.

## 1. Description
This repository implements the SwiFUN (SwiFUN). 
- Our code offers the following things.
  - Trainer based on PyTorch Lightning for running SwiFT and SwiFUN (same as Swin UNETR).
  - Data preprocessing/loading pipelines for 4D fMRI datasets.


## 2. How to install
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/Transconnectome/SwiFUN.git

# install project   
cd SwiFT
conda env create -f envs/py39.yaml
conda activate py39
 ```

## 3. Project Structure
Our directory structure looks like this:

```
├── notebooks                    <- Useful Jupyter notebook examples are given (TBU)
├── output                       <- Experiment log and checkpoints will be saved here 
├── project                 
│   ├── module                   <- Every module is given in this directory
│   │   ├── models               <- Models (Swin fMRI Transformer)
│   │   ├── utils                
│   │   │    ├── data_module.py  <- Dataloader & codes for matching fMRI scans and target variables
│   │   │    └── data_preprocessing_and_load
│   │   │        ├── datasets.py           <- Dataset Class for each dataset
│   │   │        └── preprocessing.py      <- Preprocessing codes for step 6
│   │   └── pl_classifier.py    <- LightningModule
│   └── main.py                 <- Main code that trains and tests the 4DSwinTransformer model
│
├── test                 
│   ├── module_test_swin.py     <- Code for debugging SwinTransformer
│   └── module_test_swin4d.py   <- Code for debugging 4DSwinTransformer
│ 
├── sample_scripts                     <- Example shell scripts for training
│
├── .gitignore                  <- List of files/folders ignored by git
├── export_DDP_vars.sh          <- setup file for running torch DistributedDataParallel (DDP) 
└── README.md
```

<br>

## 4. Train model

### 4.0 Quick start

- Single forward & backward pass for debugging SwinTransformer4D model.

```bash
cd SwiFUN/
python test/module_test_swin4d.py
 ```  

### 4.1 Arguments for trainer
You can check the arguments list by using -h
 ```bash
python project/main.py --data_module dummy --classifier_module default -h
```

### 4.2 Hidden Arguments for PyTorch lightning
pytorch_lightning offers useful arguments for training. For example, we used `--max_epochs` and `--default_root_dir` in our experiments. We recommend the user refer to the following link to check the argument lists.

([https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer](https://lightning.ai/docs/pytorch/1.8.6/common/trainer.html))

### 4.3 Commands/scripts for running classification/regression tasks
- Training SwiFT in an interactive way
  
 ```bash
# interactive
cd SwiFUN
bash sample_scripts/sample_train_swifun.sh
```
This bash script was tested on the server cluster (Linux) with 8 RTX 3090 GPUs.
You should correct the following lines.

[to be updated]
 ```bash
cd {path to your 'SwiFUN' directory}
source /usr/anaconda3/etc/profile.d/conda.sh (init conda) # might change if you have your own conda.
conda activate {conda env name}
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name {dataset_name} --image_path {path to the image data}' # This script assumes that you have preprocessed HCP dataset. You may run the codes anyway with "--dataset_name Dummy"
DEFAULT_ARGS='--project_name {neptune project name}'
export NEPTUNE_API_TOKEN="{Neptune API token allocated to each user}"
export CUDA_VISIBLE_DEVICES={usable GPU number}
 ```

- Training SwiFUN with Slurm (if you run the codes at Slurm-based clusters)
Please refer to the [tutorial](https://slurm.schedmd.com/sbatch.html) for Slurm commands.
 ```bash
cd SwiFUN
sbatch sample_scripts/sample_train_swifun.slurm
```

## 5. Loggers
We offer two options for loggers.
- Tensorboard (https://www.tensorflow.org/tensorboard)
   - Log & model checkpoints are saved in `--default_root_dir`
   - Logging test code with Tensorboard is not available.
- Neptune AI (https://neptune.ai/)
   - Generate a new workspace and project on the Neptune website.
      - Academic workspace offers 200GB of storage and collaboration for free. 
   - export NEPTUNE_API_TOKEN="YOUR API TOKEN" in your script.
   - specify the "--project_name" argument with your Neptune project name. ex) "--project_name user-id/project"


## 6. How to prepare your own dataset
These preprocessing codes are implemented based on the initial repository by GonyRosenman [TFF](https://github.com/GonyRosenman/TFF)

To make your own dataset, you should execute either of the minimal preprocessing steps:
- fMRIprep [Preprocessing with fMRIprep](https://fmriprep.org/en/stable/)
- FSL [UKB Preprocessing pipeline](https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf)

 * We ensure that each brain is registered to the MNI space, and the whole brain mask is applied to remove non-brain regions. 
 * We are investigating how additional preprocessing steps to remove confounding factors such as head movement impact performance.

After the minimal preprocessing steps, you should perform additional preprocessing to use SwiFT. (You can find the preprocessing code at 'project/module/utils/data_preprocessing_and_load/preprocessing.py')
- normalization: voxel normalization(not used) and whole-brain z-normalization (mainly used)
- change fMRI volumes to floating point 16 to save storage and decrease IO bottleneck.
- each fMRI volume is saved separately as torch checkpoints to facilitate window-based training.
- remove non-brain(background) voxels that are over 96 voxels.
   - you should open your fMRI scans to determine the level that does not cut out the brain regions
   - you can use `nilearn` to visualize your fMRI data. (official documentation: [here](https://nilearn.github.io/dev/index.html))
  ```python
  from nilearn import plotting
  from nilearn.image import mean_img
  
  plotting.view_img(mean_img(fmri_filename), threshold=None)
  ```
   - if your dimension is under 96, you can pad non-brain voxels at 'datasets.py' files.

* refer to the annotation in the 'preprocessing.py' code to adjust it for your own datasets.

The resulting data structure is as follows:
```
├── {Dataset name}_MNI_to_TRs                 
   ├── img                  <- Every normalized volume is located in this directory
   │   ├── sub-01           <- subject name
   │   │  ├── frame_0.pt    <- Each torch pt file contains one volume in a fMRI sequence (total number of pt files = length of fMRI sequence)
   │   │  ├── frame_1.pt
   │   │  │       :
   │   │  ├── frame_{T}.pt  <- the last volume in an fMRI sequence (length T)
   │   │  └── global_stats.pt  <- min, max, mean value of fMRI for the subject
   │   └── sub-02              
   │   │  ├── frame_0.pt    
   │   │  ├── frame_1.pt
   │   │  ├──     :
   └── metadata
       └── metafile.csv     <- file containing target variable
```

## 7. Define the Dataset class for your own dataset.
* The data loading pipeline works by processing image and metadata at 'project/module/utils/data_module.py' and passing the paired image-label tuples to the Dataset classes at 'project/module/utils/data_preprocessing_and_load/datasets.py.'
* you should implement codes for combining image path, subject_name, and target variables at 'project/module/utils/data_module.py'
* you should define Dataset Class for your dataset at 'project/module/utils/data_preprocessing_and_load/datasets.py.' In the Dataset class (__getitem__), you should specify how many background voxels you would add or remove to make the volumes shaped 96 * 96 * 96.


### Citation   
```
```   

