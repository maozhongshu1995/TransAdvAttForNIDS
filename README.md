# TransAdvAttForNIDS
# Citation

This repository contains code to reproduce results from the paper:

Exploring Transferable Adversarial Attacks for Deep Learning-Based Network Intrusion Detection

Still under review…


> - This repository does not yet include the models' training details or the attack algorithms' implementation. If the paper is accepted and published, we commit to releasing all training details and attack algorithm implementations.
> - The results in the papers that can be reproduced now are: Section 5.2, 5.3, and 5.7. 
>   - It may also include parts 5.4 and 5.5, but due to the large amount of data, the upload speed cannot be guaranteed; we can only do our best to upload it.
>   - All codes and instructions will be completed before April 20th.

# 1 Introduction
Our work focuses on the generation method of AAT in black-box scenarios and proposes SPTS and DGM. This repository contains all the code necessary to **reproduce the experiments presented in the paper**, as well as methods for **custom network training and generating adversarial attack traffic (AAT)**. The repository structure and a brief overview of each folder are provided below:
```
TransAdvAttForNIDS/
├── generating_AAT/                  # Code for generating AAT
├── reproduce_experiments_results/   # Scripts to reproduce each subsection of Section 5
├── storage/                         # Need downloading
│   ├── AAT/                         # Generated AAT
│   │   ├── surrogate model/         # Surrogate model names
│   │   │   └── attacks/             # Attack names
│   ├── dataset/                     # Datasets
│   └── pre-trained_models/          
│       ├── adv_train_with_SPTS/     # Adversarial training with SPTS
│       ├── normal_adv_train/        # Normal adversarial training
│       └── normal_train/            # Normal train
├── training_model/                  # Codes for training NIDS models
├── utils/                           # Utility functions and helper scripts
```
To reproduce the experimental results presented in the paper, follow the instructions in **Preliminaries** and **Reproducing Results in Our Paper**. To customize the training of networks and generate AAT, refer to the instructions in **Details of Training Models** and **Generating AAT**. For customizing adversarial training, follow the instructions in **Details of Adversarial Training**. 

Please make sure to complete the setup steps described in **Preliminaries** before proceeding with any experiments.

# 2 Preliminaries

## 2.1 Environment
If you would like to use the datasets we provide, the following environment setup is required:
- Ubuntu 18.04.6 LTS
- Python 3.9.13
- PyTorch 2.5.1+cu121
- Pandas 2.2.3
- Numpy 2.0.2

If you would like to use a custom dataset, the following additional environment setup is required:
- Scapy 2.6.1
- CICFlowMeter

## 2.2 Dataset, AAT, and Pre-trained models
1. We provide the datasets, AAT, and pre-trained models used to reproduce the results in the paper. These can be downloaded from XXX. **After downloading, replace the `storage` directory in the repository with the downloaded `storage`**.
2. If you would like to download the original PCAP files for the datasets, please refer to CIC-IDS-2018 and TON_IoT.

# 3 Reproducing results in our paper
1. Before reproducing the experimental results, ensure that the directory structure under `storage` matches the one presented in the **Introduction**. Then, navigate to the `reproduce_experiments_results` directory:
   ```
   cd reproduce_experiments_results
   ```
2. All scripts in the `reproduce_experiments_results` directory are named according to their corresponding subsection numbers. For example, `5_2_eval_NIDSs_performance.py` is used to reproduce the results of Section 5.2 in the paper. You can run the script with the following command:
   ```
   python 5_2_eval_NIDSs_performance.py
   ```
   ```
   python 5_3_adv_attacks_against_NIDSs.py
   ```
   ```
   ...
   ```
   - **All scripts are configured by default to run on the CIC-IDS-2018 dataset**. If you want to switch to a different dataset, change `dataset_name = 'ids18'` to `dataset_name = 'ton'` in the script. (You can easily search for the string `dataset_name = 'ids18'` — each script contains only one occurrence.)
   - **The script `5_7_mitigating_adv_attacks.py` is configured by default to use the normal adversarial training models**. If you would like to use the adversarial training with SPTS models, change `adv_train = 'normal_adv_train'` to `adv_train = 'adv_train_with_SPTS'` in line 42.

# 4 Details of training models
...

# 5 Generating AAT
...

# 6 Details of adversarial training
...

# 7 Other tips
...

# 8 Acknowledgments