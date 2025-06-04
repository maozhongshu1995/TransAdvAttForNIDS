# 1 Introduction

To make the repository easier to navigate and to help readers reproduce the experiments in the paper, we grouped all experiments into three categories:
1. Reproducing the manuscript’s results.
    
    We provide pre-processed datasets and the corresponding generated AAT, enabling users to reproduce almost all results reported in the manuscript.
    
2. Custom training and AAT generation.
    
    To address concerns about our prepared data, we provide scripts that enable users to train their own models and generate AA. Although we did not fix random seeds, repeated experiments confirm that omitting a seed does not affect the manuscript’s core findings or conclusions.
    
3. Mapping AAT to practical packets (including TANTRA).
    
    This set of experiments shows how to convert the generated AAT back into actual network packets and also includes scripts for generating AAT with TANTRA.
    
In light of the considerations above, this README is organized as follows: Section 2 covers the preliminaries—environment setup, dataset downloads, and auxiliary tools; Section 3 explains how to reproduce the manuscript’s results; Sections 4 and 5 describe custom model training and AAT generation, respectively; Section 6 details the procedure for mapping the generated AAT back to real network packets; and Section 7 offers supplementary notes.

# 2 Preliminaries

1. Environment setup
    - Ubuntu 18.04.6 LTS
    - Python 3.9.13
    - PyTorch 2.5.1+cu121
    - Pandas 2.2.3
    - Numpy 2.0.2
    - Dpkt 1.9.8
    - Matplotlib 3.9.4
    - Seaborn 0.13.2
2. Dataset：
    - [CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html)
    - [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets)
3. [CICFlowMeter (CFM)](https://github.com/UNBCIC/CICFlowMeter): If CICFlowMeter is not installed correctly, the experiments in Section 6 (**Mapping AAT to practical packets**) cannot be executed. 

The pre-trained models and pre-processed dataset can be downloaded here (all models and datasets will be uploaded within a few hours). There are 11 archive files in total and extract each one individually. Note that **5_4_4.part1.rar** and **5_4_4.part2.rar** belong to the same archive; you can extract them with: `unrar x 5_4_4.rar`. Create a new, empty directory anywhere (you can choose any name), then move the 10 extracted folders into this directory. Then, open **TransAdvAttForNIDS/utils.py** and update the **STORAGE_DIR** variable (line 9) to the path of that new directory. Within the **STORAGE_DIR** directory, create an empty folder named **adv_pcap**.

Before running the code, create three empty directories—named `output`, `output2`, and `output3`—inside the `TransAdvAttForNIDS/` directory. These empty directories will store intermediate files generated during execution. 

# 3 Reproducing the manuscript’s results

We have included all data generated as well as the corresponding scripts. Simply running these scripts will recreate every table and figure reported, including Tables 5–18 and Figures 2–9.

1. Taking Table 5 as an example, the detailed steps are as follows:
    1. Change to the target directory.
        
        ```bash
        cd TransAdvAttForNIDS/reproduce_experiments_results
        ```
        
    2. Run the script.
        
        ```bash
        python 5_2-Table_5.py
        ```
2. All scripts in the `reproduce_experiments_results` directory can be run directly without any additional parameters.
3. Please note that the results for Tables 15 and 16 may be similar to, but not identical to, those presented in the paper. These tables require real-time measurement of AAT generation time, and the generation process itself involves inherent randomness. So, each runtime can produce slightly different outcomes.
   
# 4 Training NIDSs
We do not introduce new theories or methods for improving NIDS performance, so training the NIDS is not a contribution of our paper. Nevertheless, to eliminate potential concerns, we have supplied training scripts. Unlike “Reproducing the manuscript’s results”, we have not written a separate script for every target NIDS or surrogate model. Because the training procedure is the same, we provide a demo script that trains MLP-t and MLP-s on the TON_IoT dataset. Complete implementations of the other model architectures are available in the folders `TransAdvAttForNIDS/surrogate_model_with_var_input_fea.py` and `TransAdvAttForNIDS/target_models_with_78_fea.py`. If you wish to train models with different architectures, modify the script accordingly.

## 4.1 Training target NIDSs and surrogate models

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Run the script.
    
    ```bash
    python training.py
    ```
    
3. Test the trained model.
    
    ```bash
    python verifying.py
    ```
    
4. The default dataset is TON_IoT, and the default model is the MLP-t with 66 input features. To switch to an MLP-s, set `model_type = 's'` in **training.py** (line 43). By default, the MLP-s uses 60 input features, matching the configuration described in Section 5.3.1 of the manuscript.
5. After 10 epochs, the model will be saved to `STORAGE_DIR/custom/pre-trained_models`.
6. If you want to train an MLP-t with 78 input features or a surrogate model with fewer input features, you must adjust the model architecture. The corresponding definitions are located in **TransAdvAttForNIDS/utils/target_models_with_78_fea.py** and **TransAdvAttForNIDS/utils/surrogate_model_with_var_input_fea.py**, respectively.
   
## 4.2 Normal adversarial training

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Run the script.
    
    ```bash
    python normal_adv_training.py
    ```
    
3. After 10 training epochs, the model will be saved to `STORAGE_DIR/custom/pre-trained_models`.
4. The default configurations are training an MLP-t with 66 input features on the TON_IoT. If you want to train a different model or switch to the CIC-IDS-2018, adjust the model name, dataset path, min–max scaling values, and input features. These parameters are clearly identified in lines 62–69 of the script.
   
## 4.3 Adversarial training with SPTS

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Run the script.
    
    ```bash
    python adv_training_with_SPTS.py
    ```
    
3. After 10 training epochs, the model will be saved to `STORAGE_DIR/custom/pre-trained_models`.

4. The default configurations are training an MLP-t with 66 input features on the TON_IoT. If you want to train a different model or switch to the CIC-IDS-2018, adjust the model name, dataset path, min–max scaling values, and input features. These parameters are clearly identified in lines 84–91 of the script.
   
# 5 Generating AAT

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/generate_AAT
    ```
    
2. Run the script.
    
    ```bash
    python generate_aat.py
    ```
    
3. Test the AAT.
    
    ```bash
    python test_aat.py
    ```
    
4. By default, the script uses the TON_IoT, an MLP-s surrogate model, the MI-FGSM attack with 7 iterations, and a step size of 140. These parameters are defined in lines 74–86 and can be modified to generate a custom AAT.
5. The generated AAT is saved as **aat.csv** in `STORAGE_DIR/custom/output`.

# 6 Mapping AAT to practical packets

This set of experiments maps the generated AAT back to practical packets and then re-extracts features with CFM. Note that the attack flows were further filtered, leaving a total of 36,980 flows (see the first paragraph of Section 5.3.2 in the manuscript). Because the procedure involves multiple steps, the scripts are numbered to indicate the correct execution order.

## 6.1 For SPTS

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/map_AAT_to_pkts
    ```
    
2. Extract features with CFM. While CFM yields the complete feature set used by the target NIDS, we intentionally employ a subset to simulate the attacker’s limited knowledge.
    
    ```bash
    python 0_built_features_with_cfm_over_raw_att_pcap.py
    ```
    
3. Generate the AAT. Unlike the procedure in Section 5 (Generating AAT), the AAT produced here retains only the essential fields—‘Flow ID’, ‘Src IP’, ‘Src Port’, ‘Dst IP’, ‘Dst Port’, ‘Protocol’, 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd IAT Max', and 'Fwd IAT Min’.
    
    ```bash
    python 1_generate_aat.py
    ```
    
4. Process the generated AAT by computing the differences between the original traffic and the AAT.
    
    ```bash
    python 2_process_aat.py
    ```
    
5. Modify packets.
    
    ```bash
    python 3_modify_pcap.py
    ```
    
6. Re-extract features with CFM.
    
    ```bash
    python 4_re-extract_features_with_cfm.py
    ```
    
7. Test the re-extracted features.
    
    ```bash
    python 5_test_aat.py
    ```
    
8. By default, the scripts use the MLP-s (trained in Section 4) surrogate model and MI-FGSM attack with 7 iterations, and a step size of 140. To customize these settings, modify lines 93–105 in **1_generate_aat.py**.

## 6.2 For TANTRA

TANTRA trains an LSTM to learn normal traffic patterns. We adopt exactly the architecture and hyperparameters specified in the original paper, so no additional training script is included; instead, we provide the pre-trained model.

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/TANTRA
    ```
    
2. Modify the *Timestamp* of each attack packet according to the trained LSTM model.
    
    ```bash
    python 0_modify_pkts.py
    ```
    
3. Re-extract features with CFM.
    
    ```bash
    python 1_re-extract_features_with_cfm.py
    ```
    
4. Test the re-extracted features.
    
    ```bash
    python 2_test_aat.py
    ```

# 7 Supplementary Notes

We also supply dataset-preprocessing scripts located in the `TransAdvAttForNIDS/dataset_preprocess` directory.

1. The `divide_dataset_into_target_and_surrogate.py` script divides the dataset into two subsets—one for the target NIDS and one for the surrogate model.
2. The `split_dataset_into_train_and_test.py` script splits each subset into training and test sets according to a specified ratio.
3. The `sampling_training_dataset.py` script performs random oversampling on the training dataset.
4. The script `build_input_features.py` extracts the model’s input features.
5. The `build_minmax.py` script extracts the maximum and minimum values from the training dataset; these values are then used for normalization.
6. Please note that the above scripts cannot be run directly. To execute them, you must specify the input and output file paths within each script. Their code and logic are straightforward, so no further explanation is provided.
7. For the TON_IoT, we recommend using the pre‐labeled CSV files we provide, since extracting features from the raw PCAP files and labeling them with CFM is not trivial.