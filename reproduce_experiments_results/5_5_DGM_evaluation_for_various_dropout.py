import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net
import torch
from torch.utils.data import DataLoader

def main(dataset_name, sm_name, tm_name, attack, num_copy, dropout_rate):
    # hyper
    dev = torch.device('cuda')
    batch_size = 128

    # print(f"Reading csv...")
    fp_data = os.path.join(project_root_dir, 'storage', 'AAT', sm_name, attack, f'{num_copy}_{dropout_rate}.csv')
    fp_minmax = os.path.join(project_root_dir, 'storage', 'dataset', f'{dataset_name}_minmax_{tm_name[-1]}.csv')
    fp_fea = os.path.join(project_root_dir, 'storage', 'dataset', f'fea_{tm_name[-1]}.csv')
    dataset = CustomDataset(fp_data, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    fp_model = os.path.join(project_root_dir, 'storage', 'pre-trained_models', 'normal_train', f'{tm_name}.pth')
    net = load_net(tm_name, fp_model)
    net.to(dev)

    net.eval()
    TP, FP, TN, FN, curr_iter = 0, 0, 0, 0, 0
    for flows, labels in dataloader:
        flows, labels = flows.to(dev), labels.to(dev)
        curr_iter += len(labels)

        with torch.no_grad():
            pred = net(flows).argmax(1)

        TP += ((pred == 1) & (labels == 1)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()
        acc = TP / (TP + FN)
        print(f"\r {sm_name}->{tm_name}, {attack}, {num_copy}-{dropout_rate}, Progress:{curr_iter}/{len(dataset)}, Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    dataset_name = 'ids18'      # 'ton' 'ids18'
    attack = 'DGM'
    num_copy = 5
    dropout_rates = [0.2, 0.4, 0.6, 0.8]

    model_structures = ['mlp', 'cnn', 'rescnn', 'lstm', 'Selfattention']
    sm_names, tm_names = [], []
    for onestruc in model_structures:
        sm_names.append(f'{dataset_name}_{onestruc}_s')
        tm_names.append(f'{dataset_name}_{onestruc}_t')

    list_midx = []
    for onemodel in sm_names:
        for dropout_rate in dropout_rates:
            list_midx.append((onemodel, dropout_rate))

    midx = pd.MultiIndex.from_tuples(list_midx, names=['Model', 'Dropout_rate'])
    df = pd.DataFrame([[0.] * len(tm_names)] * len(list_midx), index=midx, columns=tm_names)

    print(df)
    for sm_name in sm_names:
        for dropout_rate in dropout_rates:
            acc_sum = 0
            for tm_name in tm_names:
                acc = main(dataset_name, sm_name, tm_name, attack, num_copy, dropout_rate)
                df.loc[(sm_name, dropout_rate), tm_name] = round(acc * 100, 1)
                acc_sum += acc
            df.loc[(sm_name, dropout_rate), 'Average'] = round((acc_sum / len(tm_names)) * 100, 1)
        print(df)
