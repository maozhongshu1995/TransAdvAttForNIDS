import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net
import torch
from torch.utils.data import DataLoader

def main(dataset_name, model_name):
    dev = torch.device('cuda')
    batch_size = 128

    fp_data = os.path.join(project_root_dir, 'storage', 'dataset', f'{dataset_name}_verifying_{model_name[-1]}.csv')
    fp_minmax = os.path.join(project_root_dir, 'storage', 'dataset', f'{dataset_name}_minmax_{model_name[-1]}.csv')
    fp_fea = os.path.join(project_root_dir, 'storage', 'dataset', f'fea_{model_name[-1]}.csv')
    dataset = CustomDataset(fp_data, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    fp_model = os.path.join(project_root_dir, 'storage', 'pre-trained_models', 'normal_train', f'{model_name}.pth')
    net = load_net(model_name, fp_model)
    net.to(dev)

    net.eval()
    TP, FP, TN, FN, curr_iter = 0, 0, 0, 0, 0
    acc, pre, rec, f1 = None, None, None, None
    for flows, labels in dataloader:
        flows, labels = flows.to(dev), labels.to(dev)
        curr_iter += len(labels)

        with torch.no_grad():
            pred = net(flows).argmax(1)
        TP += ((pred == 1) & (labels == 1)).sum().item()
        FP += ((pred == 1) & (labels == 0)).sum().item()
        TN += ((pred == 0) & (labels == 0)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()

        acc = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) !=0 else 0
        pre = TP / (TP + FP) if (TP + FP) != 0 else 0
        rec = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) != 0 else 0

        print(f"\r {model_name}, Progress:{curr_iter}/{len(dataset)}, TP|FP|TN|FN:{TP}|{FP}|{TN}|{FN}, Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}", end="")
    print()
    return acc, pre, rec, f1

if __name__ == '__main__':
    dataset_name = 'ids18'      # 'ton' 'ids18'
    model_structures = ['mlp', 'cnn', 'rescnn', 'lstm', 'Selfattention']
    model_types = ['t', 's']

    model_names = []
    for model_type in model_types:
        for model_structure in model_structures:
            model_names.append(f'{dataset_name}_{model_structure}_{model_type}')

    res = pd.DataFrame([[0.] * 4] * len(model_names), columns=['acc', 'pre', 'rec', 'f1'], index=model_names)
    print(res)
    for model_name in model_names:
        acc, pre, rec, f1 = main(dataset_name, model_name)
        res.loc[model_name, 'acc'] = round(acc * 100, 1)
        res.loc[model_name, 'pre'] = round(pre * 100, 1)
        res.loc[model_name, 'rec'] = round(rec * 100, 1)
        res.loc[model_name, 'f1'] = round(f1 * 100, 1)
        print(res)

