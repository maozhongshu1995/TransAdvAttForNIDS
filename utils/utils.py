from torch.utils.data import Dataset
import pandas as pd
import torch
from utils.surrogate_models import mlp_s, cnn_s, ResCNN_s, lstm_s, SelfAttention_s
from utils.target_models import mlp_t, cnn_t, ResCNN_t, lstm_t, SelfAttention_t

def normalize_df(df, df_minmax):
    return ((df - df_minmax.loc[0]) / (df_minmax.loc[1] - df_minmax.loc[0])).fillna(0)

class CustomDataset(Dataset):
    def __init__(self, fp_data, fp_minmax, fp_fea):
        self.read_csv_and_precess_df(fp_data, fp_minmax, fp_fea)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def read_csv_and_precess_df(self, fp_data, fp_minmax, fp_fea):
        df_data = pd.read_csv(fp_data, header=0, index_col=False)
        df_minmax = pd.read_csv(fp_minmax, header=0, index_col=False)
        list_col = pd.read_csv(fp_fea, header=0, index_col=False).columns.tolist()
        
        df = df_data[list_col]
        df = normalize_df(df, df_minmax)
        self.data = torch.from_numpy(df.values).float()

        pos = df_data['Label'] == 'Benign'
        df_data.loc[pos, 'label'] = 0
        df_data.loc[~pos, 'label'] = 1
        self.label = df_data['label'].astype(int).values.tolist()

def load_net(model_name:str, fp_model:str):
    net = None
    if model_name[-1] == 's':
        if '_mlp_s' in model_name:
            net = mlp_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_cnn_s' in model_name:
            net = cnn_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_rescnn_s' in model_name:
            net = ResCNN_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_lstm_s' in model_name:
            net = lstm_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_Selfattention_s' in model_name:
            net = SelfAttention_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        return net
    else:
        if '_mlp_t' in model_name:
            net = mlp_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_cnn_t' in model_name:
            net = cnn_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_rescnn_t' in model_name:
            net = ResCNN_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_lstm_t' in model_name:
            net = lstm_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        if '_Selfattention_t' in model_name:
            net = SelfAttention_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True))
        return net