from torch.utils.data import Dataset
import pandas as pd
import torch
from utils.surrogate_models import mlp_s, cnn_s, ResCNN_s, lstm_s, SelfAttention_s
from utils.target_models import mlp_t, cnn_t, ResCNN_t, lstm_t, SelfAttention_t
from utils.target_models_with_78_fea import mlp_t_78, cnn_t_78, ResCNN_t_78, lstm_t_78, SelfAttention_t_78
from utils.surrogate_model_with_var_input_fea import mlp_s_varfea

STORAGE_DIR = ''

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

def init_net(model_type, model_name:str):
    net = None
    if model_type == 't':
        if model_name == 'mlp_t':
            net = mlp_t()
        if model_name == 'cnn_t':
            net = cnn_t()
        if model_name == 'rescnn_t':
            net = ResCNN_t()
        if model_name == 'lstm_t':
            net = lstm_t()
        if model_name == 'Selfattention_t':
            net = SelfAttention_t()
    if model_type == 's':
        if model_name == 'mlp_s':
            net = mlp_s()
        if model_name == 'cnn_s':
            net = cnn_s()
        if model_name == 'rescnn_s':
            net = ResCNN_s()
        if model_name == 'lstm_s':
            net = lstm_s()
        if model_name == 'Selfattention_s':
            net = SelfAttention_s()
    return net

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

def get_var(n, x1, x2, rwa_var, x_mean, delta1, delta2, delta_mean):
    var = (2 * (x1 * delta1 - x_mean * delta1 + x2 * delta2 - x_mean * delta2) + (delta1 - delta_mean) ** 2 + (delta2 - delta_mean) ** 2 + (n - 2) * (delta_mean ** 2) + n * rwa_var) / n
    return var

def rectify_adv_flows(adv_flows:pd.DataFrame, flows:pd.DataFrame, pert:pd.DataFrame):
    # first level

    # adv_flows.loc[adv_flows['Fwd Pkt Len Max'] > 1460, ['Fwd Pkt Len Max']] = adv_flows['Fwd Pkt Len Max'] - pert['Fwd Pkt Len Max']
    adv_flows['Fwd Pkt Len Max'] = adv_flows['Fwd Pkt Len Max'].clip(upper=1460)

    adv_flows['Fwd Pkt Len Min'] = adv_flows['Fwd Pkt Len Min'].clip(lower=0)
    adv_flows['Fwd IAT Min'] = adv_flows['Fwd IAT Min'].clip(lower=1)

    delta1 = adv_flows['Fwd Pkt Len Max'] - flows['Fwd Pkt Len Max']
    delta2 = adv_flows['Fwd Pkt Len Min'] - flows['Fwd Pkt Len Min']
    delta3 = adv_flows['Fwd IAT Max'] - flows['Fwd IAT Max']
    delta4 = adv_flows['Fwd IAT Min'] - flows['Fwd IAT Min']

    # second level
    adv_flows['Flow Duration'] = flows['Flow Duration'] + (delta3 + delta4)
    adv_flows['TotLen Fwd Pkts'] = flows['TotLen Fwd Pkts'] + (delta1 + delta2)
    adv_flows['Fwd Pkt Len Mean'] = flows['Fwd Pkt Len Mean'] + ((delta1 + delta2) / adv_flows['Tot Fwd Pkts'])
    adv_flows['Fwd Pkt Len Std'] = get_var(flows['Tot Fwd Pkts'], flows['Fwd Pkt Len Max'], flows['Fwd Pkt Len Min'], flows['Fwd Pkt Len Std']**2, flows['Fwd Pkt Len Mean'], delta1, delta2, adv_flows['Fwd Pkt Len Mean']-flows['Fwd Pkt Len Mean']) ** (1/2)

    adv_flows['Fwd IAT Tot'] = flows['Fwd IAT Tot'] + (delta3 + delta4)
    adv_flows['Fwd IAT Mean'] = flows['Fwd IAT Mean'] + ((delta3 + delta4) / (flows['Tot Fwd Pkts'] - 1))
    adv_flows['Fwd IAT Std'] = get_var(flows['Tot Fwd Pkts'] - 1, flows['Fwd IAT Max'], flows['Fwd IAT Min'], flows['Fwd IAT Std']**2, flows['Fwd IAT Mean'], delta3, delta4, adv_flows['Fwd IAT Mean']-flows['Fwd IAT Mean']) ** (1/2)

    adv_flows['Pkt Len Mean'] = flows['Pkt Len Mean'] + ((delta1 + delta2) / (flows['Tot Fwd Pkts'] + flows['Tot Bwd Pkts']))
    adv_flows['Pkt Len Std'] = get_var(flows['Tot Fwd Pkts'] + flows['Tot Bwd Pkts'], flows['Fwd Pkt Len Max'], flows['Fwd Pkt Len Min'], flows['Pkt Len Var'], flows['Pkt Len Mean'], delta1, delta2, adv_flows['Pkt Len Mean']-flows['Pkt Len Mean']) ** (1/2)
    adv_flows['Pkt Len Var'] = adv_flows['Pkt Len Std'] ** 2
    adv_flows['Pkt Len Min'] = adv_flows[['Fwd Pkt Len Min', 'Bwd Pkt Len Min']].min(axis=1)
    adv_flows['Pkt Len Max'] = adv_flows[['Fwd Pkt Len Max', 'Bwd Pkt Len Max']].max(axis=1)

    adv_flows['Pkt Size Avg'] = flows['Pkt Size Avg'] + ((delta1 + delta2) / (flows['Tot Fwd Pkts'] + flows['Tot Bwd Pkts']))

    # third level
    adv_flows['Flow Byts/s'] = (adv_flows['TotLen Fwd Pkts'] + adv_flows['TotLen Bwd Pkts']) / (adv_flows['Flow Duration']/1000000)
    adv_flows['Flow Pkts/s'] = (adv_flows['Tot Fwd Pkts'] + adv_flows['Tot Bwd Pkts']) / (adv_flows['Flow Duration']/1000000)
    adv_flows['Fwd Pkts/s'] = adv_flows['Tot Fwd Pkts'] / (adv_flows['Flow Duration']/1000000)
    adv_flows['Bwd Pkts/s'] = adv_flows['Tot Bwd Pkts'] / (adv_flows['Flow Duration']/1000000)

    adv_flows['Subflow Fwd Byts'] = adv_flows['TotLen Fwd Pkts']