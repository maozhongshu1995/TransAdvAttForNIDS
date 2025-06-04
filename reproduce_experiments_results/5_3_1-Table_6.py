import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import STORAGE_DIR

fp = os.path.join(STORAGE_DIR, 'dataset', 'ids18_raw_att.csv')
df = pd.read_csv(fp, header=0, index_col=None)
res = df['Label'].value_counts().to_dict()
res['All'] = len(df)

for key in res.keys():
    print(f'{key}: {res[key]}')