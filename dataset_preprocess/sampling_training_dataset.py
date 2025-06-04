import pandas as pd

fp = ''
path_to_save_dataset = ''

df = pd.read_csv(fp, header=0, index_col=None)

df_ben = df.loc[df['Label']=='Benign']
df_att = df.loc[df['Label']!='Benign']

if len(df_ben) > len(df_att):
    num = len(df_ben)
    df_att = df_att.sample(num, replace=True, axis=0).reset_index(drop=True)
elif len(df_ben) < len(df_att):
    num = len(df_att)
    df_ben = df_ben.sample(num, replace=True, axis=0).reset_index(drop=True)
else:
    pass

df_res = pd.concat([df_ben, df_att], axis=0)
df_res.to_csv(path_to_save_dataset, header=True, index=False)
