import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_dir not in sys.path: sys.path.append(project_root_dir)
import torch
import pandas as pd
from utils.utils import rectify_adv_flows
import time


def MIFGSM(model, lossfn, flows, labels, mask1, step, step_length, device, min_val, max_val):

    t1 = time.perf_counter()
    momentum = 0.
    mu = 1.5
    adv_flows = flows.copy(deep=True)
    
    for _ in range(step):
        adv_df = adv_flows.copy(deep=True)
        
        # 对数据进行规范化
        adv_df = (adv_df - min_val) / (max_val - min_val)
        adv_df = adv_df.fillna(0)
        adv_tensor = torch.from_numpy(adv_df.values).float().to(device)
        
        labels_tensor = torch.ones(adv_tensor.size(0), dtype=torch.long).to(device)

        # 根据算法，生成扰动方向
        adv_tensor.requires_grad_(True)
        # mask_loc1 = torch.from_numpy(np.random.choice([0, 1], size=adv_tensor.shape, p=[0.5, 0.5])).to(device)
        # # loss = lossfn(model(adv_tensor * mask_loc1), labels_tensor)

        loss = lossfn(model(adv_tensor), labels_tensor)
        loss.backward()

        momentum = mu * momentum + adv_tensor.grad / torch.norm(adv_tensor.grad, p=1)
        perturbation_direction = torch.sign(momentum) * mask1

        # 生成扰动，并在未规范化的流量上进行修改，生成对抗流量
        # 之所以不能对规范化的流量进行逆规范化，是因为数值损失（类似于图像压缩和解压缩过程中损失），例如bot流量的持续时间为557 us，逆规范化后就会变成0
        pert = step_length * perturbation_direction
        pert = pd.DataFrame(pert.to("cpu").numpy(), columns=adv_flows.columns)

        # 最小值减小，最大值增加
        pert.loc[pert['Fwd IAT Max'] < 0, ['Fwd IAT Max']] = 0.
        pert.loc[pert['Fwd IAT Min'] > 0, ['Fwd IAT Min']] = 0.
        pert.loc[pert['Fwd Pkt Len Max'] < 0, ['Fwd Pkt Len Max']] = 0.
        pert.loc[pert['Fwd Pkt Len Min'] > 0, ['Fwd Pkt Len Min']] = 0.

        # 对流量进行修正
        adv_flows += pert.values
        rectify_adv_flows(adv_flows, flows, pert)

    t2 = time.perf_counter()
    res_time = t2 - t1

    res_payload = (adv_flows['Fwd Pkt Len Max'] - flows['Fwd Pkt Len Max']).sum()
    res_iat = (adv_flows['Fwd IAT Max'] - flows['Fwd IAT Max']).sum()

    # 统计没有修改的流量
    # are_equal = (flows.reset_index(drop=True) == adv_flows.reset_index(drop=True)).all(axis=1)
    # count_equal_rows = are_equal.sum()
    # print(count_equal_rows)
        
    return adv_flows, (res_time, res_payload, res_iat)
