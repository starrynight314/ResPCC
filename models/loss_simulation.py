import random
import torch
import os
import sys
sys.path.append(os.getcwd())
from sim2net.packet_loss.gilbert_elliott import GilbertElliott  # pyright: ignore[reportMissingImports]


PRHK_LOOKUP_TABLE = {
    10:          [0.378,   0.883,   0.81,    0.938], # 使用了您原来的参数，假设它对应30%丢包率
    # 5:          [ 0.09, 0.8, 0.77, 0.97],
    # 20:           [ 0.2, 0.4, 0.56, 0.92],
    # 15:
    5:  [0.184,   0.920,      0.900,       0.960],
    15: [0.373,   0.800,      0.700,       0.920], 
    20: [0.466,   0.700,      0.650,       0.900], 
    25: [0.520,   0.600,      0.600,       0.880], 
    30: [0.480,   0.450,      0.550,       0.860],
    35: [0.511,   0.350,      0.520,       0.840],
    40: [0.550,   0.300,      0.480,       0.820]
}
#输入目标丢包率和[c,h,w] 返回mask[c] 
def Gilbert_Elliott_Channel_Zeroing(target_loss_rate, channel_num, device):
    available_rates = list(PRHK_LOOKUP_TABLE.keys())
    closest_rate = min(available_rates, key=lambda rate: abs(rate - target_loss_rate))
    prhk_params = PRHK_LOOKUP_TABLE[closest_rate]
    mask = torch.ones(channel_num, device=device)
    ge = GilbertElliott(prhk_params)

    for i in range(channel_num):
        pl = int(ge.packet_loss()) #对每个通道进行丢包模拟
        if pl == 1:
            mask[i] = 0
    #tensor_copy 丢包之后的数据
    return mask.float()