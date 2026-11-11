import torch
import numpy as np
import pickle
from tqdm import tqdm

import torch
import numpy as np
import pickle
from tqdm import tqdm

def generate_hw_lookup_table(
    M_min=1,
    M_max=18500,
    max_dim=2320,
    step=4,
    area_margin=1.1,
    save_path='./hw_lookup.pkl'
):

    lookup = {}
    max_area = int(M_max * area_margin)

    possible_hw = []
    for H in range(step, max_dim + step, step):
        for W in range(step, max_dim + step, step):
            area = H * W
            if area <= max_area:
                aspect_ratio = max(H, W) / min(H, W)
                possible_hw.append((H, W, area, aspect_ratio, -H))

    possible_hw.sort(key=lambda x: (x[2], x[4], x[3]))

    for M in tqdm(range(M_min, M_max + 1), desc="Building lookup table"):
        candidates = [hw for hw in possible_hw if hw[2] >= M]
        if not candidates:
            raise ValueError(f"No valid (H,W) found for M = {M}")
        H, W, _, _, _ = candidates[0] 
        lookup[M] = (H, W)

    with open(save_path, 'wb') as f:
        pickle.dump(lookup, f)
    print(f"Lookup table saved to {save_path} (range: {M_min}-{M_max})")
    print(f"Total entries: {len(lookup)}")
    return lookup

if __name__ == "__main__":
    lookup = generate_hw_lookup_table(max_dim=200, area_margin=1.15)
