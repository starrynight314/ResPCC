import sys
import os
import random

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

dataset_total_packet = 16000  # 总包数，可改
channel = 1  # 每个 loss 模式生成几个序列
target_loss_rates = [0.20]  # 5%,10%,20%,30%

def generate_random_loss(pe, time, output_dir):
    loss_sequence = []
    for _ in range(dataset_total_packet):
        pl = 1 if random.random() < pe else 0
        loss_sequence.append(pl)
    out_path = os.path.join(output_dir, f"channel{time+1}.txt")
    with open(out_path, 'w') as f:
        f.write(str(loss_sequence))

if __name__ == "__main__":
    for pe in target_loss_rates:
        output_dir = os.path.join('./random_channel/', f"{int(pe*100)}")
        os.makedirs(output_dir, exist_ok=True)
        for j in range(channel):
            generate_random_loss(pe, j, output_dir)
        # 打印实际丢包率
        file_path = os.path.join(output_dir, "channel1.txt")
        with open(file_path, 'r') as f:
            seq = eval(f.read())
        actual_pe = sum(seq) / len(seq)
        print(f"{int(pe*100)}% target, actual loss rate: {actual_pe:.6f}")