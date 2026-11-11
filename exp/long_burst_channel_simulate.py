import sys
import os
import random

# -----------------------------
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
# -----------------------------

dataset_total_packet = 16000
channel = 1
target_loss_rates = [0.05, 0.10, 0.20, 0.30]

min_burst = 8
max_burst = 20

base_dir = "./burst_channel"


def max_consecutive_ones(seq):
    """统计真实最大连续丢包长度"""
    max_len = 0
    cur = 0
    for v in seq:
        if v == 1:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len


def can_place_burst(seq, start, length):
    """
    判断是否可以在 [start, start+length) 放置一个 burst。
    要求：
    1. 区间内部必须全是 0，不能和已有 burst 重叠；
    2. 左右边界最好也不能紧贴已有 1，避免两个 burst 合并成更长连续丢包。
    """
    n = len(seq)
    end = start + length

    if start < 0 or end > n:
        return False

    if any(seq[start:end]):
        return False

    if start > 0 and seq[start - 1] == 1:
        return False

    if end < n and seq[end] == 1:
        return False

    return True


def generate_long_burst_loss(pe, time, output_dir, max_try=200000):
    """
    生成长爆发丢包序列。
    每个 burst 长度随机在 [min_burst, max_burst]。
    严格保证最终最大连续丢包长度 <= max_burst。
    """
    total_packets_to_drop = int(round(pe * dataset_total_packet))
    loss_sequence = [0] * dataset_total_packet
    dropped = 0
    tries = 0

    while dropped < total_packets_to_drop:
        tries += 1
        if tries > max_try:
            raise RuntimeError(
                f"Failed to generate sequence after {max_try} tries. "
                f"Current dropped={dropped}, target={total_packets_to_drop}. "
                f"Try reducing PLR or min_burst, or increasing dataset_total_packet."
            )

        remaining = total_packets_to_drop - dropped

        # 最后剩余不足 min_burst 时，允许生成短 burst，保证精确达到目标丢包率
        if remaining < min_burst:
            burst_length = remaining
        else:
            burst_length = random.randint(min_burst, min(max_burst, remaining))

        start = random.randint(0, dataset_total_packet - burst_length)

        if not can_place_burst(loss_sequence, start, burst_length):
            continue

        for i in range(start, start + burst_length):
            loss_sequence[i] = 1

        dropped += burst_length

        # 安全检查：如果超过 max_burst，立刻报错
        current_max = max_consecutive_ones(loss_sequence)
        if current_max > max_burst:
            raise RuntimeError(
                f"Generated invalid sequence: max consecutive loss = {current_max} > {max_burst}"
            )

    final_max = max_consecutive_ones(loss_sequence)
    actual_pe = sum(loss_sequence) / len(loss_sequence)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"channel{time}.txt")

    with open(out_path, "w") as f:
        f.write(str(loss_sequence))

    return out_path, actual_pe, final_max


if __name__ == "__main__":
    random.seed(2026)

    for pe in target_loss_rates:
        pe_int = int(round(pe * 100))
        output_dir = os.path.join(base_dir, str(pe_int))
        os.makedirs(output_dir, exist_ok=True)

        for j in range(channel):
            file_path, actual_pe, max_consec = generate_long_burst_loss(
                pe=pe,
                time=j,
                output_dir=output_dir
            )

            print(
                f"{pe_int}% target | "
                f"actual loss rate: {actual_pe:.6f} | "
                f"max consecutive loss: {max_consec} | "
                f"saved to: {file_path}"
            )