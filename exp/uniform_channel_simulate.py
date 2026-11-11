import os
import random


#target_pe = [20] # target packet loss rate in percentage
target_pe = [0.3]

dataset_total_packet = 80
channel = 1 # number of channels for each pe


def generate_txt(pe, pe_str, time):
    # 1 for packet loss, 0 for packet received
    uniform_channel = [1 if random.random() < pe else 0 for _ in range(dataset_total_packet)]
    out_path = './uniform_channel/' + pe_str + '/channel' + str(time) + '.txt'
    with open(out_path, "w") as f:
        f.write(str(uniform_channel))


for i in range(len(target_pe)):
    output_dir = os.path.join('./uniform_channel', str(target_pe[i]))
    os.makedirs(output_dir, exist_ok=True)
    for j in range(channel):
        #generate_txt(target_pe[i] / 100, str(target_pe[i]), j)
        generate_txt(target_pe[i] , str(target_pe[i]), j)


# # check Pe
# with open("./uniform_channel/20/channel3.txt", 'r') as f:
#     gilbert_channel = f.read()
#     gilbert_channel = eval(gilbert_channel)
# cout_0 = 0
# cout_1 = 0
# for i in gilbert_channel[:600]:
#     if i == 0:
#         cout_0 += 1
#     else:
#         cout_1 += 1
# print("0:", cout_0 / (cout_0 + cout_1))
# print("1:", cout_1 / (cout_0 + cout_1))

for pe in target_pe:
    channel_dir = os.path.join('./uniform_channel', str(pe))
    for i in range(channel):
        file_path = os.path.join(channel_dir, f"channel{i}.txt")
        with open(file_path, 'r') as f:
            gilbert_channel_str = f.read()
            gilbert_channel_list = eval(gilbert_channel_str)
        cout_0 = 0
        cout_1 = 0
        for value in gilbert_channel_list[:600]:
            if value == 0:
                cout_0 += 1
            else:
                cout_1 += 1
        total_count = cout_0 + cout_1
        if total_count > 0:
            ratio_0 = cout_0 / total_count
            ratio_1 = cout_1 / total_count
        file_identifier = f"{pe}_channel{i}"
        print(f"{file_identifier}_lossless_ratio: {ratio_0:.6f}")
        print(f"{file_identifier}_loss_ratio (loss rate): {ratio_1:.6f}")
        print("---") # 添加分隔符，让输出更清晰
