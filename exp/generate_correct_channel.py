#生成无丢包情况下的每个包的丢包情况模拟

dataset_total_packet = 56800
channel = []
out_path = "./gilbert_channel/0/channel0.txt"
for i in range(dataset_total_packet):
    channel.append(0)
with open(out_path, 'w') as f:
    f.write(str(channel))