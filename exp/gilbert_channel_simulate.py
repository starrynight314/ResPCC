import sys
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from sim2net.packet_loss.gilbert_elliott import GilbertElliott


prhks = [[0.184,0.920,0.900,0.960],[0.378,0.883, 0.810, 0.938],[0.466,0.700,0.650,0.900],[0.480,0.450,0.550,0.860],[0.550,0.300,0.480,0.820]]
target_pr = [] # target packet loss rate in percentage
channel = 10 # number of channels for each prhk 
length = 36000 # length of each channel, should be larger than total packets in test dataset
def generate_txt(prhk, pe, time):
    """
    generate a GE packet loss sequence and save to txt
    prhk: GE model parameters
    pe: target packet loss rate (folder name)
    time: channel index
    """
    ge = GilbertElliott(prhk)
    gilbert_channel = []
    
    # generate sequence of 0/1 indicating success/loss
    for i in range(length):
        pl = int(ge.packet_loss())  # 1: loss, 0: success
        gilbert_channel.append(pl)
    
    # save sequence to txt file
    out_path = './exp/gilbert_channel/' + pe + '/channel' + str(time) + '.txt'
    with open(out_path, 'w') as f:
        f.write(str(gilbert_channel))

if __name__ == "__main__":
    # compute target packet loss for each parameter set and generate sequences
    for i in range(len(prhks)):
        pe = ((1-prhks[i][3]) * prhks[i][1] + (1 - prhks[i][2]) * prhks[i][0]) / (prhks[i][0] + prhks[i][1])
        print("Pe: ", pe)
        target_pr.append(int(round(pe * 100)))
        
        # create folder for this target packet loss
        output_dir = os.path.join('./exp/gilbert_channel', str(target_pr[i]))
        os.makedirs(output_dir, exist_ok=True)
        
        # generate multiple sequences for the same parameter set
        for j in range(channel):
            generate_txt(prhks[i], str(target_pr[i]), j)

    # compute actual loss ratio for each sequence
    for pe in target_pr:
        channel_dir = os.path.join('./exp/gilbert_channel', str(pe))
        for i in range(channel):
            file_path = os.path.join(channel_dir, f"channel{i}.txt")
            
            # read generated sequence
            with open(file_path, 'r') as f:
                gilbert_channel_str = f.read()
            gilbert_channel_list = eval(gilbert_channel_str)
            
            # count successful and lost packets
            cout_0 = 0
            cout_1 = 0
            for value in gilbert_channel_list[:length]:
                if value == 0:
                    cout_0 += 1
                else:
                    cout_1 += 1
            
            # compute ratios
            total_count = cout_0 + cout_1
            if total_count > 0:
                ratio_0 = cout_0 / total_count
                ratio_1 = cout_1 / total_count
            
            # print statistics
            file_identifier = f"{pe}"
            print(f"{file_identifier}_lossless_ratio: {ratio_0:.6f}")
            print(f"{file_identifier}_loss_ratio (loss rate): {ratio_1:.6f}")
            print("---")