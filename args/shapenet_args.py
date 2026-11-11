import argparse
from models.utils import str2bool


def parse_shapenet_args():
    parser = argparse.ArgumentParser(description='Model Arguments')

    # dataset name
    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for backbone')
    parser.add_argument('--aux_lr', default=1e-3, type=float, help='learning rate for entropy model')
    parser.add_argument('--betas', default=(0.9, 0.999), type=float, help='betas for adam optimizer')
    # lr scheduler
    parser.add_argument('--lr_decay_step', default=10, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    # dataset
    parser.add_argument('--train_data_path', default='./data/shapenet/shapenet_train_cube32.pkl', type=str, help='path to train dataset')
    parser.add_argument('--train_cube_size', default=32, type=int, help='cube size of train dataset')
    parser.add_argument('--train_map_size', default=128, type=int, help='map size of train dataset')
    parser.add_argument('--val_data_path', default='./data/shapenet/shapenet_val_cube32.pkl', type=str, help='path to val dataset')
    parser.add_argument('--val_cube_size', default=32, type=int, help='cube size of val dataset')
    parser.add_argument('--val_map_size', default=128, type=int, help='map size of val dataset')
    parser.add_argument('--test_data_path', default='./data/shapenet/shapenet_test_cube32.pkl', type=str, help='path to test dataset')
    parser.add_argument('--test_cube_size', default=32, type=int, help='cube size of test dataset')
    parser.add_argument('--test_map_size', default=128, type=int, help='map size of test dataset')
    # train
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--print_freq', default=1000, type=int, help='frequency of printing training losses and metrics')
    parser.add_argument('--save_freq', default=5, type=int, help='frequency of saving model checkpoints')
    parser.add_argument('--output_path', default='./output/shapenet', type=str, help='path to save checkpoints')
    parser.add_argument('--log_dir', default='./log/shapenet', type=str, help='path to save training logs')

    # model
    parser.add_argument('--in_fdim', default=3, type=int, help='input dimension, may contain attributes')
    parser.add_argument('--k', default=8, type=int, help='knearest neighbor')
    parser.add_argument('--downsample_rate', default=[1/5, 1/5, 1/5], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[16, 16, 16], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--layer_num', default=3, type=int, help='downsample/upsmaple stage')
    parser.add_argument('--dim', default=64, type=int, help='feature dimension')
    parser.add_argument('--hyper_channels',default =48, type=int, help='hyper dimension')
    parser.add_argument('--hidden_dim', default=128, type=int, help='hiddem dimension')
    parser.add_argument('--ngroups', default=4, type=int, help='groups for groupnorm')
    # loss
    parser.add_argument('--chamfer_coe', default=0.5, type=float, help='chamfer loss coefficient for intermediate point clouds')
    parser.add_argument('--pts_num_coe', default=5e-7, type=float, help='pts num loss coefficient')
    parser.add_argument('--bpp_lambda', default=1e-3, type=float, help='bpp loss coefficient')
    parser.add_argument('--mean_distance_coe', default=5e1, type=float, help='mean distance loss coefficient')
    parser.add_argument('--std_distance_coe', default=5e1, type=float, help='std distance loss coefficient')
    parser.add_argument('--density_coe', default=1e-4, type=float, help='density loss coefficient')
    # test
    parser.add_argument('--model_path', default='', type=str, help='path to ckpt')
    parser.add_argument('--peak_point', default=3.464101615137755, type=float, help='peak value for PSNR calculation')
    parser.add_argument('--loss_rate', default=10, type=int, help='packet loss rate')
    parser.add_argument('--results', default='./results/shapenet', type=str, help='path to save test logs and reconstructed results')


    args = parser.parse_args()

    return args