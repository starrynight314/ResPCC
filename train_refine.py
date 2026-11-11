import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import sys
sys.path.append(os.getcwd())
import logging
from compressai.optimizers import net_aux_optimizer
import time
import argparse
from datetime import datetime
import torch.optim as optim
from pynvml import *
import threading
from models.utils import  AverageMeter, str2bool
from dataset.dataset import CompressDataset
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from torch.optim.lr_scheduler import StepLR 
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.autoencoder import AutoEncoder 
chamfer_dist = chamfer_3DDist()

def setup_logger(name, log_file, level=logging.INFO):
    file_handler = logging.FileHandler(log_file, mode='a')        
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger

def train(args):
    start = time.time()
    if args.batch_size > 1:
        print('The performance will degrade if batch_size is larger than 1!')

    log_dir = args.log_dir 
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_FT"
    log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    iter_logger = setup_logger('train_iter_logger', os.path.join(log_dir, 'train_iterations.log'))
    epoch_logger = setup_logger('train_epoch_logger', os.path.join(log_dir, 'train_epochs.log'))
    val_logger = setup_logger('val_loss_logger', os.path.join(log_dir, 'validation.log'))

    train_dataset = CompressDataset(data_path=args.train_data_path, map_size=128,cube_size=args.train_cube_size, batch_size=args.batch_size)
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=60000)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    print('train data has been loading')
    val_dataset = CompressDataset(data_path=args.val_data_path,map_size=128, cube_size=args.val_cube_size, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size)
    print('val data has been loading')

    # set up folders for checkpoints
    str_time = datetime.now().isoformat()
    print('Experiment Time:', str_time)
    checkpoint_dir = os.path.join(args.output_path, str_time, 'ckpt')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    
    model = AutoEncoder(args).cuda()
    print('Training Arguments:', args)
    print('Model Architecture:', model)
    trainable_params = []
    for name, param in model.named_parameters():
        if "refinement_net" in name or "decoder" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    scheduler_steplr = StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_chamfer_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading model weights from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda')
            model_state = checkpoint['model_state_dict']
            model.load_state_dict(model_state, strict=False)
            model.update(force=True)
            print("=> Weights loaded successfully")
            best_val_chamfer_loss = checkpoint.get('best_val_chamfer_loss', float('inf'))

    for epoch in range(0, args.epochs):
        epoch_loss = AverageMeter()
        epoch_chamfer_loss = AverageMeter()
        epoch_density_loss = AverageMeter()
        epoch_pts_num_loss = AverageMeter()
        epoch_bpp_loss = AverageMeter()
        epoch_aux_loss = AverageMeter()  
        epoch_xyz_bpp = AverageMeter()
        epoch_feat_bpp = AverageMeter()
        epoch_total_bpp = AverageMeter()

        model.train()
        accum_steps = 8
        clip_norm = 1.0

        for i, input_dict in enumerate(train_loader): 
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            # input: (b, 3, n) 
            input = input.permute(0, 2, 1).contiguous()
            normals = input_dict['normals'].cuda().permute(0, 2, 1).contiguous()
            input = torch.cat((input, normals), dim=1) #[b,6,n]

            # model forward 
            decompressed_xyzs, loss, loss_items, bpp = model(input)
            epoch_loss.update(loss.item()) 
            epoch_chamfer_loss.update(loss_items['chamfer_loss'])
            epoch_density_loss.update(loss_items['density_loss'])
            epoch_pts_num_loss.update(loss_items['pts_num_loss'])
            epoch_bpp_loss.update(loss_items['bpp_loss'])
            epoch_xyz_bpp.update(loss_items['xyz_bpp'])
            epoch_feat_bpp.update(loss_items['feat_bpp'])
            epoch_total_bpp.update(loss_items['real_bpp'])

            (loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                torch.nn.utils.clip_grad_norm_(model.z_eblock.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad()

            if (i+1) % args.print_freq == 0:
                print("train epoch: %d/%d, iters: %d/%d, loss: %f, avg chamfer loss: %f, "
                      "avg density loss: %f, avg pts num loss: %f,avg bpp taotal: %f,avg bpp xyz: %f,avg bpp feats: %f, avg bpp loss: %f, avg aux loss: %f" %
                      (epoch+1, args.epochs, i+1, len(train_loader), epoch_loss.get_avg(), epoch_chamfer_loss.get_avg(),
                       epoch_density_loss.get_avg(), epoch_pts_num_loss.get_avg(), epoch_total_bpp.get_avg(),epoch_xyz_bpp.get_avg(),epoch_feat_bpp.get_avg(),epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))
                log_message = ("train epoch: %d/%d, iters: %d/%d, loss: %f, avg chamfer loss: %f, "
                      "avg density loss: %f, avg pts num loss: %f,avg bpp taotal: %f,avg bpp xyz: %f,avg bpp feats: %f,avg bpp loss: %f, avg usage: %f,avg aux loss: %f" %
                      (epoch+1, args.epochs, i+1, len(train_loader), epoch_loss.get_avg(), epoch_chamfer_loss.get_avg(),
                       epoch_density_loss.get_avg(), epoch_pts_num_loss.get_avg(), epoch_total_bpp.get_avg(),epoch_xyz_bpp.get_avg(),epoch_feat_bpp.get_avg(), epoch_bpp_loss.get_avg(), epoch_usage.get_avg(),epoch_aux_loss.get_avg()))
                iter_logger.info(log_message)

        scheduler_steplr.step()
        interval = time.time() - start
        print("train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
              "avg density loss: %f, avg pts num loss: %f,  avg bpp taotal: %f,avg bpp xyz: %f,avg bpp feats: %f,avg bpp loss: %f, avg aux loss: %f" %
              (epoch+1, args.epochs, interval/60, interval%60, epoch_loss.get_avg(), epoch_chamfer_loss.get_avg(),
               epoch_density_loss.get_avg(), epoch_pts_num_loss.get_avg(),epoch_total_bpp.get_avg(),epoch_xyz_bpp.get_avg(),epoch_feat_bpp.get_avg(),epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))
        log_message = ("train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
              "avg density loss: %f, avg pts num loss: %f,avg bpp taotal: %f,avg bpp xyz: %f,avg bpp feats: %f,avg bpp loss: %f, avg usage: %f,avg aux loss: %f" %
              (epoch+1, args.epochs, interval/60, interval%60, epoch_loss.get_avg(), epoch_chamfer_loss.get_avg(),
               epoch_density_loss.get_avg(), epoch_pts_num_loss.get_avg(), epoch_total_bpp.get_avg(),epoch_xyz_bpp.get_avg(),epoch_feat_bpp.get_avg(), epoch_bpp_loss.get_avg(), epoch_usage.get_avg(), epoch_aux_loss.get_avg()))
        epoch_logger.info(log_message)

        model.eval()
        val_chamfer_loss = AverageMeter()
        val_bpp_total = AverageMeter()
        val_bpp_feats = AverageMeter()
        val_bpp_xyzs = AverageMeter()
        with torch.no_grad():
            for input_dict in val_loader:
                # xyzs: (b, n, c)
                input = input_dict['xyzs'].cuda()
                # (b, c, n)
                input = input.permute(0, 2, 1).contiguous()
                normals = input_dict['normals'].cuda().permute(0, 2, 1).contiguous()
                input = torch.cat((input, normals), dim=1)

                # gt_xyzs
                gt_xyzs = input[:, :3, :].contiguous()

                # model forward
                decompressed_xyzs, loss, loss_items, bpp = model(input)
                # calculate val loss and bpp 
                d1, d2, _, _ = chamfer_dist(gt_xyzs.permute(0, 2, 1).contiguous(),
                                            decompressed_xyzs.permute(0, 2, 1).contiguous())
                chamfer_loss = d1.mean() + d2.mean()
                val_chamfer_loss.update(chamfer_loss.item())
                val_bpp_total.update(loss_items['real_bpp'])
                val_bpp_feats.update(loss_items['feat_bpp'])
                val_bpp_xyzs.update(loss_items['xyz_bpp'])

        # print loss 
        print("val epoch: %d/%d, val total bpp: %f,val feats bpp: %f,val xyz bpp: %f, val chamfer loss: %f" %
              (epoch+1, args.epochs, val_bpp_total.get_avg(), val_bpp_feats.get_avg(),val_bpp_xyzs.get_avg(),val_chamfer_loss.get_avg()))
        log_message = ("val epoch: %d/%d, val total bpp: %f,val feats bpp: %f,val xyz bpp: %f, val chamfer loss: %f" %
              (epoch+1, args.epochs, val_bpp_total.get_avg(), val_bpp_feats.get_avg(),val_bpp_xyzs.get_avg(), val_chamfer_loss.get_avg()))
        val_logger.info(log_message)

        # save checkpoint
        cur_val_chamfer_loss = val_chamfer_loss.get_avg()
        
        if  cur_val_chamfer_loss < best_val_chamfer_loss or (epoch+1) % args.save_freq == 0:
            model_name = 'ckpt-best.pth' if cur_val_chamfer_loss < best_val_chamfer_loss else 'ckpt-epoch-%02d.pth' % (epoch+1)
            model_path = os.path.join(checkpoint_dir, model_name)
            checkpoint_data = {
                'epoch': epoch + 1, 
                'best_val_chamfer_loss': best_val_chamfer_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_steplr.state_dict()
            }
            torch.save(checkpoint_data, model_path)
        
        if cur_val_chamfer_loss < best_val_chamfer_loss:
            best_val_chamfer_loss = cur_val_chamfer_loss
        

def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    parser.add_argument('--batch_size', default=1, type=int, help='the performance will degrade if batch_size is larger than 1!')
    parser.add_argument('--downsample_rate', default=[1/5, 1/5, 1/5], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[16, 16, 16], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=1e-3, type=float, help='bpp loss coefficient')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['shapenet', 'semantickitti']

    if train_args.dataset == 'shapenet':
        model_args = parse_shapenet_args()
    else:
        model_args = parse_semantickitti_args()

    reset_model_args(train_args, model_args)

    train(model_args)
