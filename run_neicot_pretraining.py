# --------------------------------------------------------
# Based on MAE, CPC, BEiT, timm, DINO and DeiT code bases
# https://github.com/pengzhiliang/MAE-pytorch
# https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import datetime
from email.mime import image
from turtle import Turtle
from torchsummary import summary
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from thop import profile
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pretraining_dataset, build_dataset, get_dataset, HyperX
from engine_for_pretraining import train_one_epoch, t_SNE
from utils import NativeScalerWithGradNormCount as NativeScaler, sample_gt, plot
import utils
import modeling_pretrain
from torch.utils import data
import warnings
from masking_generator import RandomMaskingGenerator
from sklearn.manifold import TSNE

# 忽略警告
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch11_11', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--use_model', default='neicot_pre', type=str)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--input_size', default=11, type=int,
                        help='images input size for backbone')
    parser.add_argument('--control_patch', default=5, type=int,
                        help='Must be divisible by 200')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--normlize_target', default=False, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dataset', type=str, default='IndianPines',)
    parser.add_argument('--folder', type=str, default='../../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--sampling_mode', type=str, default='random',
                           help="Sampling mode (random sampling or disjoint, default:  fixed)")
    group_dataset.add_argument('--training_percentage', type=float, default=0.1,
                           help="Percentage of samples to use for training")
    group_dataset.add_argument('--train_gt', action='store_true',
                           help="Samples use of training")
    group_dataset.add_argument('--test_gt', action='store_true',
                           help="Samples use of testing")
    group_dataset.add_argument('--validation_percentage', type=float, default=0.1,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")        
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--patch_size', type=int,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
    group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
    
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--sigma', default=1, type=float)
    parser.add_argument('--cp_loss', default=0.5, type=float)
    parser.add_argument('--random_mask', default=False)
    parser.add_argument('--gauss_std', default=0, type=float)

    return parser.parse_args()

def main(args):
    utils.init_distributed_mode(args)
    
    # hyperspectral
    DATASET = args.dataset
    FOLDER = args.folder
    PATCH_SIZE = args.patch_size
    hyperparams = vars(args)
    img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)    
    N_BANDS = img.shape[-1]
    N_CLASSES = len(LABEL_VALUES)
   
    hyperparams.update(
        {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'supervision':'full', 'center_pixel':True})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    print("hyperparams:",hyperparams)
    print("********cp_loss is : {}".format(args.cp_loss))
    
    # 加载高光谱数据，拿全部的数据用于预训练
    train_gt = gt
    
    mask = np.unique(train_gt)
    tmp = []
    for v in mask:
        tmp.append(np.sum(train_gt==v))
    print("类别：", mask)
    print("训练集大小:", tmp)
    
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)       # 固定随机种子
    model = get_model(args, hyperparams)
    
    
    dataset_train = HyperX(img, train_gt, **hyperparams)    
    
    num_tasks = utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks 
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker    # 一个加快数据读取速度的操作
    )
    
    model.to(device)        # 将模型加载到指定设备上
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():       # 不进行梯度回传，打印模型经过的层
        for input, _, mask in data_loader_train:
            break
        summary(model, input_size = (input.size()[1:], mask.shape), depth=8, device='cpu')

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    
    loss_scaler = NativeScaler()
    print("打印loss_scaler",loss_scaler)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs): 
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=PATCH_SIZE,
            normlize_target=args.normlize_target,
            use_model=hyperparams['use_model'],
            cp_loss=hyperparams['cp_loss'],
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                print("------save model success------")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}
        print("log_stats:",log_stats)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_model(args, hyperspetral):
    print(f"Creating model: {args.model}")  
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        n_bands=hyperspetral['n_bands'],
        depth=hyperspetral['depth'],
        use_model=hyperspetral['use_model'],
        patch_size=hyperspetral['patch_size'],
        sigma=hyperspetral['sigma'],
    )
    return model


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
