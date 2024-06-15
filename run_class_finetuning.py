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
import numpy as np
from numpy import *
import time
import torch
import torch.backends.cudnn as cudnn
from thop import profile

from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from torchsummary import summary

from timm.models import create_model
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_dataset, get_dataset, HyperF
from engine_for_finetuning import train_one_epoch, evaluate, test_pre, test
from utils import NativeScalerWithGradNormCount as NativeScaler, sample_gt, metrics, plot, plot_correlation_matrix, display_goundtruth
import utils as utils
import modeling_finetune
import warnings
from sklearn.manifold import TSNE

# 忽略警告
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('MAE fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--run', type=int, default=15, help="Running times.")

    # Model parameters
    parser.add_argument('--model', default='vit_3_patch', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    parser.add_argument('--control_patch', default=5, type=int,
                        help='Must be divisible by 200')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

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
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='None', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=17, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    
    
    # 高光谱上的参数
    parser.add_argument('--dataset', type=str, default='IndianPines')
    
    parser.add_argument('--folder', type=str, default='../../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")

    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--sampling_mode', type=str, default='fixed',
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
    group_dataset.add_argument('--load_data', type=str, default=None,
                           help="Samples use of training")          
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--patch_size', type=int,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
    group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--model_name', type=str, default='conv_vit',
                        help="conv_vit or neicot_liner")
    parser.add_argument('--sigma', default=3, type=float)
    parser.add_argument('--cp_loss', default=0.5, type=float)
    parser.add_argument('--save_path', default='',
                        help='similar matrix save path')
    parser.add_argument('--gauss_std', default=0.2, type=float)

    

    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed: # 分布式框架
        try:
            import deepspeed    # type: ignore
            from deepspeed import DeepSpeedConfig   # type: ignore
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None
    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)
    RUN = args.run
    DATASET = args.dataset
    FOLDER = args.folder
    LOAD_DATA = args.load_data
    PATCH_SIZE = args.patch_size
    TRAINING_PERCENTAGE = args.training_percentage
    SAMPLING_MODE = args.sampling_mode
    hyperparams = vars(args)
    img, gt, LABEL_ALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
    N_BANDS = img.shape[-1]
    N_CLASSES = len(LABEL_ALUES)
    args.nb_classes = N_CLASSES
    
    hyperparams.update(
    {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'supervision':'full', 'center_pixel':True})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    acc_dataset = np.zeros([RUN, 1])
    A = np.zeros([RUN, N_CLASSES-1])
    start_time_all = time.time()
    seed = [4, 1, 2, 3, 5]

    for i in range(RUN):
        # 加载高光谱数据
        if LOAD_DATA:
            train_gt_file = '../../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
            val_gt_file  = '../../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
            train_gt = np.load(train_gt_file, 'r')     
            val_gt = np.load(val_gt_file, 'r')   
        else:
            train_gt, val_gt = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE, sample_nums=5)
        
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        mask = np.unique(val_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(val_gt==v))
        device = torch.device(args.device)

        torch.manual_seed(seed[i])
        np.random.seed(seed[i])

        dataset_train = HyperF(img, train_gt, **hyperparams)
        dataset_val = HyperF(img, val_gt, **hyperparams)

        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        log_writer = None 
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        if dataset_val is not None:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
        else:
            data_loader_val = None
            
        mixup_fn = None

        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            n_bands=hyperparams['n_bands'],
            n_classes=hyperparams['n_classes'],
            depth=hyperparams['depth'],
            model_name123=hyperparams['model_name'],
            patch_size=hyperparams['patch_size'],
            sigma=hyperparams['sigma']
        )

        # with torch.no_grad():       # 打印网络结构
        #     for input, _ in data_loader_train:
        #         break
        #     summary(model, input.size()[1:], depth=4, device='cpu')
        
        if args.finetune:           # 判断是否有预训练好的模型，有就加载预训练好的模型
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load ckpt from %s" % args.finetune)  # 加载预训练过的模型参数
            checkpoint_model = None
            for model_key in args.model_key.split('|'): 
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()     # state_dict变量存放训练过程中需要学习的权重和偏执系数
            
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = model.pos_embed.shape[-2] - num_patches
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed
            
            utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
        for name, param in model.named_parameters():        # 设置只更新分类头（线性分类）
            if name not in ['head.weight', 'head.bias']:
                param.requires_grad = False
        
        model.to(device)
        model_ema = None
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print("Model = %s" % str(model_without_ddp))  # 打印网络结构参数
        print('number of params:', n_parameters)

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        args.lr = args.lr * total_batch_size / 64
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(dataset_train))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_layers = model_without_ddp.get_num_layers()
        print("num_layers:",num_layers)
        if args.layer_decay < 1.0:      # args.layer_decay  default=0.75
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None
        
        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))    # 赋值的显示

        skip_weight_decay_list = model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)

        if args.distributed:    # True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        
        criterion = torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(criterion))

        # 判断是否已有训练过的模型直接加载
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

        if args.eval:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            exit(0)
        
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        # 开始训练
        for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Training the network"):
            
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, data_loader_val, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                model_name=hyperparams['model_name'], cp_loss=hyperparams['cp_loss'],run=hyperparams['epochs'],
            )

        # 取出样本计算混淆矩阵并得到结果
        test_stat, prediction, targets = test_pre(data_loader_val, model, device, model_name=hyperparams['model_name'])
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stat['acc1']:.1f}%") 
        results = metrics(prediction, targets, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        acc_dataset[i,0] = results['Accuracy']
        print("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        print("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        print("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        print("\nKappa:\n{:.4f}".format(results['Kappa']))
        A[i] = results['F1 scores'][1:]
        print("acc_dataset {}".format(acc_dataset))
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    
    OA_std = np.std(acc_dataset)
    OAMean = np.mean(acc_dataset)
    AA_std = np.std(A,1)
    AAMean = np.mean(A,1)
    
    p = []
    print("{}数据集的结果如下".format(DATASET))
    for item,std in zip(AAMean,AA_std):
        p.append(str(round(item*100,2))+"+-"+str(round(std,2)))
    print(("AAMean {:.2f} +-{:.2f}".format(np.mean(AAMean)*100,np.mean(AA_std))))
    print("OAMean {:.2f} +-{:.2f}".format(OAMean, OA_std))

    total_time_all = time.time() - start_time_all
    total_time_str_all = str(datetime.timedelta(seconds=int(total_time_all)))
    print('Training time {}'.format(total_time_str_all))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
