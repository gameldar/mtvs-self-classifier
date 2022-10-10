import argparse
import random

import torch
import numpy as np
import sys
import os
from model import Model
from lstm import CNN, LSTM
import shutil
from apex import parallel
from apex.parallel.LARC import LARC
from loss import Loss
from utils import MTSAugmentation, MTSDataset

from torch.cuda.amp import GradScaler
import time
from torch.cuda.amp import autocast
import json
import torch.backends.cudnn as cudnn


def init_random(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)


def create_base_model(num_cols, num_features):
    return CNN(num_cols, num_features)

def get_params_groups(model, args):
    if not args.no_bias_wd and args.bbone_wd is None:
        return model.parameters()
    else:
        regularized = []
        not_regularized = []
        bbone_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (name.endswith(".bias") or len(param.shape) == 1) and args.no_bias_wd:
                not_regularized.append(param)
            elif args.bbone_wd is not None and 'backbone' in name:
                bbone_regularized.append(param)
            else:
                regularized.append(param)

        param_groups = [{'params': regularized}]
        if len(not_regularized):
            param_groups.append({'params': not_regularized, 'weight_decay': 0.})
        if len(bbone_regularized):
            param_groups.append({'params': bbone_regularized, 'weight_decay': args.bbone_wd})

    return param_groups


def load_checkpoint(pretrained, rm_pretrained_cls, start_epoch, model):
    if pretrained is not None:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # load state dictionary
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                # remove classifier if necessary
                if rm_pretrained_cls and 'cls_' in k:
                    del state_dict[k]

                # remove module. prefix
                elif k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]

            start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert len(msg.missing_keys) == 0, "missing_keys: {}".format(msg.missing_keys)
            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))


def save_checkpoint(state, is_best, is_milestone, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
        print('Best model was saved.')
    if is_milestone:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_{}.pth.tar'.format(state['epoch'])))
        print('Milestone {} model was saved.'.format(state['epoch']))
def resume_checkpoint(resume, save_path, start_epoch, model, optimizer):
    last_model_path = os.path.join(save_path, 'model_last.pth.tar')
    if not resume and os.path.isfile(last_model_path):  # automatic resume
        resume = last_model_path
    best_loss = 1e10
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return start_epoch, best_loss



class PrintMultiple(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

# taken from DINO
def cosine_scheduler_with_warmup(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    final_value = base_value if final_value is None else final_value
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_lr(optimizer, lr_schedule, iteration):
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_schedule[iteration]

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms
def train(loader, model, scaler, criterion, optimizer, lr_schedule, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()

    end = time.time()
    for i, (samples, indices) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cos:
            # update learning rate
            adjust_lr(optimizer, lr_schedule, iteration=epoch * len(loader) + i)

        optimizer.zero_grad()

        samples = [x.cuda(non_blocking=True) for x in samples]
        indices = indices.cuda(non_blocking=True)
        with autocast(enabled=args.use_amp):
            embds = model(samples, return_embds=True)

            probs = model(embds, return_embds=False)

            with autocast(enabled=False):
                loss = criterion(probs)

        assert not torch.isnan(loss), 'loss is nan!'

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
            _ = clip_gradients(model, args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        # record loss
        loss = loss.detach()
        losses.update(loss.item(), probs[0][0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == args.print_freq - 1:
            #target = probs[0][1].clone().detach().argmax(dim=1)
            #unique_predictions = torch.unique(target).shape[0]
            #print(f'number of unique predictions (cls 0): {unique_predictions}')
            progress.display(i)

    return losses.avg

parser = argparse.ArgumentParser(description='MTS Self-Supervised Classifier')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints, and logs')
parser.add_argument('--seed', default=100, type=int,
                   help='seed for initialising training. ')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=3, type=int,
                    help='number of MLP hidden layers')
parser.add_argument('--cls-size', type=int, default=[1000], nargs='+',
                    help='size of classification layer. can be a list if cls-size > 1')
parser.add_argument('--use-bn', action='store_true',
                    help='use batch normalization layers in MLP')
parser.add_argument('--fixed-cls', action='store_true',
                    help='use a fixed classifier')
parser.add_argument('--no-leaky', action='store_true',
                    help='use regular relu layers instead of leaky relu in MLP')
parser.add_argument('--data-size', default=450, type=int,
                    help='length of input data size')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--rm-pretrained-cls', action='store_true',
                    help='ignore classifier when loading pretrained model (used for initializing imagenet subset)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--row-tau', default=0.1, type=float,
                    help='row softmax temperature (default: 0.1)')
parser.add_argument('--col-tau', default=0.05, type=float,
                    help='column softmax temperature (default: 0.05)')
parser.add_argument('--eps', type=float, default=1e-12,
                    help='small value to avoid division by zero and log(0)')
parser.add_argument('--no-bias-wd', action='store_true',
                    help='do not regularize biases nor Norm parameters')
parser.add_argument('--bbone-wd', type=float, default=None,
                    help='backbone weight decay. if set to None weight_decay is used for backbone as well.')
parser.add_argument('--sgd', action='store_true',
                    help='use SGD optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--lars', action='store_true',
                    help='use LARS optimizer')
parser.add_argument('--lr', '--learning-rate', default=4.8, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--final-lr', default=None, type=float,
                    help='final learning rate (None for constant learning rate)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='linear warmup epochs (default: 10)')
parser.add_argument('--start-warmup', default=0.3, type=float,
                    help='initial warmup learning rate')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--use-amp', action='store_true',
                    help='use automatic mixed precision')
parser.add_argument('--clip-grad', type=float, default=0.0,
                    help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
                    help optimization for larger ViT architectures. 0 for disabling.""")
parser.add_argument('-p', '--print-freq', default=16, type=int,
                    metavar='N', help='print frequency (default: 16)')
parser.add_argument('--summary-file', default='../results-summary.csv', type=str,
                    help='save file for test best results summary information')

def main():
    if not torch.cuda.is_available():
        raise Exception("GPU not available, aborting.")

    args = parser.parse_args()

    init_random(args.seed)

    if not os.path.exists(args.save_path):
       os.makedirs(args.save_path)
    sys.stdout = PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))
    print(f'Executing: python {" ".join(sys.argv)}')
    with open(os.path.join(args.save_path, 'configuration.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    base_model = create_base_model(args.data_size, args.hidden_dim)
    classifier = LSTM(args.hidden_dim)

    model = Model(base_model=base_model,
                  classifier=classifier,
                  dim=args.dim,
                  cls_size=args.cls_size)

    # restore checkpoint if asked for
    load_checkpoint(args.pretrained, args.rm_pretrained_cls, args.start_epoch, model)
    # allow loading data in parallel if there are multiple GPUs
    # move it to cuda based parameters
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    criterion = Loss(row_tau=args.row_tau, col_tau=args.col_tau, eps=args.eps).cuda(None)
    params_groups = get_params_groups(model, args)
    if args.sgd:
        optimizer = torch.optim.SGD(params_groups, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params_groups, args.lr,
                                      weight_decay=args.weight_decay)

    if args.lars:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    args.start_epoch, best_loss = resume_checkpoint(args.resume, args.save_path, args.start_epoch, model, optimizer)
    cudnn.benchmark = True
    #traindir = os.path.join(args.data, 'train')
    transform = MTSAugmentation()
    import pandas as pd
    df = pd.read_parquet(os.path.join(args.data, 'train.parquet'))
    df = df.drop(columns=['Timestamp'])
    #df = df.loc[0:2999, :]
    #dataset = MTSDataset(traindir, transform=transform)
    dataset = MTSDataset(df, transform=transform, has_labels=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    # schedulers
    lr_schedule = cosine_scheduler_with_warmup(base_value=args.lr,
                                                     final_value=args.final_lr,
                                                     epochs=args.epochs,
                                                     niter_per_ep=len(loader),
                                                     warmup_epochs=args.warmup_epochs,
                                                     start_warmup_value=args.start_warmup)

    # mixed precision
    scaler = GradScaler(enabled=args.use_amp, init_scale=2. ** 14)
    with open(os.path.join(args.save_path, 'epoch_loss.csv'), 'a+') as loss_file:
        if args.start_epoch == 0:
              print("epoch,loss", file=loss_file)
        for epoch in range(args.start_epoch, args.epochs):
            loss_i = train(loader, model, scaler, criterion, optimizer, lr_schedule, epoch, args)
            print(f"{epoch},{loss_i}", file=loss_file, flush=True)

            is_best = True if epoch == 0 else loss_i < best_loss
            best_loss = loss_i if epoch == 0 else min(loss_i, best_loss)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, is_milestone=(epoch + 1) % 25 == 0,
            filename=os.path.join(args.save_path, 'model_last.pth.tar'))
    output_header = not os.path.exists(args.summary_file)
    with open(args.summary_file, 'a+') as summary_file:
        if output_header:
            print("Test name, epochs, best loss, start epoch, batch size, dim, hidden dim, cls size, sgd, lr, row tau, col tau, cos", file=summary_file)
        print(f'{args.save_path}, {args.epochs}, {best_loss}, {args.start_epoch}, {args.batch_size}, {args.dim}, {args.hidden_dim}, {args.cls_size}, {args.sgd}, {args.lr}, {args.row_tau}, {args.col_tau}', file=summary_file)

if __name__ == '__main__':
    main()
