import argparse
import time
import math
from os import path, makedirs
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import numpy as np
from simsiam.loader import TwoCropsTransform
from simsiam.model_factory import SimSiam
from simsiam.criterion import SimSiamLoss
from simsiam.validation import KNNValidation
from augmentations.simsiam_aug import SimSiamTransform
from PIL import Image

from knn_monitor import knn_monitor
parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--test_cmd', type=str, default='', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)

parser.add_argument('--arch', default='resnet18', help='model name is used for training')

parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup_epoch epochs')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--loss_version', default='simplified', type=str,
                    choices=['simplified', 'original'],
                    help='do the same thing but simplified version is much faster. ()')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--eval_freq', default=5, type=int, help='evaluate model frequency')
parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')

parser.add_argument('--base_lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--soft_loss_rate', type=float, default=0.1, help='soft_loss_rate')

parser.add_argument('--N', type=int, default=2)
parser.add_argument('--M', type=int, default=9)

args = parser.parse_args()

args.learning_rate = args.base_lr * args.batch_size / 256

def main():
    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)

    trial_dir = path.join(args.exp_dir, args.trial)
    logger = SummaryWriter(trial_dir)
    print(vars(args))

    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    cifar_mean_std = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    train_transforms = SimSiamTransform(args.img_dim, mean_std=cifar_mean_std, N=args.N, M=args.M)

    train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=train_transforms)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    model = SimSiam(args)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = SimSiamLoss(args.loss_version)


    knn_test_transform = transforms.Compose([
        transforms.Resize(int(args.img_dim * (8 / 7)), interpolation=Image.BICUBIC),  # 224 -> 256
        transforms.CenterCrop(args.img_dim),
        transforms.ToTensor(),
        transforms.Normalize(*cifar_mean_std)
    ])
    knn_train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=knn_test_transform)
    knn_test_set = datasets.CIFAR10(root=args.data_root,
                                 train=False,
                                 download=True,
                                 transform=knn_test_transform)
    memory_loader = DataLoader(dataset=knn_train_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(dataset=knn_test_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True

    start_epoch = 1
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # routine
    best_acc = 0.0
    # validation = KNNValidation(args, model.encoder)
    for epoch in range(start_epoch, args.epochs+1):

        adjust_learning_rate(optimizer, epoch, args)
        print("Training...")

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        logger.add_scalar('Loss/train', train_loss, epoch)

        if epoch % args.eval_freq == 0:
            print("Validating...")
            # val_top1_acc = validation.eval()
            val_top1_acc = knn_monitor(model.backbone, memory_loader, test_loader, epoch, k=200, hide_progress=True)

            print('Top1: {}'.format(val_top1_acc))

            # save the best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc

                save_checkpoint(epoch, model, optimizer, best_acc,
                                path.join(trial_dir, '{}_best.pth'.format(args.trial)),
                                'Saving the best model!')
            logger.add_scalar('Acc/val_top1', val_top1_acc, epoch)

        # save the model
        if epoch % args.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, val_top1_acc,
                            path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
                            'Saving...')

    print('Best accuracy:', best_acc)

    # save model
    save_checkpoint(epoch, model, optimizer, val_top1_acc,
                    path.join(trial_dir, '{}_last.pth'.format(args.trial)),
                    'Saving the model at the last epoch.')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            images[3] = images[3].cuda(args.gpu, non_blocking=True)

        # compute output
        # outs = model(im_aug1=images[0], im_aug2=images[1])
        z1, p1 = model.forward_single(im_aug1=images[0])
        z2, p2 = model.forward_single(im_aug1=images[1])
        z11, p11 = model.forward_single(im_aug1=images[2])
        z21, p21 = model.forward_single(im_aug1=images[3])
        loss = criterion(z1, z2, p1, p2)
        one_way_loss = criterion.forward_simgle(p11, z1)
        one_way_loss += criterion.forward_simgle(p21, z2)
        reverse_loss = criterion.forward_simgle(p1, z11)
        reverse_loss += criterion.forward_simgle(p2, z21)
        print(f'epoch:{epoch}, loss:{loss}, one_way_loss:{one_way_loss}, reverse_loss:{reverse_loss}')
        loss += args.soft_loss_rate * reverse_loss + (1 - args.soft_loss_rate) * one_way_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    warmup_epoch = args.warmup_epoch
    # cosine lr schedule
    warmup_lr_schedule = np.linspace(0, lr, warmup_epoch)

    if epoch < warmup_epoch:
        lr = warmup_lr_schedule[epoch]
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epoch) / (args.epochs - warmup_epoch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'epoch:{epoch}, lr: {lr}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
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


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer


if __name__ == '__main__':
    main()
    if args.test_cmd:
        test_cmd = args.test_cmd.replace('~blank~', ' ').replace('@', '>') + ' &'
        print(test_cmd)
        os.system(test_cmd)



