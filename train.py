#!/usr/bin/env python3
# coding=utf-8
from torch.optim import Optimizer

from data import *
from utils import str2bool, init_torch_tensor
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
# import visdom


def init_parser():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'barcode'], type=str,
                        help='VOC or COCO or barcode')
    parser.add_argument('--dataset_root', default='', help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='Pretrained base model')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='weights', help='Directory for saving checkpoint models')
    parser.add_argument("--img_list_file_path", default="./barcode/CorrectDetect.txt", type=str, help="a file path")
    args = parser.parse_args()
    return parser, args


parser, args = init_parser()


# viz = visdom.Visdom(env=u'barcode', use_incoming_socket=True)


def train():
    print(args)
    # if args.dataset == 'COCO':
    #     if args.dataset_root == VOC_ROOT:
    #         if not os.path.exists(COCO_ROOT):
    #             parser.error('Must specify dataset_root if specifying dataset')
    #         print("WARNING: Using default COCO dataset_root because --dataset_root was not specified.")
    #         args.dataset_root = COCO_ROOT
    #     cfg = coco
    #     dataset = COCODetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    # elif args.dataset == 'VOC':
    #     if args.dataset_root == COCO_ROOT:
    #         parser.error('Must specify dataset if specifying dataset_root')
    #     cfg = voc
    #     dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    # el
    cfg = barcode_cfg_dict
    if args.dataset == 'barcode':
        cfg = barcode_cfg_dict
        dataset = BARCODEDetection(img_list_path=args.img_list_file_path,
                                   transform=SSDAugmentation(cfg['min_dim'], MEANS))
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print(f'Resuming training, loading {args.resume}...')
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(os.path.join(args.save_folder, args.basenet))
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        pass
    # vis_title = 'SSD.PyTorch on ' + dataset.name
    # vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
    # iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
    # epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    # begin epoch rewrite
    for epoch_num in range(0, cfg['max_epoch'] + 1):
        epoch_begin = time.time()
        loc_loss = 0
        conf_loss = 0
        # if args.visdom:
        # update_vis_plot(epoch_num, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
        # reset epoch loss counters
        t0 = time.time()
        if epoch_num in (280, 350, 400):
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        print(f"epoch:{epoch_num} at time {time.time()}")
        for iter_int, iteration in enumerate(iter(data_loader)):
            images, targets = iteration
            with torch.no_grad():
                if args.cuda:
                    images = Variable(images.cuda())
                    targets = [Variable(ann.cuda()) for ann in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(ann) for ann in targets]
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.data.item()
            conf_loss += loss_c.data.item()
            if iter_int % 10 == 0:
                print(f'iter  {repr(iter_int)} || Loss: {loss.data.item()} ||', flush=True)
                print(f'timer: {time.time() - t0} sec.', flush=True)
                print(f'loc_loss {float(loc_loss)}. conf_loss {conf_loss}.', flush=True)
                t0 = time.time()
            # if args.visdom:
            # update_vis_plot(iter_int, loss_l.data.item(), loss_c.data.item(), iter_plot, epoch_plot, 'append')
        if epoch_num % 5 == 0:
            print(f'Saving state, iter: {epoch_num}', flush=True)
            torch.save(ssd_net.state_dict(), f'{args.save_folder}/ssd300_{args.dataset}/{repr(epoch_num)}.pth')
    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'{args.dataset}.pth'))


def adjust_learning_rate(optimizer: Optimizer, gamma: float, step: int) -> None:
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


# def create_vis_plot(_xlabel, _ylabel, _title, _legend):
#     return viz.line(
#         X=torch.zeros((1,)).cpu(),
#         Y=torch.zeros((1, 3)).cpu(),
#         opts=dict(
#             xlabel=_xlabel,
#             ylabel=_ylabel,
#             title=_title,
#             legend=_legend
#         )
#     )
#
#
# def update_vis_plot(iteration: int, loc: float, conf: float, window1, window2, update_type, epoch_size=1) -> None:
#     viz.line(
#         X=torch.ones((1, 3)).cpu() * iteration,
#         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
#         win=window1,
#         update=update_type
#     )
#     # initialize epoch plot on first iteration
#     # if iteration == 0:
#     #     viz.line(
#     #         X=torch.zeros((1, 3)).cpu(),
#     #         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
#     #         win=window2,
#     #         update=True
#     #     )


if __name__ == '__main__':
    os.makedirs(f'{args.save_folder}/ssd300_{args.dataset}')
    init_torch_tensor(args.cuda)
    train()
