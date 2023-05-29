from __future__ import print_function
import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
import random
from utils import *
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataloader import Home_Dataset
import models
import utils

def forward_FDA(rois, targets):
    unk_pixel_list = []
    k_pixel_list = []
    loss_mining_dict = {}
    bs, c, h, w = rois.size()
    p_xk_x = 1.0

    for idx, roi in enumerate(rois):
        roi_label = targets[idx]  # bs x 1 (0-24)
        roi_flatten = roi.view(c, -1).permute(1, 0)

        model.classifier.apply(fix_bn)
        # constant > 0 is grl
        pixel_logits = model.classifier(roi_flatten, adaption=True, constant=-1 * args.mining_grl, pooling=False)
        pixel_scores = pixel_logits.softmax(-1)
        model.classifier.apply(enable_bn)

        sorted_value, sorted_index = pixel_scores[:, :-1].sort(-1, descending=True)
        k_mask = (sorted_index[:, :1] == roi_label).sum(-1).bool()
        unk_mask = ~((sorted_index[:, :args.topk] == roi_label).sum(-1).bool())

        if k_mask.any() and unk_mask.any() and k_mask.sum() > unk_mask.sum():
            unk_pixels = pixel_logits[unk_mask]
            k_pixels = pixel_logits[k_mask]
            unk_pixel_list.append(unk_pixels)
            k_pixel_list.append(k_pixels)

    if len(unk_pixel_list) > 1:

        num_lk = len(torch.cat(k_pixel_list))
        num_pu = len(torch.cat(unk_pixel_list))

        p_xu_x = num_pu / (num_pu + num_lk)
        p_xk_x = num_lk / (num_pu + num_lk)

        mined_scores = torch.cat(unk_pixel_list).softmax(-1)[:, -1]
        loss_mining_unk = p_xu_x * criterion_bce(mined_scores, torch.tensor([args.mining_th] * len(mined_scores)).cuda())
        loss_mining_dict.update(loss_mining_s=loss_mining_unk)
    return loss_mining_dict, p_xk_x


def forward_DCA(rois, all_layers=False, domain='source'):
    domain_label = 1.0 if domain == 'source' else 0.0

    bs, c, h, w = rois.size()
    rois_flatten = rois.permute(0, 2, 3, 1).contiguous().view(-1, c)  # bs, h, w ,c

    model.classifier.apply(fix_bn)
    if not all_layers:
        with torch.no_grad():
            scores = model.classifier(rois_flatten, pooling=False).softmax(-1).detach()
    else:
        scores, rois_flatten = model.classifier(rois_flatten, pooling=False, return_feat=True)
        scores = scores.softmax(-1).detach()

    model.classifier.apply(enable_bn)

    target = torch.full((rois_flatten.size(0),),
                        domain_label,
                        dtype=torch.float,
                        device=rois_flatten.device)

    weight_unk = scores[:, -1]
    weight_k = scores[:, :-1].sum(-1)

    adv_k = model.adv_k(rois_flatten, args.adv_grl)
    adv_unk = model.adv_unk(rois_flatten, args.adv_grl)

    loss_adv_k = (criterion_bce_red(adv_k, target) * weight_k).mean()
    loss_adv_unk = (criterion_bce_red(adv_unk, target) * weight_unk).mean()

    return dict(loss_adv_k=loss_adv_k, loss_adv_unk=loss_adv_unk)

def train(epoch):
    model.train()
    home_loader_iter = iter(train_loader)
    for batch_idx in range(len(train_loader)):

        data_s, target_s, data_t, target_t, _ = home_loader_iter.next()

        data_s, target_s = data_s.cuda(), target_s.long().cuda(non_blocking=True)
        data_t, target_t = data_t.cuda(), target_t.long().cuda(non_blocking=True)

        loss_dict_s = {}
        loss_dict_t = {}

        # source domain 
        rois = model.generator(data_s)
        output_s = model.classifier(rois)
        loss_cls_s = criterion_ce(output_s, target_s)
        loss_dict_s.update(loss_cls_s=loss_cls_s)

        loss_mining_s, p_xk_x = forward_FDA(rois, target_s)

        loss_dict_s.update(loss_mining_s)
        loss_dict_s.update(loss_cls_s = loss_cls_s * p_xk_x)

        loss_align_s = forward_DCA(rois, all_layers=args.all_layer_adv, domain='source')
        loss_dict_s.update(loss_align_s)

        loss_s = sum(loss for loss in loss_dict_s.values())
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0, norm_type=2.0)

        opts.step()
        opts.zero_grad()

        # target domain
        rois = model.generator(data_t)
        output_t = model.classifier(rois, constant=args.bp_grl, adaption=True)
        score_unk = output_t.softmax(-1)[:, -1]
        loss_bp_t = criterion_bce(score_unk, torch.tensor([args.bp_th] * len(score_unk)).cuda())
        loss_dict_t.update(loss_bp_t=loss_bp_t)

        loss_align_t = forward_DCA(rois, all_layers=args.all_layer_adv, domain='target')
        loss_dict_t.update(loss_align_t)

        loss_t = sum(loss for loss in loss_dict_t.values())
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0, norm_type=2.0)

        opts.step()
        opts.zero_grad()

        if batch_idx % args.log_interval == 0:
            print_and_log(
                '[Epoch: {} {}/{} ({:.0f}%)] [Loss_s: {:.3f}, Loss_t: {:.3f}], [lr:{:.4f}] {}'.format(
                    epoch,
                    batch_idx * args.batch_size, len(train_dataset.target_image),
                    100. * batch_idx / len(train_loader),
                    loss_s.item(),
                    loss_t.item(),
                    opts.param_groups[0]['lr'],
                    {**process_dict(loss_dict_s), **process_dict(loss_dict_t)}
                )
            )

def warm_up_train(epoch):
    model.train()

    home_loader_iter = iter(train_loader)
    for batch_idx in range(len(train_loader)):

        data_s, target_s, data_t, target_t, _ = home_loader_iter.next()
        data_s, target_s = data_s.cuda(), target_s.long().cuda(non_blocking=True)
        data_t, target_t = data_t.cuda(), target_t.long().cuda(non_blocking=True)

        loss_dict_s = {}
        rois = model.generator(data_s)
        output_s = model.classifier(rois)

        loss_cls_s = criterion_ce(output_s, target_s)
        loss_dict_s.update(loss_cls_s=loss_cls_s)

        loss_s = sum(loss for loss in loss_dict_s.values())
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0, norm_type=2.0)
        opts.step()
        opts.zero_grad()

        rois = model.generator(data_t)
        output_t = model.classifier(rois, constant=args.bp_grl, adaption=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0, norm_type=2.0)
        opts.step()
        opts.zero_grad()

        if batch_idx % args.log_interval == 0:
            print_and_log(
                '[Warm Up: Epoch: {} {}/{} ({:.0f}%)] [Loss_s: {:.3f}], [lr:{:.4f}] {}'.format(
                    epoch,
                    batch_idx * args.batch_size, len(train_dataset.target_image),
                    100. * batch_idx / len(train_loader),
                    loss_s.item(),
                    opts.param_groups[0]['lr'],
                    {**process_dict(loss_dict_s)}
                )
            )

def test(epoch):
    model.eval()
    pred_y = []
    true_y = []
    correct = 0
    print('Epoch:{} inference'.format(epoch))
    with torch.no_grad():
        for batch_idx, (_, _, data, target, _) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda(non_blocking=True).long()
            rois = model.generator(data)
            output = model.classifier(rois).softmax(-1)
            pred = output.argmax(-1)
            for i in range(len(pred)):
                pred_y.append(pred[i].item())
                true_y.append(target[i].item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(len(pred_y), len(true_y))
    OS_star, unk = utils.cal_acc(true_y, pred_y, NUM_CLASSES, tf_writer, epoch, test_dataset.class_name)
    OS = (OS_star * (NUM_CLASSES - 1) + unk) / NUM_CLASSES
    HOS = 2 * unk * OS_star / (OS_star + unk)
    print('\nOS*: {}, unk: {}, OS: {}, HOS: {}\n'.format(OS_star, unk, OS, HOS))
    return OS, OS_star, unk, HOS

def initialization(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp_path = args.save_path + '/{}/'.format(args.exp_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    tf_writer = SummaryWriter(exp_path + 'tf_logs')

    # backup key files
    log_file = '{}/train_{}.log'.format(exp_path, args.exp_name)
    shutil.copyfile('./train.py', exp_path + '/train.py')
    shutil.copyfile('./models.py', exp_path + '/models.py')
    shutil.copyfile('./dataloader.py', exp_path + '/dataloader.py')
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logging.info(args)
    return tf_writer

if __name__ == '__main__':
    NUM_CLASSES = 26
    # Training settings
    parser = argparse.ArgumentParser(description='Openset-DA')
    parser.add_argument('--exp-name', help='dataset root')
    parser.add_argument('--data-root', default = './OfficeHome/',
                        help='dataset root')
    parser.add_argument('--partition', default='train',
                        help='train or test')
    parser.add_argument('--source', choices=['A', 'C', 'P', 'R', 'M'], default='A',
                        help='source domain')
    parser.add_argument('--target', choices=['A', 'C', 'P', 'R', 'K'], default='C',
                        help='target domain')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=75, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--num-works', default=4, type=int, help='num_works for dataloader')
    parser.add_argument('--known_class', type=float, default=25, metavar='TH', help='known_class')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--gpu', default='0', type=str, metavar='GPU', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--bp-th', type=float, default=0.5, metavar='TH', help='threshold (default: 0.5)')
    parser.add_argument('--bp-grl', type=float, default=0.5, metavar='TH', help='grl adversarial weight (default: 0.5)')
    parser.add_argument('--seed', default=42, type=int, help='save path')
    parser.add_argument('--warm_up_epoch', default=10, type=int, help='source-domain pretraining')

    parser.add_argument('--topk', default=3, type=int, help='select potential unk regions')
    parser.add_argument('--mining_th', default=1.0, type=float, metavar='TH', help='unk label')
    parser.add_argument('--mining_grl', type=float, default=0.2, metavar='TH', help='grad scaler (default: 0.2)')

    parser.add_argument('--all-layer-adv', action='store_true', default=False, help='align all layers')
    parser.add_argument('--adv-grl', type=float, default=0.1, metavar='TH', help='grl adversarial weight (default: 0.1)')
    parser.add_argument('--save-path', default='./rerun/officehome/', help='save path')
    parser.add_argument('--test', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    args = parser.parse_args()
    tf_writer = initialization(args)

    train_dataset = Home_Dataset(
        root=args.data_root,
        partition='train',
        source=args.source,
        target=args.target
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_works
    )
    test_dataset = Home_Dataset(
        root=args.data_root,
        partition='test',
        source=args.source,
        target=args.target)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_works
    )

    model = models.Net(args).cuda()
    opts = torch.optim.SGD(model.parameters(), args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=True)

    if args.test:
        print("=> loading checkpoint '{}'".format(args.test))
        state_dict = torch.load(args.test)
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}'".format(args.test))
        test(1)
        os._exit(0)

    logging.info('-'*20 + 'START TRAINING' + '-'*20)
    criterion_bce = nn.BCELoss()
    criterion_bce_red = nn.BCELoss(reduction='none')
    criterion_ce = nn.CrossEntropyLoss()

    HOS_bank = 0
    checkpoint_path = 'tmp'
    
    def format_results(x, p=1):
        return round(x *100, p)

    save_results_root = args.save_path + 'best_results/'

    if not os.path.exists(save_results_root):
        os.makedirs(save_results_root)

    txt_file_1 = save_results_root +'/{}'.format(args.source + '_' + args.target+ '.txt')
    txt_file_2 = save_results_root +'/latex_{}'.format(args.source + '_' + args.target+ '.txt')

    for epoch in range(1, args.epochs + 1):
        
        logging.info('-'*20 + 'EPOCH {}'.format(epoch)+ '-'*20 )
        if epoch < args.warm_up_epoch:
            # warm-up training with base-class images in the source domain
            warm_up_train(epoch)
        else:
            train(epoch)

        OS, OS_star, unk, HOS = test(epoch)
        res = [format_results(x) for x in [OS, OS_star, unk, HOS]]
        key = 'Epoch: {}'.format(epoch)
        value = 'Epoch: {}: OS: {}, OS*: {}, unk: {}, HOS: {} '.format(epoch, OS, OS_star, unk, HOS)
        logging.info(value)

        if HOS > HOS_bank:
            HOS_bank = HOS
            with open(txt_file_1, 'a') as f:
                f.write( 'Epoch: {}: OS: {}, OS*: {}, unk: {}, HOS: {} \n'.format(epoch, res[0], res[1], res[2], res[3]))
            with open(txt_file_2, 'a') as f:
                f.write( '{} & {} & {}  \n'.format(  res[1], res[2], res[3]))
            dirs = args.save_path + '/{}/models/'.format(args.exp_name)
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            torch.save(model.state_dict(), dirs + f'{args.exp_name}_{epoch}_OS*{OS_star}_unk{unk}_HOS{HOS}.pt')
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            checkpoint_path =  dirs + f'{args.exp_name}_{epoch}_OS*{OS_star}_unk{unk}_HOS{HOS}.pt'
