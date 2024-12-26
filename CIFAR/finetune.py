# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet
import torch.distributions as dist
import random
import copy

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.imagenet_rc_loader import ImageNet
    from utils.tin597_loader import TIN597
    from utils.validation_dataset import validation_split
    from utils.criterion import Entropy_Loss, Energy_Loss, DUL_Loss

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(net, original_net, train_loader_in, train_loader_out, optimizer, criterion, state, args):
    net.train()  # enter train mode
    original_net.eval()
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in tqdm(zip(train_loader_in, train_loader_out), total=len(train_loader_in)):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        output = net(data)
        ID_output = output[:len(in_set[0])]
        OOD_output = output[len(in_set[0]):]
        if args.method == 'DUL':
            original_output = original_net(data)
            original_ID_output = original_output[:len(in_set[0])]
            original_OOD_output = original_output[len(in_set[0]):]

        # backward
        optimizer.zero_grad()
        if args.method == 'DUL':
            loss = criterion(ID_output, OOD_output, target, original_ID_output, original_OOD_output)
        else:
            loss = criterion(ID_output, OOD_output, target)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg


# test function
def test(test_loader, net, state):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

def get_args():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OOD detection methods.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
    parser.add_argument('--data_root', type=str, default='/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/', help='Folder to load datasets.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose ID dataset between CIFAR-10, CIFAR-100.')
    parser.add_argument('--aux', type=str, default='imagenet', choices=['imagenet', 'TIN597'],
                    help='Choose auxilary OOD dataset between imanaget and TIN597.')
    # Backbone
    parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['wrn'], help='Choose architecture.')
    # Finetune methods
    parser.add_argument('--method', type=str, default='DUL',
                    choices=['DUL', 'Energy', 'Entropy', 'WOODS', 'SCONE'], help='Choose architecture.')
    # DUL specific
    parser.add_argument('--dul_m_in', type=float, default=-430., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--dul_m_out', type=float, default=-370., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--gamma', type=float, default=2, help='DUL regularization strength')
    parser.add_argument('--tau', type=float, default=2, help='DUL norm')
    # Energy specific
    parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
    # WOODS specific
    parser.add_argument('--in_constraint_weight', type=float, default=1,
                    help='weight for in-distribution penalty in loss function')
    parser.add_argument('--out_constraint_weight', type=float, default=1,
                    help='weight for out-of-distribution penalty in loss function')
    parser.add_argument('--ce_constraint_weight', type=float, default=1,
                    help='weight for classification penalty in loss function')
    parser.add_argument('--false_alarm_cutoff', type=float,
                    default=0.05, help='false alarm cutoff')
    parser.add_argument('--lr_lam', type=float, default=1, help='learning rate for the updating lam (SSND_alm)')
    parser.add_argument('--ce_tol', type=float,
                    default=2, help='tolerance for the loss constraint')
    parser.add_argument('--penalty_mult', type=float,
                    default=1.5, help='multiplicative factor for penalty method')
    parser.add_argument('--constraint_tol', type=float,
                    default=0, help='tolerance for considering constraint violated')
    parser.add_argument('--alpha', type=float, default=0.05, help='number of labeled samples')
    # OOD regularization strength, this hyperparameter shared by all the methods
    parser.add_argument('--lamb', default=0.1, type=float, help='weight of OOD detection loss')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='The initial learning rate.')
    parser.add_argument('--ID_batch_size', type=int, default=128, help='Batch size for ID.')
    parser.add_argument('--OOD_batch_size', type=int, default=256, help='Batch size for auxiliary OOD.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='/apdcephfs/private_yangqyzhang/code/CIFAR/snapshots/baseline/', help='Checkpoint path to resume')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=16, help='Pre-fetching threads.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    if args.method in ['DUL', 'Energy']:
        args.lamb = 0.05
    elif args.method == 'Entropy':
        args.lamb = 0.5

    directory = "{save}/{dataset}_{seed}".format(save=args.save, dataset=args.dataset, seed=args.seed)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_state_file = os.path.join(directory, 'train_args.txt')
    fw = open(save_state_file, 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print(state, file=fw)
    fw.close()

    set_random_seed(args.seed)

    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data_in = dset.CIFAR10('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-10/', train=True, transform=train_transform)
        test_data = dset.CIFAR10('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-10', train=False, transform=test_transform)
        num_classes = 10
    else:
        train_data_in = dset.CIFAR100('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-100/', train=True, transform=train_transform)
        test_data = dset.CIFAR100('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-100/', train=False, transform=test_transform)
        num_classes = 100

    if args.aux == 'imagenet':
        ood_data = ImageNet(transform=trn.Compose(
            [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
            trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))
        # Do not shuffle ImageNet-RC to keep locality for efficient tuning
        shuffle = False
    elif args.aux == 'TIN597':
        ood_data = TIN597(root = '/dataset/tin597/test',
            transform=trn.Compose(
            [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
            trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))
        shuffle = True

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.ID_batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)

    train_loader_out = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.OOD_batch_size, shuffle=shuffle,
        num_workers=args.prefetch, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.ID_batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)


    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    original_net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

    
    # load model
    pretrain_model_path = os.path.join(args.load, '{}_wrn_baseline_epoch_199.pt'.format(args.dataset))
    if os.path.isfile(pretrain_model_path):
        net.load_state_dict(torch.load(pretrain_model_path))
        original_net.load_state_dict(torch.load(pretrain_model_path))
        print('Model loaded.')
    else:
        print(pretrain_model_path)
        assert False, "Could not resume."

    if args.ngpu > 0:
        net.cuda()
        original_net.cuda()

    cudnn.benchmark = True  # fire on all cylinders

    if args.method in ['Entropy', 'Energy', 'DUL']:
        optimizer = torch.optim.SGD(
            net.parameters(), state['learning_rate'], momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        
    elif args.method in ['SCONE','WOODS']:
        # use logistic regression in optimization for SCONE and WOODS
        logistic_regression = nn.Linear(1, 1)
        logistic_regression.cuda()
        
        optimizer = torch.optim.SGD(
            list(net.parameters()) + list(logistic_regression.parameters()),
            state['learning_rate'], momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        
        # make in_constraint a global variable
        in_constraint_weight = args.in_constraint_weight

        # make loss_ce_constraint a global variable
        ce_constraint_weight = args.ce_constraint_weight

        # create the lagrangian variable for lagrangian methods
        lam, lam2 = torch.tensor(0).float(), torch.tensor(0).float()
        lam, lam2 = lam.cuda(), lam2.cuda()


    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))
    
    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    with open(os.path.join(args.save, args.dataset + args.model + str(args.seed) +
                                  '_{}_training_results.csv'.format(args.method)), 'w') as f:
        f.write('epoch,time(s),train_loss, test_loss, test_error(%)\n')

    if args.method == 'Entropy':
        criterion = Entropy_Loss(lamb=args.lamb)
    elif args.method == 'Energy':
        criterion = Energy_Loss(lamb=args.lamb, m_in=args.m_in, m_out=args.m_out)
    elif args.method == 'DUL':
        criterion = DUL_Loss(lamb=args.lamb, gamma = args.gamma, dul_m_in=args.dul_m_in, dul_m_out=args.dul_m_out, tau=args.tau)
        
    # /////////////// Training ///////////////
    print('Beginning Training\n')

    # Main loop
    for epoch in range(0, args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()

        train(net, original_net, train_loader_in, train_loader_out, optimizer, criterion, state, args)
        test(test_loader, net, state)

        # Save model
        if not os.path.exists(os.path.join(args.save, args.dataset, args.method)):
            os.makedirs(os.path.join(args.save, args.dataset, args.method))
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset, args.method, args.model + '_' + str(args.seed) +
                            '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset, args.method, args.model + '_' + str(args.seed) +
                                 '_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)
        
        # Show results

        with open(os.path.join(args.save, args.dataset, args.method, args.model + '_' + str(args.seed) +
                                          '_training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
            ))

        # # print state with rounded decimals
        # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'])
        )
