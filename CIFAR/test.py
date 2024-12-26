import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
import time
from PIL import Image as PILImage
import random
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score, auc, precision_recall_curve
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    from utils.additional_transform import AddGaussianNoise

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# outliers are considered as positive and should be of higher scores
def compute_fnr(out_scores, in_scores, fpr_cutoff=.05):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    fpr, fnr, thresholds = det_curve(y_true=y_true, y_score=y_score)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))

    fpr_at_fpr_cutoff = fpr[idx]
    fnr_at_fpr_cutoff = fnr[idx]

    if fpr_at_fpr_cutoff > 0.1:
        fnr_at_fpr_cutoff = 1.0

    return fnr_at_fpr_cutoff

def compute_auroc(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc

def compute_aupr(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)

    return aupr


# caculate differetial entropy of dirichlet distribution
def diff_entropy(alphas):
    alpha0 = torch.sum(alphas, dim=1)
    return torch.sum(
            torch.lgamma(alphas)-(alphas-1)*(torch.digamma(alphas)-torch.digamma(alpha0).unsqueeze(1)),
            dim=1) - torch.lgamma(alpha0)


def get_ood_scores(net, loader, in_dist=False):
    
    _score, _right_score, _wrong_score = [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # Random sample data from Semantic OOD datasets for efficiency
            # You can certainty use all samples for testing
            if batch_idx >= 10000 // args.test_bs and in_dist is False:
                break

            data = data.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.scoring_function == 'entropy':
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            elif args.scoring_function == 'energy':
                _score.append(to_np(-torch.logsumexp(output, dim=1)))
            elif args.scoring_function == 'maxlogits':
                _score.append(-np.max(to_np(output).astype(float), axis=1))
            elif args.scoring_function == 'diff_entropy':
                _score.append(to_np(diff_entropy(F.relu(output)+1)))
            else:
                # using softmax probability by default
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.scoring_function == 'entropy':
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                elif args.scoring_function == 'energy':
                    _right_score.append(to_np((- torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np(( - torch.logsumexp(output, dim=1)))[wrong_indices])
                elif args.scoring_function == 'maxlogits':
                    _right_score.append(-np.min(to_np(output).astype(float), axis=1)[right_indices])
                    _wrong_score.append(-np.max(to_np(output).astype(float), axis=1)[wrong_indices])
                elif args.scoring_function == 'diff_entropy':
                    _right_score.append(to_np(diff_entropy(F.relu(output)+1).float())[right_indices])
                    _wrong_score.append(to_np(diff_entropy(F.relu(output)+1).float())[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
                
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:10000].copy()

def get_and_print_results(net, ood_loader, ID_score):

    print('Test on {}, {} samples in total.'.format(os.path.basename(ood_loader.dataset.root), len(ood_loader.dataset)))
    aurocs, auprs, fprs = [], [], []
    out_score = get_ood_scores(net, ood_loader, in_dist=False)
    aurocs.append(compute_auroc(out_score, ID_score))
    auprs.append(compute_aupr(out_score, ID_score))
    fprs.append(compute_fnr(out_score, ID_score))
    
    print_measures(np.mean(aurocs), np.mean(auprs), np.mean(fprs))

    return aurocs, auprs, fprs


def get_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Setup
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--scoring_function', '--score', type=str, default='diff_entropy',
                        choices=['msp', 'energy', 'entropy', 'maxlogits', 'diff_entropy'], help='Choose architecture.')
    parser.add_argument('--method', '-m', type=str, choices=['Baseline', 'Entropy', 'Energy', 'DUL'], default='DUL', help='Method name.')
    # Loading details
    parser.add_argument('--epoch', type=int, default=19, help='the epochs of finetune stage')
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    parser.add_argument('--load', '-l', type=str, default='/apdcephfs/private_yangqyzhang/code/CIFAR/snapshots/', help='Checkpoint path to resume')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--num_classes', default=10, type=int, help='total number of layers')
    # Dataset
    parser.add_argument('--severity', default=5.0, type=float, help='noise severity')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--model', type=str, default='wrn',
                    choices=['wrn'], help='Choose architecture.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(), trn.Normalize(mean, std)])
    COV_transform = trn.Compose([AddGaussianNoise(amplitude=args.severity), trn.ToTensor(), trn.Normalize(mean, std)])
    normal_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    
    if args.dataset == 'cifar10':
        ID_train_data = dset.CIFAR10('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-10/', train=True, transform=train_transform)
        COV_data = dset.CIFAR10('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-10', train=False, transform=COV_transform)
        ID_test_data = dset.CIFAR10('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-10', train=False, transform=normal_transform)
        num_classes = 10
    else:
        ID_train_data = dset.CIFAR100('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-100/', train=True, transform=train_transform)
        COV_data = dset.CIFAR100('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-100/', train=False, transform=COV_transform)
        ID_test_data = dset.CIFAR100('/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/cifar-100', train=False, transform=normal_transform)
        num_classes = 100


    COV_loader = torch.utils.data.DataLoader(COV_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)
    ID_loader = torch.utils.data.DataLoader(ID_test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

    set_random_seed(args.seed)
    
    # Create model
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    
    # Restore
    path = os.path.join(args.load, args.dataset, args.method, "{}_{}_epoch_{}.pt".format(args.model, args.seed, args.epoch))
    if os.path.isfile(path):
        net.load_state_dict(torch.load(path))
        print('model loaded')
    else:
        print(args.load)
        assert False, "could not resume"

    net.eval()
    net.cuda()

    # cudnn.benchmark = True  # fire on all cylinders
    # randomly sample some OOD test data to reduce computation
    num_of_ood_test_samples = len(ID_test_data)

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    ID_score, ID_right_score, ID_wrong_score = get_ood_scores(net, ID_loader, in_dist=True)
    
    num_right = len(ID_right_score)
    num_wrong = len(ID_wrong_score)
    print('ID Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
    IDErrorRate = 100 * num_wrong / (num_wrong + num_right)
    
    COV_score, COV_right_score, COV_wrong_score = get_ood_scores(net, COV_loader, in_dist=True)

    num_right = len(COV_right_score)
    num_wrong = len(COV_wrong_score)
    print('Covariate-shifted Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
    CovariateErrorRate = 100 * num_wrong / (num_wrong + num_right)
    
    # /////////////// OOD Detection ///////////////
    auroc_list, aupr_list, fpr_list = [], [], []
    COV_auroc_list, COV_aupr_list, COV_fpr_list = [], [], [] 

    
    # /////////////// Textures ///////////////
    dtd = dset.ImageFolder(root="/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/OOD/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    svhn = svhn.SVHN(root='/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/OOD/SVHN', split="test",
                     transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]), download=False)

    place365 = dset.ImageFolder(root="/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/OOD/place365",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    lsun = dset.ImageFolder(root = "/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/OOD/LSUN",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    lsun_r = dset.ImageFolder(root = "/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/OOD/LSUN_resize/",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    isun = dset.ImageFolder(root="/apdcephfs_cq11/share_2934111/qingyangzhang/dataset/OOD/iSUN",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    
    for SEM_OOD_data in [dtd, svhn, place365, lsun, lsun_r, isun]:
        SEM_OOD_loader = torch.utils.data.DataLoader(SEM_OOD_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)
        auroc, aupr, fpr = get_and_print_results(net, SEM_OOD_loader, ID_score)
        auroc_list.extend(auroc)
        aupr_list.extend(aupr)
        fpr_list.extend(fpr)

    # /////////////// Summarize Results ///////////////

    print('\n\nMean Test Results')
    auroc = np.mean(auroc_list)
    aupr = np.mean(aupr_list)
    fpr = np.mean(fpr_list)
    recall_level = 0.95
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))