import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from torch.optim import lr_scheduler
import argparse

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from logger import Logger
import densenet
import higher


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=300, type=int)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--lr_update_frequency', default=100, type=int)
    parser.add_argument('--num_lr_updates', default=1, type=int)
    parser.add_argument('--num_lr_inner_steps', default=5, type=int)
    

    args = parser.parse_args()
    return args


def make_dataloaders(batch_size=64, num_workers=4, pin_memory=True):
    means = [0.53129727, 0.5259391, 0.52069134]
    stdevs = [0.28938246, 0.28505746, 0.27971658]
    norm_trans = transforms.Normalize(means, stdevs)

    train_trans = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm_trans
        ]
    )
    test_trans = transforms.Compose([transforms.ToTensor(), norm_trans])

    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    train_loader = DataLoader(
        datasets.CIFAR10(
            root='cifar10', train=True, download=True, transform=train_trans
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    train_lr_loader = DataLoader(
        datasets.CIFAR10(
            root='cifar10', train=True, download=True, transform=train_trans
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = DataLoader(
        datasets.CIFAR10(
            root='cifar10', train=False, download=True, transform=test_trans
        ),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, train_lr_loader, test_loader


def train_lr(model, lr, opt, lr_opt, data_loader, num_steps, device):
    lr_opt.zero_grad()
    with higher.innerloop_ctx(
        model,
        opt,
        copy_initial_weights=False,
        track_higher_grads=True,
    ) as (fmodel, diffopt):
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= num_steps:
                break
            x, y = x.to(device), y.to(device)
            y_hat = fmodel(x)
            loss = F.nll_loss(y_hat, y)
            diffopt.step(loss, override={'lr': lr})
            
        param_sum = sum(p.sum() for p in fmodel.parameters())
        grad = torch.autograd.grad(param_sum, lr)
        # need to manually set grad 
        for (g, pg) in zip(grad, lr):
            pg.grad = g
    lr_opt.step()


def train(
    step, epoch, model, lr, data_loader, lr_data_loader, opt, lr_opt,
    device, lr_update_frequency, num_lr_updates, num_lr_inner_steps, L
):
    model.train()
    start_time = time.time()
    for x, y in data_loader:
        step += 1

        if step % lr_update_frequency == 0:
            for _ in range(num_lr_updates):
                train_lr(
                    model, lr, opt, lr_opt, lr_data_loader,
                    num_lr_inner_steps, device
                )
            # set new learning rate for each param group
            for lr_idx, (pg, g_lr) in enumerate(zip(opt.param_groups, lr)):
                pg['lr'] = g_lr.item()
                L.log('train/learning_rate_%d' % lr_idx, g_lr.item())
                

        x, y = x.to(device), y.to(device)
        
        opt.zero_grad()
        y_hat = model(x)
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        opt.step()

        prediction = y_hat.max(1)[1]
        accuracy = prediction.eq(y).sum().item()

        L.log('train/loss', loss)
        L.log('train/accuracy', 100. * accuracy / x.shape[0])

        if step % 100 == 0:
            #L.log('train/learning_rate', scheduler.get_lr()[0])
            L.log('train/duration', time.time() - start_time)
            L.log('train/epoch', epoch)
            L.dump(step)
            start_time = time.time()

    L.log('train/duration', time.time() - start_time)
    L.log('train/epoch', epoch)
    L.dump(step)

    return step


def test(step, epoch, model, data_loader, device, L):
    model.eval()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model(x)
        loss = F.nll_loss(y_hat, y, reduction='sum')
        prediction = y_hat.max(1)[1]
        accuracy = prediction.eq(y).sum().item()

        L.log('test/loss', loss, n=x.shape[0])
        L.log('test/accuracy', 100. * accuracy, n=x.shape[0])

    L.log('test/epoch', epoch)
    L.dump(step)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    work_dir = make_dir(args.work_dir)
    L = Logger(work_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = densenet.DenseNet(
        growth_rate=32,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        num_classes=10
    )
    model = model.to(device)

    param_groups = [{'params': p, 'lr': args.lr} for p in model.parameters()]
    opt = optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    lr = nn.ParameterList([nn.Parameter(x.to(device)) for x in higher.optim.get_trainable_opt_params(opt)['lr']])

    train_loader, train_lr_loader, test_loader = make_dataloaders(
        batch_size=args.batch_size
    )

    lr_opt = optim.Adam(lr)

    step = 0
    for epoch in range(1, args.num_epochs + 1):
        step = train(
            step, epoch, model, lr, train_loader, train_lr_loader, opt, lr_opt,
            device, args.lr_update_frequency, args.num_lr_updates, args.num_lr_inner_steps, L
        )
        test(step, epoch, model, test_loader, device, L)
        scheduler.step()


if __name__ == '__main__':
    main()
