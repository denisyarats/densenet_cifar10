import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from torch.optim import lr_scheduler
import argparse

import torchvision.transforms as tr
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
    parser.add_argument('--meta_batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=300, type=int)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--meta_update_freq', default=1, type=int)
    parser.add_argument('--meta_num_updates', default=1, type=int)
    parser.add_argument('--meta_num_inner_steps', default=1, type=int)

    args = parser.parse_args()
    return args


def make_transform(train=True):
    means = [0.53129727, 0.5259391, 0.52069134]
    stdevs = [0.28938246, 0.28505746, 0.27971658]

    transforms = []
    if train:
        transforms.append(tr.RandomCrop(32, padding=4))
        transforms.append(tr.RandomHorizontalFlip())

    transforms.append(tr.ToTensor())
    transforms.append(tr.Normalize(means, stdevs))

    return tr.Compose(transforms)


def make_dataset(train=True):
    transform = make_transform(train)

    dataset = datasets.CIFAR10(
        root='cifar10', train=train, download=True, transform=transform
    )
    return dataset


def make_loaders(batch_size=64):
    train_dset = make_dataset(train=True)
    test_dset = make_dataset(train=False)

    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        test_dset, batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, test_loader


def make_meta_loaders(batch_size=64, train_ratio=0.8):
    dataset = make_dataset(train=True)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dset, test_dset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        test_dset, batch_size=batch_size, shuffle=True, **kwargs
    )

    return train_loader, test_loader



class MetaTrainer(object):
    def __init__(
        self, model, init_lr, momentum, weight_decay, batch_size,
        meta_batch_size, meta_update_freq, meta_num_updates,
        meta_num_inner_steps, device
    ):
        super().__init__()

        self.model = model.to(device)
        
        self.meta_update_freq = meta_update_freq
        self.meta_num_updates = meta_num_updates
        self.meta_num_inner_steps = meta_num_inner_steps
        self.device = device

        param_groups = [
            {
                'params': p,
                'lr': init_lr
            } for p in self.model.parameters()
        ]
        self.opt = optim.SGD(
            param_groups,
            lr=init_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.learnable_lr =  higher.optim.get_trainable_opt_params(self.opt, device=self.device)['lr']
        self.lr_opt = optim.Adam(self.learnable_lr)

        self.train_loader, self.test_loader = make_loaders(batch_size)
        self.meta_train_loader, self.meta_test_loader = make_meta_loaders(
            meta_batch_size
        )

    def meta_train_iter(self, step, epoch, L):
        self.lr_opt.zero_grad()
        with higher.innerloop_ctx(
            self.model,
            self.opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            override={'lr': self.learnable_lr}
        ) as (fmodel, diffopt):
            for batch_idx, (x, y) in enumerate(self.meta_train_loader):
                if batch_idx >= self.meta_num_inner_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                y_hat = fmodel(x)
                loss = F.nll_loss(y_hat, y)
                diffopt.step(loss)

            test_loss = 0
            for batch_idx, (x, y) in enumerate(self.meta_test_loader):
                if batch_idx >= self.meta_num_inner_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                y_hat = fmodel(x)
                test_loss += F.nll_loss(y_hat, y)
            
            grads = torch.autograd.grad(test_loss, self.learnable_lr)
            # need to manually set grad
            for param, grad in zip(self.learnable_lr, grads):
                param.grad = grad
        self.lr_opt.step()

        # set new learning rate for each param group
        for i, (group, lr) in enumerate(
            zip(self.opt.param_groups, self.learnable_lr)
        ):
            group['lr'] = lr.item()
            L.log('train/learning_rate_%d' % i, lr.item())

    def train_iter(self, step, epoch, L):
        self.model.train()
        start_time = time.time()
        for x, y in self.train_loader:
            step += 1

            if step % self.meta_update_freq == 0:
                for _ in range(self.meta_num_updates):
                    self.meta_train_iter(step, epoch, L)

            x, y = x.to(self.device), y.to(self.device)

            self.opt.zero_grad()
            y_hat = self.model(x)
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            self.opt.step()

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

    def test_iter(self, step, epoch, L):
        self.model.eval()
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
            loss = F.nll_loss(y_hat, y, reduction='sum')
            prediction = y_hat.max(1)[1]
            accuracy = prediction.eq(y).sum().item()

            L.log('test/loss', loss, n=x.shape[0])
            L.log('test/accuracy', 100. * accuracy, n=x.shape[0])

        L.log('test/epoch', epoch)
        L.dump(step)

    def train(self, num_epochs, L):
        step = 0
        for epoch in range(1, num_epochs + 1):
            self.train_iter(step, epoch, L)
            self.test_iter(step, epoch, L)


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
        growth_rate=16,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        num_classes=10
    )
    model = model.to(device)

    trainer = MetaTrainer(
        model=model,
        init_lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        meta_batch_size=args.meta_batch_size,
        meta_update_freq=args.meta_update_freq,
        meta_num_updates=args.meta_num_updates,
        meta_num_inner_steps=args.meta_num_inner_steps,
        device=device
    )

    trainer.train(args.num_epochs, L)


if __name__ == '__main__':
    main()
