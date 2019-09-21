import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from torch.optim import lr_scheduler
import argparse
import uuid

import torchvision.transforms as tr
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from logger import Logger
import densenet
import higher

DATA_FOLDER = '/private/home/denisy/workspace/research/densenet_cifar10/cifar10'


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
    parser.add_argument('--meta_batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=300, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--meta_num_train_steps', default=1, type=int)
    parser.add_argument('--meta_num_test_steps', default=1, type=int)
    parser.add_argument('--meta_grad_clip', default=100, type=float)
    parser.add_argument('--split_ratio', default=0.05, type=float)

    args = parser.parse_args()
    return args


def make_transform(augument):
    means = [0.53129727, 0.5259391, 0.52069134]
    stdevs = [0.28938246, 0.28505746, 0.27971658]

    transforms = []
    if augument:
        transforms.append(tr.RandomCrop(32, padding=4))
        transforms.append(tr.RandomHorizontalFlip())

    transforms.append(tr.ToTensor())
    transforms.append(tr.Normalize(means, stdevs))

    return tr.Compose(transforms)


def make_loaders(batch_size, meta_batch_size, split_ratio=0.05):
    train_trans = make_transform(augument=True)
    valid_trans = make_transform(augument=False)
    test_trans = make_transform(augument=False)

    train_dataset = datasets.CIFAR10(
        root=DATA_FOLDER,
        train=True,
        download=True,
        transform=train_trans,
    )

    valid_dataset = datasets.CIFAR10(
        root=DATA_FOLDER,
        train=True,
        download=True,
        transform=valid_trans,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_FOLDER,
        train=False,
        download=True,
        transform=test_trans,
    )

    split_size = int(len(train_dataset) * (1 - split_ratio))
    

    idxs = np.random.permutation(len(train_dataset))
    train_idxs = idxs[:split_size]
    valid_idxs = idxs[split_size:]
    
    train_sampler = SubsetRandomSampler(train_idxs)
    valid_sampler = SubsetRandomSampler(valid_idxs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    meta_train_loader = DataLoader(
        valid_dataset,
        batch_size=meta_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    meta_test_loader = DataLoader(
        valid_dataset,
        batch_size=meta_batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader, meta_train_loader, meta_test_loader


class MetaTrainer(object):
    def __init__(
        self, model, init_lr, momentum, weight_decay, batch_size,
        meta_batch_size, meta_num_train_steps, meta_num_test_steps,
        meta_grad_clip, split_ratio, device
    ):
        super().__init__()

        self.model = model.to(device)

        self.meta_num_train_steps = meta_num_train_steps
        self.meta_num_test_steps = meta_num_test_steps
        self.meta_grad_clip = meta_grad_clip
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

        self.learnable_lr = higher.optim.get_trainable_opt_params(
            self.opt, device=self.device
        )['lr']
        self.lr_opt = optim.Adam(self.learnable_lr)

        self.train_loader, self.test_loader, self.meta_train_loader, self.meta_test_loader = make_loaders(
            batch_size, meta_batch_size, split_ratio=split_ratio
        )

        #self.train_loader, self.test_loader = make_loaders(batch_size)
        #self.meta_train_loader, self.meta_test_loader = make_meta_loaders(
        #    meta_batch_size
        #)

    def meta_train_iter(self, step, epoch, L):
        self.model.train()
        self.lr_opt.zero_grad()
        start_time = time.time()
        with higher.innerloop_ctx(
            self.model,
            self.opt,
            copy_initial_weights=True,
            track_higher_grads=True,
            device=self.device,
            override={'lr': self.learnable_lr}
        ) as (fmodel, diffopt):
            train_loss = 0
            for i, (train_x, train_y) in enumerate(self.meta_train_loader):
                if i >= self.meta_num_train_steps:
                    break
                train_x, train_y = train_x.to(self.device), train_y.to(
                    self.device
                )
                # meta train step
                train_y_hat = fmodel(train_x)
                loss = F.cross_entropy(train_y_hat, train_y)
                train_loss += loss
                diffopt.step(train_loss)

            test_loss = 0
            for i, (test_x, test_y) in enumerate(self.meta_test_loader):
                if i >= self.meta_num_test_steps:
                    break
                test_x, test_y = test_x.to(self.device
                                           ), test_y.to(self.device)
                # meta test step
                test_y_hat = fmodel(test_x)
                test_loss += F.cross_entropy(test_y_hat, test_y)

            if self.meta_num_test_steps > 0:
                test_loss.backward()

                L.log(
                    'meta/train_loss',
                    train_loss.item() / self.meta_num_train_steps, step
                )
                L.log(
                    'meta/test_loss',
                    test_loss.item() / self.meta_num_test_steps, step
                )

        torch.nn.utils.clip_grad_norm_(self.learnable_lr, self.meta_grad_clip)
        self.lr_opt.step()

        # set minimum lr
        for lr in self.learnable_lr:
            lr.data.clamp_min_(0.001)

        # set new learning rate for each param group
        higher.optim.apply_trainable_opt_params(
            self.opt, {'lr': self.learnable_lr}
        )
        lrs = np.array([lr.item() for lr in self.learnable_lr])
        L.log_histogram('meta/learning_rate', lrs, step)
        for i, lr in enumerate(lrs):
            L.log('meta/lr_%d' % i, lr, step)

        L.log('meta/duration', time.time() - start_time, step)
        L.log('meta/epoch', epoch, step)
        L.dump(step)

    def train_iter(self, step, epoch, L):
        self.model.train()
        start_time = time.time()
        for x, y in self.train_loader:
            step += 1

            x, y = x.to(self.device), y.to(self.device)

            self.opt.zero_grad()
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            self.opt.step()

            prediction = y_hat.max(1)[1]
            accuracy = prediction.eq(y).sum().item()

            L.log('train/loss', loss, step)
            L.log('train/accuracy', 100. * accuracy / x.shape[0], step)

            if step % 100 == 0:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/epoch', epoch, step)
                L.dump(step)
                start_time = time.time()

        L.log('train/duration', time.time() - start_time, step)
        L.log('train/epoch', epoch, step)
        L.dump(step)

        return step

    def test_iter(self, step, epoch, L):
        self.model.eval()
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y, reduction='sum')
            prediction = y_hat.max(1)[1]
            accuracy = prediction.eq(y).sum().item()

            L.log('test/loss', loss, step, n=x.shape[0])
            L.log('test/accuracy', 100. * accuracy, step, n=x.shape[0])

        L.log('test/epoch', epoch, step)
        L.dump(step)

    def train(self, num_epochs, L):
        step = 0
        for epoch in range(1, num_epochs + 1):
            step = self.train_iter(step, epoch, L)
            self.test_iter(step, epoch, L)
            self.meta_train_iter(step, epoch, L)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    work_dir = make_dir(args.work_dir)
    L = Logger(work_dir, use_tb=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = densenet.DenseNet(
        growth_rate=12,
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
        meta_num_train_steps=args.meta_num_train_steps,
        meta_num_test_steps=args.meta_num_test_steps,
        meta_grad_clip=args.meta_grad_clip,
        split_ratio=args.split_ratio,
        device=device
    )

    trainer.train(args.num_epochs, L)


if __name__ == '__main__':
    main()
