import numpy as np
import time
import torch
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


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


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
            root='cifar10', train=True, download=False, transform=train_trans
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = DataLoader(
        datasets.CIFAR10(
            root='cifar10', train=False, download=False, transform=test_trans
        ),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, test_loader


def train(step, epoch, model, data_loader, opt, scheduler, device, L):
    model.train()
    start_time = time.time()
    for x, y in data_loader:
        step += 1
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_hat = model(x)
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        opt.step()

        prediction = y_hat.max(1)[1]
        accuracy = prediction.eq(y).sum().item()

        L.log('train/loss', loss, step)
        L.log('train/accuracy', 100. * accuracy / x.shape[0], step)

        if step % 100 == 0:
            L.log_histogram('train/learning_rate', np.array(scheduler.get_lr()), step)
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/epoch', epoch, step)
            L.dump(step)
            start_time = time.time()

    L.log('train/duration', time.time() - start_time, step)
    L.log('train/epoch', epoch, step)
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

        L.log('test/loss', loss, step, n=x.shape[0])
        L.log('test/accuracy', 100. * accuracy, step, n=x.shape[0])

    L.log('test/epoch', epoch, step)
    L.dump(step)


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
        growth_rate=32,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        num_classes=10
    )
    model = model.to(device)


    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.MultiStepLR(
        opt, milestones=[150, 225], gamma=0.1
    )

    train_loader, test_loader = make_dataloaders(batch_size=args.batch_size)

    step = 0
    for epoch in range(1, args.num_epochs + 1):
        step = train(
            step, epoch, model, train_loader, opt, scheduler, device, L
        )
        test(step, epoch, model, test_loader, device, L)
        scheduler.step()


if __name__ == '__main__':
    main()
