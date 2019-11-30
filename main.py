
from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lista_dataset import LISTADataset
from lista_model import LISTA

''' Tensorboard 画图 '''
import os
try:
    import tensorboardX
    summary_writer = tensorboardX.SummaryWriter(log_dir='./exp/tf_logs')
except:
    summary_writer = None

criterion = torch.nn.MSELoss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (y_, x_) in enumerate(train_loader):
        y_, x_ = y_.to(device), x_.to(device)
        optimizer.zero_grad()

        xhs_ = model(y_)

        loss = 0
        for t in range(model._T):
            loss += criterion(xhs_[t+1], x_)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(y_), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            if summary_writer:
                global_step = epoch * len(train_loader) + batch_idx
                summary_writer.add_scalar('train_loss', loss.item(), global_step=global_step)

def test(args, model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    test_denom = 0
    with torch.no_grad():
        for y_, x_ in test_loader:
            y_, x_ = y_.to(device), x_.to(device)
            # 算二范数
            denom = torch.norm(x_)
            test_denom += denom.item()
            xhs_ = model(y_)
            # for t in range(model._T):
            #     loss = criterion(xhs_[t + 1], x_)
            #     test_loss += loss.item()
            loss = criterion(xhs_[model._T], x_)
            test_loss += loss.item()

    test_dB = 10 * math.log10(test_loss / test_denom)
    if summary_writer:
        summary_writer.add_scalar('test_loss', test_dB,global_step=epoch)

    print('\nTest set: test_dB: {:.4f}\n'.format(
        test_dB))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")


    # dataset
    train_datasets = LISTADataset(train=True)
    test_datasets = LISTADataset(train=False)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=args.batch_size, shuffle=True, **kwargs)

    # model
    A = train_datasets.A
    T = 16
    lam = 0.4
    model = LISTA(A, T, lam)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader,epoch)

    if args.save_model:
        torch.save(model.state_dict(), "lista.pt")
    summary_writer.close()


if __name__ == '__main__':
    main()