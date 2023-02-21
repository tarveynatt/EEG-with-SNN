from torchvision import datasets, models, transforms
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models import VGG
from transfer import fuse, transfer_snn, normalize_weight, transfer_cq
from plot import plot_loss
from time import ctime
import pickle

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def train(model, device, train_loader, optimizer, criterion, scheduler):
    model.train()
    loss_ = []
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())
    scheduler.step(sum(loss_)/len(loss_))
    return sum(loss_)/len(loss_)


def test(model, device, test_loader, criterion):
    model.eval()
    loss_ = []
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # onehot = nn.functional.one_hot(target, 10)
            output = model(input)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_.append(loss.item())
    acc = 100. * correct / len(test_loader.dataset)
    return sum(loss_)/len(loss_), acc


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch CIFAR100 VGG16')
    parser.add_argument('--cpu', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=2023, help='set random seed for training')
    parser.add_argument('--batch-size', type=int, default=256, help='set batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=512, help='set batch size for testing')
    parser.add_argument('--lr', type=float, default=1e-4, help='set learning rate for training')
    parser.add_argument('--epoch', type=int, default=128, help='set number of epochs for training')
    parser.add_argument('--category', type=int, default=10, help='set number of categories for classification')
    parser.add_argument('--time-window', type=int, default=10, help='set time window for SNN')
    parser.add_argument('--resume', type=str, default='./models/CNN_CQ.pkl', help='resume model from check point')
    parser.add_argument('--vgg', type=int, default=16, help='set the sturcture of VGG model')
    parser.add_argument('--stage', type=str, default='CNN', help='set the stage of training')

    # args = parser.parse_args()
    
    args = parser.parse_known_args()[0]

    return args


def make_data(args):
    train_trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    test_trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR10(root='./datasets/CIFAR/', train=True, download=True, transform=train_trans)
    cifar_test = datasets.CIFAR10(root='./datasets/CIFAR/', train=False, download=True, transform=test_trans)
    
    train_loader = DataLoader(cifar_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader

def save_model(args, current, device, epoch, loss, state_dict, acc):
    seed = args.seed
    test_batch_size = args.test_batch_size
    train_batch_size = args.train_batch_size
    lr = args.lr
    
    state = {'time': current,
             'seed': seed,
             'device': device,
             'learning rate': lr,
             'train batch size': train_batch_size,
             'test batch size': test_batch_size,
             'model structure': f'VGG{args.vgg}',
             'model type': args.stage,
             'epoch': epoch,
             'loss': loss,
             'accuracy': acc,
             'model state dict': state_dict}
    
    current = current.replace(' ', '-').replace(':', '-')
    torch.save(state, f'./models/VGG{args.vgg}_{args.stage}_{current}.mdl')

def main():
    # initialization
    args = parse_args()
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(device)
    torch.manual_seed(args.seed)


    # prepare datasets
    train_loader, test_loader = make_data(args)


    # step 1: train a normal CNN
    t_state_dict = {}
    t_loss = 1e3
    t_epoch = 0
    t_acc = 0

    CNN = VGG(vgg='VGG16', category=10)
    CNN.to(device)

    optimizer = Adam(CNN.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    criterion = nn.CrossEntropyLoss()

    loss_train = []
    loss_test = []
    for epoch in tqdm(range(1, args.epoch+1)):
        loss = train(CNN, device, train_loader, optimizer, criterion, scheduler)
        loss_train.append(loss)
        loss, acc = test(CNN, device, test_loader, criterion)
        loss_test.append(loss)

        if loss < t_loss:
            t_state_dict = CNN.state_dict()
            t_loss = loss
            t_epoch = epoch
            t_acc = acc


    current = ctime()
    
    plot_loss(loss_train, 'CNN Train Loss', current)
    plot_loss(loss_test, 'CNN Test Loss', current)

    save_model(args, current, device, t_epoch, t_loss, t_state_dict, t_acc)

    # # step 2: train a CNN with ReLU replaced to Clamp and Quantize
    # temp_loss = 1e3

    # CNN = VGG(vgg='VGG16', category=10)
    # CNN.load_state_dict(torch.load('./models/CNN.mdl'))
    # CNN.to(device)
    
    # CNN_CQ = VGG(cq=True, category=10)
    # transfer_cq(CNN, CNN_CQ)
    # CNN_CQ.to(device)

    # optimizer = Adam(CNN_CQ.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    # criterion = nn.CrossEntropyLoss()

    # loss_train = []
    # loss_test = []
    # for epoch in tqdm(range(1, args.epoch+1)):
    #     loss = train(CNN_CQ, device, train_loader, optimizer, criterion, scheduler)
    #     loss_train.append(loss)
    #     loss = test(CNN_CQ, device, test_loader, criterion)
    #     loss_test.append(loss)

    #     if loss < temp_loss:
    #         temp_state_dict = CNN_CQ.state_dict()

    # plot_loss(loss_train, 'CNN_CQ Train Loss')
    # plot_loss(loss_test, 'CNN_CQ Test Loss')
    # torch.save(temp_state_dict, './models/CNN_CQ.mdl')

    # CNN_CQ = VGG(cq=True, category=10)
    # CNN_CQ.load_state_dict(torch.load('./models/CNN_CQ.mdl'))
    # CNN_CQ.to(device)
    # # step 3: transfer weight to SNN with weight and bias normalization
    # SNN = VGG(vgg='VGG16', spike=True, category=10)
    # SNN.to(device)
    # # CNN_CQ.load_state_dict(torch.load('./models/CNN_CQ.mdl'))
    # # CNN_CQ = fuse(CNN_CQ)
    # transfer_snn(CNN_CQ, SNN)
    # torch.save(SNN.state_dict(), './models/SNN.mdl')

    # criterion = nn.CrossEntropyLoss()

    # # with torch.no_grad():
    # #     normalize_weight(SNN.features)
    # loss = test(SNN, device, test_loader, criterion)
    # print(loss)
    # loss = test(CNN_CQ, device, test_loader, criterion)
    # print(loss)
    # plot_loss([loss], 'SNN Test Loss')


if __name__ == '__main__':
    main()