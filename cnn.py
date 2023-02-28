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
from plot import plot_graph
from time import ctime
from spikingjelly.clock_driven import ann2snn
import numpy as np

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(model, device, train_loader, optimizer, criterion, scheduler):
    model.train().to(device)
    loss_ = []
    correct = 0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).float().sum().item()
        loss_.append(loss.item())

    scheduler.step(sum(loss_)/len(loss_))

    acc = 100. * correct / len(train_loader.dataset)
    loss = sum(loss_)/len(loss_)
    return loss, acc


def test(model, device, test_loader, criterion):
    model.eval().to(device)
    loss_ = []
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # onehot = nn.functional.one_hot(target, 10)
            output = model(input)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).float().sum().item()
            loss_.append(loss.item())

    acc = 100. * correct / len(test_loader.dataset)
    loss = sum(loss_) / len(loss_)
    return loss, acc


def test_snn(model, device, test_loader, T):
    model.eval().to(device)
    total = 0.
    # correct = 0
    corrects = np.zeros(T)
    # losses = [0.] * T

    with torch.no_grad():
        for input, target in tqdm(test_loader):
            input, target = input.to(device), target.to(device)
            for m in model.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            for t in range(T):
                if t == 0:
                    output = model(input)
                else:
                    output += model(input)
                # loss = criterion(output, target)
                # losses[t] += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                corrects[t] += pred.eq(target.view_as(pred)).float().sum().item()
            # correct += pred.eq(target.view_as(pred)).float().sum().item()
            total += output.shape[0]
    return corrects / total


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch CIFAR100 VGG16')
    parser.add_argument('--cpu', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=2023, help='set random seed for training')
    parser.add_argument('--batch-size', type=int, default=128, help='set batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=500, help='set batch size for testing')
    parser.add_argument('--lr', type=float, default=1e-4, help='set learning rate for training')
    parser.add_argument('--epoch', type=int, default=200, help='set number of epochs for training')
    parser.add_argument('--category', type=int, default=10, help='set number of categories for classification')
    parser.add_argument('--T', type=int, default=500, help='set inference steps for SNN')
    parser.add_argument('--resume', type=str, default=None, help='resume model from check point')
    parser.add_argument('--vgg', type=int, default=16, help='set the sturcture of VGG model')
    parser.add_argument('--stage', type=str, default='SNN', help='set the stage of training')

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
    test_loader = DataLoader(cifar_test, batch_size=args.test_batch_size, shuffle=True)

    return train_loader, test_loader


def save_model(args, current, device, epoch, loss, state_dict, acc):
    seed = args.seed
    test_batch_size = args.test_batch_size
    train_batch_size = args.batch_size
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
    
    
    current = current[4:-5].replace(' ', '_').replace(':', '-')
    torch.save(state, f'./models/VGGCQ{args.vgg}_{args.stage}_{round(acc, 2)}_{current}.mdl')
    # torch.save(state, f'./models/VGG{args.vgg}_{args.stage}_{round(acc[-1], 2)}_{current}.mdl')


def main():
    # initialization
    args = parse_args()
    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # prepare datasets
    train_loader, test_loader = make_data(args)


    # step 1: train a normal CNN
    t_state_dict = {}
    t_loss = 0
    t_epoch = 0
    t_acc = 0

    CNN = VGG(vgg='VGG16', category=10)
    CNN.to(device)

    optimizer = Adam(CNN.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    criterion = nn.CrossEntropyLoss()

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    for epoch in tqdm(range(args.epoch)):
        loss, acc = train(CNN, device, train_loader, optimizer, criterion, scheduler)
        loss_train.append(loss)
        acc_train.append(acc)

        loss, acc = test(CNN, device, test_loader, criterion)
        loss_test.append(loss)
        acc_test.append(acc)

        if acc > t_acc:
            t_acc = acc
            t_state_dict = CNN.state_dict()
            t_loss = loss
            t_epoch = epoch


    current = ctime()
    
    plot_graph(loss_train, 'CNN Train Loss', current)
    plot_graph(loss_test, 'CNN Test Loss', current)
    plot_graph(acc_train, 'CNN Train Accuracy', current)
    plot_graph(acc_test, 'CNN Test Accuracy', current)

    save_model(args, current, device, t_epoch, t_loss, t_state_dict, t_acc)


    # # step 2: convert to snn
    # CNN = VGG('VGG16', category=10)
    # CNN.to(device)
    # state_dict = torch.load('./models/VGG16_CNN_86.79_Thu-Feb-23-21-16-51-2023.mdl')
    # # print(state_dict['accuracy'], state_dict['epoch'])
    
    # CNN.load_state_dict(state_dict['model state dict'])
    # criterion = nn.CrossEntropyLoss()

    # loss, acc = test(CNN, device, test_loader, criterion)
    # print('CNN accuracy', acc)

    # converter = ann2snn.Converter(mode='99.9%', dataloader=train_loader)
    # SNN = converter(CNN)
    
    # acc = test_snn(SNN, device, test_loader, args.T)
    # print('SNN Accuracy', acc[-1])
    
    # current = ctime()

    # plot_graph(acc, 'SNN Accuracy', current)
    # # plot_loss(loss, 'SNN loss', current)

    # save_model(args, current, device, 0, 0, SNN.state_dict(), acc)




    # step 2: train a CNN with ReLU replaced to Clamp and Quantize
    t_state_dict = {}
    t_loss = 0
    t_epoch = 0
    t_acc = 0

    CNN = VGG(vgg='VGG16', category=10)
    CNN.load_state_dict(torch.load('./models/VGG16_SNN_86.41_Feb_28_18-31-21.mdl'))
    CNN.to(device)

    CNN_CQ = VGG(cq=True, category=10)
    transfer_cq(CNN, CNN_CQ)
    CNN_CQ.to(device)

    optimizer = Adam(CNN.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    criterion = nn.CrossEntropyLoss()

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    for epoch in tqdm(range(args.epoch)):
        loss, acc = train(CNN, device, train_loader, optimizer, criterion, scheduler)
        loss_train.append(loss)
        acc_train.append(acc)

        loss, acc = test(CNN, device, test_loader, criterion)
        loss_test.append(loss)
        acc_test.append(acc)

        if acc > t_acc:
            t_acc = acc
            t_state_dict = CNN.state_dict()
            t_loss = loss
            t_epoch = epoch


    current = ctime()
    
    plot_graph(loss_train, 'CNNCQ Train Loss', current)
    plot_graph(loss_test, 'CNNCQ Test Loss', current)
    plot_graph(acc_train, 'CNNCQ Train Accuracy', current)
    plot_graph(acc_test, 'CNNCQ Test Accuracy', current)

    save_model(args, current, device, t_epoch, t_loss, t_state_dict, t_acc)


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