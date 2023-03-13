from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np

target = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
          [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
          [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]

PATH = './eeg_feature_smooth/'


class EEG(Dataset):
    def __init__(self, split=32, train=True):
        super().__init__()

        self.train = train

        with open(f'{PATH}de_LDS.pkl', 'rb') as f:
            raw_data = pickle.load(f)

        raw_target = raw_data['labels']

        self.data = []
        self.target = []
        for i in range(72):
            l = raw_data[str(i)].shape[1]
            # split train / test dataset with 0.7
            l = round(l * 0.7)
            if train:
                temp_data = raw_data[str(i)][:, :l, :]
            else:
                temp_data = raw_data[str(i)][:, l:, :]
            
            l = temp_data.shape[1]
            temp_data = temp_data.transpose(2, 1, 0)
            # split eeg with length 3p0, discard the remainder
            for j in range(l // split):
                self.target.append(raw_target[i])
                self.data.append(temp_data[:, j*split:(j+1)*split, :])

    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)


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
    

def make_eeg(args):
    eeg_train = EEG(train=True)
    eeg_test = EEG(train=False)

    train_loader = DataLoader(eeg_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(eeg_test, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == '__main__':
    loader = make_eeg()
    for d, t in loader:
        print(d.shape)