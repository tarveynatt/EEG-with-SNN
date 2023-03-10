from os import listdir
from scipy.io import loadmat
from mne.filter import filter_data


PATH = './eeg_raw_data'
fs = 1000
target = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
          [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
          [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
# 把target写在文件名后面

channels = [

              'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 

 'F7',  'F5',  'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8',

'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',

 'T7',  'C5',  'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8',

'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',

 'P7',  'P5',  'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8',

       'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 

              'CB1',  'O1',  'OZ',  'O2', 'CB2'              
          
          ]


def load_eeg():
    file_name = []
    for i in range(1, 4):
        name = listdir(f'{PATH}/{i}')
        for n in name:
            file_name.append(f'{i}/{n}')

    return file_name


def preprocess(name, fs, target):
    # 62 channels data of one trail of one subject
    sig = loadmat(f'{PATH}/{name}')
    theta = filter_data(data=sig, sfreq=fs, l_freq=4, h_freq=8, verbose=False)
    alpha = filter_data(data=sig, sfreq=fs, l_freq=8, h_freq=14, verbose=False)
    beta = filter_data(data=sig, sfreq=fs, l_freq=14, h_freq=31, verbose=False)
    gamma = filter_data(data=sig, sfreq=fs, l_freq=31, h_freq=50, verbose=False)




if __name__ == '__main__':
    name = load_eeg()