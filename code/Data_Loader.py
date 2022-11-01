import torch
import os
from torch.utils.data import Dataset
import glob
import numpy as np
from itertools import combinations

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
path = "../cross_validation_raw/f1_40/training"

# fall and non_fall data with 0 or 1 labels for npy
class RadarDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data, self.label = self.readRadarData(self.path)

    def readRadarData(self, path):
        folder_list = os.listdir(path)
        print(folder_list)

        all_data = []
        label = []
        l = None
        for folder in folder_list:
            if folder[:3] == 'fal':
                l = 0
            else:
                l = 1

            folder_path = os.path.join(path, folder)
            files_path = glob.glob('{}/*'.format(folder_path))
            for file_path in files_path:
                data = np.load(file_path, allow_pickle=True)
                data[0] = (data[0] - data[0].min()) / (data[0].max() - data[0].min()) * 2 - 1  # normalize to [-1,1]
                data[1] = (data[1] - data[1].min()) / (data[1].max() - data[1].min()) * 2 - 1
                all_data.append(data)
                label.append(l)

        print("finish")
        all_data = np.array(all_data)
        label = np.array(label)
        print("max_label: ", label.max())
        index = np.arange(len(label))
        np.random.shuffle(index)
        all_data = all_data[index]
        label = label[index]

        all_data = torch.Tensor(all_data)
        label = torch.Tensor(label)

        print(all_data.shape)
        print(label.shape)

        return all_data, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# fall and non_fall data with 0 or 1 labels for npy
class RadarImageDataset(Dataset):
    def __init__(self, path, mode=0):
        self.path = path
        self.mode = mode
        self.data, self.label = self.readRadarData(self.path, self.mode)

    def readRadarData(self, path, mode):
        folder_list = os.listdir(path)
        print(folder_list)

        all_data = []
        label = []
        l = None
        for folder in folder_list:
            if mode:
                if folder[:3] == 'fal':
                    l = 0
                elif folder[:3] == 'ben' or folder[:3] == 'sit' or folder[:3] == 'jum':
                    l = 2
                else:
                    l = 1
            else:
                if folder[:3] == 'fal':
                    l = 0
                else:
                    l = 1

            folder_path = os.path.join(path, folder)
            files_path = glob.glob('{}/*'.format(folder_path))
            for file_path in files_path:
                data = np.load(file_path, allow_pickle=True)
                data = (data - data.min()) / (data.max() - data.min()) * 2 - 1  #scale to [-1, 1]
                all_data.append(data)
                label.append(l)

        print("finish")
        all_data = np.array(all_data)
        label = np.array(label)
        print("max_label: ", label.max())
        index = np.arange(len(label))
        np.random.shuffle(index)
        all_data = all_data[index]
        label = label[index]

        all_data = torch.Tensor(all_data)
        label = torch.Tensor(label)

        print(all_data.shape)
        print(label.shape)

        return all_data, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def readRadarData(path):
    folder_list = os.listdir(path)
    print(folder_list)

    fall_forward_data = []
    fall_backward_data = []
    fall_aside_data = []
    fall_data = []
    non_fall_data = []
    pair_fall_data = []
    test_fall_data = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    test_acc_data = []
    test_acc_label = []
    l = None
    for folder in folder_list:
        folder_path = os.path.join(path, folder)
        files_path = glob.glob('{}/*'.format(folder_path))
        for file_path in files_path:
            data = np.load(file_path, allow_pickle=True)
            data[0] = (data[0] - data[0].min()) / (data[0].max() - data[0].min()) * 2 - 1  # normalize to [-1,1]
            data[1] = (data[1] - data[1].min()) / (data[1].max() - data[1].min()) * 2 - 1
            if folder[:3] == 'fal':
                # fall_data.append(data)
                if folder[5] == 'a':
                    fall_aside_data.append(data)
                elif folder[5] == 'b':
                    fall_backward_data.append(data)
                elif folder[5] == 'f':
                    fall_forward_data.append(data)
            else:
                non_fall_data.append(data)

    print("finish")
    # print("len of fall_data: ", len(fall_data))
    print("len of non_fall_data: ", len(non_fall_data))
    # fall_data = np.array(fall_data)
    fall_forward_data = np.array(fall_forward_data)
    fall_backward_data = np.array(fall_backward_data)
    fall_aside_data = np.array(fall_aside_data)
    non_fall_data = np.array(non_fall_data)

    index1_1 = np.arange(len(fall_forward_data))
    index1_2 = np.arange(len(fall_backward_data))
    index1_3 = np.arange(len(fall_aside_data))
    np.random.shuffle(index1_1)
    np.random.shuffle(index1_2)
    np.random.shuffle(index1_3)
    fall_forward_data = fall_forward_data[index1_1]
    fall_backward_data = fall_backward_data[index1_2]
    fall_aside_data = fall_aside_data[index1_3]

    index2 = np.arange(len(non_fall_data))
    np.random.shuffle(index2)
    non_fall_data = non_fall_data[index2]

    # fall data pairs, label 0
    print("non_fall_data shape: ", non_fall_data.shape)

    # train fall data
    for i in range(90):
        pair_fall_data.append(fall_forward_data[i])  # list the fall samples used for training
        pair_fall_data.append(fall_backward_data[i])
        pair_fall_data.append(fall_aside_data[i])
    pair_fall_data = np.array(pair_fall_data)
    print("pair_fall_data shape: ", pair_fall_data.shape)

    # test fall data
    for i in range(90, 90 + 60):
        test_fall_data.append(fall_forward_data[i])  # list the fall samples used for training
        test_fall_data.append(fall_backward_data[i])
        test_fall_data.append(fall_aside_data[i])
    test_acc_data = test_acc_data + test_fall_data
    test_acc_label = [0 for i in range(len(test_fall_data))]
    test_fall_data = np.array(test_fall_data)
    print("test_fall_data shape: ", test_fall_data.shape)

    # test nonfall data
    for i in range(150, 150 + 60):
        test_acc_data.append(non_fall_data[i])
        test_acc_label.append(1)

    # train fall-fall pair
    for p in combinations(range(90), 2):
        fall_data_pair = np.concatenate((fall_forward_data[p[0], :, :], fall_forward_data[p[1], :, :]), axis=0)
        train_data.append(fall_data_pair)
        train_label.append(0)
        fall_data_pair = np.concatenate((fall_backward_data[p[0], :, :], fall_backward_data[p[1], :, :]), axis=0)
        train_data.append(fall_data_pair)
        train_label.append(0)
        fall_data_pair = np.concatenate((fall_aside_data[p[0], :, :], fall_aside_data[p[1], :, :]), axis=0)
        train_data.append(fall_data_pair)
        train_label.append(0)

    # shuffle again to build fall and non_fall pair
    index3 = np.arange(len(pair_fall_data))
    np.random.shuffle(index3)
    pair_fall_data = pair_fall_data[index3]

    # train fall-nonfall pair
    for p in combinations(range(150), 2):  # choose 2 index from 0~100 to build the pair data
        fall_nonfall_pair = np.concatenate((pair_fall_data[p[0], :, :], non_fall_data[p[1], :, :]), axis=0)
        train_data.append(fall_nonfall_pair)
        train_label.append(1)

    # build the test dataset
    for p in combinations(range(90, 90 + 60), 2):
        fall_data_pair = np.concatenate((fall_forward_data[p[0], :, :], fall_forward_data[p[1], :, :]), axis=0)
        test_data.append(fall_data_pair)
        test_label.append(0)
        fall_data_pair = np.concatenate((fall_backward_data[p[0], :, :], fall_backward_data[p[1], :, :]), axis=0)
        test_data.append(fall_data_pair)
        test_label.append(0)
        fall_data_pair = np.concatenate((fall_aside_data[p[0], :, :], fall_aside_data[p[1], :, :]), axis=0)
        test_data.append(fall_data_pair)
        test_label.append(0)

    for p in combinations(range(60), 2):  # choose 2 index from 0~100 to build the pair data
        fall_nonfall_pair = np.concatenate((test_fall_data[p[0], :, :], non_fall_data[150 + p[1], :, :]), axis=0)
        # np.save(fall_nonfall_pair_save_path + "fall_nonfall_pair" + str(i), fall_nonfall_pair)
        test_data.append(fall_nonfall_pair)
        test_label.append(1)

    # get all data and label
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_acc_data = np.array(test_acc_data)
    test_acc_label = np.array(test_acc_label)

    index4 = np.arange(len(train_label))
    np.random.shuffle(index4)
    train_data = train_data[index4]
    train_label = train_label[index4]

    index5 = np.arange(len(test_label))
    np.random.shuffle(index5)
    test_data = test_data[index5]
    test_label = test_label[index5]

    index6 = np.arange(len(test_acc_label))
    np.random.shuffle(index6)
    test_acc_data = test_acc_data[index6]
    test_acc_label = test_acc_label[index6]

    print("train_data shape: ", train_data.shape)
    print("train_label shape: ", train_label.shape)
    print("test_data shape: ", test_data.shape)
    print("test_label shape: ", test_label.shape)
    print("test_acc_data shape: ", test_acc_data.shape)
    print("test_acc_label shape: ", test_acc_label.shape)

    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label)

    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label)

    test_acc_data = torch.Tensor(test_acc_data)
    test_acc_label = torch.Tensor(test_acc_label)

    return train_data, train_label, test_data, test_label, test_acc_data, test_acc_label

# fall data pairs, fall and non_fall data pairs with 1 or 0 labels for npy
# fall_forward is a pair,fall_aside is a pair,fall_backward is a pair
# train dataset and test dataset are bulit from different samples
class RadarDatasetPair(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    # step1: data reading
    print("start reading data!")
    train_data, train_label, test_data, test_label, test_acc_data, test_acc_label = readRadarData(path)
    train_dataset = RadarDatasetPair(train_data, train_label)
    test_dataset = RadarDatasetPair(test_data, test_label)
    print(len(train_dataset))
    print(len(test_dataset))

