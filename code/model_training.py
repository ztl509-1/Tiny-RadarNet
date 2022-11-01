import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Data_Loader import RadarDataset, RadarDatasetPair, readRadarData
from tSNE import tsne
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from random_seed import setup_seed
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
path = '../cross_validation_raw/f441_480/training'
val_path = "../cross_validation_raw/f441_480/testing"
decay = 4e-5 * 10

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(Block, self).__init__()

        self.branch1 = nn.Sequential(
            # depthwise conv
            nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel,
                      bias=False),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(inplace=True),

        )
        self.branch2 = nn.Sequential(
            # depthwise conv
            nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel,
                      bias=False),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(inplace=True),

        )
    def channel_shuffle(self, x, group):
        batch_size, channel, dim = x.shape
        assert channel % group == 0
        channel_per_group = channel // group
        x = x.view(batch_size, group, channel_per_group, dim)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, channel, dim)

        return x

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return self.channel_shuffle(out, 2)


# model for softmax(good for validation)
class RadarNet(nn.Module):
    def __init__(self, classes=2):
        super(RadarNet, self).__init__()
        self.d = 1
        self.final_out_channel = 64
        self.final_average_pooling = 8

        def conv_bn(in_chs, out_chs, stride):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_chs, out_channels=out_chs, kernel_size=7,  stride=stride, bias=False, dilation=2),
                nn.BatchNorm1d(out_chs),
                nn.LeakyReLU(inplace=True),
            )

        self.model = nn.Sequential(
            # 2,4800 --> 8, 1598
            conv_bn(2, 16, 4),
            # 8, 1598 --> 16, 532
            Block(in_channel=16, out_channel=32, kernel_size=5, stride=3),
            # 16, 532 --> 32, 176
            Block(in_channel=32, out_channel=64, kernel_size=5, stride=3),
            # 32, 176 --> 64, 58
            nn.AdaptiveAvgPool1d(self.final_average_pooling),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.final_average_pooling * self.final_out_channel * self.d, 4),
            nn.LeakyReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.final_average_pooling*self.final_out_channel)
        x1 = self.fc1(x)
        return x1

def metric_loss(fea1, fea2, labels, margin=2.0):
    distance = F.pairwise_distance(fea1, fea2)

    zero_feature = torch.zeros(fea1.shape).cuda()
    zero_distance = (F.pairwise_distance(fea1, zero_feature) + F.pairwise_distance(fea2, zero_feature))/2.
    if labels.is_cuda:
        zero_loss = (1 - labels).float() * zero_distance
        c_loss = (1 - labels).float() * distance
        d_loss = (labels).float() * torch.clamp(torch.tensor(margin).cuda() - distance, min=0.0)
        loss = torch.mean(c_loss + d_loss + zero_loss)
    return loss


def training(model, train_dataset, test_dataset, epochs=10, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=0.00001)

    best_loss = 100
    train_loss_ls = []
    test_loss_ls = []

    for epoch in range(epochs):
        model.train()

        i = 0
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_dataset:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            feature1 = model(images[:, :2, :])
            feature2 = model(images[:, 2:, :])
            loss = metric_loss(feature1, feature2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total_samples += len(labels)
            i += 1

        train_acc = total_correct / total_samples
        train_loss = total_loss / i
        train_loss_ls.append(train_loss)
        print("epoch: ", epoch, "train_acc: ", train_acc, "total_loss: ", train_loss)
        test_loss = testing(model, test_dataset)
        test_loss_ls.append(test_loss)
        lr_scheduler.step(int(test_loss*100))

        # save the model
        if best_loss > test_loss:
            best_loss = test_loss
            best_model = model

    plt.plot(train_loss_ls)
    plt.plot(test_loss_ls)
    plt.show()
    return best_model


def testing(model, test_dataset):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    i = 0
    for batch in test_dataset:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()

        feature1 = model(images[:, :2, :])
        feature2 = model(images[:, 2:, :])

        loss = metric_loss(feature1, feature2, labels)
        total_loss += loss.item()
        total_samples += len(labels)
        i += 1

    total_loss = total_loss / i
    print("test_acc: ", total_correct / total_samples, "loss: ", total_loss)

    return total_loss


def validating_testing_model(model, test_dataset, mode=1):
    model.eval()
    print(1)
    total_loss = 0
    total_correct = 0
    total_samples = 0
    threshold = 1.0

    i = 0
    for batch in test_dataset:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()

        feature = model(images)

        fea = torch.abs(feature).sum(dim=1)
        preds = (fea > threshold).long()
        total_correct += preds.eq(labels.long()).sum().item()
        total_samples += len(labels)
        i += 1

    if mode:
        tsne(feature.cpu().detach(), labels.cpu().detach())

    print("valid_acc: ", total_correct / total_samples, "loss: ", total_loss)
    class_names = ['fall', 'non-fall']
    print(classification_report(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                target_names=class_names,
                                digits=4))
    return total_loss


def validateMSE(model, dataset):
    model.eval()

    label_list = np.empty((0,))
    MSE_list = np.empty((0,))

    for batch in dataset:

        images, labels = batch
        images = images.cuda()
        labels = labels.detach().numpy()

        label_list = np.append(label_list, labels, axis=0)
        feature = model(images)
        feature = feature.cpu().detach().numpy()
        feature = np.abs(feature).sum(axis = 1)
        feature = feature / feature.max()


        MSE_list = np.append(MSE_list, feature, axis=0)

    label_list = label_list.reshape(-1,)
    MSE_list = MSE_list.reshape(-1,)
    AUC = roc_auc_score(label_list, MSE_list)
    print("AUC: ", AUC)
    fpr, tpr, thresholds = metrics.roc_curve(label_list, MSE_list, pos_label=1)
    # plt.plot(fpr, tpr)
    # plt.show()
    return AUC, fpr, tpr, thresholds

if __name__ == "__main__":
    setup_seed(2)
    # step1: data reading
    print("start reading data!")
    train_data, train_label, test_data, test_label, test_acc_data, test_acc_label = readRadarData(path)
    train_dataset = RadarDatasetPair(train_data, train_label)
    test_dataset = RadarDatasetPair(test_data, test_label)
    test_acc_dataset = RadarDatasetPair(test_acc_data, test_acc_label)
    valid_dataset = RadarDataset(val_path)
    val_size = len(valid_dataset)
    print("dataset_len: ", len(train_dataset))

    # put into the Dataloader
    batch_size = 64*8
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                drop_last=True)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_acc_dataset = torch.utils.data.DataLoader(test_acc_dataset, batch_size=len(test_acc_dataset), shuffle=False, drop_last=True)

    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size= val_size, shuffle=True,
                                                drop_last=True)
    # step2: model building
    print("build the model!")
    feature_encoder = RadarNet().cuda()
    model = training(feature_encoder, train_dataset, test_dataset, epochs=100, learning_rate=0.001)
    validating_testing_model(model, test_acc_dataset, mode=0)
    validateMSE(model, test_acc_dataset)
    validating_testing_model(model, valid_dataset, mode=0)
    validateMSE(model, valid_dataset)

