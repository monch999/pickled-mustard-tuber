# -*- coding = utf-8 -*-
# @Time : 2021/10/13 9:24
# @Author : 自在清风
# @File : run.py
# @software: PyCharm
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import UNetDataset
from unet4p import Unet4P


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        log_softmax = nn.LogSoftmax()
        logpt = log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def accuracy(output, target):
    total_markup = 0
    total_intersect = 0
    _, output = output.data.max(dim=1)
    output[output > 0] = 1
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    intersect = output * target
    total_intersect += intersect.sum()
    total_markup += target.sum()
    result = total_intersect / total_markup
    
    return result


class UNetTrainer(object):
    """net train"""

    def __init__(self, start_epoch=0, save_dir='', resume='', num_classes=2, color_dim=1):
        self.net = Unet4P(color_dim=color_dim, num_classes=num_classes)
        self.start_epoch = start_epoch if start_epoch != 0 else 1
        self.save_dir = os.path.join(r'model_4P', save_dir)  # Model storage path
        # self.loss = CrossEntropyLoss()
        self.loss = FocalLoss()
        self.num_classes = num_classes

        if resume:
            checkpoint = torch.load(resume)  # load model
            if self.start_epoch == 0:
                self.start_epoch = checkpoint['epoch'] + 1
            if not self.save_dir:
                self.save_dir = checkpoint['save_dir']
            self.net.load_state_dict(checkpoint['state_dir'])  # load model parameters

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def train(self, train_loader, var_loader, lr=0.001, weight_decay=1e-4, epochs=300, save_freq=1):
        """
        train model
        :param train_loader: train dataset
        :param var_loader: validation dataset
        :param lr: learn rate
        :param weight_decay: weight decay
        :param epochs:
        :param save_freq: Save data step size
        :return:
        """
        self.logfile = os.path.join(self.save_dir, 'log.txt')
        sys.stdout = Logger(self.logfile)
        self.epochs = epochs
        self.lr = lr

        optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=weight_decay)  # Adam optimizer, SGD
        for epoch in range(self.start_epoch, epochs + 1):
            self.train_(train_loader, epoch, optimizer, save_freq)  # train model
            self.validate_(var_loader, epoch)  # Validate Model

    def train_(self, data_loader, epoch, optimizer, save_freq):
        """
        train
        :param data_loader: train dataset
        :param epoch:
        :param optimizer:
        :param save_freq:
        :return:
        """
        start_time = time.time()

        if torch.cuda.device_count() > 1:
            self.net = DataParallel(self.net)
        self.net.to(device)
        self.net.train()

        metrics = []
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data = Variable(data.to(device))
            target = Variable(target.to(device))

            output = self.net(data)

            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
            target = target.view(-1)
            loss_output = self.loss(output, target)  # Calculate loss value

            optimizer.zero_grad()
            loss_output.requires_grad_(True)
            loss_output.backward()  # Back Propagation
            optimizer.step()  # Weight update

            loss_output = loss_output.item()
            acc = accuracy(output, target)
            metrics.append([loss_output, acc])
        if epoch % save_freq == 0:
            if 'module' in dir(self.net):  # Returns the properties of net
                state_dict = self.net.module.state_dict()
            else:
                state_dict = self.net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({  # Save Model Data
                'epoch': epoch,
                'save_dir': self.save_dir,
                'state_dir': state_dict
            }, os.path.join(self.save_dir, 'unet4p%03d.pth' % epoch))

        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        writer.add_scalar('loss', np.mean(metrics[:, 0]), epoch)
        writer.add_scalar('accuracy', np.mean(metrics[:, 1]), epoch)
        writer.close()
        self.print_metrics(metrics, 'Train', end_time - start_time, epoch)

    def validate_(self, data_loader, epoch):
        """
        validation
        :param data_loader: Validation Data
        :param epoch: Learning frequency
        :return:
        """
        start_time = time.time()

        self.net.eval()
        metrics = []
        for i, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                data = Variable(data.to(device))
                target = Variable(target.to(device))

            output = self.net(data)
            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
            target = target.view(-1)
            loss_output = self.loss(output, target)

            loss_output = loss_output.item()
            acc = accuracy(output, target)
            metrics.append([loss_output, acc])
        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        self.print_metrics(metrics, 'Validation', end_time - start_time)

    def print_metrics(self, metrics, phase, time, epoch=-1):
        """
        Save metrics
        :param metrics: Loss value and correct ratio
        :param phase: Data types, divided into validation and training types
        :param time: Time taken to save data once
        :param epoch:
        :return:
        """
        if epoch != -1:
            print('Epoch: {}'.format(epoch))
        print(phase)
        print('loss %2.4f, accuracy %2.4f, time %2.2f' % (np.mean(metrics[:, 0]), np.mean(metrics[:, 1]), time))
        if phase != 'Train':
            print()


class UNetTester(object):
    """test data"""

    def __init__(self, model_path, target_path, color_dim=3, num_classes=2, img_mod=0):
        self.net = Unet4P(color_dim=color_dim)
        checkpoint = torch.load(model_path)
        self.target_dir = target_path
        self.color_dim = color_dim
        self.num_classes = num_classes
        self.net.load_state_dict(checkpoint['state_dir'])
        self.net.to(device)
        self.net.eval()
        self.mod = img_mod

    def test(self, testImage_dir):
        cracks_files = glob.glob(os.path.join(testImage_dir, '*.png'))
        print(len(cracks_files), 'imgs.')
        for cracks_file in tqdm(cracks_files):  # Show read progress bar
            name = os.path.basename(cracks_file)
            save_path = os.path.join(self.target_dir, name)

            data = cv2.imread(cracks_file, self.mod)

            output = self._test(data)
            cv2.imwrite(save_path, output)

    def _test(self, data):
        data = data.astype(np.float32) / 255.0
        data = np.expand_dims(data, 0)
        if self.mod:
            data = data.transpose((0, 3, 1, 2))
        input = torch.from_numpy(data)
        height = input.size()[-2]
        width = input.size()[-1]
        with torch.no_grad():
            input = Variable(input.to(device))

        output = self.net(input)
        output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
        _, output = output.data.max(dim=1)
        output[output > 0] = 255
        output = output.view(height, width)
        output = output.cpu().numpy()
        return output


if __name__ == '__main__':
    mode = input('Enter network mode, where t represents training mode and others represent testing mode:')
    # Visualize loss values and accuracy
    writer = SummaryWriter('./runs')
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'using device: {device}')
    if mode == 't':
        mask = glob.glob(r'data/train/fusion_data/mask/*.png')
        N = int(len(mask) * 0.8)
        train_data = DataLoader(UNetDataset(mask_list=mask[:N], phase='train'),
                                batch_size=16, shuffle=True, num_workers=8)
        validate_data = DataLoader(UNetDataset(mask_list=mask[N:], phase='train'),
                                   batch_size=16, shuffle=True, num_workers=8)
        crack_segment = UNetTrainer(save_dir='model', color_dim=3)
        crack_segment.train(train_data, validate_data)
    else:
        crack_testNet = UNetTester(model_path=r'model_4P/hd.pth',
                                   target_path=r'data/detected_result/hd', img_mod=1)
        crack_testNet.test(testImage_dir=r'data/test/test_hd_image')
