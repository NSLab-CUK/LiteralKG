# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 save_from=1,
                 save_best_model=False):

        print("Training GPU")

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        self.save_best_model = save_best_model

        self.model = model.cuda()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=None)
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.save_from = save_from
        self.checkpoint_dir = checkpoint_dir
        self.loss_data = []
        self.min_loss = None

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss

            self.loss_data.append(res)

            if self.save_best_model and epoch + 1 >= self.save_from:

                if self.min_loss is None:
                    self.min_loss = res
                elif self.min_loss > res:
                    self.min_loss = res
                    print("Epoch %d has finished, saving..." % (epoch))
                    self.model.module.save_checkpoint(os.path.join(self.checkpoint_dir + "min_model_" + str(epoch) + ".ckpt"))

            print("\nLoss in epoch %d | loss: %f" % (epoch, res))
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0 and epoch + 1 >= self.save_from:
                    print("Epoch %d has finished, saving..." % (epoch))
                    self.model.module.save_checkpoint(os.path.join(self.checkpoint_dir + "model_" + str(epoch) + ".ckpt"))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return self.loss_data

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir