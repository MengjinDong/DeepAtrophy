#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:24:35 2018

@author: mengjin
"""
import io
import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import utils
from torch import autograd
import matplotlib.pyplot as plt
import csv
import gc
import random
from itertools import zip_longest
from data_aug_gpu import TPSRandomSampler3D
from matplotlib import pyplot
from tensorboardX import SummaryWriter

pyplot.switch_backend('agg')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)  # want top k most predicted values
        batch_size = target.size(0)  # batch_size = 250

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(
            target.view(1, -1).expand_as(pred))  # pred: 250*1, compare with target value, get #correct samples

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # number of correct predictions in each top k
            #            print("correct_k = ", correct_k)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, model_name='checkpoint'):
    file_name = model_name + ".pth.tar"
    torch.save(state, file_name)
    if is_best:
        best_name = model_name + "_best.pth.tar"
        shutil.copyfile(file_name, best_name)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, tolerance=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            tolerance (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_prec1 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.tolerance = tolerance

    def __call__(self, state, score, model):

        # score = - acc

        print('model_name is: ', model)

        if self.best_prec1 is None:
            self.best_prec1 = score
            state['best_prec1'] =  self.best_prec1
            self.save_checkpoint(state, is_best=True, model_name=model)

        elif score < self.best_prec1 + self.tolerance:
            state['best_prec1'] = self.best_prec1
            self.save_checkpoint(state, is_best=False, model_name=model)
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_prec1 = score
            self.save_checkpoint(state, is_best=True, model_name=model)
            self.counter = 0

        print("Best precision is: ", self.best_prec1)
        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

    def save_checkpoint(self, state, is_best, model_name='checkpoint'):
        file_name = model_name + ".pth.tar"
        torch.save(state, file_name)
        if is_best:
            best_name = model_name + "_best.pth.tar"
            shutil.copyfile(file_name, best_name)

def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          sample_size,
          print_freq,
          writer,
          range_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # for all three measurements, can only use loss to visualize performance.
    losses0 = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    loss_all = AverageMeter()

    output_date_diff1 = AverageMeter()
    output_date_diff2 = AverageMeter()

    height, width, depth = sample_size

    tps_sampler = TPSRandomSampler3D(height, width, depth,
                                     vertical_points=10, horizontal_points=10, depth_points=10,
                                     rotsd=10.0, scalesd=0.0, transsd=0.0,
                                     warpsd=(0.0, 0.0),
                                     cache_size=0, pad=False)

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        sample_batched['image'] = sample_batched['image'].cuda(non_blocking=True)
        sample_batched['label_date_diff1'] = sample_batched['label_date_diff1'].cuda(non_blocking=True)
        sample_batched['label_date_diff2'] = sample_batched['label_date_diff2'].cuda(non_blocking=True)
        sample_batched['date_diff_ratio'] = sample_batched['date_diff_ratio'].cuda(non_blocking=True)
        sample_batched['label_time_interval'] = sample_batched['label_time_interval'].cuda(non_blocking=True)

        with torch.no_grad():
            sample_batched['image'] = tps_sampler(sample_batched['image'])

        # # switch to train mode
        # model.train()
        out_t_order1, out_t_order2, out_range = model(sample_batched['image'])
        num_batches = sample_batched['image'].size(0)

        loss0 = criterion[0](out_t_order1, sample_batched['label_date_diff1'].long())
        loss1 = criterion[0](out_t_order2, sample_batched['label_date_diff2'].long())
        loss2 = criterion[1](out_range, sample_batched['label_time_interval'].long())  # long for NLLLoss

        loss = loss0 + loss1 + range_weight * loss2

        # loss = loss1
        losses0.update(loss0.item(), num_batches)
        losses1.update(loss1.item(), num_batches) # statistics of loss, losses0 and losses1 are AverageMeters
        losses2.update(loss2.item(), num_batches)
        loss_all.update(loss.item(), num_batches)

        res1, pred1 = accuracy(out_t_order1, sample_batched['label_date_diff1'].long())
        output_date_diff1.update(res1[0], num_batches)
        
        res2, pred2 = accuracy(out_range, sample_batched['label_time_interval'].long())
        output_date_diff2.update(res2[0], num_batches)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with autograd.detect_anomaly():
            loss.backward()  # back prop loss to intermediate layers
            optimizer.step()

        gc.collect()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\n'
                  'Time {batch_time.val!s:4s} ({batch_time.avg!s:4s})\n'
                  'Data {data_time.val!s:4s} ({data_time.avg!s:4s})\n'
                  'Loss0(BCE loss for t order) {Loss0.val!s:5s} ({Loss0.avg!s:5s})\n'
                  'Loss1(BCE loss for t order) {Loss1.val!s:5s} ({Loss1.avg!s:5s})\n'
                  'Loss2(CrossEntropy loss for range) {Loss2.val!s:5s} ({Loss2.avg!s:5s})\n'
                  'Accuracy t_order {output_date_diff1.val!s:5s} ({output_date_diff1.avg!s:5s})\n'
                  'Accuracy range {output_date_diff2.val!s:5s} ({output_date_diff2.avg!s:5s})\n'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, Loss0=losses0,
                Loss1=losses1,
                Loss2=losses2,
                output_date_diff1=output_date_diff1, 
                output_date_diff2=output_date_diff2))
            print('-' * 15)

    writer.add_scalar('Train/Loss(BCE loss for t order)', losses0.avg, epoch)
    writer.add_scalar('Train/Loss(BCE loss for t order)', losses1.avg, epoch)
    writer.add_scalar('Train/Loss(CrossEntropy loss for range)', losses2.avg, epoch)
    writer.add_scalar('Train/Loss(all)', loss_all.avg, epoch)
    writer.add_scalar('Train/Accuracy t_order', output_date_diff1.avg, epoch)
    writer.add_scalar('Train/Accuracy range', output_date_diff2.avg, epoch)

    print(' * Training Prec@1 {loss.avg!s:5s}'.format(loss=output_date_diff1))
    print(' * Training Prec@2 {loss.avg!s:5s}'.format(loss=output_date_diff2))

    print('num_train = {loss_all.count!s:5s}'.format(loss_all=loss_all))


def validate(val_loader,
             model,
             criterion,
             model_name,
             epoch=0,
             writer=None,
             range_weight = 1,
             print_freq = 20):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # for all three measurements, can only use loss to visualize performance.
    losses0 = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    loss_all = AverageMeter()

    output_date_diff1 = AverageMeter()
    output_date_diff2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    csv_name = model_name + ".csv"
    if os.path.isfile(csv_name):
        os.remove(csv_name)
    with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(["bl_fname1", "bl_fname2", "subjectID", "side", "stage", "bl_time1", "bl_time2",
                     "fu_time1", "fu_time2", "date_diff1", "date_diff2", "label_date_diff1", "label_date_diff2",
                     "label_time_interval", "pred_date_diff1", "pred_date_diff2", "pred_time_interval",
                     "score0", "score1", "score2", "score3", "score4", "score5", ])

    with torch.no_grad():
        end = time.time()
        for i, sample_batched in enumerate(val_loader):

            data_time.update(time.time() - end)

            sample_batched['image'] = sample_batched['image'].cuda(non_blocking=True)
            sample_batched['label_date_diff1'] = sample_batched['label_date_diff1'].cuda(non_blocking=True)
            sample_batched['label_date_diff2'] = sample_batched['label_date_diff2'].cuda(non_blocking=True)
            sample_batched['date_diff_ratio'] = sample_batched['date_diff_ratio'].cuda(non_blocking=True)
            sample_batched['label_time_interval'] = sample_batched['label_time_interval'].cuda(non_blocking=True)

            # model returns a list: [seg, out_jac, out_date_diff]
            out_t_order1, out_t_order2, out_range = model(sample_batched['image'])
            num_batches = sample_batched['image'].size(0)

            loss0 = criterion[0](out_t_order1, sample_batched['label_date_diff1'].long())
            loss1 = criterion[0](out_t_order2, sample_batched['label_date_diff2'].long())
            loss2 = criterion[1](out_range, sample_batched['label_time_interval'].long())  # long for NLLLoss

            loss = loss0 + loss1 + range_weight * loss2

            # loss = loss1
            losses0.update(loss0.item(), num_batches)
            losses1.update(loss1.item(), num_batches)  # statistics of loss, losses0 and losses1 are AverageMeters
            losses2.update(loss2.item(), num_batches)
            loss_all.update(loss.item(), num_batches)

            res1, pred1 = accuracy(out_t_order1, sample_batched['label_date_diff1'].long())
            output_date_diff1.update(res1[0], num_batches)

            res2, pred2 = accuracy(out_range, sample_batched['label_time_interval'].long())
            output_date_diff2.update(res2[0], num_batches)

            res3, pred3 = accuracy(out_t_order2, sample_batched['label_date_diff1'].long())

            bl_fname1 = sample_batched['bl_fname1']
            bl_fname2 = sample_batched['bl_fname2']
            bl_time1 = sample_batched['bl_time1']
            bl_time2 = sample_batched['bl_time2']
            fu_time1 = sample_batched['fu_time1']
            fu_time2 = sample_batched['fu_time2']

            date_diff1 = sample_batched['date_diff1']
            date_diff2 = sample_batched['date_diff2']
            label_date_diff1 = sample_batched['label_date_diff1']
            label_date_diff2 = sample_batched['label_date_diff2']
            label_time_interval = sample_batched['label_time_interval']
            stage = sample_batched['stage']

            subjectID = sample_batched['subjectID']
            side = sample_batched['side']

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\n'
                      'Time {batch_time.val!s:4s} ({batch_time.avg!s:4s})\n'
                      'Data {data_time.val!s:4s} ({data_time.avg!s:4s})\n'
                      'Loss0(BCE loss for t order) {Loss0.val!s:5s} ({Loss0.avg!s:5s})\n'
                      'Loss1(BCE loss for t order) {Loss1.val!s:5s} ({Loss1.avg!s:5s})\n'
                      'Loss2(CrossEntropy loss for atrophy) {Loss2.val!s:5s} ({Loss2.avg!s:5s})\n'
                      'Accuracy t_order {output_date_diff1.val!s:5s} ({output_date_diff1.avg!s:5s})\n'
                      'Accuracy range {output_date_diff2.val!s:5s} ({output_date_diff2.avg!s:5s})\n'
                    .format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    data_time=data_time, Loss0=losses0, Loss1=losses1, Loss2=losses2,
                    output_date_diff1=output_date_diff1, output_date_diff2=output_date_diff2))
                print('-' * 15)

            d = [bl_fname1, bl_fname2, subjectID, side, stage, bl_time1, bl_time2,
                 fu_time1, fu_time2, date_diff1.cpu().numpy(), date_diff2.cpu().numpy(),
                 label_date_diff1.cpu().numpy(), label_date_diff2.cpu().numpy(),
                 label_time_interval.cpu().numpy(),
                 np.transpose(pred1.cpu().numpy()).squeeze(1),
                 np.transpose(pred3.cpu().numpy()).squeeze(1),
                 np.transpose(pred2.cpu().numpy()).squeeze(1), ]
            d.extend(np.transpose(out_t_order1.cpu().numpy()))
            d.extend(np.transpose(out_t_order2.cpu().numpy()))
            d.extend(np.transpose(out_range.cpu().numpy()))


            export_data = zip_longest(*d, fillvalue='')
            with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(export_data)
            myfile.close()

        writer.add_scalar('Test/Loss(BCE loss for t order)', losses0.avg, epoch)
        writer.add_scalar('Test/Loss(BCE loss for t order)', losses1.avg, epoch)
        writer.add_scalar('Test/Loss(CrossEntropy loss for range)', losses2.avg, epoch)
        writer.add_scalar('Test/Loss(all)', loss_all.avg, epoch)
        writer.add_scalar('Test/Accuracy t_order', output_date_diff1.avg, epoch)
        writer.add_scalar('Test/Accuracy range', output_date_diff2.avg, epoch)

        print('Test overall: [{0}]\n'
              'Time {batch_time.val!s:4s} ({batch_time.avg!s:4s})\n'
              'Data {data_time.val!s:4s} ({data_time.avg!s:4s})\n'
              'Loss0(BCE loss for t order) {Loss0.val!s:5s} ({Loss0.avg!s:5s})\n'
              'Loss1(BCE loss for t order) {Loss1.val!s:5s} ({Loss1.avg!s:5s})\n'
              'Loss2(CrossEntropy loss for atrophy) {Loss2.val!s:5s} ({Loss2.avg!s:5s})\n'
              'Accuracy t_order {output_date_diff1.val!s:5s} ({output_date_diff1.avg!s:5s})\n'
              'Accuracy range {output_date_diff2.val!s:5s} ({output_date_diff2.avg!s:5s})\n'
                .format(
                len(val_loader), batch_time=batch_time, data_time=data_time,
                Loss0=losses0, Loss1=losses1, Loss2=losses2,
                output_date_diff1=output_date_diff1, output_date_diff2=output_date_diff2))

        print(' * Overall Prec@1 {output_date_diff.avg!s:5s}'.format(output_date_diff=output_date_diff1))
        print(' * Overall Prec@2 {output_date_diff.avg!s:5s}'.format(output_date_diff=output_date_diff2))

        print('num_test = {loss_all.count!s:5s}'.format(loss_all=loss_all))

    return output_date_diff1.avg

def validate_pair(val_loader,
             model,
             criterion,
             model_name,
             epoch=0,
             writer=None,
             print_freq=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # for all three measurements, can only use loss to visualize performance.
    losses0 = AverageMeter()

    loss_all = AverageMeter()

    output_date_diff1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    csv_name = model_name + ".csv"
    if os.path.isfile(csv_name):
        os.remove(csv_name)
    with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(["bl_fname1", "subjectID", "side", "stage", "bl_time1",
                     "fu_time1", "date_diff1", "label_date_diff1",
                     "pred_date_diff1",
                     "score0", "score1", "score2", "score3", "score4"])

    with torch.no_grad():
        end = time.time()
        for i, sample_batched in enumerate(val_loader):

            data_time.update(time.time() - end)

            sample_batched['image'] = sample_batched['image'].cuda(non_blocking=True)
            sample_batched['label_date_diff1'] = sample_batched['label_date_diff1'].cuda(non_blocking=True)

            # model returns a list: [seg, out_jac, out_date_diff]
            out_t_order_full1 = model(sample_batched['image'])
            out_t_order1 = out_t_order_full1[:, 0:2]
            num_batches = sample_batched['image'].size(0)

            loss0 = criterion[0](out_t_order1, sample_batched['label_date_diff1'].long())

            losses0.update(loss0.item(), num_batches)

            res1, pred1 = accuracy(out_t_order1, sample_batched['label_date_diff1'].long())
            output_date_diff1.update(res1[0], num_batches)


            bl_fname1 = sample_batched['bl_fname1']
            bl_time1 = sample_batched['bl_time1']
            fu_time1 = sample_batched['fu_time1']

            date_diff1 = sample_batched['date_diff1']
            label_date_diff1 = sample_batched['label_date_diff1']
            stage = sample_batched['stage']

            subjectID = sample_batched['subjectID']
            side = sample_batched['side']

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\n'
                      'Time {batch_time.val!s:4s} ({batch_time.avg!s:4s})\n'
                      'Data {data_time.val!s:4s} ({data_time.avg!s:4s})\n'
                      'Loss0(BCE loss for t order) {Loss0.val!s:5s} ({Loss0.avg!s:5s})\n'
                      'Accuracy t_order {output_date_diff1.val!s:5s} ({output_date_diff1.avg!s:5s})\n'
                    .format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    data_time=data_time, Loss0=losses0,
                    output_date_diff1=output_date_diff1))
                print('-' * 15)

            d = [bl_fname1, subjectID, side, stage, bl_time1,
                 fu_time1, date_diff1.cpu().numpy(),
                 label_date_diff1.cpu().numpy(),
                 np.transpose(pred1.cpu().numpy()).squeeze(1)]
            d.extend(np.transpose(out_t_order_full1.cpu().numpy()))

            export_data = zip_longest(*d, fillvalue='')
            with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(export_data)
            myfile.close()

        writer.add_scalar('Test_pair/Loss(BCE loss for t order)', losses0.avg, epoch)
        writer.add_scalar('Test_pair/Accuracy t_order', output_date_diff1.avg, epoch)

        print('Test_pair overall: [{0}]\n'
              'Time {batch_time.val!s:4s} ({batch_time.avg!s:4s})\n'
              'Data {data_time.val!s:4s} ({data_time.avg!s:4s})\n'
              'Loss0(BCE loss for t order) {Loss0.val!s:5s} ({Loss0.avg!s:5s})\n'
              'Accuracy t_order {output_date_diff1.val!s:5s} ({output_date_diff1.avg!s:5s})\n'
                .format(
                len(val_loader), batch_time=batch_time, data_time=data_time,
                Loss0=losses0,
                output_date_diff1=output_date_diff1,))

        print(' * Overall Prec@1 {output_date_diff.avg!s:5s}'.format(output_date_diff=output_date_diff1))

        print('num_test_pair = {loss_all.count!s:5s}'.format(loss_all=loss_all))

    return output_date_diff1.avg

