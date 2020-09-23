#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:28:29 2018

@author: mengjin
ResNet training from python example

"""
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import shutil
import time
import warnings
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import data_loading_longi_3d as long
import utility as util
import longi_models
from tensorboardX import SummaryWriter
from generate_model import generate_model
from param import parse_opts
import csv

import matplotlib
matplotlib.use('qt5agg') # MUST BE CALLED BEFORE IMPORTING plt, or qt5agg

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

args = parse_opts()

best_prec1 = 0

# sample_size = [48, 80, 64] # 245760
sample_size = [args.input_D, args.input_H, args.input_W]

# Use a tool at comet.com to keep track of parameters used
hyper_params = vars(args)

experiment = Experiment(api_key="8pO1ROZQ8g3OqjqYpnZX7DCdR",
                        project_name="UNET3D", workspace="wuxiaoxiao")
# End of using comet


print("ROOT is ", args.ROOT)

if not os.path.exists(args.ROOT + '/Model'):
    os.makedirs(args.ROOT + '/Model')

if not os.path.exists(args.ROOT + '/log'):
    os.makedirs(args.ROOT + '/log')

def main():
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_prec1, sample_size
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        print("Current Device is ", torch.cuda.get_device_name(0))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model2:
    if args.pretrained:
        print("=> Model (date_diff): using pre-trained model '{}_{}'".format(args.model, args.model_depth))
        pretrained_model = models.__dict__[args.arch](pretrained=True)
    else:
        if args.model_type == 2:
            print("=> Model (date_diff regression): creating model '{}_{}'".format(args.model, args.model_depth))
            pretrained_model = generate_model(args) # good for resnet
            save_folder = "{}/Model/{}{}".format(args.ROOT, args.model, args.model_depth)

    model = longi_models.ResNet_interval(pretrained_model, args.num_date_diff_classes, args.num_reg_labels)

    criterion0 = torch.nn.CrossEntropyLoss().cuda(args.gpu) # for STO loss
    criterion1 = torch.nn.CrossEntropyLoss().cuda(args.gpu) # for RISI loss

    criterion = [criterion0, criterion1]
    start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # all models optionally resume from a checkpoint
    if args.resume_all:
        if os.path.isfile(args.resume_all):
            print("=> Model_all: loading checkpoint '{}'".format(args.resume_all))
            checkpoint = torch.load(args.resume_all, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            start_epoch = checkpoint['epoch']
            print("=> Model_all: loaded checkpoint '{}' (epoch {})"
              .format(args.resume_all, checkpoint['epoch']))
    

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            
    print("batch-size = ", args.batch_size)
    print("epochs = ", args.epochs)
    print("range-weight (weight of range loss) = ", args.range_weight)
    cudnn.benchmark = True
    print(model)
    
    # Data loading code
    traingroup = ["train"]
    evalgroup = ["eval"]
    testgroup = ["test"]

    train_augment = ['normalize', 'flip', 'crop'] # 'rotate',
    test_augment = ['normalize', 'crop']
    eval_augment = ['normalize', 'crop']

    train_stages = args.train_stages.strip('[]').split(', ')
    test_stages = args.test_stages.strip('[]').split(', ')
    eval_stages = args.eval_stages.strip('[]').split(', ')
    #############################################################################
    # test-retest analysis

    trt_stages = args.trt_stages.strip('[]').split(', ')
        
    model_pair = longi_models.ResNet_pair(model.modelA, args.num_date_diff_classes)
    torch.cuda.set_device(args.gpu)
    model_pair = model_pair.cuda(args.gpu)

    if args.resume_all:
        model_name = args.resume_all[:-8]

    else:
        model_name = save_folder + "_" + time.strftime("%Y-%m-%d_%H-%M")+ \
                     traingroup[0] + '_' + args.train_stages.strip('[]').replace(', ', '')

    data_name = args.datapath.split("/")[-1]

    log_name = (args.ROOT
                + "/log/"
                + args.model+ str(args.model_depth)
                + "/" + data_name
                + "/" + time.strftime("%Y-%m-%d_%H-%M"))
    writer = SummaryWriter(log_name)

    trt_dataset = long.LongitudinalDataset3DPair(
        args.datapath,
        testgroup,
        args.datapath + "/test_retest_list.csv",
        trt_stages,
        test_augment,
        args.max_angle,
        args.rotate_prob,
        sample_size)

    trt_loader = torch.utils.data.DataLoader(
        trt_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print("\nEvaluation on Test-Retest Set: ")

    util.validate_pair(trt_loader,
                       model_pair,
                       criterion,
                       model_name + "_test_retest",
                       args.epochs,
                       writer,
                       args.print_freq)

    ##########################################################################

    train_dataset = long.LongitudinalDataset3D(
            args.datapath,
            traingroup,
            args.datapath + "/train_list.csv",
            train_stages,
            train_augment, # advanced transformation: add random rotation
            args.max_angle,
            args.rotate_prob,
            sample_size)
    
    eval_dataset = long.LongitudinalDataset3D(
            args.datapath,
            evalgroup,
            args.datapath + "/eval_list.csv",
            eval_stages,
            eval_augment,
            args.max_angle,
            args.rotate_prob,
            sample_size)

    test_dataset = long.LongitudinalDataset3D(
            args.datapath,
            testgroup,
            args.datapath + "/test_list.csv",
            test_stages,
            test_augment,
            args.max_angle,
            args.rotate_prob,
            sample_size)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            # sampler = train_sampler,
            num_workers=args.workers, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers, pin_memory=True)

    data_name = args.datapath.split("/")[-1]

    if args.resume_all:
        model_name = args.resume_all[:-8]

    else:
        model_name = save_folder + "_" + time.strftime("%Y-%m-%d_%H-%M")+ \
                     traingroup[0] + '_' + args.train_stages.strip('[]').replace(', ', '')

    # Use a tool at comet.com to keep track of parameters used
    # log model name, loss, and optimizer as well
    hyper_params["loss"] = criterion
    hyper_params["optimizer"] = optimizer
    hyper_params["model_name"] = model_name
    hyper_params["save_folder"] = save_folder
    experiment.log_parameters(hyper_params)
    # End of using comet

    log_name = (args.ROOT
                + "/log/"
                + args.model+ str(args.model_depth)
                + "/" + data_name
                + "/" + time.strftime("%Y-%m-%d_%H-%M"))
    writer = SummaryWriter(log_name)

    if args.evaluate:
        print("\nEVALUATE before starting training: ")
        util.validate(eval_loader,
                      model,
                      criterion,
                      model_name + "_eval",
                      writer=writer,
                      range_weight = args.range_weight)

    # training the model
    if start_epoch < args.epochs - 1:
        print("\nTRAIN: ")
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            util.adjust_learning_rate(optimizer, epoch, args.lr)

            # train for one epoch
            util.train(train_loader,
                       model,
                       criterion,
                       optimizer,
                       epoch,
                       sample_size,
                       args.print_freq,
                       writer,
                       range_weight = args.range_weight)

            # evaluate on validation set
            if epoch % args.eval_freq == 0:
                csv_name = model_name + "_eval.csv"
                if os.path.isfile(csv_name):
                    os.remove(csv_name)
                prec = util.validate(eval_loader,
                                     model,
                                     criterion,
                                     model_name + "_eval",
                                     epoch,
                                     writer,
                                     range_weight = args.range_weight)

                if args.early_stop:

                    early_stopping = util.EarlyStopping(patience=args.patience, tolerance=args.tolerance)

                    early_stopping({
                                'epoch': epoch + 1,
                                'arch1': args.arch1,
                                'arch2': args.model2 + str(args.model2_depth),
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                            },
                            prec,
                            model_name)

                    print("=" * 50)

                    if early_stopping.early_stop:
                        print("Early stopping at epoch", epoch, ".")
                        break

                else:
                    # remember best prec@1 and save checkpoint
                    is_best = prec > best_prec1
                    best_prec1 = max(prec, best_prec1)
                    util.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.model + str(args.model_depth),
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    },
                    is_best,
                    model_name)

    if args.test:
        print("\nTEST: ")
        util.validate(test_loader,
                      model,
                      criterion,
                      model_name + "_test",
                      args.epochs,
                      writer,
                      range_weight=args.range_weight)

        print("\nEvaluation on Train Set: ")
        util.validate(train_loader,
                      model,
                      criterion,
                      model_name + "_train",
                      args.epochs,
                      writer,
                      range_weight=args.range_weight)

    #############################################################################################################

    # test on only the basic sub-network (STO loss)
    model_pair = longi_models.ResNet_pair(model.modelA, args.num_date_diff_classes)
    torch.cuda.set_device(args.gpu)
    model_pair = model_pair.cuda(args.gpu)

    if args.test_pair:

        train_pair_dataset = long.LongitudinalDataset3DPair(
                args.datapath,
                traingroup,
                args.datapath + "/train_pair_list.csv",
                train_stages,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)

        train_pair_loader = torch.utils.data.DataLoader(
                train_pair_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)

        print("\nEvaluation on Train Pair Set: ")

        util.validate_pair(train_pair_loader,
                      model_pair,
                      criterion,
                      model_name + "_train_pair_update",
                      args.epochs,
                      writer,
                      args.print_freq)

        test_pair_dataset = long.LongitudinalDataset3DPair(
                args.datapath,
                testgroup,
                args.datapath + "/test_pair_list.csv",
                test_stages,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)

        test_pair_loader = torch.utils.data.DataLoader(
                test_pair_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)

        print("\nEvaluation on Test Pair Set: ")

        util.validate_pair(test_pair_loader,
                      model_pair,
                      criterion,
                      model_name + "_test_pair_update",
                      args.epochs,
                      writer,
                      args.print_freq)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == '__main__':
    main()
