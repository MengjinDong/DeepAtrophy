#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 23:15:26 2019


Data Loading and Processing Tutorial
====================================
modified from:
 `Sasank Chilamkurthy <https://chsasank.github.io>`_


"""
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import glob
import csv
from pathlib import Path
from datetime import datetime
import data_aug_cpu
from itertools import permutations

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


class LongitudinalDataset3D(Dataset):
    """ AD longitudinal dataset."""

    def __init__(self, root_dir, groups, csv_out, stages, augment=None,
                 max_angle = 0, rotate_prob = 0.5, output_size = [1, 1, 1]):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            groups: a list specifying the stages included in the dataset
                    (0 = A- NC,   1 = A+ NC,   2 = A- eMCI, 3 = A+ eMCI,
                     4 = A- lMCI, 5 = A+ lMCI, 6 = A- AD,   7 = A+ AD   )
            csv_out: output training or test statistics
            augment: a list specifying augmentation methods applied to this dataset
            max_angle: the maximum angles applied in random rotation
            rotate_prob: probability of applying random rotation
            output_size: the desired output size after random cropping,
                         in this experiment it is [48, 80, 64]
        """
        
        self.root_dir = root_dir
        self.groups = groups
        self.csv_out = csv_out
        self.augment = augment
        self.date_format = "%Y-%m-%d"
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob
        self.output_size = output_size

        print("stages = ", stages)
            
        if os.path.exists(csv_out):
            os.remove(csv_out)


        with open(csv_out, 'w') as filename:
            for group in groups:
                for stage in stages:
                    print(group, stage)
                    wr = csv.writer(filename, lineterminator='\n')
                    subject_list = self.root_dir + "/" + group + "/subject_list" + stage + ".csv"
                    if os.path.exists(subject_list):
                        with open(subject_list) as f:
                            for subjectID in f:
                                subjectID = subjectID.strip('\n')
                                for side in ["left", "right"]:
                                    # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                                    scan_list = glob.glob(self.root_dir + "/" + group + "/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.npy")
                                    perm = permutations(range(0, len(scan_list)), 2)
                                    for bl_item1, bl_item2 in list(perm):
                                        bl_item1 = scan_list[bl_item1]
                                        bl_item2 = scan_list[bl_item2]

                                        fu_item1 = Path(bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                        mask_item1 = Path(bl_item1.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))
                                        if fu_item1.exists() and mask_item1.exists():
                                            fname1 = bl_item1.split("/")[-1]
                                            bl_time1 = datetime.strptime(fname1.split("_")[3], self.date_format)
                                            fu_time1 = datetime.strptime(fname1.split("_")[4], self.date_format)
                                            date_diff1 = (fu_time1 - bl_time1).days
                                            label_date_diff1 = float(np.greater(date_diff1, 0))
                                        else:    continue

                                        fu_item2 = Path(bl_item2.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                        mask_item2 = Path(
                                            bl_item2.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))

                                        if fu_item2.exists() and mask_item2.exists():
                                            fname2 = bl_item2.split("/")[-1]
                                            bl_time2 = datetime.strptime(fname2.split("_")[3], self.date_format)
                                            fu_time2 = datetime.strptime(fname2.split("_")[4], self.date_format)
                                            date_diff2 = (fu_time2 - bl_time2).days
                                            label_date_diff2 = float(np.greater(date_diff2, 0))
                                        else:    continue

                                        date_diff_ratio = abs(date_diff1 / date_diff2)

                                        if date_diff_ratio < 0.5:
                                            label_time_interval = 0
                                        elif date_diff_ratio < 1:
                                            label_time_interval = 1
                                        elif date_diff_ratio < 2:
                                            label_time_interval = 2
                                        else:
                                            label_time_interval = 3

                                        if abs(date_diff1) > abs(date_diff2):
                                            if date_diff1 > 0:
                                                if bl_time2 <= fu_time1 and bl_time2 >= bl_time1 and fu_time2 <= fu_time1 and fu_time2 >= bl_time1:
                                                    wr.writerow(
                                                        [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage, date_diff1, date_diff2,
                                                         label_date_diff1, label_date_diff2, label_time_interval, subjectID, side])
                                            else:
                                                if bl_time2 <= bl_time1 and bl_time2 >= fu_time1 and fu_time2 <= bl_time1 and fu_time2 >= fu_time1:
                                                    wr.writerow(
                                                        [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage, date_diff1, date_diff2,
                                                         label_date_diff1, label_date_diff2, label_time_interval, subjectID, side])

                                        elif abs(date_diff1) < abs(date_diff2):
                                            if date_diff2 > 0:
                                                if bl_time1 <= fu_time2 and bl_time1 >= bl_time2 \
                                                        and fu_time1 <= fu_time2 and fu_time1 >= bl_time2:
                                                    wr.writerow(
                                                        [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                         date_diff1, date_diff2,
                                                         label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                         side])
                                            else:
                                                if bl_time1 <= bl_time2 and bl_time1 >= fu_time2 \
                                                        and fu_time1 <= bl_time2 and fu_time1 >= fu_time2:
                                                    wr.writerow(
                                                        [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                         date_diff1, date_diff2,
                                                         label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                         side])


        with open(csv_out, 'r') as f:
            reader = csv.reader(f)
            self.image_frame = list(reader)
            

        
    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):

        random_bl1 = self.image_frame[idx][0]
        random_bl2 = self.image_frame[idx][1]
        bl_time1 = self.image_frame[idx][2]
        fu_time1 = self.image_frame[idx][3]
        bl_time2 = self.image_frame[idx][4]
        fu_time2 = self.image_frame[idx][5]
        stage = self.image_frame[idx][6]

        date_diff1 = float(self.image_frame[idx][7])
        date_diff2 = float(self.image_frame[idx][8])

        label_date_diff1 = float(self.image_frame[idx][9])
        label_date_diff2 = float(self.image_frame[idx][10])

        label_time_interval = float(self.image_frame[idx][11])
        subjectID = self.image_frame[idx][12]
        side = self.image_frame[idx][13]

        random_bl1 = ''.join(random_bl1)
        random_fu1 = random_bl1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
        random_mask1 = random_bl1.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1)

        random_bl2 = ''.join(random_bl2)
        random_fu2 = random_bl2.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
        random_mask2 = random_bl2.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1)

        bl_cube1 = np.load(random_bl1)
        fu_cube1 = np.load(random_fu1)
        mask_cube1 = np.load(random_mask1)

        bl_cube2 = np.load(random_bl2)
        fu_cube2 = np.load(random_fu2)
        mask_cube2 = np.load(random_mask2)

        # print("len_image_list = ", len(image_list))

        if 'normalize' in self.augment:
            [bl_cube1, fu_cube1] = data_aug_cpu.Normalize([bl_cube1, fu_cube1])
            [bl_cube2, fu_cube2] = data_aug_cpu.Normalize([bl_cube2, fu_cube2])

        image_list1 = [bl_cube1, fu_cube1, mask_cube1]
        image_list2 = [bl_cube2, fu_cube2, mask_cube2]

        # flip: left right 1/6; up down 1/6; front back 1/6; no flipping 1/6
        if 'flip' in self.augment:
            image_list1 = data_aug_cpu.randomFlip3d(image_list1)
            image_list2 = data_aug_cpu.randomFlip3d(image_list2)

        # Random 3D rotate image
        if 'rotate' in self.augment and self.max_angle > 0:
            image_list1 = data_aug_cpu.randomRotation3d(image_list1, self.max_angle, self.rotate_prob)
            image_list2 = data_aug_cpu.randomRotation3d(image_list2, self.max_angle, self.rotate_prob)

        if 'crop' in self.augment:
            image_list1 = data_aug_cpu.randomCrop3d(image_list1, self.output_size)
            image_list2 = data_aug_cpu.randomCrop3d(image_list2, self.output_size)

        bl_cube1 = image_list1[0]
        fu_cube1 = image_list1[1]
        bl_cube2 = image_list2[0]
        fu_cube2 = image_list2[1]


        bl_cube1 = torch.from_numpy(bl_cube1[np.newaxis, :, :, :].copy()).float()
        bl_cube2 = torch.from_numpy(bl_cube2[np.newaxis, :, :, :].copy()).float()
        fu_cube1 = torch.from_numpy(fu_cube1[np.newaxis, :, :, :].copy()).float()
        fu_cube2 = torch.from_numpy(fu_cube2[np.newaxis, :, :, :].copy()).float()

        sample = {}

        # wrap up prepared images for network input
        input_im1 = np.concatenate(\
            (bl_cube1, fu_cube1, bl_cube2, fu_cube2), axis=0)
        sample['image'] = input_im1

        sample["bl_fname1"] = random_bl1
        sample["bl_fname2"] = random_bl2

        sample["bl_time1"] = bl_time1
        sample["bl_time2"] = bl_time2

        sample["fu_time1"] = fu_time1
        sample["fu_time2"] = fu_time2

        sample['stage'] = stage

        sample['date_diff1'] = date_diff1
        sample['date_diff2'] = date_diff2

        sample['label_date_diff1'] = torch.from_numpy(np.array(label_date_diff1).copy()).float()
        sample['label_date_diff2'] = torch.from_numpy(np.array(label_date_diff2).copy()).float()
        sample['label_time_interval'] = torch.from_numpy(np.array(label_time_interval).copy()).float()

        sample['subjectID'] = subjectID
        sample['side'] = side
        sample['date_diff_ratio'] = torch.from_numpy(np.array(abs(date_diff1/date_diff2)).copy()).float()

        return sample


class LongitudinalDataset3DPair(Dataset):
    """ AD longitudinal dataset."""

    def __init__(self, root_dir, groups, csv_out, stages, augment=None,
                 max_angle=0, rotate_prob=0.5, output_size=[1, 1, 1]):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_in: blindeddays.txt, used to extract date difference
            csv_out: output training or test statistics
        """

        self.root_dir = root_dir
        self.groups = groups
        self.csv_out = csv_out
        self.augment = augment
        self.date_format = "%Y-%m-%d"
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob
        self.output_size = output_size

        print("stages = ", stages)

        if os.path.exists(csv_out):
            os.remove(csv_out)

        with open(csv_out, 'w') as filename:
            for group in groups:
                for stage in stages:
                    print(group, stage)
                    wr = csv.writer(filename, lineterminator='\n')

                    subject_list = self.root_dir + "/" + group + "/subject_list" + stage + ".csv"
                    if os.path.exists(subject_list):
                        with open(subject_list) as f:
                            for subjectID in f:
                                subjectID = subjectID.strip('\n')
                                for side in ["left", "right"]:
                                    # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                                    scan_list = glob.glob(
                                        self.root_dir + "/" + group + "/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.npy")

                                    for bl_item1 in list(scan_list):

                                        fu_item1 = Path(
                                            bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                        if fu_item1.exists():
                                            fname1 = bl_item1.split("/")[-1]
                                            bl_time1 = datetime.strptime(fname1.split("_")[3], self.date_format)
                                            fu_time1 = datetime.strptime(fname1.split("_")[4], self.date_format)
                                            date_diff1 = (fu_time1 - bl_time1).days
                                            label_date_diff1 = float(np.greater(date_diff1, 0))
                                        else:
                                            continue
                                        wr.writerow(
                                            [bl_item1, bl_time1, fu_time1, stage,
                                             date_diff1,
                                             label_date_diff1, subjectID,
                                             side])

        with open(csv_out, 'r') as f:
            reader = csv.reader(f)
            self.image_frame = list(reader)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):

        random_bl1 = self.image_frame[idx][0]
        bl_time1 = self.image_frame[idx][1]
        fu_time1 = self.image_frame[idx][2]
        stage = self.image_frame[idx][3]

        date_diff1 = float(self.image_frame[idx][4])

        label_date_diff1 = float(self.image_frame[idx][5])

        subjectID = self.image_frame[idx][6]
        side = self.image_frame[idx][7]

        random_bl1 = ''.join(random_bl1)
        random_fu1 = random_bl1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
        random_mask1 = random_bl1.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1)

        bl_cube1 = np.load(random_bl1)
        fu_cube1 = np.load(random_fu1)
        mask_cube1 = np.load(random_mask1)

        if 'normalize' in self.augment:
            [bl_cube1, fu_cube1] = data_aug_cpu.Normalize([bl_cube1, fu_cube1])

        image_list1 = [bl_cube1, fu_cube1, mask_cube1]

        # flip: left right 1/6; up down 1/6; front back 1/6; no flipping 1/6
        if 'flip' in self.augment:
            image_list1 = data_aug_cpu.randomFlip3d(image_list1)

        # Random 3D rotate image
        if 'rotate' in self.augment and self.max_angle > 0:
            image_list1 = data_aug_cpu.randomRotation3d(image_list1, self.max_angle, self.rotate_prob)

        if 'crop' in self.augment:
            image_list1 = data_aug_cpu.randomCrop3d(image_list1, self.output_size)

        bl_cube1 = image_list1[0]
        fu_cube1 = image_list1[1]

        bl_cube1 = torch.from_numpy(bl_cube1[np.newaxis, :, :, :].copy()).float()
        fu_cube1 = torch.from_numpy(fu_cube1[np.newaxis, :, :, :].copy()).float()

        sample = {}

        # wrap up prepared images for network input
        input_im1 = np.concatenate( \
            (bl_cube1, fu_cube1), axis=0)

        sample['image'] = input_im1
        sample["bl_fname1"] = random_bl1
        sample["bl_time1"] = bl_time1
        sample["fu_time1"] = fu_time1
        sample['stage'] = stage
        sample['date_diff1'] = date_diff1
        sample['label_date_diff1'] = torch.from_numpy(np.array(label_date_diff1).copy()).float()
        sample['subjectID'] = subjectID
        sample['side'] = side

        return sample
