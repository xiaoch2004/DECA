import cv2
import os
import glob
import shutil
import numpy as np
import scipy.io as sio
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import numpy as np
import numpy.matlib as matlab
import scipy.signal as signal
import scipy.fftpack as fft
# from scipy.misc import imresize





def load_decoder(path,shapes):
    nn = sio.loadmat(path)
    weights = []
    biases = []
    for i in range(len(shapes)):
        weights.append(nn['w{}'.format(i+1)].astype('float32'))
        biases.append(nn['b{}'.format(i+1)][0].astype('float32'))
    return weights, biases



def sequencewise_mean_image_subtraction(input, seqlens, axis=0):
    """
    sequence-wise mean image removal
    :param input: input sequences
    :param seqlens: sequence lengths
    :param axis: axis to apply mean image removal
    :return: mean removed input sequences
    """
    mean_subtracted = np.zeros(input.shape, input.dtype)
    start = 0
    end = 0
    for len in seqlens:
        end += len
        seq = input[start:end, :]
        mean_image = np.sum(seq, axis, input.dtype) / len
        mean_subtracted[start:end, :] = seq - mean_image
        start += len
    return mean_subtracted

def normalize_input(input, centralize=True, quantize=False):
    """
    samplewise normalize input
    :param input: input features
    :param centralize: apply 0 mean, std 1
    :param quantize: rescale values to fall between 0 and 1
    :return: normalized input
    """
    def center(item):
        item = item - item.mean()
        std = np.std(item)
        if std == 0:
            std = 1.00
        item = item / std
        return item

    def rescale(item):
        min = np.min(item)
        max = np.max(item)
        dif = max - min
        if dif == 0:
            dif = 1.00
        item = (item - min) / dif
        return item

    for i, item in enumerate(input):
        if centralize:
            input[i] = center(item)
        if quantize:
            input[i] = rescale(item)
    return input

def split_seq_data(X, y, subjects, video_lens, train_ids, val_ids, test_ids, delete_categories=[]):
    """
    Splits the data into training and testing sets
    :param X: input X
    :param y: target y
    :param subjects: array of video -> subject mapping
    :param video_lens: array of video lengths for each video
    :param train_ids: list of subject ids used for training
    :param val_ids: list of subject ids used for validation
    :param test_ids: list of subject ids used for testing
    :return: split data
    """
    # construct a subjects data matrix offset
    X_feature_dim = X.shape[1]
    train_X = np.empty((0, X_feature_dim), dtype='float32')
    val_X = np.empty((0, X_feature_dim), dtype='float32')
    test_X = np.empty((0, X_feature_dim), dtype='float32')
    train_y = np.empty((0,), dtype='int')
    val_y = np.empty((0,), dtype='int')
    test_y = np.empty((0,), dtype='int')
    train_vidlens = np.empty((0,), dtype='int')
    val_vidlens = np.empty((0,), dtype='int')
    test_vidlens = np.empty((0,), dtype='int')
    train_subjects = np.empty((0,), dtype='int')
    val_subjects = np.empty((0,), dtype='int')
    test_subjects = np.empty((0,), dtype='int')
    previous_subject = 1
    subject_video_count = 0
    current_video_idx = 0
    current_data_idx = 0
    populate = False
    for idx, subject in enumerate(subjects):
        if previous_subject == subject:  # accumulate
            subject_video_count += 1
        else:  # populate the previous subject
            populate = True
        if idx == len(subjects) - 1:  # check if it is the last entry, if so populate
            populate = True
            previous_subject = subject
        if populate:
            # slice the data into the respective splits
            end_video_idx = current_video_idx + subject_video_count
            subject_data_len = int(np.sum(video_lens[current_video_idx:end_video_idx]))
            end_data_idx = current_data_idx + subject_data_len
            if previous_subject in train_ids:
                train_X = np.concatenate((train_X, X[current_data_idx:end_data_idx]))
                train_y = np.concatenate((train_y, y[current_data_idx:end_data_idx]))
                train_vidlens = np.concatenate((train_vidlens, video_lens[current_video_idx:end_video_idx]))
                train_subjects = np.concatenate((train_subjects, subjects[current_video_idx:end_video_idx]))
            if previous_subject in val_ids:
                val_X = np.concatenate((val_X, X[current_data_idx:end_data_idx]))
                val_y = np.concatenate((val_y, y[current_data_idx:end_data_idx]))
                val_vidlens = np.concatenate((val_vidlens, video_lens[current_video_idx:end_video_idx]))
                val_subjects = np.concatenate((val_subjects, subjects[current_video_idx:end_video_idx]))
            if previous_subject in test_ids:
                test_X = np.concatenate((test_X, X[current_data_idx:end_data_idx]))
                test_y = np.concatenate((test_y, y[current_data_idx:end_data_idx]))
                test_vidlens = np.concatenate((test_vidlens, video_lens[current_video_idx:end_video_idx]))
                test_subjects = np.concatenate((test_subjects, subjects[current_video_idx:end_video_idx]))
            previous_subject = subject
            current_video_idx = end_video_idx
            current_data_idx = end_data_idx
            subject_video_count = 1
            populate = False
    if len(delete_categories) > 0:
        # Delete some categories from training set and valid set
        vid_idx = 0
        vid_upper_buffer = train_vidlens[0]
        data_idx = 0
        frame_count = 0
        idx = 0
        train_delete_idx = np.empty((0,), dtype=int)
        train_delete_vididx = np.empty((0,), dtype=int)
        while idx < train_y.shape[0]:
            if train_y[idx] not in delete_categories:
                frame_count += 1
                if frame_count >= vid_upper_buffer: # new video
                    vid_idx += 1
                    if vid_idx < train_vidlens.shape[0]:
                        vid_upper_buffer += train_vidlens[vid_idx]
                idx += 1
                continue
            else:
                #print("deleted y: {}, pre: {}, aft: {}, idx:{} vid_idx:{}".format(train_y[idx:idx+train_vidlens[vid_idx]], train_y[idx-1], train_y[idx+train_vidlens[vid_idx]], idx, vid_idx))
                train_delete_idx = np.concatenate((train_delete_idx, np.arange(idx, idx+train_vidlens[vid_idx])))
                train_delete_vididx = np.append(train_delete_vididx, vid_idx)
                idx += train_vidlens[vid_idx]
                frame_count += train_vidlens[vid_idx]
                vid_idx += 1
                if vid_idx < train_vidlens.shape[0]:
                    vid_upper_buffer += train_vidlens[vid_idx]
        #print("total deleted items number:{}, target items number:{}, deleted videos:{}".format(train_delete_idx.shape[0], np.where(train_y>=8)[0].shape[0], train_delete_vididx.shape[0]))
        train_X = np.delete(train_X, train_delete_idx, 0)
        train_y = np.delete(train_y, train_delete_idx)
        train_vidlens = np.delete(train_vidlens, train_delete_vididx)
        train_subjects = np.delete(train_subjects, train_delete_vididx)

        vid_idx = 0
        vid_upper_buffer = val_vidlens[0]
        data_idx = 0
        frame_count = 0
        idx = 0
        val_delete_idx = np.empty((0,), dtype=int)
        val_delete_vididx = np.empty((0,), dtype=int)
        while idx < val_y.shape[0]:
            if val_y[idx] not in delete_categories:
                frame_count += 1
                if frame_count >= vid_upper_buffer: # new video
                    vid_idx += 1
                    if vid_idx < val_vidlens.shape[0]:
                        vid_upper_buffer += val_vidlens[vid_idx]
                idx += 1
                continue
            else:
                #print("deleted y: {}, pre: {}, aft: {}, idx:{} vid_idx:{}".format(val_y[idx:idx+val_vidlens[vid_idx]], val_y[idx-1], val_y[idx+val_vidlens[vid_idx]], idx, vid_idx))
                val_delete_idx = np.concatenate((val_delete_idx, np.arange(idx, idx+val_vidlens[vid_idx])))
                val_delete_vididx = np.append(val_delete_vididx, vid_idx)
                idx += val_vidlens[vid_idx]
                frame_count += val_vidlens[vid_idx]
                vid_idx += 1
                if vid_idx < val_vidlens.shape[0]:
                    vid_upper_buffer += val_vidlens[vid_idx]
        #print("total deleted items number:{}, target items number:{}, deleted videos:{}".format(val_delete_idx.shape[0], np.where(val_y>=8)[0].shape[0], val_delete_vididx.shape[0]))
        val_X = np.delete(val_X, val_delete_idx, 0)
        val_y = np.delete(val_y, val_delete_idx)
        val_vidlens = np.delete(val_vidlens, val_delete_vididx)
        val_subjects = np.delete(val_subjects, val_delete_vididx)

    return train_X, train_y, train_vidlens, train_subjects, \
           val_X, val_y, val_vidlens, val_subjects, \
           test_X, test_y, test_vidlens, test_subjects

def pad_frames(X, total_frames):
    current_frame_num = X.shape[0]
    if total_frames <= current_frame_num:
        return X
    else:
        pad_front_num = int(np.floor((total_frames - current_frame_num)/2))
        pad_end_num = total_frames - current_frame_num - pad_front_num
        X_new = np.zeros((total_frames ,X.shape[1]))
        X_new[:pad_front_num] = np.tile(X[0], [pad_front_num,1])
        X_new[pad_front_num:-pad_end_num] = X
        X_new[-pad_end_num:] = np.tile(X[-1], [pad_end_num,1])
        return X_new

def pad_frame_and_reshape(X, y, vidlens, subjects, size):
    max_length = np.max(vidlens)
    num_videos = len(vidlens)
    X_new = np.zeros((max_length*num_videos, X.shape[1]))
    y_new = np.zeros(num_videos,)
    start_x = 0
    start_new = 0
    for idx, length in enumerate(vidlens):
        end_x = start_x + length
        end_new = start_new + max_length
        X_new[start_new:end_new] = pad_frames(X[start_x:end_x], max_length)
        y_new[idx] = y[start_x]
        start_new = end_new
        start_x = end_x
    return X_new.reshape(num_videos, max_length, size[0], size[1], order='F'), y_new, int(max_length)*np.ones((len(vidlens),), dtype=int), subjects

