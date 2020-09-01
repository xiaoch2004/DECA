import cv2
import os
import glob
import shutil
import numpy as np
import pickle
import scipy.io as sio
from collections import OrderedDict
import pickle
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from models import *
from data_model_loader import *
from utils import *
from loss_datagen import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default="0", help='Please write which gpu to use 0 is for cuda:0 and so one \
                        if you want to use CPU specify -1 ')

    parser.add_argument('--one_stream_view', type=str, default="1", help='1stream view \
    to train like 1, 2 and so on .   \
    1 --> 0deg, 2 --> 30deg, 3 --> 45deg, 4 --> 60deg, 5 --> 90deg')

    parser.add_argument('--save_path', type=str, default='../results/lipnet', help='path where \
    saved_model loss, acc and predictions with true lables .npy files will be saved for both test train and val ')

    parser.add_argument('--pretrained_encoder_path', type=str, default='../pretrained_encoder/', help='path where pretrained encoder \
    is found ')

    parser.add_argument('--data_pickle_path', type=str,  default='../data/oulu_processed_pad.pkl', help='path where \
    data in OuluVS2 video format and processed by pre_process_oulu.py  is stored ')


    parser.add_argument('--iteration', type=int, default=1, help='if running multiple times add iteration to distinguish between \
                        different iterations ( can write a bash script and run this \
                        multiple times and save all the output to different folder by setting this to iteration no ')

    parser.add_argument('--num_epoch', type=int, default=20, help='no of epochs ')
    parser.add_argument('--num_classes', type=int, default=10, help='no of classes ')



    args = parser.parse_args()

    view =args.one_stream_view
    view=int(view)

    train_views = [1,2,3]
    test_views = [4,5]
    string = ""
    for i in range(len(train_views)):
        if i == 0:
            string += str(train_views[i])
        else:
            string += (","+ str(train_views[i]))

    if view == -1:
        save_path=args.save_path+"/lipnet_train_views_"+string
    else:
        save_path=args.save_path+"/lipnet_train_views_"+str(view)


    if args.iteration:
        save_path+="_iteration_"+str(args.iteration)

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_path+"/models",exist_ok=True)
    os.makedirs(save_path+"/predictions_truelabels",exist_ok=True)


# for setting gpu and Cpu
#
# either this
# gpu_no=0
# torch.cuda.set_device(gpu_no)
#and use .cuda()
#
# or this
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# and replacing every .cuda() with .to(device)

    gpu=args.gpu
    device_name='cpu'

    if gpu>=0:
        device_name='cuda:'+str(gpu)
    device = torch.device(device_name)


    shape= [2000,1000,500,50]
    nonlinearities= ["rectify","rectify","rectify","linear"]
    # preprocessing options
    reorderdata= False
    diffimage= False
    meanremove= True
    samplewisenormalize= True
    featurewisenormalize= False

    # [lstm_classifier]
    windowsize= 3
    lstm_size= 450
    output_classes= args.num_classes

    matlab_target_offset= True

    #[training]
    learning_rate= 0.0003
#     num_epoch= 40
    num_epoch=args.num_epoch
    epochsize= 105*2
    batchsize= 10

    #35*30=1050
    #35*39=1365

    print("no of epochs:",num_epoch)

    train_subject_ids = [1,2,3,5,7,10,11,12,14,16,17,18,19,20,21,23,24,25,27,28,31,32,33,35,36,37,39,40,41,42,45,46,47,48,53]
    val_subject_ids = [4,13,22,38,50]
    test_subject_ids = [6,8,9,15,26,30,34,43,44,49,51,52]

    delete_categories = []

    pretrained_encoder_path=args.pretrained_encoder_path
    data_pickle_path=args.data_pickle_path

    with open(data_pickle_path, "rb") as myFile:
        data_processed= pickle.load(myFile)

    imagesize = [44,50]
    train_data = {}
    train_data['dataMatrix'] = np.concatenate(tuple([data_processed[i]['dataMatrix'] for i in train_views]))
    train_data['targetsVec'] = np.concatenate(tuple([data_processed[i]['targetsVec'] for i in train_views]))
    train_data['subjectsVec'] = np.concatenate(tuple([data_processed[i]['subjectsVec'] for i in train_views]))
    train_data['videoLengthVec'] = np.concatenate(tuple([data_processed[i]['videoLengthVec'] for i in train_views]))

    test_data = {}
    test_data['dataMatrix'] = np.concatenate(tuple([data_processed[i]['dataMatrix'] for i in test_views]))
    test_data['targetsVec'] = np.concatenate(tuple([data_processed[i]['targetsVec'] for i in test_views]))
    test_data['subjectsVec'] = np.concatenate(tuple([data_processed[i]['subjectsVec'] for i in test_views]))
    test_data['videoLengthVec'] = np.concatenate(tuple([data_processed[i]['videoLengthVec'] for i in test_views]))

    print('constructing end to end model...\n')

    pretrained_encoder_isTrue=False
    pre_trained_encoder_variables = 0

    #pretrained_encoder_isTrue=True
    #pre_trained_encoder_variables = load_decoder(stream1, shape)

    input_dimensions = 44*50

    network=deltanet_majority_vote(device, pretrained_encoder_isTrue, \
                pre_trained_encoder_variables, shape, nonlinearities, input_dimensions, windowsize, lstm_size, args.num_classes)

    # network = LipNet(dropout_p=0.5)


    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    print("Network Architecture",network)
    print("\n")
    for key, value in network.state_dict().items():
        print(key,value.shape)
    print("\nModel on GPU",next(network.parameters()).is_cuda)


    train_data_matrix = train_data['dataMatrix'].astype('float32')
    train_targets_vec = train_data['targetsVec'].reshape((-1,))
    train_subjects_vec = train_data['subjectsVec'].reshape((-1,))
    train_vidlen_vec = train_data['videoLengthVec'].reshape((-1,))

    test_data_matrix = test_data['dataMatrix'].astype('float32')
    test_targets_vec = test_data['targetsVec'].reshape((-1,))
    test_subjects_vec = test_data['subjectsVec'].reshape((-1,))
    test_vidlen_vec = test_data['videoLengthVec'].reshape((-1,))

    print("Shape of train_data_matrix: ",train_data_matrix.shape)
    print("Shape of test_data_matrix: ",test_data_matrix.shape)


    matlab_target_offset=True
    meanremove= True
    samplewisenormalize= True

    #convert to 0 order
    train_X, train_y, train_vidlens, train_subjects, \
    val_X, val_y, val_vidlens, val_subjects ,_,_,_,_ \
     = split_seq_data(train_data_matrix, train_targets_vec, train_subjects_vec, train_vidlen_vec, train_subject_ids, val_subject_ids, test_subject_ids, delete_categories=delete_categories)


    _,_,_,_,_,_,_,_, \
    test_X, test_y, test_vidlens, test_subjects =split_seq_data(test_data_matrix, test_targets_vec, test_subjects_vec, test_vidlen_vec, train_subject_ids, val_subject_ids, test_subject_ids, delete_categories=delete_categories)

    if matlab_target_offset:
        train_y -= 1
        val_y -= 1
        test_y -= 1

    if meanremove:
        train_X = sequencewise_mean_image_subtraction(train_X, train_vidlens)
        val_X = sequencewise_mean_image_subtraction(val_X, val_vidlens)
        test_X = sequencewise_mean_image_subtraction(test_X, test_vidlens)

    if samplewisenormalize:
        train_X = normalize_input(train_X)
        val_X = normalize_input(val_X)
        test_X = normalize_input(test_X)

    train_X, train_y, train_vidlens, train_subjects = pad_frame_and_reshape(train_X, train_y, train_vidlens, train_subjects, [44,50])