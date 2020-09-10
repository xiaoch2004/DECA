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

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

from models import *
from data_model_loader import *
from utils import *
from loss_datagen import *

def train(device, model, optimizer, datagen, epoch, epochsize):
    model.train()

    tloss=0
    for i in range(epochsize):
        X, y, m = next(datagen)
        # repeat targets based on max sequence len
        y = y.reshape((-1, 1))
        y = y.repeat(m.shape[-1], axis=-1)

        X = np.transpose(X[:,:,None,:,:], (0,2,1,3,4))
        X=torch.from_numpy(X).float().to(device)
        y=torch.from_numpy(y).long().to(device)
        #vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
        m=torch.from_numpy(m).float().to(device)

        optimizer.zero_grad()

        #print(X.is_cuda)
        #print(next(model.parameters()).is_cuda)
        #print(X.shape,X_s2.shape)
        #output,ordered_idx = model(X, vid_lens_batch)
        output = model(X)


        #target=torch.index_select(y,0,ordered_idx)
        #m=torch.index_select(m,0,ordered_idx)

        #loss = temporal_ce_loss(output, target,m)
        loss = temporal_ce_loss(output, y, m)

        tloss+=loss.item()

        loss.backward()

        #gradient clip, if needed add
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, tloss*1.0/epochsize))


def test(device, model,name,epoch, X_val, y_val, \
         mask_val):
    model.eval()

    y_val = y_val.reshape((-1, 1))
    y_val = y_val.repeat(mask_val.shape[-1], axis=-1)


    X_val = np.transpose(X_val[:,:,None,:,:], (0,2,1,3,4))
    X_val = torch.from_numpy(X_val).float().to(device)
    y_val=torch.from_numpy(y_val).long().to(device)
    # vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
    mask_val=torch.from_numpy(mask_val).to(device)

    #output,ordered_idx = model(X, vid_lens_batch)
    output = model(X_val)

    #y_val=torch.index_select(y_val,0,ordered_idx)
    #mask_val=torch.index_select(mask_val,0,ordered_idx)
    #y=torch.from_numpy(y).long().to(device)
    #target=torch.index_select(y,0,ordered_idx)


    #m=mask_val.float()

    loss = temporal_ce_loss(output, y_val, mask_val)

    output=output.cpu().detach().numpy()

    seq_len=output.shape[1]
    y_val=y_val[:,0].contiguous()
    mask_val=mask_val[:,:seq_len].contiguous()

    mask_val=mask_val.cpu().numpy()
    y_val=y_val.cpu().numpy()

    num_classes = output.shape[-1]

    ix = np.zeros((X_val.shape[0],), dtype='int')
    ix_top3 = np.zeros((X_val.shape[0], 3), dtype='int')
    seq_lens = np.sum(mask_val, axis=-1).astype(np.int)



    # for each example, we only consider argmax of the seq len
    votes = np.zeros((num_classes,), dtype='int')
    for i, eg in enumerate(output):
        predictions = np.argmax(eg[:seq_lens[i]], axis=-1)
#         print(predictions.shape)
        for cls in range(num_classes):
            count = (predictions == cls).sum(axis=-1)
            votes[cls] = count
        ix[i] = np.argmax(votes)
        ix_top3[i] = torch.topk(torch.from_numpy(votes),3)[1].numpy()


    c = ix == y_val
#     print(c,ix[:10],y_val[:10])
    classification_rate = np.sum(c == True) / float(len(c))

    c_top3 = np.zeros((X_val.shape[0],), dtype='int')
    for idx in range(X_val.shape[0]):
        if y_val[idx] in ix_top3[idx]:
            c_top3[idx] = 1
    classification_rate_top3 = np.sum(c_top3) / float(len(c_top3))


    print('{} Epoch: {} \tTop3Acc: {:.6f} \tAcc: {:.6f} \tLoss: {:.6f}'.format(name,
                    epoch,classification_rate_top3,classification_rate,loss.item() ))

    preds = ix
    true_labels = y_val

    return classification_rate, classification_rate_top3, loss.item(), preds, true_labels


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
    parser.add_argument('--bigru_size', type=int, default=512, help='size of bi-GRU')



    args = parser.parse_args()

    view =args.one_stream_view
    view=int(view)

    train_views = [1,2,3,4,5]
    test_views = [1,2,3,4,5]
    string = ""
    for i in range(len(train_views)):
        if i == 0:
            string += str(train_views[i])
        else:
            string += (","+ str(train_views[i]))

    save_path=args.save_path+"/train_views_"+string


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
    learning_rate= 0.0001
#     num_epoch= 40
    num_epoch=args.num_epoch
    epochsize = 126
    #batchsize = 150
    batchsize = 10

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

    # network=deltanet_majority_vote(device, pretrained_encoder_isTrue, \
    #            pre_trained_encoder_variables, shape, nonlinearities, input_dimensions, windowsize, lstm_size, args.num_classes)

    network = LipNet()

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
    val_X, val_y, val_vidlens, val_subjects = pad_frame_and_reshape(val_X, val_y, val_vidlens, val_subjects, [44,50])
    test_X, test_y, test_videns, test_subjects = pad_frame_and_reshape(test_X, test_y, test_vidlens, test_subjects, [44,50])

    print("Shape of training Dataset: ", train_X.shape)
    print("Shape of validate Dataset: ", val_X.shape)
    print("Shape of test Dataset: ", test_X.shape)


    #datagen = gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=batchsize)
    datagen = gen_cnn_batch(train_X, train_y, batchsize=batchsize, shuffle=True)
    test_datagen = gen_cnn_batch(test_X, test_y, batchsize=len(test_vidlens), shuffle=False)
    val_datagen = gen_cnn_batch(val_X, val_y, batchsize=len(val_vidlens), shuffle=False)

    X_val, y_val, mask_val = next(val_datagen)
    X_test, y_test, mask_test = next(test_datagen)

    loss_list_train=[]
    loss_list_test=[]
    loss_list_val=[]

    acc_list_train=[]
    acc_list_test=[]
    acc_list_val=[]
    tstart = time.time()

    for epoch in range(1, num_epoch+1):
        time_start = time.time()
        train(device, network, optimizer, datagen, epoch, epochsize)
        train_acc, train_loss,  predictions_train, true_label_train=(0,0,0,0)

        print("Time Taken for epoch:",epoch," ",time.time()-time_start)

        if epoch%5 == 0 or epoch > num_epoch-5:
            val_acc, val_acc_top3, val_loss, predictions_val, true_label_val = test(device, network,"val",epoch, \
                                    X_val,\
                                                                y_val, mask_val)
            test_acc, test_acc_top3, test_loss, predictions_test, true_label_test = test(device, network,"test",epoch, \
                                X_test,\
                                                                y_test, mask_test)

            print("Saved model dict at epoch:",epoch," at path ",save_path)
            state={
                'epoch': epoch + 1,
                'state_dict': network.state_dict(),

                'test_loss':             test_loss,
                'test_accuracy':         test_acc,
                'test_predictions':      predictions_test,
                'test_true_label':       true_label_test,

                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_predictions':      predictions_train,
                'train_true_label':       true_label_train,

                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_predictions':      predictions_val,
                'val_true_label':       true_label_val,

#                 'optimizer' : optimizer.state_dict(),
            }
            fname=save_path+"/models/epoch_"+str(epoch)+"_checkpoint.pth.tar"
            torch.save(state,fname)


            # np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_prediction_train.npy", predictions_train)
            np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_prediction_test.npy", predictions_test)
            np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_prediction_val.npy", predictions_val)
            # np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_true_label_train.npy", true_label_train)
            np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_true_label_test.npy", true_label_test)
            np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_true_label_val.npy", true_label_val)

            # loss_list_train.append(train_loss)
            loss_list_test.append(test_loss)
            loss_list_val.append(val_loss)

            # acc_list_train.append(train_acc)
            acc_list_test.append(test_acc)
            acc_list_val.append(val_acc)

            # np.save(save_path+"/loss_list_train.npy", loss_list_train)
            np.save(save_path+"/loss_list_test.npy", loss_list_test)
            np.save(save_path+"/loss_list_val.npy", loss_list_val)


            # np.save(save_path+"/acc_list_train.npy", acc_list_train)
            np.save(save_path+"/acc_list_test.npy", acc_list_test)
            np.save(save_path+"/acc_list_val.npy", acc_list_val)

    print("Total training time: ",time.time()-tstart)

if __name__ == "__main__":
    main()
