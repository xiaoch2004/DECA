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


# This is the script training the multiview data with 1stream model. The image size has been padded to 44*50.

def train(device, model, optimizer, datagen, epoch, epochsize, cnn_encode=False):

    model.train()
    if epoch == 5:
        print('pause')

    tloss=0
    for i in range(epochsize):
        if not cnn_encode:
            X, y, vid_lens_batch, m, batch_idxs = next(datagen)
        else:
            X, y, vid_lens_batch, m, batch_idxs = next(datagen)
        vid_lens_batch = (X.shape[1]*np.ones(X.shape[0])).astype(np.int)

        # repeat targets based on max sequence len
        y = y.reshape((-1, 1))
        y = y.repeat(m.shape[-1], axis=-1)


        # print(X.shape,y.shape, vid_lens_batch.shape, m.shape,batch_idxs.shape)
    #(10, 36, 1450)     (10, 36)        (10,)               (10, 36) (10,)
        # print(X.dtype,y.dtype,vid_lens_batch.dtype,m.dtype,batch_idxs.dtype)
    # #float32          uint8        uint8                  uint8    int64
        if cnn_encode:
            X = np.transpose(X[:,:,None,:,:], (0,2,1,3,4))
        X=torch.from_numpy(X).float().to(device)
        y=torch.from_numpy(y).long().to(device)
        vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
        m=torch.from_numpy(m).float().to(device)

#         X.to(device)
#         y.to(device)
#         vid_lens_batch.to(device)
#         m.to(device)
        # batch_idxs.to(device)

        optimizer.zero_grad()

#         print(X.is_cuda)
#         print(next(model.parameters()).is_cuda)
#         print(X.shape,X_s2.shape)
        output,ordered_idx = model(X, vid_lens_batch)

        target=torch.index_select(y,0,ordered_idx)
        m=torch.index_select(m,0,ordered_idx)

        loss = temporal_ce_loss(output, target,m)

        tloss+=loss.item()

        loss.backward()

        #gradient clip, if needed add
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, tloss*1.0/epochsize))





def test(device, model,name,epoch, datagen, data_size, batch_size, cnn_encode=False):
    model.eval()

    num_iterations = int(data_size/batch_size)
    # train/valid batchsize is 50
    for i in range(num_iterations):
        X_val, y_val, vid_lens_batch, mask_val, idxs_val = next(datagen)

        vid_lens_batch = (X_val.shape[1]*np.ones(X_val.shape[0])).astype(np.int)
        mask_val = np.ones((batch_size, X_val.shape[1]), dtype=int)

        y_target = y_val
        y = y_val.reshape((-1, 1))
        y = y.repeat(X_val.shape[1], axis=-1)

        if cnn_encode:
            X_val = np.transpose(X_val[:,:,None,:,:], (0,2,1,3,4))
            vid_lens_batch.astype(np.int)

        X=torch.from_numpy(X_val).float().to(device)
        y_val=torch.from_numpy(y_val).to(device)
        vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
        mask_val=torch.from_numpy(mask_val).to(device)


        output,ordered_idx = model(X, vid_lens_batch)


        y_val=torch.index_select(y_val,0,ordered_idx)
        mask_val=torch.index_select(mask_val,0,ordered_idx)


        y=torch.from_numpy(y).long().to(device)
        target=torch.index_select(y,0,ordered_idx)


        m=mask_val.float()

        loss = temporal_ce_loss(output, target,m)

        output=output.cpu().detach().numpy()
        y_val = y_val.cpu().detach().numpy()
        #ordered_idx = ordered_idx.cpu().detach().numpy()

        if i == 0:
            seq_len = output.shape[1]
            num_classes = output.shape[-1]
            output_all = np.zeros((data_size, output.shape[1], output.shape[-1]))
            y_all = np.zeros((data_size,), dtype=int)
            ordered_idx_all = np.zeros((data_size,), dtype=int)

        output_all[int(batch_size*i):int(batch_size*(i+1))] = output
        y_all[int(batch_size*i):int(batch_size*(i+1))] = y_val
        #ordered_idx_all[int(batch_size*i):int(batch_size*(i+1))] = ordered_idx

    # loss
    #ordered_idx_all = torch.from_numpy(ordered_idx_all).long().to(device)
    output_all=torch.from_numpy(output_all).float().to(device)
    y_all = y_all.reshape((-1, 1))
    y_all = y_all.repeat(output_all.shape[1], axis=-1)
    y_all=torch.from_numpy(y_all).to(device)
    mask_all = np.ones((output_all.shape[0], output_all.shape[1]))
    mask_all=torch.from_numpy(mask_all).to(device)

    #y_all=torch.index_select(y_all,0,ordered_idx_all)
    #mask_all=torch.index_select(mask_all,0,ordered_idx_all)

    loss = temporal_ce_loss(output_all, y_all, mask_all)

    y_all = y_all.cpu().detach().numpy()
    output_all = output_all.cpu().detach().numpy()
    mask_all = mask_all.cpu().detach().numpy()

    seq_len=output_all.shape[1]
    #y_val=y_val[:].contiguous()
    #mask_val=mask_val[:,:seq_len].contiguous()

    #mask_val=mask_val.cpu().numpy()
    #y_val=y_val.cpu().numpy()

    #num_classes = output_all.shape[-1]

    #ix = np.zeros((X_val.shape[0],), dtype='int')
    #ix_top3 = np.zeros((X_val.shape[0], 3), dtype='int')
    #seq_lens = np.sum(mask_val, axis=-1).astype(np.int)
    ix = np.zeros((output_all.shape[0],), dtype='int')
    ix_top3 = np.zeros((output_all.shape[0], 3), dtype='int')
    seq_lens = (output_all.shape[1]*np.ones((output_all.shape[0],), dtype='int')).astype(np.int)




    # for each example, we only consider argmax of the seq len
    votes = np.zeros((num_classes,), dtype='int')
    for i, eg in enumerate(output_all):
        # changed !!!
        predictions = np.argmax(eg[:seq_lens[0]], axis=-1)
#         print(predictions.shape)
        for cls in range(num_classes):
            count = (predictions == cls).sum(axis=-1)
            votes[cls] = count
        ix[i] = np.argmax(votes)
        ix_top3[i] = torch.topk(torch.from_numpy(votes),3)[1].numpy()


    y_all = y_all[:,0]
    c = ix == y_all
#     print(c,ix[:10],y_val[:10])
    classification_rate = np.sum(c == True) / float(len(c))

    c_top3 = np.zeros((output_all.shape[0],), dtype='int')
    for idx in range(output_all.shape[0]):
        if y_all[idx] in ix_top3[idx]:
            c_top3[idx] = 1
    classification_rate_top3 = np.sum(c_top3) / float(len(c_top3))


    print('{} Epoch: {} \tTop3Acc: {:.6f} \tAcc: {:.6f} \tLoss: {:.6f}'.format(name,
                    epoch,classification_rate_top3,classification_rate,loss.item() ))

    preds = ix
    true_labels = y_all

    return classification_rate, classification_rate_top3, loss.item(), preds, true_labels





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default="0", help='Please write which gpu to use 0 is for cuda:0 and so one \
                        if you want to use CPU specify -1 ')

    parser.add_argument('--one_stream_view', type=str, default="1", help='1stream view \
    to train like 1, 2 and so on .   \
    1 --> 0deg, 2 --> 30deg, 3 --> 45deg, 4 --> 60deg, 5 --> 90deg')

    parser.add_argument('--save_path', type=str, default='../results/1stream', help='path where \
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
    parser.add_argument('--cnn_encode', type=int, default=False, help='Whether to use CNN encoder ')



    args = parser.parse_args()

    view =args.one_stream_view
    view=int(view)

    cnn_encode = bool(args.cnn_encode)

    train_views = [1,2,3,4,5]
    test_views = [1,2,3,4,5]
    string = ""
    for i in range(len(train_views)):
        if i == 0:
            string += str(train_views[i])
        else:
            string += (","+ str(train_views[i]))

    if cnn_encode:
        save_path = args.save_path + "/cnn_dnn_train_view_" + string
    else:
        if view == -1:
            save_path=args.save_path+"/train_view_"+string
        else:
            save_path=args.save_path+"/train_view_"+str(view)


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
    #shape = [1024, 1024, 1024, 50]
    nonlinearities= ["rectify","rectify","rectify","linear"]
    # preprocessing options
    reorderdata= False
    diffimage= False
    meanremove= True
    samplewisenormalize= True
    featurewisenormalize= False

    # CNN channels
    channels = [32, 64, 96]

    # [lstm_classifier]
    windowsize= 3
    lstm_size= 450
    output_classes= args.num_classes

    matlab_target_offset= True

    #[training]
    learning_rate= 0.0001
#     num_epoch= 40
    num_epoch=args.num_epoch
    #epochsize= 105*2
    epochsize = 525
    batchsize= 10

    #[testing and val]
    val_batchsize = 50
    test_batchsize = 50

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

    view = -1

    if view==1:
        imagesize= [29,50]
        input_dimensions= 1450
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_frontal.mat"

    elif view==2:
        imagesize= [29,44]
        input_dimensions= 1276
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_30.mat"

    elif view==3:
        imagesize= [29,43]
        input_dimensions= 1247
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_45.mat"

    elif view==4:
        imagesize= [35,44]
        input_dimensions= 1540
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_60.mat"

    elif view==5:
        imagesize= [44,30]
        input_dimensions= 1320
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_profile.mat"

    elif view==-1: # some views
        imagesize = [44,50]
        input_dimensions = 2200
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

    #network=deltanet_majority_vote(device, pretrained_encoder_isTrue, \
    #            pre_trained_encoder_variables, shape, nonlinearities, input_dimensions, windowsize, lstm_size, args.num_classes, channels=channels, cnn_encode=cnn_encode)

    # network = LipNet(dropout_p=0.5)

    network = cnn_dnn_delta_net(device, shape, nonlinearities, 2200, windowsize, lstm_size, args.num_classes, cnn_channels=channels, conv=False)

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

    #get train accuracy
    #     X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train = next(train_datagen)


    # train_datagen= gen_lstm_batch_random(s1_train_X, s1_train_y, s1_train_vidlens, batchsize=len(s1_train_vidlens), shuffle=False)
    # X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train = next(train_datagen)


    if cnn_encode == True:
        cnn_dump_path = '../data/cnn_input_data.pkl'
        if not os.path.isfile(cnn_dump_path):

            train_X, train_y, train_vidlens, train_subjects = pad_frame_and_reshape(train_X, train_y, train_vidlens, train_subjects, [44,50])
            val_X, val_y, val_vidlens, val_subjects = pad_frame_and_reshape(val_X, val_y, val_vidlens, val_subjects, [44,50])
            test_X, test_y, test_videns, test_subjects = pad_frame_and_reshape(test_X, test_y, test_vidlens, test_subjects, [44,50])
            cnn_data = {}
            cnn_data['train_X'] = train_X
            cnn_data['train_y'] = train_y
            cnn_data['train_vidlens'] = train_vidlens
            cnn_data['train_subjects'] = train_subjects
            cnn_data['val_X'] = val_X
            cnn_data['val_y'] = val_y
            cnn_data['val_vidlens'] = val_vidlens
            cnn_data['val_subjects'] = val_subjects
            cnn_data['test_X'] = test_X
            cnn_data['test_y'] = test_y
            cnn_data['test_vidlens'] = test_vidlens
            cnn_data['test_subjects'] = test_subjects

            with open(cnn_dump_path, "wb") as myFile:
                pickle.dump(cnn_data, myFile)
        else:
            with open(cnn_dump_path, "rb") as myFile:
                cnn_data = pickle.load(myFile)
            train_X = cnn_data['train_X']
            train_y = cnn_data['train_y']
            train_vidlens = cnn_data['train_vidlens']
            train_subjects = cnn_data['train_subjects']
            val_X = cnn_data['val_X']
            val_y = cnn_data['val_y']
            val_vidlen = cnn_data['val_vidlens']
            val_subjects = cnn_data['val_subjects']
            test_X = cnn_data['test_X']
            test_y = cnn_data['test_y']
            test_vidlens = cnn_data['test_vidlens']
            test_subject = cnn_data['test_subjects']

        datagen = gen_cnn_batch(train_X, train_y, batchsize=batchsize, shuffle=False)
        test_datagen = gen_cnn_batch(test_X, test_y, batchsize=test_batchsize, shuffle=False)
        val_datagen = gen_cnn_batch(val_X, val_y, batchsize=val_batchsize, shuffle=False)
    else:
        datagen = gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=batchsize)
        val_datagen = gen_lstm_batch_random(val_X, val_y, val_vidlens, batchsize=len(val_vidlens), shuffle=False)
        test_datagen = gen_lstm_batch_random(test_X, test_y, test_vidlens, batchsize=len(test_vidlens), shuffle=False)

        # We'll use this "validation set" to periodically check progress
        X_val, y_val,  vid_lens_batch_val, mask_val, idxs_val = next(val_datagen)

        # we use the test set to check final classification rate
        X_test, y_test,  vid_lens_batch_test, mask_test, idxs_test = next(test_datagen)


    loss_list_train=[]
    loss_list_test=[]
    loss_list_val=[]

    acc_list_train=[]
    acc_list_test=[]
    acc_list_val=[]
    model=network

    tstart = time.time()

    print("Started training")

    for epoch in range(1,num_epoch+1):
        time_start = time.time()

        #(device, model, optimizer, datagen, epoch, epochsize)
        train(device, network, optimizer, datagen, epoch, epochsize, cnn_encode)


    #   train_acc, train_loss,  predictions_train, true_label_train = test(device, model,"train",epoch, X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train)
        train_acc, train_loss,  predictions_train, true_label_train=(0,0,0,0)

        #(device, model,name,epoch, X_val, y_val, vid_lens_batch, mask_val, idxs_val)

        print("Time Taken for epoch:",epoch," ",time.time()-time_start)

        if (epoch%5==0 ):
            if not cnn_encode:
                #val_acc, val_acc_top3, val_loss, predictions_val, true_label_val = test(device, model,"val",epoch, \
                #                    X_val,\
                #                                                y_val, vid_lens_batch_val, mask_val, idxs_val, cnn_encode)

                #test_acc, test_acc_top3, test_loss, predictions_test, true_label_test = test(device, model,"test",epoch, \
                #                X_test,\
                #                                                y_test, vid_lens_batch_test, mask_test, idxs_test, cnn_encode)
                val_acc, val_acc_top3, val_loss, predictions_val, true_label_val = test(device, model,"val",epoch, \
                                                                val_datagen, y_val.shape[0], len(y_val), cnn_encode)
                test_acc, test_acc_top3, test_loss, predictions_test, true_label_test = test(device, model,"test",epoch, \
                                                                test_datagen, y_test.shape[0], len(y_test), cnn_encode)

            else:
                val_acc, val_acc_top3, val_loss, predictions_val, true_label_val = test(device, model,"val",epoch, \
                                                                val_datagen, val_y.shape[0], val_batchsize, cnn_encode)
                test_acc, test_acc_top3, test_loss, predictions_test, true_label_test = test(device, model,"test",epoch, \
                                                                test_datagen, test_y.shape[0], test_batchsize, cnn_encode)
            print("Saved model dict at epoch:",epoch," at path ",save_path)
            state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),

                'test_loss':             test_loss,
                'test_accuracy':         test_acc,
                'test_accuracy_top3':    test_acc_top3,
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

    print(time.time()-tstart)


if __name__ == "__main__":
    main()
