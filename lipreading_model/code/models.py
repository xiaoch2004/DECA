import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from collections import OrderedDict
import math
import pickle
import glob




class cutoff_deltanet_majority_vote(nn.Module):
    def __init__(self, device, pretrained_encoder_isTrue, pre_trained_encoder_variables, shapes, nonlinearities, input_size,window,hidden_units,output_classes=10):
        super(cutoff_deltanet_majority_vote, self).__init__()

        print("pretrained_encoder",pretrained_encoder_isTrue)

        self.device=device
        self.window=window
        self.input_size=input_size
        self.hidden_units=hidden_units
        self.output_classes=output_classes


        #'fc1', 'fc2', 'fc3', 'bottleneck'
        if pretrained_encoder_isTrue==True:
            weights, biases = pre_trained_encoder_variables
            self.shapes=shapes
            self.nonlinearities=nonlinearities
            self.layer_encoder,_=pretrained_custom_encoder(input_size, shapes, nonlinearities, weights, biases)
        else:
#             shapes, nonlinearities = pre_trained_encoder_variables
            self.shapes=shapes
            self.nonlinearities=nonlinearities
            self.layer_encoder=custom_encoder(input_size,shapes,nonlinearities)

        self.layer_delta=delta_layer(self.device, self.window)

        # only blstm implemented

        self.layer_blstm=nn.LSTM(
            input_size=shapes[-1]*3,
            hidden_size=self.hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional =True,
        )



    def init_hidden(self,batch_size,num_layers=1,directions=2):# blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)
        c_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)


        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)


        return h_0,c_0




    def forward(self, x, x_lengths):
#         print(x.size())
#         print(x_lengths)
        batch_size=x.shape[0]
        seq_len=x.shape[1]

        h_0,c_0=self.init_hidden(batch_size)

        #(inputth_size,seqlen,input_size)
        x=self.layer_encoder(x)
#         print(x.size())
        #(inputth_size,seqlen,feature_size)
        x=self.layer_delta(x)
#         print(x.size())
        #(inputth_size,seqlen,3*feature_size)

        x_lengths_sorted,ordered_idx=x_lengths.sort(0, descending=True)

        x=torch.index_select(x,0,ordered_idx)

        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths_sorted, batch_first=True)

        X,(h_n,c_n)=self.layer_blstm(X,(h_0,c_0))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

# #         print(X.size())
#         x=self.layer_out(X)
# #         print(x.size())
# #convert label using torch.index_select(y,0,ordered_idx)
#         return x,ordered_idx
        return X,ordered_idx


def return_non_linearity(s):
    if s=="rectify":
        return nn.ReLU()
    return 0

class custom_encoder(nn.Module):
    def __init__(self,input_size, shapes, nonlinearities):
        super(custom_encoder, self).__init__()
        self.layers_list=nn.ModuleList([])
#         print(input_size, shapes[0])
        self.layers_list.append(nn.Linear(input_size, shapes[0]))
        temp_activation=return_non_linearity(nonlinearities[0])
        if temp_activation!=0:
            self.layers_list.append(temp_activation)

        for i in range(1,len(shapes)):
            self.layers_list.append(nn.Linear(shapes[i-1], shapes[i]))
            temp_activation=return_non_linearity(nonlinearities[i])
            if temp_activation!=0:
                self.layers_list.append(temp_activation)



    def forward(self, x):
#         print(x.is_cuda)
        batch_size=x.shape[0]
        seq_len=x.shape[1]
        for l in self.layers_list:
            x=l(x)
        return x

class cnn_encoder(nn.Module):
    def __init__(self, channels, dropout_p=0.5):
        super(cnn_encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, channels[0], (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(channels[0], channels[1], (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(channels[1], channels[2], (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        #x = self.conv2(x)
        #x = self.relu(x)
        #x = self.dropout3d(x)
        #x = self.pool2(x)

        #x = self.conv3(x)
        #x = self.relu(x)
        #x = self.dropout3d(x)
        #x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        return x




def pretrained_custom_encoder(input_size, shapes, nonlinearities, weights, biases):
    encoder=custom_encoder(input_size,shapes,nonlinearities)

    combined=[]
    for i in range(len(weights)):
        combined.append(weights[i])
        combined.append(biases[i])

    count=0
    new_state_dict= OrderedDict()
    for key, value in encoder.state_dict().items():
        new_state_dict[key]=torch.from_numpy(combined[count].T)
        count+=1
    encoder.load_state_dict(new_state_dict)

    return encoder,combined


class delta_layer(nn.Module):
    def __init__(self, device, window):
        self.device=device

        super(delta_layer, self).__init__()
        self.window = window

    def forward(self, inp):
        #shape (batchsize,timesteps,features_size)

        x=inp
        # utils.signals.append_delta_coeff
        y=torch.cat((x[:,0,:].reshape(x.shape[0],1,-1).repeat(1,self.window,1),x,x[:,-1,:].reshape(x.shape[0],1,-1).repeat(1,self.window,1)),1)
        z=torch.zeros(x.size()).to(self.device)

        for i in range(0,x.shape[-2]):
            for j in range(1,self.window+1):
                z[:,i,:]+=(y[:,i+self.window+j,:]-y[:,i+self.window-j,:])/(2*j)

        delta=z

        x=delta
        y=torch.cat((x[:,0,:].reshape(x.shape[0],1,-1).repeat(1,self.window,1),x,x[:,-1,:].reshape(x.shape[0],1,-1).repeat(1,self.window,1)),1)
        z=torch.zeros(x.size()).to(self.device)

        for i in range(0,x.shape[-2]):
            for j in range(1,self.window+1):
                z[:,i,:]+=(y[:,i+self.window+j,:]-y[:,i+self.window-j,:])/(2*j)

        double_delta=z
        # return shape (batchsize,timesteps,features_size*3)
        return torch.cat((inp,delta,double_delta),2)


class deltanet_majority_vote(nn.Module):
    def __init__(self,device, pretrained_encoder_isTrue, pre_trained_encoder_variables, shapes, nonlinearities, input_size,window,hidden_units,output_classes=10, channels=[32, 32, 8], cnn_encode=False):
        super(deltanet_majority_vote, self).__init__()

        print("pretrained_encoder",pretrained_encoder_isTrue)

        self.device=device
        self.window=window
        self.input_size=input_size
        self.hidden_units=hidden_units
        self.output_classes=output_classes
        self.cnn_encode=cnn_encode


        #'fc1', 'fc2', 'fc3', 'bottleneck'
        if pretrained_encoder_isTrue==True:
            weights, biases = pre_trained_encoder_variables
            self.shapes=shapes
            self.nonlinearities=nonlinearities
            self.layer_encoder,_=pretrained_custom_encoder(input_size, shapes, nonlinearities, weights, biases)
        else:
#             shapes, nonlinearities = pre_trained_encoder_variables
            self.shapes=shapes
            self.nonlinearities=nonlinearities
            self.layer_encoder=custom_encoder(input_size,shapes,nonlinearities)
            self.layer_cnn_encoder=cnn_encoder(channels)

        self.layer_delta=delta_layer(self.device,self.window)

        # only blstm implemented
        if not cnn_encode:
            self.lstm_input_size = shapes[-1]
        else:
            self.lstm_input_size = int(channels[-1])*6

        self.layer_blstm=nn.LSTM(
            input_size=self.lstm_input_size*3,
            hidden_size=self.hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional =True,
        )

        # hidden_units*2 because of 2 direction,   we watn it to do over the last dim softmax    ( not doing here )
        self.layer_out=nn.Sequential(nn.Linear(self.hidden_units*2,self.output_classes), nn.Softmax(dim=-1))



    def init_hidden(self,batch_size,num_layers=1,directions=2):# blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)
        c_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)


        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)


        return h_0,c_0




    def forward(self, x, x_lengths):
#         print(x.size())
#         print(x_lengths)
        batch_size=x.shape[0]
        seq_len = x.shape[1] if self.cnn_encode == False else x.shape[2]

        h_0,c_0=self.init_hidden(batch_size)

        #(inputth_size,seqlen,input_size)
        if self.cnn_encode:
            x = self.layer_cnn_encoder(x)
        else:
            x=self.layer_encoder(x)
#         print(x.size())
        #(inputth_size,seqlen,feature_size)
        x=self.layer_delta(x)
#         print(x.size())
        #(inputth_size,seqlen,3*feature_size)

        x_lengths_sorted,ordered_idx=x_lengths.sort(0, descending=True)

        if self.cnn_encode:
            x = x.permute(1,0,2).contiguous()

        x=torch.index_select(x,0,ordered_idx)

        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths_sorted, batch_first=True)

        X,(h_n,c_n)=self.layer_blstm(X,(h_0,c_0))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

#         print(X.size())
        x=self.layer_out(X)
#         print(x.size())
#convert label using torch.index_select(y,0,ordered_idx)
        return x,ordered_idx



class adenet_2stream(nn.Module):
    def __init__(self, device, pretrained_stream1_model_isTrue, pretrained_stream1_model1, shapes1, nonlinearities1, input_size1,
                 pretrained_stream1_model2, shapes2, nonlinearities2, input_size2 ,
                 window,hidden_units,output_classes=10):


        super(adenet_2stream, self).__init__()

        self.device=device
        self.window=window
        self.input_size1=input_size1
        self.input_size2=input_size2
        self.hidden_units=hidden_units
        self.output_classes=output_classes


        #'fc1', 'fc2', 'fc3', 'bottleneck'
        if pretrained_stream1_model_isTrue==True:
            self.stream1_model_1=pretrained_stream1_model1
            self.stream1_model_2=pretrained_stream1_model2

        else:
            self.stream1_model_1=cutoff_deltanet_majority_vote(False, None, shapes1, nonlinearities1,
                                               input_size1,window, hidden_units, 10)

            self.stream1_model_2=cutoff_deltanet_majority_vote(False, None, shapes2, nonlinearities2,
                                               input_size2,window, hidden_units, 10)



        # only blstm implemented

        self.layer_blstm_agg=nn.LSTM(
            input_size=self.hidden_units*4,
            hidden_size=self.hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional =True,
        )


        # hidden_units*2 because of 2 direction,   we watn it to do over the last dim softmax    ( not doing here )
        self.layer_out=nn.Sequential(nn.Linear(self.hidden_units*2,self.output_classes), nn.Softmax(dim=-1))



    def init_hidden(self,batch_size,num_layers=1,directions=2):# blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)
        c_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)


        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)


        return h_0,c_0



    def forward(self, x1,x2,x_lengths):
#         print(x1.size())
#         print(x_lengths)
        batch_size=x1.shape[0]
        seq_len=x1.shape[1]


        h_agg_0,c_agg_0=self.init_hidden(batch_size)

        #(inputth_size,seqlen,input_size)
        x1,ordered_idx=self.stream1_model_1(x1,x_lengths)
#         print(x.size())
        #(inputth_size,seqlen,feature_size)
        x2,ordered_idx_mod=self.stream1_model_2(x2,x_lengths)
#         print(x.size())
        #(inputth_size,seqlen,3*feature_size)

        x_lengths_sorted =torch.index_select(x_lengths,0,ordered_idx)

        x=torch.cat((x1, x2), -1)

        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths_sorted, batch_first=True)

        X,(h_agg_n,c_agg_n)=self.layer_blstm_agg(X,(h_agg_0,c_agg_0))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

#         print(X.size())
        x=self.layer_out(X)
#         print(x.size())
#convert label using torch.index_select(y,0,ordered_idx)
        return x,ordered_idx


class adenet_3stream(nn.Module):
    def __init__(self, device, pretrained_stream1_model_isTrue,\
                 pretrained_stream1_model1, shapes1, nonlinearities1, input_size1,
                 pretrained_stream1_model2, shapes2, nonlinearities2, input_size2 ,
                 pretrained_stream1_model3, shapes3, nonlinearities3, input_size3 ,
                 window,hidden_units,output_classes=10):


        super(adenet_3stream, self).__init__()

        self.device=device
        self.window=window
        self.input_size1=input_size1
        self.input_size2=input_size2
        self.input_size3=input_size3
        self.hidden_units=hidden_units
        self.output_classes=output_classes


        #'fc1', 'fc2', 'fc3', 'bottleneck'
        if pretrained_stream1_model_isTrue==True:
            self.stream1_model_1=pretrained_stream1_model1
            self.stream1_model_2=pretrained_stream1_model2
            self.stream1_model_3=pretrained_stream1_model3

        else:
            self.stream1_model_1=cutoff_deltanet_majority_vote(False, None, shapes1, nonlinearities1,
                                               input_size1,window, hidden_units, 10)

            self.stream1_model_2=cutoff_deltanet_majority_vote(False, None, shapes2, nonlinearities2,
                                               input_size2,window, hidden_units, 10)

            self.stream1_model_3=cutoff_deltanet_majority_vote(False, None, shapes3, nonlinearities3,
                                               input_size3,window, hidden_units, 10)



        # only blstm implemented

        self.layer_blstm_agg=nn.LSTM(
            input_size=self.hidden_units*6,
            hidden_size=self.hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional =True,
        )


        # hidden_units*2 because of 2 direction,   we watn it to do over the last dim softmax    ( not doing here )
        self.layer_out=nn.Sequential(nn.Linear(self.hidden_units*2,self.output_classes), nn.Softmax(dim=-1))


    def init_hidden(self,batch_size,num_layers=1,directions=2):# blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)
        c_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)


        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)


        return h_0,c_0



    def forward(self, x1, x2, x3, x_lengths):
#         print(x1.size())
#         print(x_lengths)
        batch_size=x1.shape[0]
        seq_len=x1.shape[1]


        h_agg_0,c_agg_0=self.init_hidden(batch_size)

        #(inputth_size,seqlen,input_size)
        x1,ordered_idx=self.stream1_model_1(x1,x_lengths)
#         print(x.size())
        #(inputth_size,seqlen,feature_size)
        x2,ordered_idx_2=self.stream1_model_2(x2,x_lengths)
#         print(x.size())
        #(inputth_size,seqlen,3*feature_size)
        x3,ordered_idx_3=self.stream1_model_3(x3,x_lengths)


        x_lengths_sorted =torch.index_select(x_lengths,0,ordered_idx)

        x=torch.cat((x1, x2, x3), -1)

        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths_sorted, batch_first=True)

        X,(h_agg_n,c_agg_n)=self.layer_blstm_agg(X,(h_agg_0,c_agg_0))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

#         print(X.size())
        x=self.layer_out(X)
#         print(x.size())
#convert label using torch.index_select(y,0,ordered_idx)
        return x,ordered_idx




class adenet_4stream(nn.Module):
    def __init__(self, device, pretrained_stream1_model_isTrue,\
                 pretrained_stream1_model1, shapes1, nonlinearities1, input_size1,
                 pretrained_stream1_model2, shapes2, nonlinearities2, input_size2 ,
                 pretrained_stream1_model3, shapes3, nonlinearities3, input_size3 ,
                 pretrained_stream1_model4, shapes4, nonlinearities4, input_size4 ,
                 window,hidden_units,output_classes=10):


        super(adenet_4stream, self).__init__()

        self.device=device
        self.window=window
        self.input_size1=input_size1
        self.input_size2=input_size2
        self.input_size3=input_size3
        self.input_size4=input_size4
        self.hidden_units=hidden_units
        self.output_classes=output_classes


        #'fc1', 'fc2', 'fc3', 'bottleneck'
        if pretrained_stream1_model_isTrue==True:
            self.stream1_model_1=pretrained_stream1_model1
            self.stream1_model_2=pretrained_stream1_model2
            self.stream1_model_3=pretrained_stream1_model3
            self.stream1_model_4=pretrained_stream1_model4

        else:
            self.stream1_model_1=cutoff_deltanet_majority_vote(False, None, shapes1, nonlinearities1,
                                               input_size1,window, hidden_units, 10)

            self.stream1_model_2=cutoff_deltanet_majority_vote(False, None, shapes2, nonlinearities2,
                                               input_size2,window, hidden_units, 10)

            self.stream1_model_3=cutoff_deltanet_majority_vote(False, None, shapes3, nonlinearities3,
                                               input_size3,window, hidden_units, 10)

            self.stream1_model_4=cutoff_deltanet_majority_vote(False, None, shapes4, nonlinearities4,
                                               input_size4,window, hidden_units, 10)



        # only blstm implemented

        self.layer_blstm_agg=nn.LSTM(
            input_size=self.hidden_units*8,
            hidden_size=self.hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional =True,
        )


        # hidden_units*2 because of 2 direction,   we watn it to do over the last dim softmax    ( not doing here )
        self.layer_out=nn.Sequential(nn.Linear(self.hidden_units*2,self.output_classes), nn.Softmax(dim=-1))



    def init_hidden(self,batch_size,num_layers=1,directions=2):# blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)
        c_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)


        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)


        return h_0,c_0




    def forward(self, x1, x2, x3, x4, x_lengths):
#         print(x1.size())
#         print(x_lengths)
        batch_size=x1.shape[0]
        seq_len=x1.shape[1]


        h_agg_0,c_agg_0=self.init_hidden(batch_size)

        #(inputth_size,seqlen,input_size)
        x1,ordered_idx=self.stream1_model_1(x1,x_lengths)
#         print(x.size())
        #(inputth_size,seqlen,feature_size)
        x2,ordered_idx_2=self.stream1_model_2(x2,x_lengths)
#         print(x.size())
        #(inputth_size,seqlen,3*feature_size)
        x3,ordered_idx_3=self.stream1_model_3(x3,x_lengths)

        x4,ordered_idx_4=self.stream1_model_4(x4,x_lengths)


        x_lengths_sorted =torch.index_select(x_lengths,0,ordered_idx)

        x=torch.cat((x1, x2, x3, x4), -1)

        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths_sorted, batch_first=True)

        X,(h_agg_n,c_agg_n)=self.layer_blstm_agg(X,(h_agg_0,c_agg_0))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

#         print(X.size())
        x=self.layer_out(X)
#         print(x.size())
#convert label using torch.index_select(y,0,ordered_idx)
        return x,ordered_idx




class adenet_5stream(nn.Module):
    def __init__(self, device, pretrained_stream1_model_isTrue,\
                 pretrained_stream1_model1, shapes1, nonlinearities1, input_size1,
                 pretrained_stream1_model2, shapes2, nonlinearities2, input_size2 ,
                 pretrained_stream1_model3, shapes3, nonlinearities3, input_size3 ,
                 pretrained_stream1_model4, shapes4, nonlinearities4, input_size4 ,
                 pretrained_stream1_model5, shapes5, nonlinearities5, input_size5 ,
                 window,hidden_units,output_classes=10):


        super(adenet_5stream, self).__init__()

        self.device=device
        self.window=window
        self.input_size1=input_size1
        self.input_size2=input_size2
        self.input_size3=input_size3
        self.input_size4=input_size4
        self.input_size5=input_size5
        self.hidden_units=hidden_units
        self.output_classes=output_classes


        #'fc1', 'fc2', 'fc3', 'bottleneck'
        if pretrained_stream1_model_isTrue==True:
            self.stream1_model_1=pretrained_stream1_model1
            self.stream1_model_2=pretrained_stream1_model2
            self.stream1_model_3=pretrained_stream1_model3
            self.stream1_model_4=pretrained_stream1_model4
            self.stream1_model_5=pretrained_stream1_model5

        else:
            self.stream1_model_1=cutoff_deltanet_majority_vote(device, False, None, shapes1, nonlinearities1,
                                               input_size1,window, hidden_units, 10)

            self.stream1_model_2=cutoff_deltanet_majority_vote(device, False, None, shapes2, nonlinearities2,
                                               input_size2,window, hidden_units, 10)

            self.stream1_model_3=cutoff_deltanet_majority_vote(device, False, None, shapes3, nonlinearities3,
                                               input_size3,window, hidden_units, 10)

            self.stream1_model_4=cutoff_deltanet_majority_vote(device, False, None, shapes4, nonlinearities4,
                                               input_size4,window, hidden_units, 10)

            self.stream1_model_5=cutoff_deltanet_majority_vote(device, False, None, shapes5, nonlinearities5,
                                               input_size5,window, hidden_units, 10)



        # only blstm implemented

        self.layer_blstm_agg=nn.LSTM(
            input_size=self.hidden_units*10,
            hidden_size=self.hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional =True,
        )


        # hidden_units*2 because of 2 direction,   we watn it to do over the last dim softmax    ( not doing here )
        self.layer_out=nn.Sequential(nn.Linear(self.hidden_units*2,self.output_classes), nn.Softmax(dim=-1))



    def init_hidden(self,batch_size,num_layers=1,directions=2):# blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)
        c_0 = torch.randn(num_layers*directions, batch_size, self.hidden_units)


        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)


        return h_0,c_0



    def forward(self, x1, x2, x3, x4, x5, x_lengths):
#         print(x1.size())
#         print(x_lengths)
        batch_size=x1.shape[0]
        seq_len=x1.shape[1]


        h_agg_0,c_agg_0=self.init_hidden(batch_size)

        #(inputth_size,seqlen,input_size)
        x1,ordered_idx=self.stream1_model_1(x1,x_lengths)
        x2,ordered_idx_2=self.stream1_model_2(x2,x_lengths)
        x3,ordered_idx_3=self.stream1_model_3(x3,x_lengths)
        x4,ordered_idx_4=self.stream1_model_4(x4,x_lengths)
        x5,ordered_idx_4=self.stream1_model_5(x5,x_lengths)


        x_lengths_sorted =torch.index_select(x_lengths,0,ordered_idx)

        x=torch.cat((x1, x2, x3, x4, x5), -1)

        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths_sorted, batch_first=True)

        X,(h_agg_n,c_agg_n)=self.layer_blstm_agg(X,(h_agg_0,c_agg_0))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

#         print(X.size())
        x=self.layer_out(X)
#         print(x.size())
#convert label using torch.index_select(y,0,ordered_idx)
        return x,ordered_idx


class LipNet(torch.nn.Module):
    def __init__(self, dropout_p=0.5, output_dim=10, bigru_size=256):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1  = nn.GRU(96*2*3, bigru_size, 1, bidirectional=True)
        self.gru2  = nn.GRU(bigru_size*2, bigru_size, 1, bidirectional=True)

        self.FC    = nn.Linear(bigru_size*2, output_dim)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = self.softmax(x)
        x = x.permute(1, 0, 2).contiguous()
        return x

def parameterCount(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
