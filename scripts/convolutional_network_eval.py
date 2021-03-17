import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import imageio
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join
import os;
import numpy as np
import tempfile
import sklearn.model_selection as sk
import sklearn 
import torch.nn.functional as F

import torch
from torch.utils import data
import time
import socket
import matplotlib.pyplot as plt
import platform
import glob
torch.cuda.set_device(0)

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        return tempTimeInterval;
        #print( "Elapsed time: %f seconds.\n" %tempTimeInterval )



def show(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index,:,:,:]
#       print(ID.shape)

        # Load data and get label
#        X = torch.from_numpy(ID).float().reshape((2,200,256))
#        print(X.shape)
        xi= torch.from_numpy(ID).float()
        X = torch.zeros(2,200, 256, dtype=torch.float)
        X[0,:,:]=xi[:,:,0];
        X[1,:,:]=xi[:,:,1];

        y = (np.long(self.labels[index]))

        return X, y


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs =  500
num_classes = 9
batch_size = 1
learning_rate = 0.0001
classes = [];
imgs=[];
count=0


files = glob.glob('Results/*')
for f in files:
    os.remove(f)

files = glob.glob('Stream/*')
for f in files:
    os.remove(f)

# MNIST dataset
 
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=64, kernel_size=15, stride=1, padding=2),  # (1,200,256)
            nn.BatchNorm2d(64), # (32,200,256)
            nn.ReLU(), # (32,200,256)
            nn.MaxPool2d(kernel_size=2, stride=2)) # (32,100,128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=10, stride=1, padding=2), # (32,100,128)
            nn.BatchNorm2d(64), # (64,100,128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # (64,50,64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2), # (32,100,128)
            nn.BatchNorm2d(32), # (64,100,128)
            nn.ReLU(),
            # nn.Dropout(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # (32,50,64)
        self.fc1 = nn.Linear( 32* 22* 29, 5048) #25*32*32
        self.fc1a = nn.Linear(5048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape);
        out =  self.layer2(out) 
        #print(out.shape);
        out =  self.layer3(out) 
        #print(out.shape);
        out = out.reshape(out.size(0), -1)
        #print(out.shape);
        out =  F.relu(self.fc1(out))
        out =  F.relu(self.fc1a(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
model =  ConvNet(num_classes).to(device)
os=platform.system();
print(os)
if os=='Darwin':
    model.load_state_dict(torch.load('model_8_bal3_ep50.ckpt',map_location='cpu'))
else:
    model.load_state_dict(torch.load('model_8_bal3_ep50.ckpt'))

#
 
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
 
TCP_IP = ''
TCP_PORT = 5005
BUFFER_SIZE = 4096  # Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
fx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
fx.connect(("localhost",6002))
conn, addr = s.accept()
print ('Connection address:', addr)
globalgt=0;
streamback=b'';

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
with torch.no_grad():
    while 1:

        stream=streamback;
        while 1:
            data = conn.recv(BUFFER_SIZE)
            if not data: break
            if b'#' in data: 
               stream+=data;

               l=stream.decode("utf-8").split('#');

               if(len(l)==1):
                  streamback=b'';
               else:
                  streamback=l[-1].encode();
                  stream=l[0].encode();

               break
            stream+=data;
            #print ("received:", len(stream))
        tic();

        #print ("received data:", len(stream))
        if len(stream)==0:
            exit();
        stream=stream.decode("utf-8")
        stream=stream.replace('(1,0)','');
        #print ("received data:", stream)
        l=stream.split('!');
        img = torch.zeros(1,2,200, 256, dtype=torch.float) 
#        imgdebug = torch.zeros(1,3,200, 256, dtype=torch.float)
        imgdebug = torch.zeros(1,3,200, 256, dtype=torch.float)

#        print(img.shape)
#        print(l)

        size=0

        for i in l:
            s=i.split(';')
#            print(s)

            if len(s)==2:
                r=s[0].split(',');
                g=s[1].split(',');
#                print(r)
#                print(g)

                for j in r:
                    #print (j)
                    if (j.isdigit()):
                        img[0,0,int(j)-1,size]=1.0;
 #                       imgdebug[0,0,int(j)-1,size]=1.0;
                for j in g:
                    #print (j)
                    if (j.isdigit()):
                        img[0,1,int(j)-1,size]=1.0;
 #                       imgdebug[0,1,int(j)-1,size]=1.0;
            size=size+1;

                

        #print(img.shape)
        #print(type(img))
        img = img.to(device)
        #print(img.shape)
        #print(type(img))

        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        #print (predicted)
        fx.send('{} {}'.format(predicted[0],globalgt).encode());
        imgdebug[0,0:2,:,:]=img[0,:,:,:];
        torchvision.utils.save_image(imgdebug,'Results/received{}_{}.png'.format(globalgt,predicted[0])); # comentado 02-06-2020 
        file = open('Stream/received{}_{}.txt'.format(globalgt,predicted[0]), 'w') # comentado 02-06-2020 
        file.write(stream) # comentado 02-06-2020 
        file.close() # comentado 02-06-2020 

        #print(predicted[0])
        show()
        globalgt=globalgt+1;

    conn.close()

        
