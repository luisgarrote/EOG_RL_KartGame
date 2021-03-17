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
from sklearn.utils import shuffle
import torch.nn.functional as F

import balanceClasses as kB

import torch
from torch.utils import data
import time
import sys
import glob
torch.cuda.set_device(0)

# Variante da 8_bal2 com hiperparâmetros e data-splitting modificados (stratification on)

#torch.cuda.synchronize()
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
        xi= torch.from_numpy(ID).float()
        X = torch.zeros(2,200, 256, dtype=torch.float)
        X[0,:,:]=xi[:,:,0];
        X[1,:,:]=xi[:,:,1];
#        print(X.shape)


        y = (np.long(self.labels[index]))

        return X, y


#print(os.environ["CUDA_VISIBLE_DEVICES"]);
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
if(len(sys.argv)==2):
    num_epochs =  int(sys.argv[1])
else:
    num_epochs =  50

num_classes = 9
batch_size = 55
learning_rate = 0.0001
classes = [];
imgs=[];
count=0
countClass = np.zeros(9);

#clean up 
files = glob.glob('Fails/*')
for f in files:
    os.remove(f)

nttRatio = 10; # maximum allowed non-target (Null Events) to target ratio (set to -1 to ignore)

directory='E:\\Kartgame_databases\\RGB_matfiles8\\CNN8_RGB\\'
files = [f for f in listdir(directory) if isfile(join(directory, f))];
files.sort(key=lambda x: os.path.getmtime(directory+x))

trainDirectory='E:\\Kartgame_databases\\RGB_matfiles8\\Train_Arrays_8\\'
filesTrain = [f for f in listdir(trainDirectory) if isfile(join(trainDirectory, f))];
newSave = False

#files = glob.glob('Results/*')
#for f in files:
#    os.remove(f)

# No new data files are created (load unbalanced data,
# balance it internally if necessary)
# if('X_train_bal.npy' in filesTrain):
#     X_train=np.load(directory+'X_train.npy')
#     X_test=np.load(directory+'X_test.npy')
#     y_train=np.load(directory+'y_train.npy')
#     y_test=np.load(directory+'y_test.npy')
# else:
for f in files:
    # print (f)
    if ('classes' in f):
        cl=genfromtxt(directory+f, delimiter=' ');
        for c in cl:
             classes.append(c);
    else:
        
        if('_0' in f):
           classes.append(0);
           countClass[0] = countClass[0] + 1;
        elif('_1' in f):
           classes.append(1);
           countClass[1] = countClass[1] + 1;
        elif('_2' in f):
           classes.append(2);
           countClass[2] = countClass[2] + 1;
        elif('_3' in f):
           classes.append(3);
           countClass[3] = countClass[3] + 1;
        elif('_4' in f):
           classes.append(4);
           countClass[4] = countClass[4] + 1;
        elif('_5' in f):
           classes.append(5);
           countClass[5] = countClass[5] + 1;
        elif('_6' in f):
           classes.append(6);
           countClass[6] = countClass[6] + 1;
        elif('_7' in f):
           classes.append(7);
           countClass[7] = countClass[7] + 1;
        elif('_8' in f):
           classes.append(8);
           countClass[8] = countClass[8] + 1;

        img1=imageio.imread(directory+f);
        imgs.append((img1[:,:,0:2])/255.0);

        #print ((img1[:,:,0:2]/255.0).shape);
        #print (type((img1[:,:,0:2]/255.0).shape));

    count=count+1;

print('Loaded {} files'.format(count))

train_data = np.asanyarray(imgs, dtype='float32') # Returns np.array
train_labels = np.asarray(classes, dtype=np.int32)


[train_data_bal, train_labels_bal, instClass] = kB.kartBalance(nttRatio, countClass, train_data, train_labels)


train_data_bal, train_labels_bal = shuffle(train_data_bal, train_labels_bal, random_state = 564)

X_train_bal, X_test_bal, y_train_bal, y_test_bal = sk.train_test_split(train_data_bal, train_labels_bal,  test_size=0.33, random_state=42,shuffle=True,stratify=train_labels_bal)
if(newSave==True):
    X_train, X_test, y_train, y_test = sk.train_test_split(train_data, train_labels,  test_size=0.33, random_state=42,shuffle=True)
    np.save(trainDirectory+'X_train.npy',X_train)
    np.save(trainDirectory+'X_test.npy',X_test)
    np.save(trainDirectory+'y_train.npy',y_train)
    np.save(trainDirectory+'y_test.npy',y_test)


training_set = Dataset(X_train_bal, y_train_bal)
 
validation_set = Dataset(X_test_bal, y_test_bal)
 
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=training_set,batch_size=batch_size,  shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=validation_set,batch_size=batch_size, shuffle=False)


#((200-5+2*2)/1)+1
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=64, kernel_size=11, stride=1, padding=2),  # (1,200,256)
            nn.BatchNorm2d(64), # (32,200,256)
            nn.ReLU(), # (32,200,256)
            nn.MaxPool2d(kernel_size=2, stride=2)) # (32,100,128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=2), # (32,100,128)
            nn.BatchNorm2d(64), # (64,100,128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # (64,50,64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2), # (32,100,128)
            nn.BatchNorm2d(32), # (64,100,128)
            nn.ReLU(),
            # nn.Dropout(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # (32,50,64)
        self.fc1 = nn.Linear( 32* 23* 30, 5048) #25*32*32
        self.fc1a = nn.Linear(5048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape);
        out =  self.layer2(out) 
        #print(out.shape);
        out = F.dropout( self.layer3(out) ) 
        #print(out.shape);
        out = out.reshape(out.size(0), -1)
        #print(out.shape);
        out =  F.relu(self.fc1(out)) 
        out =  F.relu(self.fc1a(out)) 
        out = F.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

model =  ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)

starttime = time.time()
localtime = time.time()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        tic()
        fx=time.time();
        images = images.to(device)
        labels = labels.to(device)
#        print (images.shape)
            

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1) % 1 == 0:
            tp=time.time()-localtime
            tot=time.time()-starttime;
            expected=(num_epochs-(epoch+1))*total_step*tp+ ((total_step-(i+1)))*tp;
            localtime = time.time()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, {:.4f}s  ({:.4f}s -- {:.4f}s)' .format(epoch+1, num_epochs, i+1, total_step, loss.item(),toc(),tot,expected))
             # call to print takes a while lol

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
conf_matrix =[ [0 for x in range( num_classes )] for y in range( num_classes ) ]  


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        x=(predicted == labels);
        correct += (predicted == labels).sum().item()
        for i in range(0,len(x)):
            if(not x[i]):
                imgdebug = torch.zeros(1,3,200, 256, dtype=torch.float)
                #print (images.shape)
                imgdebug[0,0:2,:,:]=images[i,:,:,:];
                


                torchvision.utils.save_image(imgdebug,'Fails/id{}failed{}_was_{}.png'.format(count,predicted[i],labels[i]));
            count=count+1

        #print(labels)
        #print('#############################################')
        #print(predicted)
        conf_matrix+=sklearn.metrics.confusion_matrix(predicted.cpu(), labels.cpu(),labels=[x for x in range( num_classes )]);
        #print(conf_matrix)



    print('Test Accuracy of the model on the {} test images: {} %'.format(total,100 * correct / total))
    print(conf_matrix)
# Save the model checkpoint

modelName = 'model_8B_bal{}_ep{}.ckpt'.format(nttRatio, num_epochs)

torch.save(model.state_dict(), modelName)

for p in range(0,9):
    print('Event {} --> {} instances'.format(p, instClass[p]))