import numpy as np
import sklearn 
from sklearn.utils import shuffle

# Balances training data according to the maximum balancing ratio (nttRatio) inputted
def kartBalance(nttRatio, countClass, train_data, train_labels):

	train_data_bal = np.empty([0,200,256,2])
	train_labels_bal = np.empty([0])

	popClass = np.argmax(countClass)
	nopClass = np.argmin(countClass)

	popCount = countClass[popClass]
	nopCount = countClass[nopClass]
	if(nttRatio != -1) and (popCount/nopCount > nttRatio):
	    nopRatio = np.zeros(9)
	    instClass = np.zeros(9)
	    for k in range(0, 9):
	        if(k != nopClass):
	            nopRatio[k] = countClass[k]/nopCount
	            if(nopRatio[k] > nttRatio):
	                nCountsK = int(round(nopCount * nttRatio)) # Max allowed samples for class k
	                instClass[k] = nCountsK
	                icountsK = np.where(train_labels == k)
	                train_dataK = train_data[icountsK]
	                train_labelsK = train_labels[icountsK]
	                

	                train_dataK, train_labelsK = shuffle(train_dataK, train_labelsK, random_state = 5)
	                print(nCountsK)
	                train_dataK = train_dataK[0 : nCountsK]
	                train_labelsK = train_labelsK[0 : nCountsK]

	                print(train_dataK.shape)
	                train_data_bal = np.concatenate((train_data_bal, train_dataK))
	                train_labels_bal = np.concatenate((train_labels_bal, train_labelsK))

	            else:
	                icountsK = np.where(train_labels == k)
	                train_dataK = train_data[icountsK]
	                train_labelsK = train_labels[icountsK]

	                instClass[k] = len(icountsK[0])
	                print(train_dataK.shape)
	                train_data_bal = np.concatenate((train_data_bal, train_dataK))
	                train_labels_bal = np.concatenate((train_labels_bal, train_labelsK))

	        else:
	            icountsK = np.where(train_labels == k)
	            train_dataK = train_data[icountsK]
	            train_labelsK = train_labels[icountsK]

	            instClass[k] = len(icountsK[0])
	            print(train_dataK.shape)
	            train_data_bal = np.concatenate((train_data_bal, train_dataK))
	            train_labels_bal = np.concatenate((train_labels_bal, train_labelsK))

	else:
	    train_data_bal = train_data
	    train_labels_bal = train_labels

	return train_data_bal, train_labels_bal, instClass