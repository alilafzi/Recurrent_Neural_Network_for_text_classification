#!/usr/bin/env python

##  text_classification_with_TEXTnetOrder2.py

"""
This script is based on the TEXTnetOrder2 class in DLStudio, which is a variant of
the TEXTnet.  In the forward() of TEXTnetOrder2, you will see the following: The
value of hidden at the current time instant passes through a sigmoid nonlinearity
that acts like a switch to help the network decide how much of the previous hidden to
combine with its current value.  You can think of the TEXTnetOrder2 as a stepping
stone to the full-blown GRU.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import os, sys
import copy
import torch.optim as optim
import time
import matplotlib.pyplot as plt


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

class modified_TEXTnetOrder2(DLStudio.TextClassification.TEXTnetOrder2):
    def __init__(self, input_size, hidden_size, output_size):
                super(DLStudio.TextClassification.TEXTnetOrder2, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + 3*hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + 3*hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.dropout = nn.Dropout(p=0.1)
                # for the cell
                self.linear_for_cell = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hidden, cell, cell2):
        combined = torch.cat((input, hidden, cell, cell2), 1)
        hidden = self.combined_to_hidden(combined)
        hidden = torch.tanh(hidden)                     
        out = self.combined_to_middle(combined)
        out = torch.nn.functional.relu(out)
        out = self.dropout(out)
        out = self.middle_to_out(out)
        out = self.logsoftmax(out)
        hidden_clone = hidden.clone()
#                cell = torch.tanh(self.linear_for_cell(hidden_clone))
        cell = torch.sigmoid(self.linear_for_cell(hidden_clone))
        cell2 = torch.sigmoid(self.linear_for_cell(hidden_clone))
        return out,hidden,cell, cell2       
    
    
class modified_text_classification(DLStudio.TextClassification):
    def run_code_for_training_with_TEXTnetOrder2(self, net, hidden_size):        
            filename_for_out = "training_loss_double_gating_" + str(self.dl_studio.epochs) + "epochs.txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    cell_prev = net.initialize_cell(1).to(self.dl_studio.device)
                    cell_prev_2_prev = net.initialize_cell(1).to(self.dl_studio.device)
                    cell_prev_2_prev_2_prev = net.initialize_cell(1).to(self.dl_studio.device)
                    hidden = torch.zeros(1, hidden_size)
                    hidden = hidden.to(self.dl_studio.device)
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2])
                    input = input.to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell, cell2 = net(input, hidden, cell_prev_2_prev, cell_prev_2_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        elif k == 1:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                        else:
                            #cell_prev_2_prev_2_prev = cell_prev_2_prev
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                            cell_prev_2_prev_2_prev = cell_prev
                            cell_prev = cell2
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward()        
                    optimizer.step()
                    if i % 500 == 499:    
                        avg_loss = running_loss / float(500)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            plt.figure(figsize=(10,5))
            plt.title("Training Loss vs. Iterations")
            plt.plot(training_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("training loss")
            plt.legend()
            plt.savefig("training_loss_double_gating.png")
            plt.show()
            
            
    def run_code_for_testing_with_TEXTnetOrder2(self, net, hidden_size):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    cell_prev = net.initialize_cell(1)
                    cell_prev_2_prev = net.initialize_cell(1)
                    cell_prev_2_prev_2_prev = net.initialize_cell(1)
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    input = torch.zeros(1,review_tensor.shape[2])
                    hidden = torch.zeros(1, hidden_size)         
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell, cell2 = net(input, hidden, cell_prev_2_prev, cell_prev_2_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        elif k == 1:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                        else:
                            #cell_prev_2_prev_2_prev = cell_prev_2_prev
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                            cell_prev_2_prev_2_prev = cell_prev
                            cell_prev = cell2
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)


            
            
dls = DLStudio(
#                  dataroot = "/home/kak/TextDatasets/sentiment_dataset/",
#                  dataroot = "./data/",
                  dataroot = "/content/drive/My Drive/Deep_Learning/Homeworks/HW7/DLStudio-2.0.8/Examples/data/" ,
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  1e-5,  
                  epochs = 10,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )


text_cl = modified_text_classification( dl_studio = dls )
dataserver_train = DLStudio.TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_train_40.tar.gz",
#                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
                   )
dataserver_test = DLStudio.TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()

model = modified_TEXTnetOrder2(vocab_size, hidden_size=512, output_size=2)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

text_cl.run_code_for_training_with_TEXTnetOrder2(model, hidden_size=512)

#import pymsgbox
#response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
#if response == "OK": 
#    text_cl.run_code_for_testing_with_TEXTnetOrder2(model, hidden_size=512)

text_cl.run_code_for_testing_with_TEXTnetOrder2(model, hidden_size=512)
