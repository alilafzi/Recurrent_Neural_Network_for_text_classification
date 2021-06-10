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
import statistics
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
    def initialize_cell(self, batch_size):
                weight = next(self.linear_for_cell.parameters()).data
                cell = weight.new(batch_size, self.hidden_size).zero_()
                return cell
      

class modified_text_classification(DLStudio.TextClassification):
    def run_code_for_training_with_TEXTnetOrder2(self, net, hidden_size):        
            filename_for_out = "training_loss_batching_mean_length_textnet2_binary_encoding" + str(self.dl_studio.epochs) + "epochs.txt"
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
                    cell_prev = net.initialize_cell(self.dl_studio.batch_size).to(self.dl_studio.device)
                    cell_prev_2_prev = net.initialize_cell(self.dl_studio.batch_size).to(self.dl_studio.device)
                    hidden = torch.zeros(self.dl_studio.batch_size, hidden_size)
                    hidden = hidden.to(self.dl_studio.device)
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(self.dl_studio.batch_size,review_tensor.shape[2])
                    input = input.to(self.dl_studio.device)
                    output = torch.zeros(self.dl_studio.batch_size, 2)
                    output = output.to(self.dl_studio.device)
                    for j in range(self.dl_studio.batch_size):
                        for k in range(review_tensor.shape[1]):
                            input[j,:] = review_tensor[j,k]
                            output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                            if k == 0:
                                cell_prev = cell
                            else:
                                cell_prev_2_prev = cell_prev
                                cell_prev = cell
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward()        
                    optimizer.step()
                    if i % 20 == 19:    
                        avg_loss = running_loss / float(20)
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
            plt.savefig("training_loss_batching_mean_length_textnet2_binary_encoding.png")
            plt.show()
            

class modified_sentiment_analysis(DLStudio.TextClassification.SentimentAnalysisDataset):
    def one_hotvec_for_word(self, word):
        word_index =  self.vocab.index(word)
        binary = format(word_index, 'b')
        temp_list = list(binary)
        temp_list.reverse()
        length = 16
        for i in range(length - len(temp_list)):
            temp_list.append('0')
        temp_list.reverse()
        for j in range(len(temp_list)):
            temp_list[j] = int(temp_list[j])

        return torch.FloatTensor(temp_list)
        
    def get_mean_review_length(self):
        if self.train_or_test == 'train':
            review_lengths = []
            for i in range(len(self.indexed_dataset_train)):
                review_lengths.append(len(self.indexed_dataset_train[i][0]))
            #print (len(self.indexed_dataset_train))
        elif self.train_or_test == 'test':
            review_lengths = []
            for i in range(len(self.indexed_dataset_test)):
                review_lengths.append(len(self.indexed_dataset_test[i][0]))
        return int(np.floor(statistics.mean(review_lengths)))
        
    def review_to_tensor(self, review):
        review_tensor = torch.zeros(self.get_mean_review_length(), 16)
        if len(review) <= self.get_mean_review_length():
            for i,word in enumerate(review):
                review_tensor[i,:] = self.one_hotvec_for_word(word)
        else:
            #for i,word in enumerate(self.get_mean_review_length()):
            for i in range(self.get_mean_review_length()):
                review_tensor[i,:] = self.one_hotvec_for_word(review[i])
        return review_tensor


dls = DLStudio(
#                  dataroot = "/home/kak/TextDatasets/sentiment_dataset/",
#                  dataroot = "./data/",
                  dataroot = "/content/drive/My Drive/Deep_Learning/Homeworks/HW7/DLStudio-2.0.8/Examples/data/" ,
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  1e-4,  
                  epochs = 10,
                  batch_size = 5,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )


text_cl = modified_text_classification( dl_studio = dls )
dataserver_train = modified_sentiment_analysis(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_train_3.tar.gz",
#                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
                   )
dataserver_test = modified_sentiment_analysis(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_test_3.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()

model = modified_TEXTnetOrder2(16, hidden_size=512, output_size=2)

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
