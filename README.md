# Recurrent_Neural_Network_for_text_classification
## Project goals
Here, we will be using various types of RNNs with different approaches to classify product reviews provided by Amazon, i.e. to identify whether a review is positive or negative. We define the following 3 tasks to achieve this. There is one source file (Final.ipynb) that calls each of these separate .py files associated with a subtask within the major tasks. Each of those .py files inherits required classes from DLStudio module (referenced below) and make necessary modifications accordingly.<br>

## Task 1
The traditional approach for text classification is to represent the words within a review file with one-hot encoding. However, this approach is not computationally efficient since the amount of training data relative to the trained model size (number of learnable parameters) is usually small. Therefore, more training data is required to improve the network performance that leads to even more numerical cost. As a result, since the size of the vocabulary in the review files is less than 2^16, we will represent words with binary encoding as an alternative approach to see its effect. This helps reduce the model size. The changes made in the modified class inside “text_classification_with_TEXTnet_binary_encoding.py” represent this approach. Using “!python3 text_classification_with_TEXTnet.py” within the main module “Final.py”, we call the Vanilla RNN with a one-hot representation of words in the available reviews. <br>
After running each of the aforementioned commands, we could see the model size, classification accuracy, and the resultant confusion matrix in the output of the corresponding cell within the notebook. First of all, we see that the model size (the number of learnable parameters) has been reduced by a factor of 30, approximately. We also note that the drop in the training loss is more compared to the one-hot case. Consequently, the performance on the test set in terms of overall classification accuracy and correct prediction of negative reviews is better. This improved performance makes sense as the model capacity has decreased compared to the constant amount of training data. <br>

## Task 2 


## Task 3


## Reference:
https://engineering.purdue.edu/kak/distDLS/

