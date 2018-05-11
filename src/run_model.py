from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import time
import math
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from torch import optim
from optparse import OptionParser

from dataset_iterator import *
from constants import (EMBEDDING_SIZE, DIFF_VOCAB, EMBEDDING_PATH,
                       LIMIT_ENCODE, LIMIT_DECODE, WORKING_DIR,
                       BATCH_SIZE, MAX_EPOCHS, EARLY_STOP, PRINT_FREQUENCY,
                       HIDDEN_SIZE, LEARNING_RATE)
from seq2seq import Encoder, Decoder, Seq2Seq


class run_model:

    def __init__(self, dataset, model):

        """ The model is initializer with the hyperparameters.

            Args:
                config : Config() object for the hyperparameters.
        """
        self.model = model

        # Vocabulary and datasets are initialized.
        self.dataset = dataset

    def run_epoch(self, epoch_number, fp = None):

        """ Defines the per epoch run of the model

            Args:
                epoch_number: The current epoch number
                sess:       :  The current tensorflow session.

            Returns
                total_loss : Value of loss per epoch

        """
        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.model.train()
        steps_per_epoch = int(math.ceil(float(self.dataset.datasets["train"].number_of_samples)\
                                        / float(BATCH_SIZE)))
        total_loss = 0
        for step in xrange(1, steps_per_epoch+1):
            train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq,\
            max_content, max_title, max_query = self.dataset.next_batch(self.dataset.datasets["train"],
                                                                        BATCH_SIZE, True)
            #Todo embeddings - glove
            optimizer.zero_grad()
            outputs = self.model(train_content, train_title, BATCH_SIZE, max_title)
            loss = F.cross_entropy(outputs[1:].view(-1, self.dataset.length_vocab_decode()),
                               trg[1:].contiguous().view(-1)) #ignore index pad
            loss.backward()
            clip_grad_norm(self.model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.data[0]
            duration = time.time() - start_time
            if step == steps_per_epoch or step%PRINT_FREQUENCY == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss.data[0], duration))
                sys.stdout.flush()
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.print_titles(sess, self.dataset.datasets["train"], 2)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                self.print_titles(sess,self.dataset.datasets["valid"], 2)    
                sys.stdout.flush()

        return float(total_loss)/ float(steps_per_epoch)

    def do_eval(self,sess, data_set):

        """ Does a forward propogation on the data to know how the model's performance is.
             This will be mainly used for valid and test dataset.

            Args:
                sess : The current tensorflow session
                data_set : The datset on which this should be evaluated.

            Returns
                Loss value : loss value for the given dataset.
        """  
        self.model.eval()
        total_loss = 0
        steps_per_epoch = int(math.ceil(float(data_set.number_of_samples) / float(BATCH_SIZE)))
        for step in xrange(1, steps_per_epoch+1):
            train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq,\
            max_content, max_title, max_query = self.dataset.next_batch(self.dataset.datasets["train"],
                                                                        BATCH_SIZE, True)
            outputs = self.model(train_content, train_title, BATCH_SIZE, max_title)
            loss = F.cross_entropy(outputs[1:].view(-1, self.dataset.length_vocab_decode()),
                                   trg[1:].contiguous().view(-1)) #ignore index pad
            total_loss += loss.data[0]
        return total_loss/steps_per_epoch



    def print_titles_in_files(self, sess, data_set):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """
        return None

    def print_titles(self, sess, data_set, total_examples):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """
        return None


    def run_training(self):

        """ Train the graph for a number of epochs 
        """
        best_val_loss = float('inf')
        best_val_epoch = 0
        for epoch in range(1, MAX_EPOCHS+1):
            print ("Epoch: " + str(epoch))
            start = time.time()
            train_loss = self.run_epoch(epoch)
            valid_loss = self.do_eval(epoch, self.dataset.datasets["valid"])
            print ("Training Loss:{}".format(train_loss))
            print ("Validation Loss:{}".format(valid_loss))

            if valid_loss <= best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if not os.path.isdir(".save"):
                    os.makedirs(".save")
                torch.save(self.model.state_dict(), './.save/seq2seq_%d.pt' % (epoch))
            if epoch == MAX_EPOCHS:
                torch.save(self.model.state_dict(), './.save/seq2seq_lastepoch.pt')
            if epoch - best_val_epoch > EARLY_STOP:
                print ("Results are getting no better. Early Stopping")
                break
            print ("Total time:{}".format(time.time() - start))
        # Todo - Restore the best model and evaluate
        test_loss = self.do_eval(self.dataset.datasets["test"])
        print ("Test Loss:{}".format(test_loss))
        self.print_titles_in_files(self.dataset.datasets["test"])


    def test_batch_dataset(self):

        train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq, \
        max_content, max_title,max_query = self.dataset.next_batch(self.dataset.datasets["train"],
                                                                   BATCH_SIZE, True)

        print(max_query)
        return None

def main():
    #Dataset
    dataset = PadDataset(WORKING_DIR, EMBEDDING_SIZE, diff_vocab = DIFF_VOCAB, embedding_path = EMBEDDING_PATH,\
                  limit_encode = LIMIT_ENCODE, limit_decode = LIMIT_DECODE)
    encoder_vocab_size = dataset.length_vocab_encode()
    decoder_vocab_size = dataset.length_vocab_decode()
    #Initialising Model
    encoder = Encoder(encoder_vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE)
    decoder = Decoder(EMBEDDING_SIZE, HIDDEN_SIZE, decoder_vocab_size)
    seq2seqwattn = Seq2Seq(encoder, decoder)

    run_this = run_model(dataset, seq2seqwattn)
    run_this.run_training()
    # run_this.test_batch_dataset()
    

if __name__ == '__main__':
    main()
