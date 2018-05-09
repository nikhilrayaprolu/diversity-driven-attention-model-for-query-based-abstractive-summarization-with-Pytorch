from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import math
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from optparse import OptionParser
from dataset_iterator import *
from constants import (EMBEDDING_SIZE, DIFF_VOCAB, EMBEDDING_PATH,
                       LIMIT_ENCODE, LIMIT_DECODE, WORKING_DIR,
                       BATCH_SIZE)

import os


class run_model:

    def __init__(self, wd):

        """ The model is initializer with the hyperparameters.

            Args:
                config : Config() object for the hyperparameters.
        """


        # Vocabulary and datasets are initialized.
        self.dataset = PadDataset(wd, EMBEDDING_SIZE, diff_vocab = DIFF_VOCAB, embedding_path = EMBEDDING_PATH,\
				  limit_encode = LIMIT_ENCODE, limit_decode = LIMIT_DECODE)


    def run_epoch(self, epoch_number, sess, fp = None):

        """ Defines the per epoch run of the model

            Args:
                epoch_number: The current epoch number
                sess:       :  The current tensorflow session.

            Returns
                total_loss : Value of loss per epoch

        """
        return None


    def do_eval(self,sess, data_set):

        """ Does a forward propogation on the data to know how the model's performance is.
             This will be mainly used for valid and test dataset.

            Args:
                sess : The current tensorflow session
                data_set : The datset on which this should be evaluated.

            Returns
                Loss value : loss value for the given dataset.
        """  

        return None



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
        return None

    def test_batch_dataset(self):

        train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq, \
        max_content, max_title,max_query = self.dataset.next_batch(self.dataset.datasets["train"],
                                                                   BATCH_SIZE, True)

        print(max_query)
        return None

def main():

    run_this = run_model(WORKING_DIR)
    run_this.test_batch_dataset()
    

if __name__ == '__main__':
    main()
