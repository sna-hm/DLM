import logging
import numpy as np
from numpy import load
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from spektral.utils.convolution import normalized_adjacency

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils import batch_iterator
from spektral.utils.data import Batch

class EvaluateModel:
    def __init__(self):
        self.tmp_dir = "tmp"
        self.model_now = load_model('moraphishdet.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool': MinCutPool, 'GlobalAvgPool': GlobalAvgPool})
        self.model_new = load_model(self.tmp_dir + '/model/moraphishdet.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool': MinCutPool, 'GlobalAvgPool': GlobalAvgPool})

    def evaluate(self, model):
        # Load data
        X_test_A = load(self.tmp_dir + '/datasets/te_ma.npy', allow_pickle=True)
        X_test_B, A_test_B, y_test_B = load(self.tmp_dir + '/datasets/te_feat.npy', allow_pickle=True), list(load(self.tmp_dir + '/datasets/te_adj.npy', allow_pickle=True)), load(self.tmp_dir + '/datasets/te_class.npy', allow_pickle=True)
        A_test_B = [normalized_adjacency(a) for a in A_test_B]

        # Model evaluation function
        def evaluate(A_list, X_list, y_list, X_A_list, ops, batch_size):
            batches = batch_iterator([A_list, X_list, y_list, X_A_list], batch_size=batch_size)
            output = []
            for b in batches:
                X, A, I = Batch(b[0], b[1]).get('XAI')
                A = sp_matrix_to_sp_tensor(A)
                y = b[2]
                X_A = b[3]
                pred = model([X_A, X, A, I], training=False)
                outs = [o(y, pred) for o in ops]
                output.append(outs)
            return np.mean(output, 0)

        ################################################################################
        # PARAMETERS
        ################################################################################
        batch_size = 1             # Batch size. NOTE: it MUST be 1 when using MinCutPool and DiffPool
        loss_fn = model.loss_functions[0]
        acc_fn = lambda x, y: K.mean(tf.keras.metrics.categorical_accuracy(x, y))

        ################################################################################
        # EVALUATE MODEL
        ################################################################################

        def prf1(A_list, X_list, y_list, X_A_list, batch_size):
            batches = batch_iterator([A_list, X_list, y_list, X_A_list], batch_size=batch_size)
            y_pred = []
            y_true = []
            for b in batches:
                X, A, I = Batch(b[0], b[1]).get('XAI')
                A = sp_matrix_to_sp_tensor(A)
                X_A = b[3]
                y_pred.append(np.argmax(model([X_A, X, A, I], training=False)))
                y_true.append(np.argmax(b[2]))
            return y_pred,y_true

        test_loss, test_acc = evaluate(A_test_B, X_test_B, y_test_B, X_test_A, [loss_fn, acc_fn], batch_size=batch_size)
        y_pred, y_true = prf1(A_test_B, X_test_B, y_test_B, X_test_A, batch_size=batch_size)

        precision = metrics.precision_score(y_true, y_pred) * 100
        recall = metrics.recall_score(y_true, y_pred) * 100
        f1 = metrics.f1_score(y_true, y_pred) * 100
        test_acc = test_acc * 100

        logging.warning("+++++++++=RESULTS=+++++++++")
        logging.warning("Accuracy: %.2f%%" % test_acc)
        logging.warning("Precision: %.2f%%" % precision)
        logging.warning("Recall: %.2f%%" % recall)
        logging.warning("F1-Score: %.2f%%" % f1)
        logging.warning("Loss: %.4f" % test_loss)
        logging.warning("+++++++++++++==++++++++++++")

        logging.warning("Evaluation process done")

        return test_acc, test_loss

    def get_accuracy(self):
        acc_now, loss_now = self.evaluate(self.model_now)
        acc_new, loss_new = self.evaluate(self.model_new)

        return acc_now, acc_new, loss_now, loss_new
