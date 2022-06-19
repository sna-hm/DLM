import os
import sys
import codecs
import pickle
import gensim
import urllib
import logging

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from connection import DBConnection

import numpy as np
from numpy import load
from numpy import save
from numpy import asarray
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils import batch_iterator
from spektral.utils.data import Batch

import networkx as nx

from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

class RetrainModel:
    def __init__(self):
        # Using sys.setrecursionlimit() method
        sys.setrecursionlimit(3000)
        self.data_dir = "data"
        self.tmp_dir = "tmp"
        self.doc2vec = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
        self.df_tr, self.df_val, self.df_te = self.getData()
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.node = 0
        self.nodes_list = []
        self.attr_list = []

    def getData(self):
        try:
            connection =  DBConnection("moraphishdet").get_connection()
            cursor = connection.cursor()

            cursor.callproc('sp_set_dl_training_data')
            connection.commit()
            cursor.close()

        except mysql.connector.Error as error:
            print("Failed to execute stored procedure: {}".format(error))

        df_tr = pd.read_sql('SELECT * FROM tmp_tr_data', con= DBConnection("moraphishdet").get_connection())
        df_val = pd.read_sql('SELECT * FROM tmp_val_data', con= DBConnection("moraphishdet").get_connection())
        df_te = pd.read_sql('SELECT * FROM tmp_te_data', con= DBConnection("moraphishdet").get_connection())

        return df_tr, df_val, df_te

    def convert(self, list):
        return tuple(list)

    def callTag(self, tag):
        node_from = self.node
        self.node = self.node + 1

        for tag in tag.children:
            attr_node = 1
            if (tag.name is not None):
                self.nodes_list.append(self.convert([str(node_from), str(self.node)]))
                self.attr_list.append(str(self.node) + ",nname," + str(tag.name))
                self.attr_list.append(str(self.node) + ",value," + "")
                for attr in tag.attrs:
                    self.nodes_list.append(self.convert([str(self.node), str(self.node) + "_" + str(attr_node)]))
                    self.attr_list.append(str(self.node) + "_" + str(attr_node) + ",nname," + str(attr))
                    self.attr_list.append(str(self.node) + "_" + str(attr_node) + ",value," + str(tag.get(attr)))
                    attr_node = attr_node + 1
                self.callTag(tag)

    def read_corpus(self, fname, tokens_only=False):
        for i, line in enumerate(fname):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def get_udst_inputs(self, url, html_file, phishing_flag):
        f = codecs.open(Path(self.data_dir + '/' + html_file), 'r', encoding='utf-8', errors='ignore')
        soup = BeautifulSoup(f)
        self.callTag(soup)

        self.nodes_list.pop(0)
        self.attr_list.pop(1)
        self.attr_list.insert(1, '1, value, ' + url)

        G = nx.Graph()
        G.add_edges_from(self.nodes_list)

        A = nx.adjacency_matrix(G) # N*N Adjacency matrix

        df_features = pd.DataFrame(0.0, index=G.nodes(), columns=['nname', 'value'])

        attr_val_list = []
        for x in self.attr_list:
            y = x.split(',')
            if y[1] == "value":
                attr_val_list.append(y[2])

        for x in self.attr_list:
            y = x.split(',')
            test_corpus = list(self.read_corpus([y[2]], tokens_only=True))
            vector = self.doc2vec.infer_vector(test_corpus[0])

            if y[1] == "nname":
                df_features.loc[y[0]]["nname"] = vector

            else:
                df_features.loc[y[0]]["value"] = vector

        X = df_features.values # N*d Feature Matrix
        y = to_categorical(phishing_flag, num_classes=2).tolist() # Label

        X_ = np.array([np.array(ai, dtype=np.float32) for ai in X.tolist()])

        url_token = self.url_tokenizer([url])[0]

        return X_, A, y, url_token, url

    def url_tokenizer(self, url):
        url_int_tokens = self.tokenizer.texts_to_sequences(url)
        return sequence.pad_sequences(url_int_tokens, maxlen=150, padding='post')

    def fetch_tr_data(self):
        for _, data in self.df_tr.iterrows():
            try:
                X, A, y, X_A, _ = self.get_udst_inputs(data['url'], data['website'], data['result'])
            except:
                logging.warning(data['url'] + " | " + data['website'] + " is having an error.")
                self.reset()
                continue

            with open(Path(self.tmp_dir + '/datasets/tr_feat.pkl'), 'ab') as f1:
                pickle.dump(X, f1)
            with open(Path(self.tmp_dir + '/datasets/tr_adj.pkl'), 'ab') as f2:
                pickle.dump(A, f2)
            with open(Path(self.tmp_dir + '/datasets/tr_class.pkl'), 'ab') as f3:
                pickle.dump(y, f3)
            with open(Path(self.tmp_dir + '/datasets/tr_ma.pkl'), 'ab') as f4:
                pickle.dump(X_A, f4)

            self.reset()

        for _type_ in ['adj','class','feat','ma']:
            file = Path(self.tmp_dir + '/datasets/tr_' + _type_ + '.pkl')
            data = []

            with open(file, 'rb') as f:
                try:
                    while True:
                        data.append(pickle.load(f))
                except EOFError:
                    pass

            # save numpy array as npy file
            if _type_ == 'feat':
                data_feat = np.array([np.array(ai, dtype=np.float32) for ai in data])
                save(self.tmp_dir + '/datasets/tr_feat.npy', data_feat)
            elif _type_ == 'adj':
                data_adj = asarray(data)
                save(self.tmp_dir + '/datasets/tr_adj.npy', data_adj)
            elif _type_ == 'class':
                data_class = asarray(data, dtype=np.float32)
                save(self.tmp_dir + '/datasets/tr_class.npy', data_class)
            elif _type_ == 'ma':
                data_ma = asarray(data)
                save(self.tmp_dir + '/datasets/tr_ma.npy', data_ma)
            data = None
            os.remove(file)

    def fetch_val_data(self):
        for _, data in self.df_val.iterrows():
            try:
                X, A, y, X_A, _ = self.get_udst_inputs(data['url'], data['website'], data['result'])
            except:
                logging.warning(data['url'] + " | " + data['website'] + " is having an error.")
                self.reset()
                continue

            with open(Path(self.tmp_dir + '/datasets/val_feat.pkl'), 'ab') as f1:
                pickle.dump(X, f1)
            with open(Path(self.tmp_dir + '/datasets/val_adj.pkl'), 'ab') as f2:
                pickle.dump(A, f2)
            with open(Path(self.tmp_dir + '/datasets/val_class.pkl'), 'ab') as f3:
                pickle.dump(y, f3)
            with open(Path(self.tmp_dir + '/datasets/val_ma.pkl'), 'ab') as f4:
                pickle.dump(X_A, f4)

            self.reset()

        for _type_ in ['adj','class','feat','ma']:
            file = Path(self.tmp_dir + '/datasets/val_' + _type_ + '.pkl')
            data = []

            with open(file, 'rb') as f:
                try:
                    while True:
                        data.append(pickle.load(f))
                except EOFError:
                    pass

            # save numpy array as npy file
            if _type_ == 'feat':
                data_feat = np.array([np.array(ai, dtype=np.float32) for ai in data])
                save(self.tmp_dir + '/datasets/val_feat.npy', data_feat)
            elif _type_ == 'adj':
                data_adj = asarray(data)
                save(self.tmp_dir + '/datasets/val_adj.npy', data_adj)
            elif _type_ == 'class':
                data_class = asarray(data, dtype=np.float32)
                save(self.tmp_dir + '/datasets/val_class.npy', data_class)
            elif _type_ == 'ma':
                data_ma = asarray(data)
                save(self.tmp_dir + '/datasets/val_ma.npy', data_ma)
            data = None
            os.remove(file)

    def fetch_te_data(self):
        for _, data in self.df_te.iterrows():
            try:
                X, A, y, X_A, _ = self.get_udst_inputs(data['url'], data['website'], data['result'])
            except:
                logging.warning(data['url'] + " | " + data['website'] + " is having an error.")
                self.reset()
                continue

            with open(Path(self.tmp_dir + '/datasets/te_feat.pkl'), 'ab') as f1:
                pickle.dump(X, f1)
            with open(Path(self.tmp_dir + '/datasets/te_adj.pkl'), 'ab') as f2:
                pickle.dump(A, f2)
            with open(Path(self.tmp_dir + '/datasets/te_class.pkl'), 'ab') as f3:
                pickle.dump(y, f3)
            with open(Path(self.tmp_dir + '/datasets/te_ma.pkl'), 'ab') as f4:
                pickle.dump(X_A, f4)

            self.reset()

        for _type_ in ['adj','class','feat','ma']:
            file = Path(self.tmp_dir + '/datasets/te_' + _type_ + '.pkl')
            data = []

            with open(file, 'rb') as f:
                try:
                    while True:
                        data.append(pickle.load(f))
                except EOFError:
                    pass

            # save numpy array as npy file
            if _type_ == 'feat':
                data_feat = np.array([np.array(ai, dtype=np.float32) for ai in data])
                save(self.tmp_dir + '/datasets/te_feat.npy', data_feat)
            elif _type_ == 'adj':
                data_adj = asarray(data)
                save(self.tmp_dir + '/datasets/te_adj.npy', data_adj)
            elif _type_ == 'class':
                data_class = asarray(data, dtype=np.float32)
                save(self.tmp_dir + '/datasets/te_class.npy', data_class)
            elif _type_ == 'ma':
                data_ma = asarray(data)
                save(self.tmp_dir + '/datasets/te_ma.npy', data_ma)
            data = None
            os.remove(file)

    def train(self):
        # Load data
        X_train_A = load(self.tmp_dir + '/datasets/tr_ma.npy', allow_pickle=True)
        X_test_A = load(self.tmp_dir + '/datasets/te_ma.npy', allow_pickle=True)
        X_val_A = load(self.tmp_dir + '/datasets/val_ma.npy', allow_pickle=True)

        X_train_B, A_train_B, y_train_B = load(self.tmp_dir + '/datasets/tr_feat.npy', allow_pickle=True), list(load(self.tmp_dir + '/datasets/tr_adj.npy', allow_pickle=True)), load(self.tmp_dir + '/datasets/tr_class.npy', allow_pickle=True)
        X_test_B, A_test_B, y_test_B = load(self.tmp_dir + '/datasets/te_feat.npy', allow_pickle=True), list(load(self.tmp_dir + '/datasets/te_adj.npy', allow_pickle=True)), load(self.tmp_dir + '/datasets/te_class.npy', allow_pickle=True)
        X_val_B, A_val_B, y_val_B = load(self.tmp_dir + '/datasets/val_feat.npy', allow_pickle=True), list(load(self.tmp_dir + '/datasets/val_adj.npy', allow_pickle=True)), load(self.tmp_dir + '/datasets/val_class.npy', allow_pickle=True)

        # Preprocessing adjacency matrices for convolution
        A_train_B = [normalized_adjacency(a) for a in A_train_B]
        A_val_B = [normalized_adjacency(a) for a in A_val_B]
        A_test_B = [normalized_adjacency(a) for a in A_test_B]

        # Load pre-trained model
        model = load_model('moraphishdet.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool': MinCutPool,
                                                        'GlobalAvgPool': GlobalAvgPool})
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
        epochs = 500               # Number of training epochs
        es_patience = 20           # Patience for early stopping
        learning_rate = 1e-5       # Learning rate
        batch_size = 1             # Batch size. NOTE: it MUST be 1 when using MinCutPool and DiffPool

        #model.summary()

        # Training setup
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = model.loss_functions[0]
        acc_fn = lambda x, y: K.mean(tf.keras.metrics.categorical_accuracy(x, y))

        # Training function
        @tf.function(experimental_relax_shapes=True)
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, acc_fn(targets, predictions)

        ################################################################################
        current_batch = 0
        model_loss = 0
        model_loss_values = []
        model_val_loss_values = []
        model_acc = 0
        model_acc_values = []
        model_val_acc_values = []
        best_val_loss = np.inf
        best_weights = None
        patience = es_patience
        batches_in_epoch = np.ceil(y_train_B.shape[0] / batch_size)

        ################################################################################
        # FITTING MODEL
        ################################################################################

        logging.warning('Fitting model')

        batches = batch_iterator([A_train_B, X_train_B, y_train_B, X_train_A], batch_size=batch_size, epochs=epochs)
        for b in batches:
            X_, A_, I_ = Batch(b[0], b[1]).get('XAI')
            A_ = sp_matrix_to_sp_tensor(A_)
            y_ = b[2]
            X_A_ = b[3]
            outs = train_step([X_A_, X_, A_, I_], y_)

            model_loss += outs[0]
            model_acc += outs[1]
            current_batch += 1
            if current_batch % batches_in_epoch == 0:
                model_loss /= batches_in_epoch
                model_acc /= batches_in_epoch

                # Compute validation loss and accuracy
                val_loss, val_acc = evaluate(A_val_B, X_val_B, y_val_B, X_val_A, [loss_fn, acc_fn], batch_size=batch_size)
                logging.warning('Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
                      .format(current_batch // batches_in_epoch, model_loss, model_acc, val_loss, val_acc))

                # Check if loss improved for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = es_patience
                    logging.warning('New best val_loss {:.3f}'.format(val_loss))
                    best_weights = model.get_weights()
                else:
                    patience -= 1
                    if patience == 0:
                        logging.warning('Early stopping (best val_loss: {})'.format(best_val_loss))
                        break
                model_loss_values.append(model_loss)
                model_acc_values.append(model_acc)
                model_val_loss_values.append(val_loss)
                model_val_acc_values.append(val_acc)
                model_loss = 0
                model_acc = 0

        ################################################################################
        # EVALUATE MODEL
        ################################################################################

        # Load best model
        model.set_weights(best_weights)

        # Save model
        model.save(self.tmp_dir + '/model/moraphishdet.h5')

        # Test model
        logging.warning('Testing model')

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

        with open(Path(self.tmp_dir + '/model/model_acc.pkl'), 'wb') as f1:
            pickle.dump(model_acc_values,f1)

        with open(Path(self.tmp_dir + '/model/model_loss.pkl'), 'wb') as f2:
            pickle.dump(model_loss_values,f2)

        with open(Path(self.tmp_dir + '/model/model_val_acc.pkl'), 'wb') as f3:
            pickle.dump(model_val_acc_values,f3)

        with open(Path(self.tmp_dir + '/model/model_val_loss.pkl'), 'wb') as f4:
            pickle.dump(model_val_loss_values,f4)

        confusion_m = confusion_matrix(y_true, y_pred)
        logging.warning(confusion_m)

        logging.warning("DL retrain process completed")
