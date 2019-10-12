import numpy as np
import os
import matplotlib.pyplot as plt
from tree import *
import pickle
from collections import OrderedDict
import tensorflow as tf
import time

"""
Parameters of network
"""
learning_rate = 0.01
regularization = 0.1
NO_EPOCHS = 10


class RNN():

    def load_data(self):
        """
            Loads training data in correct format
        """

        data = []
        testPercent = 1.0
        data_dic = 'probni/'
        cls = 0

        for p_file in os.listdir(data_dic):
            files = pickle.load(open(data_dic + p_file, 'rb'))
            cls += 1
            for tree_list in files:
                for tree in tree_list:
                    data.append(tree)

        splitIndex = int(round(testPercent * len(data)))

        self.training_data = data[:splitIndex]
        self.validate_data = data[splitIndex:]

        print("Loaded ", cls, "clusters.")

    def add_variables(self):
        """
        Add model variables:
        For PROJECTION LAYER (PROJECTION)
            Wp - projection matrix used for projection layer
            bp - bias used for projection layer

        For RECURSIVE LAYER (RECURSIVE)
            Wt - transition matrix used for forward propagation vector features
            bt - bias for forward propagation of vector features

        For REGRESSION LAYER (REGRESSION)
            Wr1 - regression matrix used for calc. salience scores
            Wr2 - regression matrix used for calc. salience scores with raw word features
            Wr3 - regression matrix used for calc. salience scores with raw sentence features
            br - bias for calc. salience scores (regression process)
        """

        global Wr1_reg, Wr2_reg, Wr3_reg, Wt_reg, Wp_reg, br_reg, bt_reg, bp_reg

        with tf.variable_scope('PROJECTION'):

            tf.get_variable('Wp', dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(Wp_reg, dtype=tf.float32), shape=[15, 8]))
            tf.get_variable("bp", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(bp_reg, dtype=tf.float32), shape=[1, 8]))

        with tf.variable_scope("RECURSIVE"):
            tf.get_variable("Wt", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(Wt_reg, dtype=tf.float32), shape=[16, 8]))
            tf.get_variable("bt", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(bt_reg, dtype=tf.float32), shape=[1, 8]))

        with tf.variable_scope("REGRESSION"):
            tf.get_variable("Wr1", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(Wr1_reg, dtype=tf.float32), shape=[8, 1]))
            tf.get_variable("Wr2", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(Wr2_reg, dtype=tf.float32), shape=[15, 1]))
            tf.get_variable("Wr3", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(Wr3_reg, dtype=tf.float32), shape=[14, 1]))
            tf.get_variable("br", dtype=tf.float32, initializer=tf.reshape(tf.convert_to_tensor(br_reg, dtype=tf.float32), shape=[1, 1]))


    def projection_layer(self, raw_word_tensor):
        """
        Projection layer : raw features -> hidden features
        :param raw_word_tensor:
        :return: hidden_feature_tensor
        """
        hidden_feature_tensor = None

        with tf.variable_scope('PROJECTION', reuse=tf.AUTO_REUSE):
            Wp = tf.get_variable('Wp')
            bp = tf.get_variable("bp")

            #assert raw_word_tensor.shape.as_list()[1] == Wp.shape.as_list()[0]
            hidden_feature_tensor = tf.tanh(tf.add(tf.matmul(raw_word_tensor, Wp), bp))

        #assert isinstance(hidden_feature_tensor, tf.Tensor)
        return hidden_feature_tensor

    def recursive_layer(self, left_feature_tensor, right_feature_tensor):
        """
        Recursive layer : [left_feature, right_feature] -> parent_features
        :param left_feature_tensor:
        :param right_feature_tensor:
        :return: parent_feature_tensor
        """
        parent_feature_tensor = None
        in_tensor = tf.concat([left_feature_tensor, right_feature_tensor], 1)

        with tf.variable_scope("RECURSIVE", reuse=True):
            Wt = tf.get_variable("Wt")
            bt = tf.get_variable("bt")

            #assert in_tensor.shape.as_list()[1] == Wt.shape.as_list()[0]
            parent_feature_tensor = tf.tanh(tf.add(tf.matmul(in_tensor, Wt), bt))

        #assert isinstance(parent_feature_tensor, tf.Tensor)
        return parent_feature_tensor

    def regression_layer(self, feature_tensor, word_raw_tensor=None, sentence_raw_tensor=None, tag="rest"):
        """
        Regression layer : calc. salience score
        :param feature_tensor:
        :param word_raw_tensor:
        :param sentence_raw_tensor:
        :param tag:
        :return: salience_score_tensor
        """
        salience_score_tensor = None

        if tag == "rest":
            with tf.variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                br = tf.get_variable("br")

                #assert feature_tensor.shape.as_list()[1] == Wr1.shape.as_list()[0]
                salience_score_tensor = tf.sigmoid(tf.add(tf.matmul(feature_tensor, Wr1), br))

        if tag == "pre-terminal":
            with tf.variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                Wr2 = tf.get_variable("Wr2")
                br = tf.get_variable("br")

                #assert feature_tensor.shape.as_list()[1] == Wr1.shape.as_list()[0]
                #assert word_raw_tensor.shape.as_list()[1] == Wr2.shape.as_list()[0]
                salience_score_tensor = tf.sigmoid(tf.add(tf.add(tf.matmul(feature_tensor, Wr1), tf.matmul(word_raw_tensor, Wr2)), br))

        if tag == "root":
            with tf.variable_scope("REGRESSION", reuse=True):
                Wr1 = tf.get_variable("Wr1")
                Wr3 = tf.get_variable("Wr3")
                br = tf.get_variable("br")

                #assert feature_tensor.shape.as_list()[1] == Wr1.shape.as_list()[0]
                #assert sentence_raw_tensor.shape.as_list()[1] == Wr3.shape.as_list()[0]
                salience_score_tensor = tf.sigmoid(tf.add(tf.add(tf.matmul(feature_tensor, Wr1), tf.matmul(sentence_raw_tensor, Wr3)), br))

        #assert isinstance(salience_score_tensor, tf.Tensor)
        return salience_score_tensor

    def inference(self, node, sentence_raw_tensor):
        """
        Build computation graph for given node
        :param node:
        :param sentence_raw_tensor:
        :return: feature_tensors_dict, salience_tensors_dict
        """
        # print("KURAC")
        feature_tensors_dict = OrderedDict()
        salience_tensors_dict = OrderedDict()

        if node.label == "ROOT":
            left_ftrs, left_sal = self.inference(node.left, sentence_raw_tensor=sentence_raw_tensor)

            feature_tensor = left_ftrs[node.left]
            salience_tensor = left_sal[node.left]

            feature_tensors_dict.update(left_ftrs)
            feature_tensors_dict[node] = feature_tensor

            salience_tensors_dict.update(left_sal)
            salience_tensors_dict[node] = salience_tensor

        if node.parent is not None:
            if node.parent.label == "ROOT":
                #assert node.left is not None
                left_ftrs, left_sal = self.inference(node.left, sentence_raw_tensor=None)
                #assert node.right is not None
                right_ftrs, right_sal = self.inference(node.right, sentence_raw_tensor=None)

                #assert isinstance(sentence_raw_tensor, tf.Tensor)
                #assert sentence_raw_tensor.shape.as_list() == [1, 14]
                #assert left_ftrs[node.left].shape.as_list() == [1, 8]
                #assert right_ftrs[node.right].shape.as_list() == [1, 8]

                feature_tensor = self.recursive_layer(left_feature_tensor=left_ftrs[node.left], right_feature_tensor=right_ftrs[node.right])
                salience_tensor = self.regression_layer(feature_tensor=feature_tensor, word_raw_tensor=None, sentence_raw_tensor=sentence_raw_tensor, tag="root")

                feature_tensors_dict.update(left_ftrs)
                feature_tensors_dict[node] = feature_tensor
                feature_tensors_dict.update(right_ftrs)

                salience_tensors_dict.update(left_sal)
                salience_tensors_dict[node] = salience_tensor
                salience_tensors_dict.update(right_sal)

        if node.isPreTerminal is True:
            word_raw_tensor = tf.convert_to_tensor(node.left.feature, dtype=tf.float32)
            #assert word_raw_tensor.shape.as_list() == [15, ]
            word_raw_tensor = tf.reshape(word_raw_tensor, shape=[1, 15])
            #assert isinstance(word_raw_tensor, tf.Tensor)

            feature_tensor = self.projection_layer(raw_word_tensor=word_raw_tensor)
            salience_tensor = self.regression_layer(feature_tensor=feature_tensor, word_raw_tensor=word_raw_tensor, sentence_raw_tensor=None, tag="pre-terminal")

            feature_tensors_dict[node] = feature_tensor
            salience_tensors_dict[node] = salience_tensor

        if node.isPreTerminal is not True and node.label != "ROOT" and node.parent.label != "ROOT":
            left_ftrs, left_sal = self.inference(node.left, sentence_raw_tensor=None)
            right_ftrs, right_sal = self.inference(node.right, sentence_raw_tensor=None)

            #assert left_ftrs[node.left].shape.as_list() == [1, 8]
            #assert right_ftrs[node.right].shape.as_list() == [1, 8]

            feature_tensor = self.recursive_layer(left_feature_tensor=left_ftrs[node.left], right_feature_tensor=right_ftrs[node.right])
            salience_tensor = self.regression_layer(feature_tensor=feature_tensor, word_raw_tensor=None, sentence_raw_tensor=None, tag="rest")

            feature_tensors_dict.update(left_ftrs)
            feature_tensors_dict[node] = feature_tensor
            feature_tensors_dict.update(right_ftrs)

            salience_tensors_dict.update(left_sal)
            salience_tensors_dict[node] = salience_tensor
            salience_tensors_dict.update(right_sal)

        return feature_tensors_dict, salience_tensors_dict

    def loss(self, true_salience, calc_salience):
        """
        Loss function for salience scores
        :param true_salience:
        :param calc_salience:
        :return: loss
        """
        Wr1 = None
        Wr2 = None
        Wr3 = None
        Wt = None
        Wp = None
        with tf.variable_scope("REGRESSION", reuse=True):
            Wr1 = tf.get_variable("Wr1")
            Wr2 = tf.get_variable("Wr2")
            Wr3 = tf.get_variable("Wr3")

        with tf.variable_scope("RECURSIVE", reuse=True):
            Wt = tf.get_variable("Wt")

        with tf.variable_scope("PROJECTION", reuse=True):
            Wp = tf.get_variable("Wp")

        suml2norms = tf.nn.l2_loss(Wr1) + tf.nn.l2_loss(Wr2) + tf.nn.l2_loss(Wr3) + tf.nn.l2_loss(Wt) + tf.nn.l2_loss(Wp)
        #print(true_salience.shape)
        #print(calc_salience.shape)

        #cross_entropy = 0
        #for idx in range(len(true_salience)):
        #    cross_entropy += -(true_salience[idx]*np.log(calc_salience[idx]) + (1-true_salience[idx])*np.log(1-calc_salience[idx]))


        #cross_entropy = tf.reduce_mean(-(tf.matmul(true_salience, tf.log(calc_salience)) + tf.matmul(1-true_salience, tf.log(1-calc_salience))))
        #cross_entropy = cross_entropy / len(true_salience)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_salience, logits=calc_salience))
        loss = cross_entropy + regularization * suml2norms
        

        return loss

    def optimizer(self, loss):
        """
        Training optimizer
        :param loss:
        :return: train_op
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

        return train_op

    def run_epoch(self, epoch):
        """
        Runs training of one epoch on whole training data set and writes learned parameters of nets
        :return: losses
        """
        losses = []
        global Wr1_reg, Wr2_reg, Wr3_reg, Wt_reg, Wp_reg, br_reg, bt_reg, bp_reg
        print("Start of epoch ", epoch)

        for idx in range(len(self.training_data)):
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    self.add_variables()
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    tree = self.training_data[idx]
                    sentence_raw_tensor = tf.convert_to_tensor(tree.sentence_features, dtype=tf.float32)
                    #assert sentence_raw_tensor.shape.as_list() == [14, ]
                    sentence_raw_tensor = tf.reshape(sentence_raw_tensor, shape=[1, 14])
                    #assert isinstance(sentence_raw_tensor, tf.Tensor)

                    try:
                        feature_dic, salience_dic = self.inference(tree.root, sentence_raw_tensor=sentence_raw_tensor)
                    except:
                        print("Puko ", idx)
                        continue

                    calc_saliences = []
                    for key, value in salience_dic.items():
                        calc_saliences.append(value)
                    true_saliences = tree.getSaliences()
                    #c_s = calc_saliences.eval()
                    l = len(true_saliences)
                    t_s = tf.convert_to_tensor(true_saliences, dtype=tf.float32)
                    #c_s = tf.convert_to_tensor(calc_saliences, dtype=tf.float32)
                    loss = self.loss(tf.reshape(t_s, shape=[l]), tf.reshape(calc_saliences, shape=[l]))
                    train_op = self.optimizer(loss)
                    train_op.run()
                    loss_read = loss.eval()
                    losses.append(loss_read)
                    #print("\t Sentence ", idx + 1, " in epoch ", epoch, " has loss: ", loss_read)

                    with tf.variable_scope("REGRESSION", reuse=True):
                        Wr1 = tf.get_variable("Wr1")
                        Wr1_reg = Wr1.eval()
                        Wr2 = tf.get_variable("Wr2")
                        Wr2_reg = Wr2.eval()
                        Wr3 = tf.get_variable("Wr3")
                        Wr3_reg = Wr3.eval()
                        br = tf.get_variable("br")
                        br_reg = br.eval()

                    with tf.variable_scope("RECURSIVE", reuse=True):
                        Wt = tf.get_variable("Wt")
                        Wt_reg = Wt.eval()
                        bt = tf.get_variable("bt")
                        bt_reg = bt.eval()

                    with tf.variable_scope("PROJECTION", reuse=True):
                        Wp = tf.get_variable("Wp")
                        Wp_reg = Wp.eval()
                        bp = tf.get_variable("bp")
                        bp_reg = bp.eval()

                    # print(Wr1_reg)

        tf.reset_default_graph()
        print("End of epoch ", epoch, " ! ")
        return losses

    def validate(self):
        """
        Runs validation after one epoch of training
        :return: losses
        """

        losses = []
        for idx in range(len(self.validate_data)):
            tree = self.validate_data[idx]
            sentence_raw_tensor = tf.convert_to_tensor(tree.sentence_features, dtype=tf.float32)
            #assert isinstance(sentence_raw_tensor, tf.Tensor)
            feature_dic, salience_dic = self.inference(tree.root, sentence_raw_tensor=sentence_raw_tensor)
            calc_salience = []
            for key, value in salience_dic.iteritems():
                calc_salience.append(value)
            true_salience = tree.getSaliences()
            loss = self.loss(true_salience, calc_salience)
            losses.append(loss)
        
        return losses

    def training(self):
        """
        Runs training and after that validation through epochs
        Takes results and weights of epoch that has lowest loss
        :return:
        """

        if not os.path.exists("./weights"):
            os.makedirs("./weights")

        if not os.path.exists("./losses"):
            os.makedirs("./losses")
            
        #train_losses = []
        mean_train_losses = []

        for epoch in range(NO_EPOCHS):
                start = time.time()
                train_loss = self.run_epoch(epoch + 1)
                # train_losses.update(train_loss)
                # print("Writing sentence losses in file...")
                # with open("./losses/all_sentence_losses", 'a') as f:
                    # np.savetxt(f, train_loss)
                print("Writting weights in file...")
                with open("./weights/w"+str(epoch), 'w') as f:
                    np.savetxt(f, Wr1_reg)
                    np.savetxt(f, Wr2_reg)
                    np.savetxt(f, Wr3_reg)
                    np.savetxt(f, br_reg)
                    np.savetxt(f, Wt_reg)
                    np.savetxt(f, bt_reg)
                    np.savetxt(f, Wp_reg)
                    np.savetxt(f, bp_reg)

                mean = np.mean(train_loss)
                print("Mean loss in epoch: ", mean)
                mean_train_losses.append(mean)
                end = time.time()
                print("Epoch took ", end-start, " to complete")

        plt.plot(mean_train_losses)
        plt.title("Losses per epoch")
        plt.show()




if __name__ == "__main__":
    start = time.time()
    print("Started!")
    Wr1_reg = np.random.normal(0.0, 0.1, [8, 1])
    Wr2_reg = np.random.normal(0.0, 0.1, [15, 1])
    Wr3_reg = np.random.normal(0.0, 0.1, [14, 1])
    br_reg = np.random.normal(0.0, 0.1, [1, 1])

    Wt_reg = np.random.normal(0.0, 0.1, [16, 8])
    bt_reg = np.random.normal(0.0, 0.1, [1, 8])

    Wp_reg = np.random.normal(0.0, 0.1, [15, 8])
    bp_reg = np.random.normal(0.0, 0.1, [1, 8])



    r =  RNN()
    r.load_data()
    r.training()

    print("Finished training!")
    end = time.time()
    print("Time = ", (int)(end-start))
