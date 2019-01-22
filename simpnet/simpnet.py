    # ========================================
# [] File Name : simpnet.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import time
from itertools import chain
from cnn_util import conv_bn_sc_relu, saf_pool
from cnn_config import *
from utils import *
from sklearn.metrics import accuracy_score, roc_auc_score

class SimpNet(object):

    def __init__(self):

        # Dropout rate
        self.keep_prob = 1.0

        # Learning rate
        self.learning_rate = 1e-5

        # Setup the data path (folder should contain train and test folders inside itself)
        self.data_path = '../data/images/ROI_eq/'
        self.main_csv = '../data/Data_Entry_2017.csv'
        self.train_val_csv = shuffle_csv('../data/train_val_list.csv')
        self.test_csv = shuffle_csv('../data/test_list.csv')

        # Number of images in each batch
        self.batch_size = 20

        # Number of classes
        self.n_classes = 14

        # Global step (times the graph seen the data)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
        # Shows the overall graph status (Trainig vs Testing)
        self.training = True

        # Which steps show the loss in each epoch
        self.skip_steps = 50

        self.result_step = 2000 / self.batch_size

        self.n_test = 25596

        self.train_list = pd.read_csv(self.main_csv)

        # Sample the data for faster training
        # self.train_list = self.train_list.sample(1000)

        self.class_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        # Initial data preprocessing
        self.preprocess()

    def preprocess(self):
        # Convert to one hot
        self.train_list['Finding Labels'] = self.train_list['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        self.all_labels = np.unique(list(chain(*self.train_list['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        self.all_labels = [x for x in self.all_labels if len(x)>0]
        for c_label in self.all_labels:
            if len(c_label)>1: # leave out empty labels
                self.train_list[c_label] = self.train_list['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    def get_data(self):

        with tf.name_scope('data'):
            
            train_data_generator = lambda: self.nih_data_generator(images_path=self.data_path, from_target=self.train_val_csv)
            test_data_generator = lambda: self.nih_data_generator(images_path=self.data_path, from_target=self.test_csv)

            train_data = tf.data.Dataset.from_generator(
            generator=train_data_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
            ).batch(self.batch_size).prefetch(2)
            
            test_data = tf.data.Dataset.from_generator(
            generator=test_data_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
            ).batch(self.batch_size).prefetch(2)

            iterator = train_data.make_initializable_iterator()

            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, [-1, CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH, CNN_INPUT_CHANNELS])

            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def nih_data_generator(self, images_path, from_target):
        with open(from_target, 'r') as f:
            for image in f.readlines():
                image = image.strip()
                img = os.path.join(images_path, image)
                a = cv2.imread(img)
                if a is None:
                    print("Unable to read image", img)
                    continue
                
                a = cv2.resize(a, (256, 256))

                # Normalize
                a = a / 255.0
                
                # print("Data", a)
                yield (np.array(a.flatten()), self.train_list.loc[self.train_list[self.train_list["Image Index"] == image].index.item(), self.all_labels].as_matrix())


    def build_network_graph(self):

        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def inference(self):

        conv1 = conv_bn_sc_relu(
            inputs=self.img,
            filters=CONV1_NUM_FILTERS,
            k_size=CONV1_FILTER_SIZE,
            stride=2,
            padding='SAME',
            scope_name='conv_1',
            keep_prob=self.keep_prob
        )
    
        conv2 = conv_bn_sc_relu(
            inputs=conv1,
            filters=CONV2_NUM_FILTERS,
            k_size=CONV2_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_2',
            keep_prob=self.keep_prob
        )

        conv3 = conv_bn_sc_relu(
            inputs=conv2,
            filters=CONV3_NUM_FILTERS,
            k_size=CONV3_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_3',
            keep_prob=self.keep_prob
        )

        conv4 = conv_bn_sc_relu(
            inputs=conv3,
            filters=CONV4_NUM_FILTERS,
            k_size=CONV4_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_4',
            keep_prob=self.keep_prob
        )

        conv5 = conv_bn_sc_relu(
            inputs=conv4,
            filters=CONV5_NUM_FILTERS,
            k_size=CONV5_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_5',
            keep_prob=self.keep_prob
        )

        pool1 = saf_pool(
            inputs=conv5,
            k_size=MAXPOOL1_SIZE,
            stride=2,
            padding='VALID',
            scope_name='saf_pool_1'
        )

        conv6 = conv_bn_sc_relu(
            inputs=pool1,
            filters=CONV6_NUM_FILTERS,
            k_size=CONV6_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_6',
            keep_prob=self.keep_prob
        )

        conv7 = conv_bn_sc_relu(
            inputs=conv6,
            filters=CONV7_NUM_FILTERS,
            k_size=CONV7_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_7',
            keep_prob=self.keep_prob
        )

        conv8 = conv_bn_sc_relu(
            inputs=conv7,
            filters=CONV8_NUM_FILTERS,
            k_size=CONV8_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_8',
            keep_prob=self.keep_prob
        )

        conv9 = conv_bn_sc_relu(
            inputs=conv8,
            filters=CONV9_NUM_FILTERS,
            k_size=CONV9_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_9',
            keep_prob=self.keep_prob
        )

        conv10 = conv_bn_sc_relu(
            inputs=conv9,
            filters=CONV10_NUM_FILTERS,
            k_size=CONV10_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_10',
            keep_prob=self.keep_prob
        )

        pool2 = saf_pool(
            inputs=conv10,
            k_size=MAXPOOL2_SIZE,
            stride=2,
            padding='VALID',
            scope_name='saf_pool_2'
        )

        conv11 = conv_bn_sc_relu(
            inputs=pool2,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_11',
            keep_prob=self.keep_prob
        )

        conv12 = conv_bn_sc_relu(
            inputs=conv11,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_12',
            keep_prob=self.keep_prob
        )

        conv13 = conv_bn_sc_relu(
            inputs=conv12,
            filters=CONV13_NUM_FILTERS,
            k_size=CONV13_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_13',
            keep_prob=self.keep_prob
        )

        global_pool = saf_pool(
            inputs=conv13,
            k_size=MAXPOOL3_SIZE,
            stride=2,
            padding='VALID',
            scope_name='global_saf_pool'
        )

        flattened = tf.layers.flatten(
            inputs=global_pool,
            name='flatten_input'
        )

        # Softmax is applied in the loss function when softmax_cross_entropy_with_logits is called
        self.logits = tf.layers.dense(
            inputs=flattened,
            units=self.n_classes,
            name='fc_final'
        )


    def loss(self):

        with tf.name_scope('loss'):
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)

            # Loss is mean of error on all dimensions
            self.loss_val = tf.reduce_mean(entropy, name='loss')


    def optimize(self):

        with tf.name_scope('optimizer'):
            # self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val, global_step=self.gstep)
            self.opt = tf.train.GradientDescentOptimizer(
                 learning_rate=self.learning_rate,
                 name='SGD'
            ).minimize(self.loss_val, global_step=self.gstep)

    def summary(self):

        with tf.name_scope('sumstep_accuracymary'):
            tf.summary.scalar('step_accuracyloss', self.loss_val)
            # tf.summary.scalar('step_accuracyaccuracy', self.step_accuracy)
            # tf.summary.histogstep_accuracyram('loss histogram', self.loss_val)
            self.summary_op = tf.summary.merge_all()

    def eval(self):

        with tf.name_scope('predict'):

            # preds = tf.round(self.logits)

            # normalize_a = tf.nn.l2_normalize(self.logits,0)        
            # normalize_b = tf.nn.l2_normalize(self.label,0)
            
            self.sigmoided_logits = tf.round(tf.cast(tf.nn.sigmoid(self.logits), tf.float32))
            self.label = tf.round(tf.cast(self.label, tf.float32))

            # self.step_accuracy=tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            # self.step_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(self.logits)), tf.round(self.label)), dtype=tf.float32))

            # self.ground_truth = tf.reduce_sum(tf.cast(self.label, tf.float32))
            # self.step_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    def train_network_one_epoch(self, sess, init, saver, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        print("[INFO]: Initialized Training...")
        self.training = True

        n_batches = 0
        total_loss = 0
        total_accuracy = 0

        appended_preds = pd.DataFrame(columns=self.class_list)
        appended_labels = pd.DataFrame(columns=self.class_list)
        
        try:
            while True:

                # Run the training graph nodes
                actual_img, preds, labels, _, step_loss, step_summary = sess.run([self.img, self.sigmoided_logits, self.label, self.opt, self.loss_val, self.summary_op])

                step += 1
                total_loss += step_loss
                n_batches += 1

                # Merge batch preds and labels
                for prediction in preds:
                    appended_preds.loc[len(appended_preds)] = prediction
                
                for label in labels:
                    appended_labels.loc[len(appended_labels)] = label
                    
                # print("preds: ", preds)
                # print("labels: ", label)
                
                if(((step + 1) % self.skip_steps) == 0):
                    writer.add_summary(step_summary, global_step=step)
                
                # Save result predictions
                if(((step + 1) % self.result_step) == 0):

                    # Save its predicted vector
                    with open('../results/vis/predicted_vectors.txt', 'a') as predfile:
                        what_to_write = "epoch pred: " + str(epoch) + ": " + str(preds[0]) + '\n'
                        gt_writer = "epoch grnd: " + str(epoch) + ": " + str(labels[0]) + '\n'
                        predfile.write(what_to_write)
                        predfile.write(gt_writer)

                # Stepwise loss
                
                # print("[LOSS - TRAIN (STEP)] at Step {0}: {1}".format(step, step_loss))
        except tf.errors.OutOfRangeError:
            pass

        # Overall loss
        print("[LOSS - TRAIN (EPOCH)] at Epoch {0}: {1}".format(epoch, total_loss/n_batches))
        # Epoch Accuracy
        
        class_scores = np.zeros(shape=len(self.class_list))

        for class_idx in range(len(self.class_list)):
            class_name = self.class_list[class_idx]
            class_scores[class_idx] = roc_auc_score(appended_labels[class_name], appended_preds[class_name])

        avg_accuracy = np.mean(class_scores)
        print("\n**************************************")
        print("[CLASS ROC-AUC SCORES]: ")
        for class_idx in range(len(self.class_list)):
            class_name = self.class_list[class_idx]
            print("{0} -> {1}".format(class_name, class_scores[class_idx]))

        print("**************************************\n")
        print("[ACCURACY - TRAIN] at Epoch {0}: {1}".format(epoch, avg_accuracy))
        print("[TIMING] Took {0} Seconds...".format(time.time() - start_time))

        return step


    def draw_class_maps(self):

        img = self.img
        logits = self.logits

        max_prop = logits[0]

        for i in logits.eval():
            candidate = logits[i]

            if candidate > max_prop:
                max_prop = candidate

        print("Maximum is: {0}".format(max_prop))

    def evaluate_network(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        print("[INFO]: Initialized Validation...")
        self.traininig = False

        n_batches = 0
        total_truths = 0

        try:
            while True:

                # Test the network
                step_accuracy, step_summary = sess.run([self.step_accuracy, self.true_predicted_count, self.total_count, self.summary_op])
                n_batches += 1
                total_truths += step_accuracy
                writer.add_summary(step_summary, global_step=step)

        except tf.errors.OutOfRangeError:
            pass
        
        print("[ACCURACY - VALIDATION] at Epoch {0}: {1}".format(epoch, (total_truths/self.n_test) * 100))
        print("[TIMING] Took {0} Seconds...".format(time.time() - start_time))



    def train(self, n_epochs):

        safe_mkdir('../results/checkpoints')
        safe_mkdir('../results/checkpoints/simpnet_train')
        train_writer = tf.summary.FileWriter('../results/graphs/simpnet_train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter('../results/graphs/simpnet_test')

        with tf.Session() as sess:

            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            # Check if there exists a training checkpoint
            saver = tf.train.Saver()

            # Restore the checkpoints (in case of any!)
            ckpt = tf.train.get_checkpoint_state('../results/checkpoints/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("[INFO]: Reloaded From Checkpoints...")
            else:
                print("[INFO]: No Checkpoint Found...")

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                # Train the model for one epoch
                step = self.train_network_one_epoch(
                   sess=sess,
                   init=self.train_init,
                   saver=saver,
                   writer=train_writer,
                   epoch=epoch,
                   step=step
                )

                # Evaluate the model after each epoch
                # self.evaluate_network(
                #     sess=sess,
                #     init=self.test_init,
                #     writer=test_writer,
                #     epoch=epoch,
                #     step=step
                # )
        train_writer.close()
        test_writer.close()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    model = SimpNet()
    model.build_network_graph()
    model.train(n_epochs=200)
