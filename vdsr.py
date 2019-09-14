import tensorflow as tf
import numpy as np

import os, time
from tqdm import tqdm

import h5py

from utils import *
from dataset import Dataset


class VDSR():
    def __init__(self, config):
        # Network setting
        self.layer_depth = config.layer_depth

        # Learning schedule
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size

        # Log interval
        self.PRINT_INTERVAL = config.print_interval
        self.EVAL_INTERVAL = config.eval_interval

        # Upscale factor, only available in test stage 
        self.scale = config.scale

        # Others
        self.CHECKPOINT_PATH = config.checkpoint_path
        self.MODEL_PATH = os.path.join(config.model_path, 'model')
        self.TRAIN_DATASET_PATH = config.train_dataset_path
        self.TRAIN_DATASET = config.train_dataset
        self.VALID_DATASET = config.valid_dataset
        self.TEST_DATASET_PATH = config.test_dataset_path
        self.TEST_DATASET = config.test_dataset
        self.RESULT_DIR = config.result_dir
        
        # Build network computational graph
        self.build_network()
        

    # Build network architecture
    def build_network(self):
        initializer_w = tf.initializers.he_normal()
        initializer_b = tf.constant_initializer(0)

        with tf.variable_scope('VDSR'):
            self.weights_t = {
                'w1': tf.get_variable('w1', [3, 3, 1, 64], initializer=initializer_w, dtype=tf.float32),
                'w{:d}'.format(self.layer_depth): tf.get_variable('w{:d}'.format(self.layer_depth), [3, 3, 64, 1], initializer=initializer_w, dtype=tf.float32)
            }
            
            self.biases_t = {
                'b1': tf.get_variable('b1', [64], initializer=initializer_b, dtype=tf.float32),
                'b{:d}'.format(self.layer_depth): tf.get_variable('b{:d}'.format(self.layer_depth), [1], initializer=initializer_b, dtype=tf.float32)
            }
            
            for i in range(2, self.layer_depth):
                self.weights_t['w{:d}'.format(i)] = tf.get_variable('w{:d}'.format(i), [3, 3, 64, 64], initializer=initializer_w, dtype=tf.float32)
                self.biases_t['b{:d}'.format(i)] = tf.get_variable('b{:d}'.format(i), [64], initializer=initializer_b, dtype=tf.float32)


    # Forward pass
    def forward_pass(self, input_t):
        conv = tf.nn.conv2d(input_t, self.weights_t['w1'], strides=[1,1,1,1], padding='SAME', name='conv1')
        conv = tf.nn.bias_add(conv, self.biases_t['b1'], name='bias_add1')
        conv = tf.nn.relu(conv)

        for i in range(2, self.layer_depth):
            conv = tf.nn.conv2d(conv, self.weights_t['w{:d}'.format(i)], strides=[1,1,1,1], padding='SAME', name='conv{:d}'.format(i))
            conv = tf.nn.bias_add(conv, self.biases_t['b{:d}'.format(i)], name='bias_add{:d}'.format(i))
            conv = tf.nn.relu(conv)

        conv = tf.nn.conv2d(conv, self.weights_t['w{:d}'.format(self.layer_depth)], strides=[1,1,1,1], padding='SAME', name='conv{:d}'.format(self.layer_depth))
        conv = tf.nn.bias_add(conv, self.biases_t['b{:d}'.format(self.layer_depth)], name='bias_add{:d}'.format(self.layer_depth))
                
        output = tf.add(conv, input_t, name='residual')
        
        return tf.clip_by_value(output, 0., 1.)
 

    # Training stage
    def train(self):
        print('\n[*] VDSR training will be started !\n')

        train_path = os.path.join(self.TRAIN_DATASET_PATH, self.TRAIN_DATASET)
        valid_path = os.path.join(self.TEST_DATASET_PATH, self.VALID_DATASET)

        if not exist_train_data(train_path):
            print('[!] No train data ready .. Please generate train data first with Matlab')
            return

        else:
            dataset = Dataset(data_path=train_path, batch_size=self.batch_size)
            
            print('[*] Successfully load train data !\n')

        self.valid_images_x2, self.valid_labels_x2 = prepare_data(path=valid_path, scale=2, is_valid=True)
        self.valid_images_x3, self.valid_labels_x3 = prepare_data(path=valid_path, scale=3, is_valid=True)
        self.valid_images_x4, self.valid_labels_x4 = prepare_data(path=valid_path, scale=4, is_valid=True)
        self.best_psnr_x2 = 0
        self.best_psnr_x3 = 0
        self.best_psnr_x4 = 0

        self.input_t = tf.placeholder(tf.float32, [None, None, None, 1], name='input')
        self.label_t = tf.placeholder(tf.float32, [None, None, None, 1], name='label')
        
        learning_rate_t = tf.Variable(self.learning_rate, dtype=tf.float32, trainable=False, name='learning_rate')

        self.output_t = self.forward_pass(self.input_t)

        # Loss Function: Mean Squared Error
        # Model predict residual image (r = y - x)
        # r - forward = y - x - forward
        loss_t = tf.reduce_mean(tf.square(self.label_t - self.output_t))
        
        # Optimizer: AdamWOptimizer
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_t, name='adam').minimize(loss_t)
        
        self.saver = tf.train.Saver()

        # Handling GPU consumption
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as self.sess:
            # initialize TensorFlow variables in session
            self.sess.run(tf.global_variables_initializer())

            dataset.start_enqueue_deamon(self.sess)

            loss = []
        
            print('[*] Start training ... Please be patient !\n')
            
            try:
                for i in range(self.epoch):
                    desc = '[*] Epoch {:d} '.format(i)
                    
                    print('[*] Epoch: [{:d}], Learning rate: [{:.4f}]\n'.format(i, self.sess.run(learning_rate_t)))

                    for idx in tqdm(range(dataset.data_index), desc=desc, leave=False, disable=True):
                        batch_images, batch_labels = self.sess.run(dataset.dequeue_many)
                        
                        _, _loss = self.sess.run([train_op, loss_t], feed_dict={self.input_t: batch_images, self.label_t: batch_labels})

                        loss.append(_loss)
                        
                        if (idx+1) % self.PRINT_INTERVAL == 0:
                            # Print loss 
                            print('[*] Epoch: [{:d}], Iteration: [{:d}/{:d}], Loss: [{:.8f}]'.format(i, idx+1, dataset.data_index, np.mean(loss)), flush=True)

                            loss = []

                        if ((idx+1) % self.EVAL_INTERVAL == 0) or ((idx+1) == dataset.data_index):
                            # Evaluate PSNR values of Set5 in multi-scale
                            self.validate(epoch=i, iteration=idx+1)

                print('[*] Training done !')

            except KeyboardInterrupt:
                print('[!] HALTED !')

            
    # Validate stage
    def validate(self, epoch, iteration):
        psnr_x2 = []
        psnr_x3 = []
        psnr_x4 = []
        for i in range(len(self.valid_images_x2)):
            forward_x2 = self.sess.run(self.output_t, feed_dict={self.input_t: self.valid_images_x2[i]})
            forward_x3 = self.sess.run(self.output_t, feed_dict={self.input_t: self.valid_images_x3[i]})
            forward_x4 = self.sess.run(self.output_t, feed_dict={self.input_t: self.valid_images_x4[i]})
                                
            psnr_x2.append(psnr(self.valid_labels_x2[i][0], forward_x2[0], shave=2))
            psnr_x3.append(psnr(self.valid_labels_x3[i][0], forward_x3[0], shave=3))
            psnr_x4.append(psnr(self.valid_labels_x4[i][0], forward_x4[0], shave=4))

        print('\n[*] Epoch: [{:d}], Iteration: [{:d}], Evaluate PSNR: [X2: {:.2f} / X3: {:.2f} / X4: {:.2f}]\n'.format(
            epoch, iteration, np.mean(psnr_x2), np.mean(psnr_x3), np.mean(psnr_x4)
            ), flush=True)
        
        if ((np.mean(psnr_x2)+np.mean(psnr_x3)+np.mean(psnr_x4)) > (self.best_psnr_x2+self.best_psnr_x3+self.best_psnr_x4)):
            print('[*] Best PSNR value updated !')
            print('[*] [X2: {:.2f} -> {:.2f} / X3: {:.2f} -> {:.2f} / X4: {:.2f} -> {:.2f}]\n'.format(
                self.best_psnr_x2, np.mean(psnr_x2), self.best_psnr_x3, np.mean(psnr_x3), self.best_psnr_x4, np.mean(psnr_x4)
                ), flush=True)
            self.best_psnr_x2 = np.mean(psnr_x2)
            self.best_psnr_x3 = np.mean(psnr_x3)
            self.best_psnr_x4 = np.mean(psnr_x4)
            # Save model
            self.saver.save(self.sess, os.path.join(self.CHECKPOINT_PATH, 'model'), write_meta_graph=False)
            print('[*] Save checkpoint\n')            


    # Test stage
    def test(self):
        print('[*] VDSR testing will be started ! ')
        t = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))

        test_path = os.path.join(self.TEST_DATASET_PATH, self.TEST_DATASET)

        test_images, test_labels = prepare_data(path=test_path, scale=self.scale)

        self.input_t = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
        self.label_t = tf.placeholder(tf.float32, [None, None, None, 1], name='labels')

        self.output_t = self.forward_pass(self.input_t)

        # Handling GPU consumption
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as self.sess:
            self.sess.run(tf.global_variables_initializer())

            # Load model
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.sess, self.MODEL_PATH)
                print('[*] Load checkpoint: {}\n'.format(self.MODEL_PATH))
            except:
                print('[!] Invalid checkpoint path\n')
                return

            results = []
            bicubic_psnr = []
            test_psnr = []
            print('[*] Start testing !')
            for idx in tqdm(range(len(test_images))):
                h, w, _ = test_images[idx].shape
                test_input_y = test_images[idx][:, :, 0]
                test_label_y = test_labels[idx][:, :, 0]

                test_input_cbcr = test_images[idx][:, :, 1:3]
                test_label_cbcr = test_labels[idx][:, :, 1:3]

                test_input_y = test_input_y.reshape([1, h, w, 1])
                test_label_y = test_label_y.reshape([1, h, w, 1])

                test_input_cbcr = test_input_cbcr.reshape([1, h, w, 2])
                test_label_cbcr = test_label_cbcr.reshape([1, h, w, 2])

                output = self.sess.run(self.output_t, feed_dict={self.input_t: test_input_y})
                    
                bicubic_psnr.append(psnr(test_label_y[0], test_input_y[0], shave=self.scale))
                test_psnr.append(psnr(test_label_y[0], output[0], shave=self.scale))

                gt = concat_ycrcb(test_label_y[0], test_label_cbcr[0])
                bicubic = concat_ycrcb(test_input_y[0], test_input_cbcr[0])
                result = concat_ycrcb(output[0], test_input_cbcr[0])
                
                path = os.path.join(os.getcwd(), self.RESULT_DIR)
                path = os.path.join(path, t)
                if not os.path.exists(path):
                    os.makedirs(path)

                save_result(path, gt, bicubic, result, idx)
        
        print('[*] Test dataset: [{}], upscale factor: [X{:d}]'.format(self.TEST_DATASET, self.scale))
        print('[*] PSNR value of ground truth and bicubic : {:.2f}'.format(np.mean(bicubic_psnr)))
        print('[*] PSNR value of ground truth and VDSR    : {:.2f}'.format(np.mean(test_psnr)))
   