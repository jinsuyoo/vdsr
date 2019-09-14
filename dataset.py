import tensorflow as tf
import numpy as np
import threading, time

import h5py


class Dataset():
    def __init__(self, data_path, batch_size):
        self.data_path = data_path

        # Patch size for training
        self.input_size = 41
        self.label_size = 41
        
        self.batch_size = batch_size
        self.queue_size = 3000

        self.open_h5py_file()

        self.make_queue()


    def open_h5py_file(self):
        self.h5py_file = h5py.File('{}.h5'.format(self.data_path), 'r')
        self.data_size = self.h5py_file['data'].shape[0]
        self.data_index = self.data_size // self.batch_size


    def make_queue(self):
        self.input_t = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 1])
        self.label_t = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 1])

        queue = tf.RandomShuffleQueue(
            capacity=self.queue_size, 
            min_after_dequeue=self.batch_size,
            dtypes=(tf.float32, tf.float32), 
            shapes=((self.input_size, self.input_size, 1), (self.label_size, self.label_size, 1)),
            name = 'random_shuffle_queue'
            )

        self.enqueue_many = queue.enqueue_many([self.input_t, self.label_t])
        self.dequeue_many = queue.dequeue_many(self.batch_size)


    def start_enqueue_deamon(self, sess):
        def enqueue_thread(sess): 
            while (True):
                for (input_t, label_t) in self.generator():
                    sess.run([self.enqueue_many], feed_dict={
                        self.input_t: input_t, 
                        self.label_t: label_t
                        })

                    time.sleep(0.0001)

        thread_number = 1
        threads = []
        for i in range(thread_number):
            t = threading.Thread(target=enqueue_thread, args=(sess,), daemon=True)
            t.start()
            threads.append(t)

        return threads


    def generator(self):
        for i in range(self.data_index):
            input_t = self.h5py_file['data'][i * self.batch_size : (i+1) * self.batch_size]
            label_t = self.h5py_file['label'][i * self.batch_size : (i+1) * self.batch_size]
            yield (input_t, label_t)
