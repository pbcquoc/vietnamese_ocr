import json
import cv2
import os, random
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import itertools
import editdistance
import json

letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
SIZE = 2560, 160
CHAR_DICT = len(letters) + 1

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):
    def __init__(self, sess, y_func, text_img_gen, text_size, num_display_words=6):
        self.y_func = y_func
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        self.text_size = text_size
        self.sess = sess

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen.next_batch())[0]
            num_proc = min(word_batch['the_inputs'].shape[0], num_left)
            # predict
            inputs = word_batch['the_inputs'][0:num_proc]
            pred = self.y_func([inputs])[0]
            decoded_res = decode_batch(pred)
            # label
            labels = word_batch['the_labels'][:num_proc].astype(np.int32)
            labels = [labels_to_text(label) for label in labels]
            
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], labels[j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(labels[j])

            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance:'
              '%.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        batch = next(self.text_img_gen.next_batch())[0]
        inputs = batch['the_inputs'][:self.num_display_words]
        labels = batch['the_labels'][:self.num_display_words].astype(np.int32)
        labels = [labels_to_text(label) for label in labels]
         
        pred = self.y_func([inputs])[0]
        pred_texts = decode_batch(pred)
        for i in range(min(self.num_display_words, len(inputs))):
            print("label: {} - predict: {}".format(labels[i], pred_texts[i]))

        self.show_edit_distance(self.text_size)

class TextImageGenerator:
    def __init__(self, img_dirpath, labels_path, img_w, img_h,
                 batch_size, downsample_factor, idxs, training=True, max_text_len=9, n_eraser=5):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.idxs = idxs
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.labels= json.load(open(labels_path)) if labels_path != None else None
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        if self.idxs is not None:
            self.img_dir = [self.img_dir[idx] for idx in self.idxs]

        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w, 3), dtype=np.float16)
        self.training = training
        self.texts = []

    def build_data(self):
        print(self.n, " Image Loading start... ", self.img_dirpath)
        for i, img_file in enumerate(self.img_dir):
            img = image.load_img(self.img_dirpath + img_file, target_size=SIZE[::-1])
            img = image.img_to_array(img)
            img = preprocess_input(img).astype(np.float16)
            self.imgs[i] = img
            if self.labels != None: 
                self.texts.append(self.labels[img_file])
            else:
                #valid mode
                self.texts.append('')
        print("Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]].astype(np.float32), self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 3], dtype=np.float32)     # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_text_len], dtype=np.float32)             # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.transpose((1, 0, 2))
                
                X_data[i] = img
                Y_data[i,:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_inputs': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)
