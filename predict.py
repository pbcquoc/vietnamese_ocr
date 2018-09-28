import os
import tensorflow as tf
from crnn import get_model
from loader import SIZE, MAX_LEN, TextImageGenerator, decode_batch
from keras import backend as K
import glob                                                                 
import argparse


def loadmodel(weight_path):
    model = get_model((*SIZE, 3), training=False, finetune=0)
    model.load_weights(weight_path)
    return model

def predict(model, datapath):
    sess = tf.Session()
    K.set_session(sess)

    batch_size = 3
    models = glob.glob('{}/best_*.h5'.format(model))
    test_generator  = TextImageGenerator(datapath, None, *SIZE, batch_size, 32, None, False, MAX_LEN)
    test_generator.build_data()

    for weight_path in models:
        
        print('load {}'.format(weight_path))
        model = loadmodel(weight_path)
        X_test = test_generator.imgs.transpose((0, 2, 1, 3))
        y_pred = model.predict(X_test, batch_size=3)
        decoded_res = decode_batch(y_pred)
        for i in range(len(test_generator.img_dir)):
            print('{}: {}'.format(test_generator.img_dir[test_generator.indexes[i]], decoded_res[i]))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='../data/ocr/model/', type=str)
    parser.add_argument('--data', default='../data/ocr/preprocess/test/', type=str)
    parser.add_argument('--device', default=2, type=int)
    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    predict(args.model, args.data)

