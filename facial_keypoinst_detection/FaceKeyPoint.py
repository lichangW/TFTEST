import tensorflow as tf
import numpy as np
import pandas as pd
import random
import sys
import cv2
import matplotlib.pyplot as pyplot

def weight_variable(shape,name=None):
    initial = tf.truncated_normal(shape,stddev=0.1,name=name)
    return tf.Variable(initial)

def bias_variable(shape,name=None):
    initial = tf.constant(0.1,shape=shape,name=name)
    return  tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

x = tf.placeholder(tf.float32,shape=[None,96,96,1],name="x")
y_ = tf.placeholder(tf.float32,shape=[None,30],name="y_")
keep_prob = tf.placeholder(tf.float32,name="keep_prob")

def model():

    W_conv1 = weight_variable([3,3,1,32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([2,2,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([2,2,64,128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([11*11*128,500])
    b_fc1 = bias_variable([500])

    h_pool3_flat = tf.reshape(h_pool3,[-1,11*11*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)

    W_fc2 = weight_variable([500,500])
    b_fc2 = bias_variable([500])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2)+b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

    W_fc3 = weight_variable([500,30])
    b_fc3 = bias_variable([30])

    y_conv = tf.add(tf.matmul(h_fc2_drop,W_fc3),b_fc3,name="y_conv")
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_-y_conv)),name="rmse")

    return y_conv,rmse


TRAIN_FILE = "training.csv"
TEST_FILE = "test.csv"
SAVE_PATH = "./result/"

VALIDATION_SIZE = 100
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10

key_point_index = {
        'left_eye_center_x':0,
        'left_eye_center_y':1,
        'right_eye_center_x':2,
        'right_eye_center_y':3,
        'left_eye_inner_corner_x':4,
        'left_eye_inner_corner_y':5,
        'left_eye_outer_corner_x':6,
        'left_eye_outer_corner_y':7,
        'right_eye_inner_corner_x':8,
        'right_eye_inner_corner_y':9,
        'right_eye_outer_corner_x':10,
        'right_eye_outer_corner_y':11,
        'left_eyebrow_inner_end_x':12,
        'left_eyebrow_inner_end_y':13,
        'left_eyebrow_outer_end_x':14,
        'left_eyebrow_outer_end_y':15,
        'right_eyebrow_inner_end_x':16,
        'right_eyebrow_inner_end_y':17,
        'right_eyebrow_outer_end_x':18,
        'right_eyebrow_outer_end_y':19,
        'nose_tip_x':20,
        'nose_tip_y':21,
        'mouth_left_corner_x':22,
        'mouth_left_corner_y':23,
        'mouth_right_corner_x':24,
        'mouth_right_corner_y':25,
        'mouth_center_top_lip_x':26,
        'mouth_center_top_lip_y':27,
        'mouth_center_bottom_lip_x':28,
        'mouth_center_bottom_lip_y':29
}

def input_data(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    cols = df.columns[:-1]

    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img,sep=' ')/255.0)

    X = np.vstack(df['Image'])
    X = X.reshape([-1,96,96,1])

    if test:
        y = None
    else:
        y = df[cols].values

    return X, y

def save_mode(saver,sess,save_path,step):
    path = saver.save(sess,save_path,global_step=step)
    print "model save in :{0}".format(path)

if __name__ == '__main__':

    sess = tf.InteractiveSession()
    y_conv,rmse = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)
    sess.run(tf.initialize_all_variables())

    X,y = input_data()
    X_valid,y_valid = X[:VALIDATION_SIZE],y[:VALIDATION_SIZE]
    X_train,y_train = X[VALIDATION_SIZE:],y[VALIDATION_SIZE:]

    best_validation_loss = 1000000.0
    current_epoch = 0
    TRAIN_SIZE = X_train.shape[0]
    train_index = range(TRAIN_SIZE)
    saver = tf.train.Saver(max_to_keep=5)

    print "begin training ...size: {0}".format(TRAIN_SIZE)

    for i in xrange(EPOCHS):
        random.shuffle(train_index)
        X_train, y_train = X_train[train_index], y_train[train_index]

        for j in xrange(0,TRAIN_SIZE,BATCH_SIZE):
            print "epoch: {0},train {1} samples done...".format(i,j)

            train_step.run(feed_dict={x:X_train[j:j+BATCH_SIZE],y_:y_train[j:j+BATCH_SIZE],keep_prob:0.5})

        validation_loss = rmse.eval(feed_dict={x:X_valid,y_:y_valid,keep_prob:1.0})
        print 'epoch {0} done! validation loss:{1}'.format(i, validation_loss * 96.0)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch=i
            save_mode(saver,sess,SAVE_PATH+"/model",i)
        elif (i-current_epoch)>=EARLY_STOP_PATIENCE:
            print "earlying stoping"
            break
