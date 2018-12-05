import os
import numpy as np
import tensorflow as tf
from model import load_model
import h5py
import sklearn.metrics
from sklearn import preprocessing
import pandas as pd
import csv

facenet_model_checkpoint =  './merge'

class FaceEncoder:
    def __init__(self):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        with self.sess.as_default():
            load_model(facenet_model_checkpoint)
            print ('load model is done')
            variable_names = [v.name for v in tf.all_variables()]
            print(variable_names)
         
    def generate_embedding(self):
        
        base_dir = "/data/tianchi"
        
        path = os.path.join(base_dir, "round1_test_a_20181109.h5")
        #path = os.path.join(base_dir, "validation.h5")
        fid = h5py.File(path, 'r')
        s1 = fid['sen1']
        s2 = fid['sen2']
        data = np.concatenate((s1,s2), axis=-1)
        
        #label = fid['label'][0:5000]
        #label = np.argmax(label, axis=1)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("img_inputs:0")
        
        label_pre = tf.get_default_graph().get_tensor_by_name("pred_label:0")

        dropout_rate_placeholder = self.sess.graph.get_tensor_by_name("dropout_rate:0")
        feed_dict = {images_placeholder:data,dropout_rate_placeholder:1}
        y_pred = self.sess.run(label_pre, feed_dict=feed_dict)
        #print( sklearn.metrics.classification_report(label, y_pred))
        
        enc = preprocessing.OneHotEncoder() 
        y_pred = np.asarray(y_pred, dtype = 'int32').reshape(y_pred.shape[0],1)
        enc.fit(y_pred)
        result=pd.DataFrame(enc.transform(y_pred).toarray().astype('uint8'))
        result.to_csv("./result.csv",index=False,sep=',',encoding="utf-8")

encoder = FaceEncoder()
encoder.generate_embedding()
