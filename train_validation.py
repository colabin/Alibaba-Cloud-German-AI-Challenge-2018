import tensorflow as tf 
import numpy as np
from net.L_Resnet_E_IR import get_resnet
import time
from tensorflow.core.protobuf import config_pb2
import tensorlayer as tl
import os
import h5py

batch_size = 256 

def data_generate(data_path, batch_size, schulffe=False):
    
    fid = h5py.File(data_path,'r')
    data_len = fid['sen1'].shape[0]
    # ceil
    c = [i for i in range(int(data_len / batch_size))]

    if schulffe:
        np.random.shuffle(c)

    for i in c:
        y_b = np.array((fid['label'][i * batch_size:(i + 1) * batch_size]))
        x_b = np.array(
        np.concatenate(
                (
                    fid['sen1'][i * batch_size:(i + 1) * batch_size],
                    fid['sen2'][i * batch_size:(i + 1) * batch_size]
                ),
                axis=3)
            )
        yield x_b, y_b, len(c)

         

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, 32,32, 18], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)


    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)

    net = get_resnet(images, 100, type='ir', w_init=w_init_method, trainable = trainable , keep_rate=dropout_rate)

    w = tf.Variable(tf.zeros([512, 17]))  # 定义w维度是:[784,10],初始值是0
    b = tf.Variable(tf.zeros([17]))  # 定义b维度是:[10],初始值是0

    logit = tf.matmul(net.outputs, w) + b
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

    wd_loss = 0
    weight_deacy = 5e-4
    for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(weights)
    for W in tl.layers.get_variables_with_name('resnet_v1_100/E_DenseLayer/W', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(W)
    for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(weights)
    for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(gamma)
    for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(weight_deacy)(alphas)

    total_loss = inference_loss + wd_loss
    lr_steps = [40000, 60000, 80000]
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

    pred = tf.nn.softmax(logit)
    pred_label = tf.argmax(pred, axis=1, name='pred_label')
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_label, labels), dtype=tf.float32))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())

    count = 0
    for i in range(10000):
        get_batch_train = data_generate(data_path="/data/tianchi/training.h5", batch_size= batch_size)
        #get_batch_test = data_generate(data_path="/data/tianchi/validation.h5", batch_size= batch_size)
        print('------------start iteration')
        i_train = 0
        while(True):
            images_train, labels_train, batch_num_train = get_batch_train.__next__()
            if(i_train==batch_num_train-1):
                print('end of training')
                break
            labels_train = np.argmax(labels_train, axis=1)
            feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.3, trainable:True }
            feed_dict.update(net.all_drop)
            start = time.time()
            _, total_loss_val, inference_loss_val, wd_loss_val, _, acc_val = \
                sess.run([train_op, total_loss, inference_loss, wd_loss, inc_op, acc],
                         feed_dict=feed_dict,
                         options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            end = time.time()
            pre_sec = batch_size / (end - start)
            i_train = i_train + 1
            if count > 0 and count % 50 == 0:
                print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
                      'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                      (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val, pre_sec))
            count += 1

            # save ckpt files
            if count > 0 and count % 3000 == 0:
                filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                filename = os.path.join('./ckpt', filename)
                saver.save(sess, filename)

            if count > 0 and count % 5000 == 0:
                accuracy = []
                pred_labels = []
                real_labels = []
                i_test = 0
                get_batch_test = data_generate(data_path="/data/tianchi/validation.h5", batch_size=batch_size)
                while (True):
                    images_test, labels_test, batch_num_test = get_batch_test.__next__()
                    if(i_test == batch_num_test-1):
                        print('testacc------------',np.sum(accuracy)/len(accuracy))
                        break
                    labels_test = np.argmax(labels_test, axis=1)
                    feed_dict = {images: images_test, labels: labels_test, dropout_rate:1, trainable:None}
                    feed_dict.update(tl.utils.dict_to_one(net.all_drop))
                    acc_val,pred_label_ = sess.run([acc,pred_label],
                                 feed_dict=feed_dict,
                                 options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                    accuracy.append(acc_val)
                    pred_labels.append(pred_label_.tolist())
                    real_labels.append(labels_test.tolist())
                    i_test = i_test + 1
                 
