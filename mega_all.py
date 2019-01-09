import tensorflow as tf
import numpy as np
#from tianchi_net import get_resnet
from L_Resnet_E_IR_fix_issue9 import get_resnet
import time
from tensorflow.core.protobuf import config_pb2
import tensorlayer as tl
import os
import h5py
import argparse

def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl



batch_size = 128

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--channel', default = 0,type =int, help='epoch to train the network')
    args = parser.parse_args()
    return args

def weighted_data_generate_with_channel(data_path,batch_size, channel=0):
    fid1 = h5py.File(data_path, 'r')
    labels1 = np.argmax(fid1['label'], 1)
    distrib1 = np.bincount(labels1)
    prob1 = 1 / distrib1[labels1].astype(float)
    prob1 /= prob1.sum()

    data_len = fid1['sen1'].shape[0]
    c = [i for i in range(int(data_len / batch_size))]

    while (True):
        bingo1 = np.random.choice(np.arange(len(labels1)), batch_size, replace=False, p=prob1)

        y_b = np.array([fid1['label'][i] for i in bingo1])
        if channel < 8 :
            x_b = np.array([fid1['sen1'][i, :, :, channel] for i in bingo1])
        else:
            x_b = np.array([fid1['sen2'][i, :, :, channel - 8] for i in bingo1])

        x_b = x_b.reshape(batch_size,32,32,1)
        yield x_b, y_b, len(c)

def data_generate_with_channel(data_path, batch_size, schulffe=False, channel=0):

    fid = h5py.File(data_path, 'r')
    data_len = fid['sen1'].shape[0]
    # ceil
    c = [i for i in range(int(data_len / batch_size))]

    if schulffe:
        np.random.shuffle(c)

    for i in c:
        y_b = np.array((fid['label'][i * batch_size:(i + 1) * batch_size]))
        if channel < 8:
            x_b = np.array(fid['sen1'][i * batch_size:(i + 1) * batch_size, :, :, channel])
        else:
            x_b = np.array(fid['sen2'][i * batch_size:(i + 1) * batch_size, :, :, channel-8])

        x_b = x_b.reshape(batch_size,32,32,1)

        yield x_b, y_b, len(c)

def data_generate(data_path, batch_size, schulffe=False):
    fid = h5py.File(data_path, 'r')
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

def weighted_data_generate(data_path, batch_size, replace=False):
    fid = h5py.File(data_path, 'r')
    labels = np.argmax(fid['label'], 1)
    distrib = np.bincount(labels)
    prob = 1 / distrib[labels].astype(float)
    prob /= prob.sum()

    data_len = fid['sen1'].shape[0]
    c = [i for i in range(int(data_len / batch_size))]

    while (True):
        bingo = np.random.choice(np.arange(len(labels)), size=batch_size, replace=replace, p=prob)
        y_b = np.array([fid['label'][i] for i in bingo])
        x_b = np.array(
            np.concatenate(
                (
                    np.array([fid['sen1'][i] for i in bingo]),
                    np.array([fid['sen2'][i] for i in bingo])
                ),
                axis=3)
        )
        yield x_b, y_b, len(c)

def data_generate_sen2(data_path, batch_size, schulffe=False):
            fid = h5py.File(data_path, 'r')
            data_len = fid['sen1'].shape[0]
            # ceil
            c = [i for i in range(int(data_len / batch_size))]

            if schulffe:
                np.random.shuffle(c)

            for i in c:
                y_b = np.array(fid['label'][i * batch_size:(i + 1) * batch_size])
                x_b = np.array(fid['sen2'][i * batch_size:(i + 1) * batch_size])
                yield x_b, y_b, len(c)

def weighted_data_generate_sen2(data_path, batch_size, replace=False):
    fid = h5py.File(data_path, 'r')
    labels = np.argmax(fid['label'], 1)
    distrib = np.bincount(labels)
    prob = 1 / distrib[labels].astype(float)
    prob /= prob.sum()

    data_len = fid['sen1'].shape[0]
    c = [i for i in range(int(data_len / batch_size))]

    while (True):
        bingo = np.random.choice(np.arange(len(labels)), size=batch_size, replace=replace, p=prob)
        y_b = np.array([fid['label'][i] for i in bingo])
        x_b = np.array([fid['sen2'][i] for i in bingo])
        yield x_b, y_b, len(c)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_parser()

    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, 32, 32,18 ], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)

    net = get_resnet(images, 100, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, 100, type='ir', w_init=w_init_method, trainable=False, reuse=True, keep_rate=dropout_rate)

    embeddings = tf.nn.l2_normalize(test_net.outputs, 1, 1e-10, name='embeddings')

    w = tf.Variable(tf.zeros([512, 17]))  # 定义w维度是:[784,10],初始值是0
    b = tf.Variable(tf.zeros([17]))  # 定义b维度是:[10],初始值是0

    logit = tf.matmul(net.outputs, w) + b
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    pred = tf.nn.softmax(logit)
    pred_label = tf.argmax(pred, axis=1, name='pred_label')
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_label, labels), dtype=tf.float32))
    #inference_loss = tf.reduce_mean(focal_loss(labels,pred))

    logit_test = tf.matmul(test_net.outputs, w) + b
    pred_test = tf.nn.softmax(logit_test)
    pred_label_test = tf.argmax(pred_test, axis=1, name='pred_label_test')
    acc_test = tf.reduce_mean(tf.cast(tf.equal(pred_label_test, labels), dtype=tf.float32))

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
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001],
                                     name='lr_schedule')

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())

    #restore_saver = tf.train.Saver()
    #restore_saver.restore(sess, '/data/tianchi/models/sen2/merge_best_split_0/InsightFace_iter_8256.ckpt')

    count = 0
    best_acc = 0
    for i in range(10000):
        epoch_flag = False
        get_batch_train = weighted_data_generate(data_path="./data/mega.h5", batch_size=batch_size)
        get_batch_test_3000 = data_generate(data_path="./data/acc4000.h5", batch_size=batch_size)
        print('------------epoch', i)
        while (True):

            images_train, labels_train, batch_num_train = get_batch_train.__next__()

            labels_train = np.argmax(labels_train, axis=1)
            feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
            feed_dict.update(net.all_drop)
            start = time.time()
            _, total_loss_val, inference_loss_val, wd_loss_val, _, acc_val = \
                sess.run([train_op, total_loss, inference_loss, wd_loss, inc_op, acc],
                         feed_dict=feed_dict,
                         options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            end = time.time()
            pre_sec = batch_size / (end - start)

            count += 1

            if count > 0 and count % 50 == 0:
                print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
                      'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                      (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val, pre_sec))

            # save ckpt files
            if count > 0 and count % 5000 == 0:
                filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                file_path = './models/20190104/mega_all/merge_split_'+ str(args.channel)
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                filename = os.path.join(file_path, filename)
                saver.save(sess, filename)

            if count>0 and count % batch_num_train == 0:
                print('end of epoch')
                break

        print('--------------validation')
        acc_3000 = []
        test_count = 0
        while (True):
            images_test_3000, labels_test_3000, batch_num_test_3000 = get_batch_test_3000.__next__()
            labels_test_3000 = np.argmax(labels_test_3000, axis=1)

            feed_dict = {images: images_test_3000, labels: labels_test_3000, dropout_rate: 1}
            feed_dict.update(tl.utils.dict_to_one(net.all_drop))

            start = time.time()
            acc_test_ =  sess.run([acc_test],
                                 feed_dict=feed_dict,
                                 options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            acc_3000.append(acc_test_)
            end = time.time()
            pre_sec = batch_size / (end - start)

            test_count += 1

            if(test_count%batch_num_test_3000==0):
                result = np.sum(acc_3000) / len(acc_3000)
                print('--------------acc_3000',result)
                if (result > best_acc):
                    print('acc_3000,', result, '----------update_best_model-----------')
                    best_acc = result
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    file_path = './models/20190104/mega_all/merge_best_split_' + str(args.channel)
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    filename = os.path.join(file_path, filename)
                    saver.save(sess, filename)
                break


