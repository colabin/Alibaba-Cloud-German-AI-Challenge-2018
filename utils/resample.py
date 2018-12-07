#用于处理样本的不平衡问题
def weighted_data_generate(data_path, batch_size,replace=False):
    fid = h5py.File(data_path,'r')
    labels=np.argmax(fid['label'],1)
    distrib = np.bincount(labels)
    prob = 1/distrib[labels].astype(float)
    prob /= prob.sum()
    stop=(len(fid['label'])/batch_size)
    while(True):
        bingo=np.random.choice(np.arange(len(labels)), size=batch_size, replace=replace,p=prob)
        y_b = np.array([fid['label'][i] for i in bingo])
        x_b = np.array(
            np.concatenate(
                    (
                        np.array([fid['sen1'][i] for i in bingo]),
                        np.array([fid['sen2'][i] for i in bingo])
                    ),
                    axis=3)
                )
        yield x_b, y_b
