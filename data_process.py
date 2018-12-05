#数据生成器
def data_generate(data_path:str, batch_size:int, schulffe:bool=False):
    while True:
        fid = h5py.File(data_path,'r')
        data_len = fid['sen1'].shape[0]
        # ceil
        c = [i for i in range(int(data_len / batch_size)+1)]

        if schulffe:
            np.random.shuffle(c)

        for i in c:
            y_b = np.array((fid['label'][i * batch_size:(i + 1) * batch_size]))
            try:
                x_b = np.array(
                    np.concatenate(
                        (
                            fid['sen1'][i * batch_size:(i + 1) * batch_size],
                            fid['sen2'][i * batch_size:(i + 1) * batch_size]
                        ),
                        axis=3)
                )
                yield (x_b, y_b)
            except:
                # the last few rows data
                x_b = np.array(
                    np.concatenate(
                        (
                            fid['sen1'][i * batch_size:],
                            fid['sen2'][i * batch_size:]
                        ),
                        axis=3)
                )
                yield (x_b, y_b)
                
#把同一个通道的像素点相加
n = train_s1.shape[0]
train_X = np.concatenate((train_s1,train_s2),axis=-1)
train_data = np.zeros([n,18])
for i in range(0,n):
   if(i%2000==0):
       print('------',i)
   for j in range(0,18):
       train_data[i,j] = np.sum(train_X[i,:,:,j])

n = validation_s1.shape[0]
validation_X = np.concatenate((validation_s1,validation_s2),axis=-1)
valid_data = np.zeros([n,18])
for i in range(0,n):
   if(i%2000==0):
       print('------',i)
   for j in range(0,18):
       valid_data[i,j] = np.sum(validation_X[i,:,:,j])
