import os
import h5py
import numpy as np
np.random.seed(8102)

def slipt(path,ouputs1,ouputs2,nums):
    fid = h5py.File(os.path.expanduser(path),'r')
    labels=np.argmax(fid['label'],1)
    distrib = np.bincount(labels)
    prob = 1/distrib[labels].astype(float)
    prob /= prob.sum()
    ll=len(labels)
    print(ll)
    select=np.random.choice(np.arange(ll), size=nums,replace=False,p=prob)
    labels= np.argmax(np.array([fid['label'][i] for i in select]),1)
    distrib2 = np.bincount(labels)
    print(distrib2)
    print("random seed is "+str(distrib2[-1]==65))
    f=h5py.File(ouputs1)
    t_index=select.tolist()
    t_index.sort()
    label=np.array(fid['label'][t_index])
    sen1=np.array(fid['sen1'][t_index])
    sen2=np.array(fid['sen2'][t_index])
    f.create_dataset('label',data=label)
    f.create_dataset('sen1',data=sen1)
    f.create_dataset('sen2',data=sen2)
    f.close()
    f=h5py.File(ouputs2)
    alltag=np.arange(ll)
    legacy=np.setdiff1d(alltag,select)
    l_index=legacy.tolist()
    l_index.sort()
    label=np.array(fid['label'][l_index])
    sen1=np.array(fid['sen1'][l_index])
    sen2=np.array(fid['sen2'][l_index])
    f.create_dataset('label',data=label)
    f.create_dataset('sen1',data=sen1)
    f.create_dataset('sen2',data=sen2)
    f.close()
    print("done")

def merge(path1,path2,output):
    f1=h5py.File(os.path.expanduser(path1),'r')
    f2=h5py.File(os.path.expanduser(path2),'r')
    mergef=h5py.File(output)
    label=np.concatenate(
    (np.array(f1['label']),np.array(f2['label'])),axis=0
    )
    sen1=np.concatenate(
    (np.array(f1['sen1']),np.array(f2['sen1'])),axis=0
    )
    sen2=np.concatenate(
    (np.array(f1['sen2']),np.array(f2['sen2'])),axis=0
    )
    mergef.create_dataset('label',data=label)
    mergef.create_dataset('sen1',data=sen1)
    mergef.create_dataset('sen2',data=sen2)
    f1.close()
    f2.close()
    mergef.close()

slipt('./data/validation.h5','./data/acc1000.h5','./data/mega1.h5',1000)
slipt('./data/training.h5','./data/acc3000.h5','./data/mega2.h5',3000)
merge('./data/acc1000.h5','./data/acc3000.h5','./data/acc4000.h5')
merge('./data/mega1.h5','./data/mega2.h5','./data/mega.h5')
