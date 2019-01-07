import glob
import numpy as np
import pandas as pd


def get_esemble():
    path_to_csvs = './results/'
    all_files = glob.glob(path_to_csvs + '*.csv')
    all_files = np.sort(all_files)
    weight_path = './weights.csv'
    weights = pd.read_csv(weight_path, header=None, sep=',')
    weights = np.array(weights)

    res = np.zeros((4838, 17))
    for i, path in enumerate(all_files):
        df = pd.read_csv(path, header=None, sep=',')
        df = np.array(df)
        res = res + df * weights[i]

    ress = np.zeros((4838, 17))
    dg = pd.DataFrame(res.astype(np.int32))
    dg.to_csv('mid_results.csv', index=False, header=None, sep=',')
    pg = np.argmax(res, 1)
    for i in range(4838):
        ress[i][pg[i]] = 1
    df = pd.DataFrame(ress.astype(np.int32))
    df.to_csv('results.csv', index=False, header=None, sep=',')


def get_fine():
    df = pd.read_csv('mid_results.csv', header=None, sep=',')
    df = np.array(df)
    base = pd.read_csv('./results/02.csv', header=None, sep=',')
    base = np.array(base)
    base = np.argmax(base, 1)
    s = np.max(df, 1)

    for i, lis in enumerate(df):
        maxlis = s[i]
        n = 0
        for t in lis:
            if t == maxlis:
                n = n + 1
        if n > 1:
            lis[base[i]] = 100
    ress = np.zeros((4838, 17))
    pg = np.argmax(df, 1)
    for i in range(4838):
        ress[i][pg[i]] = 1
    df = pd.DataFrame(ress.astype(np.int32))
    df.to_csv('fine_results.csv', index=False, header=None, sep=',')


def check(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = np.array(df)
    print(df.shape)
    print(np.sum(df))


def compare(path1, path2):
    df1 = pd.read_csv(path1, header=None, sep=',')
    df1 = np.array(df1)
    df2 = pd.read_csv(path2, header=None, sep=',')
    df2 = np.array(df2)
    print(np.sum(df1, 0))
    print(np.sum(df2, 0))
    arg1 = np.argmax(df1, 1)
    arg2 = np.argmax(df2, 1)
    diff = arg1 - arg2
    t = 0
    for i in range(len(diff)):
        if diff[i] != 0:
            # print(i)
            t = t + 1
    print(t)


#'fine_results.csv
compare('./results/00.csv','results.csv')
compare('./results/01.csv','results.csv')
# compare('./results/02.csv','./results/03.csv')
compare('./results/02.csv','results.csv')
compare('./results/03.csv','results.csv')
compare('81.csv','results.csv')
