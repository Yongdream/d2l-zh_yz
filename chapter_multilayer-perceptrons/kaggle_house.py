import hashlib
import os
import requests


def download(name, cache_dir = os.path.join('..', 'data')): 
    """下载一个DATA_HUB中的文件,返回本地文件名"""
    # os.path.join()函数用于路径拼接文件路径 
    # 存在以‘’/’’开始的参数，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃
    # 此处的路径就为/..\data    
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)   # os.makedirs() 方法用于递归创建目录
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:    
        # Python-with open() as f的用法(Favorite）
        # rb: 以二进制格式打开一个文件用于只读
            while True:
                data = f.read(1048576)  # excel最大行数：1048576
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)       # 返回文件路径
    data_dir, ext = os.path.splitext(fname) # 分割路径，返回路径名和文件扩展名的元组
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False
    fp.extractall(base_dir)     # 解压压缩包中的所有文件至指定文件夹
    return os.path.join(base_dir, folder) if folder else data_dir


# %% [markdown]
# 

# %%
%matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import seaborn as sns 

# %% [markdown]
# 

# %%
train_data = pd.read_csv("E:\\Galaxy\\House_prices\\data\\house-prices-advanced-regression-techniques\\train.csv")
test_data = pd.read_csv("E:\\Galaxy\\House_prices\\data\\house-prices-advanced-regression-techniques\\test.csv")

# %% [markdown]
# 

# %%
print(train_data.shape)
print(test_data.shape)

# %%
fig = plt.figure(figsize=(14,8))
abs(train_data.corr()['SalePrice']).sort_values(ascending=False).plot.bar()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# %%
#异常值处理ok
figure=plt.figure()
sns.pairplot(x_vars=['OverallQual','GrLivArea','YearBuilt','TotalBsmtSF'],
             y_vars=['SalePrice'],data=train_data,dropna=True)
plt.show()

# %%
#删除异常值1123
train_data = train_data.drop(train_data[(train_data['OverallQual']<5) &
                                        (train_data['SalePrice']>200000)].index)
 
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) &
                                        (train_data['SalePrice']<300000)].index)
 
train_data = train_data.drop(train_data[(train_data['YearBuilt']<1900) &
                                        (train_data['SalePrice']>400000)].index)
train_data = train_data.drop(train_data[(train_data['TotalBsmtSF']>6000) &
                                        (train_data['SalePrice']<200000)].index)

# %%
figure=plt.figure()
sns.pairplot(x_vars=['OverallQual','GrLivArea','YearBuilt','TotalBsmtSF'],
             y_vars=['SalePrice'],data=train_data,dropna=True)
plt.show()

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# %% [markdown]
# 

# %%
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

#观察数据的缺失情况
nan_index=((all_features.isnull().sum()/len(all_features))).sort_values(ascending=False)#34个缺失值

# %%
# 数据预处理 非object数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# %%
all_features.shape

# %%
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
# 将离散型特征的每一种取值都看成一种状态，若你的这一特征中有N个不相同的取值
# 我们就可以将该特征抽象成N种不同的状态
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

# %%
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# ## [**训练**]

# %%
loss = nn.MSELoss()
in_features = train_features.shape[1]   # 331

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

# %% [markdown]
# 这使得预测价格的对数与真实标签价格的对数之间出现以下均方根误差：
# 
# $$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

# %%
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# %% [markdown]
# [**我们的训练函数将借助Adam优化器**]

# %%
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    optimizer = torch.optim.Adam(net.parameters(),
                            lr = learning_rate,
                            weight_decay = weight_decay)


    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls



# %% [markdown]
# ## $K$折交叉验证

# %%
# 获取k折交叉验证某一折的训练集和验证集
def get_k_fold_data(k, i, X, y):
    assert k > 1

    fold_size = X.shape[0] // k     # 每份的个数:数据总条数/折数（组数）
    X_train, y_train = None, None   # X_train为训练集，y_valid为验证集
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:          
            # i为当前折, 则把该折作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)   # 按维数0（行）拼接
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

# %% [markdown]
# 当我们在$K$折交叉验证中训练$K$次后，[**返回训练和验证误差的平均值**]。

# %%
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, 
                                    weight_decay, batch_size) 
                                    # *是解码，变成前面返回的四个数据
                                    # [-1]返回的每一折的最后一次epoch的损失值
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k        

# %%
k, num_epochs, lr, weight_decay, batch_size = 10, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

# %%
#模型预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv("E:\\Galaxy\\House_prices\\submission.csv",index=False)   

# %%
train_and_pred(train_features,test_features,train_labels,
               test_data,num_epochs,lr,weight_decay,batch_size)  

# %%



