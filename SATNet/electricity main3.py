import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from networks import SATNet
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

print(torch.cuda.is_available())
data = pd.read_csv("../dataset/electricity.csv")
print(data.head())  #返回data的前几行数据，默认为前五行
cols = list(data.columns[0:])
print(cols)
depth = 48
X = np.zeros((len(data), depth, len(cols))) #43824*10*11
y = np.zeros((len(data), len(cols))) #43824*11
for i, name in enumerate(cols):
    for j in range(depth):
        X[:, j, i] = data[name].shift(depth - j - 1).fillna(method='bfill')
        # shift表示将整个数据往下平移，前面缺失的用NA补再用后面最近的补
        #最后生成每10行数据一组的矩阵 第一组全是第一行的数据 第二组前9行是第一行的数据
for i, name in enumerate(cols):
    y[:, i] = data[name].shift(-1).fillna(method='ffill')
train_bound = int(0.6 * (len(data))) #26294
val_bound = int(0.8 * (len(data))) #35059
X_train = X[:train_bound]
X_val = X[train_bound:val_bound]
X_test = X[val_bound:-12] #8765*10*11
y_train = y[12:train_bound+12]
y_val = y[train_bound+12:val_bound+12]
y_test = y[val_bound+12:]
X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)  #每一列的最大最小值 可以不同行
y_train_min, y_train_max = y_train.min(axis=0), y_train.max(axis=0)  #训练中的pm2.5最大值994 最小值0
X_train = (X_train - X_train_min) / (X_train_max - X_train_min + 1e-9)
X_val = (X_val - X_train_min) / (X_train_max - X_train_min + 1e-9)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min + 1e-9)
y_train = (y_train - y_train_min) / (y_train_max - y_train_min + 1e-9)
y_val = (y_val - y_train_min) / (y_train_max - y_train_min + 1e-9)
y_test = (y_test - y_train_min) / (y_train_max - y_train_min + 1e-9)
X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
y_train_t = torch.Tensor(y_train)
y_val_t = torch.Tensor(y_val)
y_test_t = torch.Tensor(y_test)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
                          #用来对tensor进行打包 产生一个元组里有两个矩阵
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
model = SATNet(X_train_t.shape[2], depth).cuda()  #11,1,128
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
                    #三个参数分别为所使用的优化器对象,每多少轮循环后更新一次学习率(lr),每次更新lr的gamma倍
from sklearn.metrics import mean_squared_error, mean_absolute_error
epochs = 1000
loss = nn.MSELoss() #均方损失 除以数量
patience = 35
min_val_loss = 9999
counter = 0
for i in range(epochs):
    mse_train = 0
    for batch_x, batch_y in train_loader: #batch_x 64*10*11
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        opt.zero_grad()
        y_pred= model(batch_x)
        l = loss(y_pred, batch_y)
        l.backward()
        mse_train += l.item() * batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output= model(batch_x)
            preds.append(output.detach().cpu().numpy()) #返回一个相同的矩阵并取消梯度
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]
    preds = np.concatenate(preds) #数组拼接
    true = np.concatenate(true)

    if min_val_loss > mse_val ** 0.5:
        min_val_loss = mse_val ** 0.5
        print("Saving...")
        torch.save(model.state_dict(), "../SATNet/model_electricity3.pt")
        counter = 0
    else:
        counter += 1

    if counter == patience:
        break
    print("Iter: ", i, "train: ", (mse_train / len(X_train_t)) ** 0.5, "val: ", (mse_val / len(X_val_t)) ** 0.5)
    if (i % 10 == 0):
        # preds = preds * (y_train_max - y_train_min) + y_train_min
        # true = true * (y_train_max - y_train_min) + y_train_min
        mse = mean_squared_error(true, preds) #均方损失 除以数量
        mae = mean_absolute_error(true, preds) #平均绝对损失
        print("lr: ", opt.param_groups[0]["lr"]) #optimizer.param_groups： 是一个list，其中的元素为字典；
        #optimizer.param_groups[0]：长度为7的字典，包括[‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]这7个参数
        print("mse: ", mse, "mae: ", mae)
        plt.figure(figsize=(20, 10))
        plt.plot(preds) #蓝色
        plt.plot(true) #橙色
        plt.savefig(f'picture/electricitytrain3{i}.png')
        plt.show()


model.load_state_dict(torch.load("../SATNet/model_electricity3.pt"))
with torch.no_grad():
    mse_val = 0
    preds = []
    true = []
    alphas = []
    betas = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        output= model(batch_x)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds = np.concatenate(preds)
true = np.concatenate(true)
# preds = preds*(y_train_max - y_train_min) + y_train_min
# true = true*(y_train_max - y_train_min) + y_train_min
mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
print('mse:',mse, 'mae',mae)
plt.figure(figsize=(20, 10))
plt.plot(preds)
plt.plot(true)
plt.savefig(f'picture/electricitytesttest3.png')
plt.show()