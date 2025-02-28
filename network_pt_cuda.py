import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from d2l import torch as d2l

import mnist_loader
from mnist_loader import MNISTDataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
numgpus =  torch.cuda.device_count()

class NetWork_PT(nn.Module):
    # 模型参数
    def __init__(self, x_dim, network_ac_type):
        super().__init__()
        self.hidden1 = nn.Linear(x_dim, 30)
        self.hidden2 = nn.Linear(30, 10)
        self.network_ac_type = network_ac_type
        # self.net = nn.Sequential(nn.Linear(x_dim, 30), F.sigmoid(), nn.Linear(30, 10), nn.sigmoid())

     # 模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        if self.network_ac_type == 'Sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = nn.ReLU()

        return self.hidden2(activation(self.hidden1(X)))

# Parameters
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6 if not use_cuda else 4*numgpus}

max_epochs = 10
learning_rate = 1e-3

def mnist_dataloader():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    train_dataset = MNISTDataset(list(training_data))
    validate_dataset = MNISTDataset(list(validation_data))
    test_dataset = MNISTDataset(list(test_data))

    train_dataloader = DataLoader(train_dataset, **params)
    validate_dataloader = DataLoader(validate_dataset, **params)
    test_dataloader = DataLoader(test_dataset, **params)

    return train_dataloader, validate_dataloader, test_dataloader

def train(loss_fn_type = 'CrossEntropyLoss', network_ac_type = 'Sigmoid'):
    model = NetWork_PT(784, network_ac_type)
    model.to(device)
        
    if loss_fn_type == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataloader, validate_dataloader, test_dataloader = mnist_dataloader()
    
    animator = d2l.Animator(xlabel='epoch', xlim=[1, max_epochs], ylim=[0.0, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(max_epochs):
        metric = d2l.Accumulator(3)
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.squeeze().to(device)
            labels = labels.squeeze().to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels.float())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # visulization of results.
            labels = d2l.argmax(labels, axis=1).cpu()
            metric.add(float(loss.cpu()), d2l.accuracy(outputs.cpu(), labels), labels.numel())
        # print(model.hidden1.data.device)
        test_acc = d2l.evaluate_accuracy(model, test_dataloader, device)
        metrics = (metric[0] / metric[2], metric[1] / metric[2], test_acc,)
        # print(metrics)
        animator.add(epoch + 1, metrics)

    print('Finished Training')

if __name__=='__main__':
    train()