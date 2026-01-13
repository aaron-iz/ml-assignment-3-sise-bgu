import torch
import torch.nn as nn
import torch.optim as optim

from utils import minibatch_generator

class PyTorchNN(nn.Module):
    """
    Implements a two hidden layer neural network using PyTorch
    Each layer has an activation function of Sigmoid
    """
    def __init__(self, num_features, num_hidden1, num_hidden2, num_classes):
        super(PyTorchNN, self).__init__()
        
        self.num_classes = num_classes
        self.fc1 = nn.Linear(num_features, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


def train_pytorch(model, X_train, y_train, X_test, y_test, num_epochs, learning_rate, batch_size):
    X_train_torch = torch.DoubleTensor(X_train)
    y_train_torch = torch.LongTensor(y_train)
    X_test_torch = torch.DoubleTensor(X_test)
    y_test_torch = torch.LongTensor(y_test)

    y_train_one_hot = nn.functional.one_hot(y_train_torch, num_classes=model.num_classes).double()
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for e in range(num_epochs):
        model.train()
        for X_train_mini, y_train_mini in minibatch_generator(X_train_torch, y_train_one_hot, batch_size):
            optimizer.zero_grad()
            outputs = model(X_train_mini)
            loss = criterion(outputs, y_train_mini)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_train_torch)
            train_pred = torch.argmax(outputs, dim=1)
            train_acc = (train_pred == y_train_torch).double().mean().item()
            
            test_outputs = model(X_test_torch)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = (test_pred == y_test_torch).double().mean().item()
        
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if e % 5 == 0:
            print(f'Epoch: {e:03d}/{num_epochs:03d} | Loss: {loss.item():.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%')
    
    return train_losses, train_accs, test_accs