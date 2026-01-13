import numpy as np
from utils import minibatch_generator, sigmoid, int_to_onehot, RANDOM_SEED

class NeuralNetMLP_OneHidden:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=RANDOM_SEED):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        y_onehot = int_to_onehot(y, self.num_classes)
        
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x
        
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        
        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)
    

def train_one_hidden(model, X_train, y_train, X_test, y_test, num_epochs, learning_rate, batch_size):
    train_losses = []
    train_accs = []
    test_accs = []
    
    for e in range(num_epochs):
        for X_train_mini, y_train_mini in minibatch_generator(X_train, y_train, batch_size):
            a_h, a_out = model.forward(X_train_mini)
            
            (d_loss__dw_out, d_loss__db_out,
            d_loss__d_w_h, d_loss__d_b_h) = model.backward(
                X_train_mini, a_h, a_out, y_train_mini)
            
            model.weight_out -= learning_rate * d_loss__dw_out
            model.bias_out -= learning_rate * d_loss__db_out
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
        
        _, a_out = model.forward(X_train)
        y_train_onehot = int_to_onehot(y_train, model.num_classes)
        loss = np.mean((a_out - y_train_onehot)**2)
        
        y_train_pred = np.argmax(a_out, axis=1)
        train_acc = np.sum(y_train == y_train_pred) / y_train.shape[0]
        
        _, a_out_test = model.forward(X_test)
        y_test_pred = np.argmax(a_out_test, axis=1)
        test_acc = np.sum(y_test == y_test_pred) / y_test.shape[0]
        
        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if e % 5 == 0:
            print(f'Epoch: {e:03d}/{num_epochs:03d} | Loss: {loss:.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%')
    
    return train_losses, train_accs, test_accs
