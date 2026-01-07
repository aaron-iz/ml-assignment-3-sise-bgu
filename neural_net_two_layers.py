import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


class NeuralNetMLP_TwoHidden:
    def __init__(self, num_features, num_hidden1, num_hidden2, num_classes, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        
        rng = np.random.RandomState(random_seed)
        
        self.weight_h1 = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden1, num_features))
        self.bias_h1 = np.zeros(num_hidden1)
        
        self.weight_h2 = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden2, num_hidden1))
        self.bias_h2 = np.zeros(num_hidden2)
        
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden2))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h1 = np.dot(x, self.weight_h1.T) + self.bias_h1
        a_h1 = sigmoid(z_h1)
        
        z_h2 = np.dot(a_h1, self.weight_h2.T) + self.bias_h2
        a_h2 = sigmoid(z_h2)
        
        z_out = np.dot(a_h2, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        
        return a_h1, a_h2, a_out

    def backward(self, x, a_h1, a_h2, a_out, y):
        y_onehot = int_to_onehot(y, self.num_classes)
        
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        
        d_loss__dw_out = np.dot(delta_out.T, a_h2)
        d_loss__db_out = np.sum(delta_out, axis=0)
        
        d_z_out__a_h2 = self.weight_out
        d_loss__a_h2 = np.dot(delta_out, d_z_out__a_h2)
        d_a_h2__d_z_h2 = a_h2 * (1. - a_h2)
        delta_h2 = d_loss__a_h2 * d_a_h2__d_z_h2
        
        d_loss__d_w_h2 = np.dot(delta_h2.T, a_h1)
        d_loss__d_b_h2 = np.sum(delta_h2, axis=0)
        
        d_z_h2__a_h1 = self.weight_h2
        d_loss__a_h1 = np.dot(delta_h2, d_z_h2__a_h1)
        d_a_h1__d_z_h1 = a_h1 * (1. - a_h1)
        delta_h1 = d_loss__a_h1 * d_a_h1__d_z_h1
        
        d_loss__d_w_h1 = np.dot(delta_h1.T, x)
        d_loss__d_b_h1 = np.sum(delta_h1, axis=0)
        
        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h2, d_loss__d_b_h2,
                d_loss__d_w_h1, d_loss__d_b_h1)


def train(model, x_train, y_train, x_valid, y_valid, num_epochs, learning_rate):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):
        a_h1, a_h2, a_out = model.forward(x_train)
        
        (d_loss__dw_out, d_loss__db_out,
         d_loss__d_w_h2, d_loss__d_b_h2,
         d_loss__d_w_h1, d_loss__d_b_h1) = model.backward(
            x_train, a_h1, a_h2, a_out, y_train)
        
        model.weight_out -= learning_rate * d_loss__dw_out
        model.bias_out -= learning_rate * d_loss__db_out
        model.weight_h2 -= learning_rate * d_loss__d_w_h2
        model.bias_h2 -= learning_rate * d_loss__d_b_h2
        model.weight_h1 -= learning_rate * d_loss__d_w_h1
        model.bias_h1 -= learning_rate * d_loss__d_b_h1
        
        y_train_onehot = int_to_onehot(y_train, model.num_classes)
        loss = np.mean((a_out - y_train_onehot)**2)
        
        y_train_pred = np.argmax(a_out, axis=1)
        train_acc = np.sum(y_train == y_train_pred) / y_train.shape[0]
        
        _, _, a_out_valid = model.forward(x_valid)
        y_valid_pred = np.argmax(a_out_valid, axis=1)
        valid_acc = np.sum(y_valid == y_valid_pred) / y_valid.shape[0]
        
        epoch_loss.append(loss)
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        
        if e % 10 == 0:
            print(f'Epoch: {e:03d}/{num_epochs:03d} | '
                  f'Loss: {loss:.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | '
                  f'Valid Acc: {valid_acc*100:.2f}%')
    
    return epoch_loss, epoch_train_acc, epoch_valid_acc


def predict(model, x):
    _, _, a_out = model.forward(x)
    predictions = np.argmax(a_out, axis=1)
    return predictions, a_out
