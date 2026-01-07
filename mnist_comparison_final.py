import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from neural_net_two_layers import NeuralNetMLP_TwoHidden, sigmoid, int_to_onehot
import matplotlib.pyplot as plt


class NeuralNetMLP_OneHidden:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 / (num_features + num_hidden)), 
            size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        self.weight_out = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 / (num_hidden + num_classes)), 
            size=(num_classes, num_hidden))
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


class NeuralNetMLP_TwoHidden_Improved:
    def __init__(self, num_features, num_hidden1, num_hidden2, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        
        self.weight_h1 = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 / (num_features + num_hidden1)),
            size=(num_hidden1, num_features))
        self.bias_h1 = np.zeros(num_hidden1)
        
        self.weight_h2 = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 / (num_hidden1 + num_hidden2)),
            size=(num_hidden2, num_hidden1))
        self.bias_h2 = np.zeros(num_hidden2)
        
        self.weight_out = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 / (num_hidden2 + num_classes)),
            size=(num_classes, num_hidden2))
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


class PyTorchNN(nn.Module):
    def __init__(self, num_features, num_hidden1, num_hidden2, num_classes):
        super(PyTorchNN, self).__init__()
        
        self.fc1 = nn.Linear(num_features, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_hidden(model, x_train, y_train, x_test, y_test, num_epochs, learning_rate):
    train_losses = []
    train_accs = []
    test_accs = []
    
    for e in range(num_epochs):
        a_h, a_out = model.forward(x_train)
        
        (d_loss__dw_out, d_loss__db_out,
         d_loss__d_w_h, d_loss__d_b_h) = model.backward(
            x_train, a_h, a_out, y_train)
        
        model.weight_out -= learning_rate * d_loss__dw_out
        model.bias_out -= learning_rate * d_loss__db_out
        model.weight_h -= learning_rate * d_loss__d_w_h
        model.bias_h -= learning_rate * d_loss__d_b_h
        
        y_train_onehot = int_to_onehot(y_train, model.num_classes)
        loss = np.mean((a_out - y_train_onehot)**2)
        
        y_train_pred = np.argmax(a_out, axis=1)
        train_acc = np.sum(y_train == y_train_pred) / y_train.shape[0]
        
        _, a_out_test = model.forward(x_test)
        y_test_pred = np.argmax(a_out_test, axis=1)
        test_acc = np.sum(y_test == y_test_pred) / y_test.shape[0]
        
        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if e % 5 == 0 or e == num_epochs - 1:
            print(f'Epoch: {e:03d}/{num_epochs:03d} | Loss: {loss:.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%')
    
    return train_losses, train_accs, test_accs


def train_two_hidden(model, x_train, y_train, x_test, y_test, num_epochs, learning_rate):
    train_losses = []
    train_accs = []
    test_accs = []
    
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
        
        _, _, a_out_test = model.forward(x_test)
        y_test_pred = np.argmax(a_out_test, axis=1)
        test_acc = np.sum(y_test == y_test_pred) / y_test.shape[0]
        
        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if e % 5 == 0 or e == num_epochs - 1:
            print(f'Epoch: {e:03d}/{num_epochs:03d} | Loss: {loss:.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%')
    
    return train_losses, train_accs, test_accs


def train_pytorch(model, x_train, y_train, x_test, y_test, num_epochs, learning_rate):
    x_train_torch = torch.FloatTensor(x_train)
    y_train_torch = torch.LongTensor(y_train)
    x_test_torch = torch.FloatTensor(x_test)
    y_test_torch = torch.LongTensor(y_test)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for e in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_pred = torch.argmax(outputs, dim=1)
            train_acc = (train_pred == y_train_torch).float().mean().item()
            
            test_outputs = model(x_test_torch)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = (test_pred == y_test_torch).float().mean().item()
        
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if e % 5 == 0 or e == num_epochs - 1:
            print(f'Epoch: {e:03d}/{num_epochs:03d} | Loss: {loss.item():.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%')
    
    return train_losses, train_accs, test_accs


def calculate_macro_auc(model, x, y, num_classes, model_type='numpy'):
    if model_type == 'numpy_one':
        _, a_out = model.forward(x)
        probabilities = a_out
    elif model_type == 'numpy_two':
        _, _, a_out = model.forward(x)
        probabilities = a_out
    elif model_type == 'pytorch':
        model.eval()
        with torch.no_grad():
            x_torch = torch.FloatTensor(x)
            outputs = model(x_torch)
            probabilities = torch.softmax(outputs, dim=1).numpy()
    
    y_onehot = int_to_onehot(y, num_classes)
    macro_auc = roc_auc_score(y_onehot, probabilities, average='macro', multi_class='ovr')
    
    return macro_auc


def main():
    print("=" * 80)
    print("MNIST Classification: Optimized Comparison")
    print("=" * 80)
    
    print("\nLoading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    num_features = X_train.shape[1]
    num_hidden_single = 500
    num_hidden1 = 350
    num_hidden2 = 150
    num_classes = 10
    num_epochs = 50
    learning_rate_numpy = 1.0
    learning_rate_pytorch = 0.001
    
    print("\n" + "=" * 80)
    print("Optimized Architecture Configuration:")
    print(f"  Single Hidden: {num_hidden_single} neurons")
    print(f"  Two Hidden: {num_hidden1} + {num_hidden2} = {num_hidden1+num_hidden2} neurons")
    print(f"  NumPy LR: {learning_rate_numpy}, PyTorch LR: {learning_rate_pytorch}")
    print(f"  Epochs: {num_epochs}")
    print("=" * 80)
    
    results = {}
    
    print("\n" + "=" * 80)
    print("1. Training Single Hidden Layer Network (500 neurons)")
    print("=" * 80)
    
    start_time = time.time()
    model_one = NeuralNetMLP_OneHidden(
        num_features=num_features,
        num_hidden=num_hidden_single,
        num_classes=num_classes,
        random_seed=123
    )
    
    losses_one, train_accs_one, test_accs_one = train_one_hidden(
        model_one, X_train, y_train, X_test, y_test, num_epochs, learning_rate_numpy)
    
    macro_auc_one = calculate_macro_auc(model_one, X_test, y_test, num_classes, 'numpy_one')
    time_one = time.time() - start_time
    
    print(f"\n✓ Final Test Accuracy: {test_accs_one[-1]*100:.2f}%")
    print(f"✓ Final Macro AUC: {macro_auc_one:.4f}")
    print(f"✓ Training Time: {time_one:.2f}s")
    
    results['one_hidden'] = {
        'accuracy': test_accs_one[-1],
        'macro_auc': macro_auc_one,
        'time': time_one,
        'losses': losses_one,
        'train_accs': train_accs_one,
        'test_accs': test_accs_one
    }
    
    print("\n" + "=" * 80)
    print("2. Training Two Hidden Layers Network (350+150 neurons)")
    print("=" * 80)
    
    start_time = time.time()
    model_two = NeuralNetMLP_TwoHidden_Improved(
        num_features=num_features,
        num_hidden1=num_hidden1,
        num_hidden2=num_hidden2,
        num_classes=num_classes,
        random_seed=123
    )
    
    losses_two, train_accs_two, test_accs_two = train_two_hidden(
        model_two, X_train, y_train, X_test, y_test, num_epochs, learning_rate_numpy)
    
    macro_auc_two = calculate_macro_auc(model_two, X_test, y_test, num_classes, 'numpy_two')
    time_two = time.time() - start_time
    
    print(f"\n✓ Final Test Accuracy: {test_accs_two[-1]*100:.2f}%")
    print(f"✓ Final Macro AUC: {macro_auc_two:.4f}")
    print(f"✓ Training Time: {time_two:.2f}s")
    
    results['two_hidden'] = {
        'accuracy': test_accs_two[-1],
        'macro_auc': macro_auc_two,
        'time': time_two,
        'losses': losses_two,
        'train_accs': train_accs_two,
        'test_accs': test_accs_two
    }
    
    print("\n" + "=" * 80)
    print("3. Training PyTorch Network (350+150 neurons, ReLU)")
    print("=" * 80)
    
    start_time = time.time()
    model_pytorch = PyTorchNN(
        num_features=num_features,
        num_hidden1=num_hidden1,
        num_hidden2=num_hidden2,
        num_classes=num_classes
    )
    
    losses_pytorch, train_accs_pytorch, test_accs_pytorch = train_pytorch(
        model_pytorch, X_train, y_train, X_test, y_test, num_epochs, learning_rate_pytorch)
    
    macro_auc_pytorch = calculate_macro_auc(model_pytorch, X_test, y_test, num_classes, 'pytorch')
    time_pytorch = time.time() - start_time
    
    print(f"\n✓ Final Test Accuracy: {test_accs_pytorch[-1]*100:.2f}%")
    print(f"✓ Final Macro AUC: {macro_auc_pytorch:.4f}")
    print(f"✓ Training Time: {time_pytorch:.2f}s")
    
    results['pytorch'] = {
        'accuracy': test_accs_pytorch[-1],
        'macro_auc': macro_auc_pytorch,
        'time': time_pytorch,
        'losses': losses_pytorch,
        'train_accs': train_accs_pytorch,
        'test_accs': test_accs_pytorch
    }
    
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<35} {'Accuracy':<12} {'Macro AUC':<12} {'Time (s)':<10}")
    print("-" * 80)
    print(f"{'1. Single Hidden (500)':<35} {results['one_hidden']['accuracy']*100:>9.2f}%  "
          f"{results['one_hidden']['macro_auc']:>10.4f}  {results['one_hidden']['time']:>8.2f}")
    print(f"{'2. Two Hidden (350+150)':<35} {results['two_hidden']['accuracy']*100:>9.2f}%  "
          f"{results['two_hidden']['macro_auc']:>10.4f}  {results['two_hidden']['time']:>8.2f}")
    print(f"{'3. PyTorch (350+150, ReLU)':<35} {results['pytorch']['accuracy']*100:>9.2f}%  "
          f"{results['pytorch']['macro_auc']:>10.4f}  {results['pytorch']['time']:>8.2f}")
    print("-" * 80)
    
    auc_improvement = (results['two_hidden']['macro_auc'] - results['one_hidden']['macro_auc']) / results['one_hidden']['macro_auc'] * 100
    print(f"\nTwo hidden vs Single: {auc_improvement:+.2f}% AUC change")
    
    best_model = max(results.keys(), key=lambda k: results[k]['macro_auc'])
    best_names = {'one_hidden': 'Single Hidden', 'two_hidden': 'Two Hidden', 'pytorch': 'PyTorch'}
    print(f"Best model: {best_names[best_model]} (AUC: {results[best_model]['macro_auc']:.4f})")
    
    print("\nGenerating plots...")
    create_plots(results)
    print("✓ Plots saved to 'comparison_results.png'")
    
    np.savez('results.npz', **results)
    print("✓ Results saved to 'results.npz'")
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)


def create_plots(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(results['one_hidden']['losses'], label='Single Hidden (500)', linewidth=2)
    ax.plot(results['two_hidden']['losses'], label='Two Hidden (350+150)', linewidth=2)
    ax.plot(results['pytorch']['losses'], label='PyTorch (ReLU)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(results['one_hidden']['test_accs'], label='Single Hidden (500)', linewidth=2)
    ax.plot(results['two_hidden']['test_accs'], label='Two Hidden (350+150)', linewidth=2)
    ax.plot(results['pytorch']['test_accs'], label='PyTorch (ReLU)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    models = ['Single\n(500)', 'Two Hidden\n(350+150)', 'PyTorch\n(ReLU)']
    accuracies = [
        results['one_hidden']['accuracy']*100,
        results['two_hidden']['accuracy']*100,
        results['pytorch']['accuracy']*100
    ]
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Test Accuracy')
    ax.set_ylim([0, 100])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    aucs = [
        results['one_hidden']['macro_auc'],
        results['two_hidden']['macro_auc'],
        results['pytorch']['macro_auc']
    ]
    bars = ax.bar(models, aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Macro AUC')
    ax.set_title('Final Macro AUC Score')
    ax.set_ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
