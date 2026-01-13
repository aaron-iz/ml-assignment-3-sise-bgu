import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt


from neural_net_one_layer import NeuralNetMLP_OneHidden, train_one_hidden
from neural_net_two_layers import NeuralNetMLP_TwoHidden, train_two_hidden
from pytorch_nn import PyTorchNN, train_pytorch
from utils import calculate_macro_auc


RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)
torch.set_default_dtype(torch.float64)
torch.manual_seed(RANDOM_SEED)


def main():
    print("=" * 80)
    print("MNIST Classification: Comparing Neural Network Architectures")
    print("=" * 80)
    
    print("\nLoading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    num_features = X_train.shape[1]
    numpy_hidden = 50
    torch_hidden = 500
    num_classes = 10
    num_epochs_numpy = 20
    num_epochs_pytorch = 50
    learning_rate_numpy = 0.1
    learning_rate_pytorch = 0.1
    batch_size = 100
    
    print("\n" + "=" * 80)
    print("Architecture Configuration:")
    print(f"  Input features: {num_features}")
    print(f"  Hidden layers for NumPy: {numpy_hidden} neurons")
    print(f"  Hidden layers for PyTorch: {torch_hidden} neurons")
    print(f"  Output classes: {num_classes}")
    print(f"  NumPy Learning rate: {learning_rate_numpy}")
    print(f"  PyTorch Learning rate: {learning_rate_pytorch}")
    print(f"  NumPy Epochs: {num_epochs_numpy}")
    print(f"  Batch size: {batch_size}")
    print(f"  PyTorch Epochs: {num_epochs_pytorch}")
    print("=" * 80)
    
    results = {}
    
    print("\n" + "=" * 80)
    print("1. Training Original Network (Single Hidden Layer)")
    print("=" * 80)
    
    start_time = time.time()
    model_one = NeuralNetMLP_OneHidden(
        num_features=num_features,
        num_hidden=numpy_hidden,
        num_classes=num_classes,
        random_seed=RANDOM_SEED
    )
    
    losses_one, train_accs_one, test_accs_one = train_one_hidden(
        model_one, X_train, y_train, X_test, y_test, num_epochs_numpy, 
        learning_rate_numpy, batch_size)
    
    macro_auc_one = calculate_macro_auc(model_one, X_test, y_test, num_classes, 'numpy_one')
    time_one = time.time() - start_time
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accs_one[-1]*100:.2f}%")
    print(f"  Macro AUC: {macro_auc_one:.4f}")
    print(f"  Training Time: {time_one:.2f}s")
    
    results['one_hidden'] = {
        'accuracy': test_accs_one[-1],
        'macro_auc': macro_auc_one,
        'time': time_one,
        'losses': losses_one,
        'train_accs': train_accs_one,
        'test_accs': test_accs_one
    }
    
    print("\n" + "=" * 80)
    print("2. Training Extended Network (Two Hidden Layers)")
    print("=" * 80)
    
    start_time = time.time()
    model_two = NeuralNetMLP_TwoHidden(
        num_features=num_features,
        num_hidden1=numpy_hidden,
        num_hidden2=numpy_hidden,
        num_classes=num_classes,
        random_seed=RANDOM_SEED
    )
    
    losses_two, train_accs_two, test_accs_two = train_two_hidden(
        model_two, X_train, y_train, X_test, y_test, num_epochs_numpy, 
        learning_rate_numpy, batch_size)
    
    macro_auc_two = calculate_macro_auc(model_two, X_test, y_test, num_classes, 'numpy_two')
    time_two = time.time() - start_time
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accs_two[-1]*100:.2f}%")
    print(f"  Macro AUC: {macro_auc_two:.4f}")
    print(f"  Training Time: {time_two:.2f}s")
    
    results['two_hidden'] = {
        'accuracy': test_accs_two[-1],
        'macro_auc': macro_auc_two,
        'time': time_two,
        'losses': losses_two,
        'train_accs': train_accs_two,
        'test_accs': test_accs_two
    }
    
    print("\n" + "=" * 80)
    print("3. Training PyTorch Network (Two Hidden Layers)")
    print("=" * 80)
    
    start_time = time.time()
    model_pytorch = PyTorchNN(
        num_features=num_features,
        num_hidden1=torch_hidden,
        num_hidden2=torch_hidden,
        num_classes=num_classes
    )
    
    losses_pytorch, train_accs_pytorch, test_accs_pytorch = train_pytorch(
        model_pytorch, X_train, y_train, X_test, y_test, num_epochs_pytorch, 
        learning_rate_pytorch, batch_size)
    
    macro_auc_pytorch = calculate_macro_auc(model_pytorch, X_test, y_test, num_classes, 'pytorch')
    time_pytorch = time.time() - start_time
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_accs_pytorch[-1]*100:.2f}%")
    print(f"  Macro AUC: {macro_auc_pytorch:.4f}")
    print(f"  Training Time: {time_pytorch:.2f}s")
    
    results['pytorch'] = {
        'accuracy': test_accs_pytorch[-1],
        'macro_auc': macro_auc_pytorch,
        'time': time_pytorch,
        'losses': losses_pytorch,
        'train_accs': train_accs_pytorch,
        'test_accs': test_accs_pytorch
    }
    
    print_comparison_summary(results)
    
    print("\nGenerating performance plots...")
    create_plots(results)
    print("Plots saved to 'comparison_results.png'")
    
    np.savez('results.npz', 
             one_hidden=results['one_hidden'],
             two_hidden=results['two_hidden'],
             pytorch=results['pytorch'])
    print("\nResults saved to 'results.npz'")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


def print_comparison_summary(results):
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Test Accuracy':<15} {'Macro AUC':<15} {'Time (s)':<10}")
    print("-" * 80)
    print(f"{'1. Single Hidden Layer':<30} {results['one_hidden']['accuracy']*100:>12.2f}%  "
          f"{results['one_hidden']['macro_auc']:>12.4f}  {results['one_hidden']['time']:>8.2f}")
    print(f"{'2. Two Hidden Layers':<30} {results['two_hidden']['accuracy']*100:>12.2f}%  "
          f"{results['two_hidden']['macro_auc']:>12.4f}  {results['two_hidden']['time']:>8.2f}")
    print(f"{'3. PyTorch (Two Hidden)':<30} {results['pytorch']['accuracy']*100:>12.2f}%  "
          f"{results['pytorch']['macro_auc']:>12.4f}  {results['pytorch']['time']:>8.2f}")
    print("-" * 80)
    
    print("\nPerformance Analysis:")
    auc_improvement = (results['two_hidden']['macro_auc'] - results['one_hidden']['macro_auc']) / results['one_hidden']['macro_auc'] * 100
    print(f"  Two hidden layers vs Single: {auc_improvement:+.2f}% AUC change")
    
    best_model = max(results.keys(), key=lambda k: results[k]['macro_auc'])
    best_name = {'one_hidden': 'Single Hidden Layer', 
                 'two_hidden': 'Two Hidden Layers', 
                 'pytorch': 'PyTorch'}[best_model]
    print(f"  Best performing model: {best_name} (AUC: {results[best_model]['macro_auc']:.4f})")
    

def create_plots(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(results['one_hidden']['losses'], label='Single Hidden Layer', linewidth=2)
    ax.plot(results['two_hidden']['losses'], label='Two Hidden Layers', linewidth=2)
    ax.plot(results['pytorch']['losses'], label='PyTorch', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(results['one_hidden']['test_accs'], label='Single Hidden Layer', linewidth=2)
    ax.plot(results['two_hidden']['test_accs'], label='Two Hidden Layers', linewidth=2)
    ax.plot(results['pytorch']['test_accs'], label='PyTorch', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy Comparison', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    models = ['Single\nHidden', 'Two\nHidden', 'PyTorch']
    accuracies = [
        results['one_hidden']['accuracy']*100,
        results['two_hidden']['accuracy']*100,
        results['pytorch']['accuracy']*100
    ]
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Test Accuracy', pad=10)
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
    ax.set_title('Final Macro AUC Score', pad=10)
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
