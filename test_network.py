import numpy as np
from neural_net_two_layers import NeuralNetMLP_TwoHidden, int_to_onehot

def test_forward_pass():
    print("Testing forward pass...")
    
    model = NeuralNetMLP_TwoHidden(
        num_features=4,
        num_hidden1=3,
        num_hidden2=2,
        num_classes=3,
        random_seed=123
    )
    
    x = np.random.randn(5, 4)
    a_h1, a_h2, a_out = model.forward(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Hidden layer 1 output shape: {a_h1.shape}")
    print(f"  Hidden layer 2 output shape: {a_h2.shape}")
    print(f"  Output shape: {a_out.shape}")
    
    assert a_h1.shape == (5, 3), "Hidden layer 1 shape incorrect"
    assert a_h2.shape == (5, 2), "Hidden layer 2 shape incorrect"
    assert a_out.shape == (5, 3), "Output shape incorrect"
    assert np.all((a_out >= 0) & (a_out <= 1)), "Output not in [0,1] range"
    
    print("  ✓ Forward pass test passed!")
    return True

def test_backward_pass():
    print("\nTesting backward pass...")
    
    model = NeuralNetMLP_TwoHidden(
        num_features=4,
        num_hidden1=3,
        num_hidden2=2,
        num_classes=3,
        random_seed=123
    )
    
    x = np.random.randn(5, 4)
    y = np.array([0, 1, 2, 0, 1])
    
    a_h1, a_h2, a_out = model.forward(x)
    
    (d_loss__dw_out, d_loss__db_out,
     d_loss__d_w_h2, d_loss__d_b_h2,
     d_loss__d_w_h1, d_loss__d_b_h1) = model.backward(x, a_h1, a_h2, a_out, y)
    
    print(f"  Output weight gradient shape: {d_loss__dw_out.shape}")
    print(f"  Output bias gradient shape: {d_loss__db_out.shape}")
    print(f"  Hidden 2 weight gradient shape: {d_loss__d_w_h2.shape}")
    print(f"  Hidden 2 bias gradient shape: {d_loss__d_b_h2.shape}")
    print(f"  Hidden 1 weight gradient shape: {d_loss__d_w_h1.shape}")
    print(f"  Hidden 1 bias gradient shape: {d_loss__d_b_h1.shape}")
    
    assert d_loss__dw_out.shape == model.weight_out.shape
    assert d_loss__db_out.shape == model.bias_out.shape
    assert d_loss__d_w_h2.shape == model.weight_h2.shape
    assert d_loss__d_b_h2.shape == model.bias_h2.shape
    assert d_loss__d_w_h1.shape == model.weight_h1.shape
    assert d_loss__d_b_h1.shape == model.bias_h1.shape
    
    print("  ✓ Backward pass test passed!")
    return True

def test_training():
    print("\nTesting training loop...")
    
    np.random.seed(123)
    x_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 4, 100)
    
    model = NeuralNetMLP_TwoHidden(
        num_features=10,
        num_hidden1=8,
        num_hidden2=6,
        num_classes=4,
        random_seed=123
    )
    
    learning_rate = 0.1
    losses = []
    
    for epoch in range(20):
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
        
        y_train_onehot = int_to_onehot(y_train, 4)
        loss = np.mean((a_out - y_train_onehot)**2)
        losses.append(loss)
    
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    
    assert losses[-1] < losses[0], "Loss did not decrease during training"
    
    print("  ✓ Training test passed!")
    return True

def test_prediction():
    print("\nTesting predictions...")
    
    model = NeuralNetMLP_TwoHidden(
        num_features=5,
        num_hidden1=4,
        num_hidden2=3,
        num_classes=3,
        random_seed=123
    )
    
    x_test = np.random.randn(10, 5)
    _, _, a_out = model.forward(x_test)
    predictions = np.argmax(a_out, axis=1)
    
    print(f"  Test samples: {x_test.shape[0]}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Unique predictions: {np.unique(predictions)}")
    
    assert predictions.shape == (10,)
    assert np.all((predictions >= 0) & (predictions < 3))
    
    prob_sums = np.sum(a_out, axis=1)
    print(f"  Probability sums range: [{prob_sums.min():.4f}, {prob_sums.max():.4f}]")
    
    print("  ✓ Prediction test passed!")
    return True

def main():
    print("="*60)
    print("Testing NeuralNetMLP_TwoHidden Implementation")
    print("="*60)
    
    try:
        test_forward_pass()
        test_backward_pass()
        test_training()
        test_prediction()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe two-hidden-layer neural network is working correctly.")
        print("Ready to proceed with MNIST classification!")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    main()
