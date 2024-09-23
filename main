import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
weights = np.zeros(2)  # Two inputs for the OR gate
bias = 0
learning_rate = 0.1
epochs = 10

# Training data for the OR gate
training_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])
labels = np.array([0, 1, 1, 1])  # Expected outputs for OR gate


# Activation function
def activation(x):
    return 1 if x >= 0 else 0


# Prediction function
def predict(inputs):
    weighted_sum = np.dot(weights, inputs) + bias
    return activation(weighted_sum)


# Track error for plotting
error_history = []

# Training the perceptron
for _ in range(epochs):
    errors = 0  # Count errors for this epoch
    for inputs, label in zip(training_inputs, labels):
        prediction = predict(inputs)
        # Update weights and bias
        update = learning_rate * (label - prediction)
        weights += update * inputs
        bias += update

        # Count errors
        if prediction != label:
            errors += 1

    error_history.append(errors)

# Test the OR gate after training
print("Testing OR gate after training:")
for x in training_inputs:
    output = predict(x)
    print(f"Input: {x}, Output: {output}")

# Plotting error history
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), error_history, marker='o', color='red')
plt.title('Training Error over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassifications')
plt.xticks(range(epochs))
plt.grid()
plt.show()
