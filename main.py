import math
import time
import keras
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Reads the mnist dataset, flattens it from 28*28 arrays to 784 element 1d arrays and returns the data.
def read_flatten_mnist():
    print("Reading files...\n")
    # Load the dataset using the applicable keras function.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten the train and test data from 28*28 2d arrays to 784 element 1d arrays.
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    # Return the data.
    return (x_train, y_train), (x_test, y_test)


# Function that takes the train data and labels and the test data and labels as input and returns the train and test
# data accuracy using the nearest class centroid classifier.
def ncc(x_train, y_train, x_test, y_test):
    # Create the nearest class centroid classifier and train it with the train data and labels.
    ncc_classifier = NearestCentroid()
    ncc_classifier.fit(x_train, y_train)
    # Calculate the train and test data accuracies.
    train_accuracy = ncc_classifier.score(x_train, y_train)
    test_accuracy = ncc_classifier.score(x_test, y_test)
    return [train_accuracy, test_accuracy]


# Function that takes the train data and labels, the test data and labels and the number of neighbors k as input and
# returns the train and test data accuracy using the k nearest neighbors classifier.
def knn(x_train, y_train, x_test, y_test, k):
    # Create the k nearest neighbor classifier and train it with the train data and labels.
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    # Calculate the train and test data accuracies.
    train_accuracy = knn_classifier.score(x_train, y_train)
    test_accuracy = knn_classifier.score(x_test, y_test)
    return [train_accuracy, test_accuracy]


# Function that calculates and returns the average accuracy given the outputs of a model and the true values.
def calculate_accuracy(output, y_true):
    predictions = np.argmax(output, axis=1)
    return np.mean(predictions == y_true)


# Function that finds a wrong prediction of the given model for the input data x and labels y and returns its index.
def find_wrong_prediction(custom_nn, x, y, print_all_predictions=False):
    print("\nLooking for a wrong prediction in the neural network:")
    if print_all_predictions:
        print()
    # The current sample (starting from zero).
    sample = 0
    # The probabilities output of the model for the current sample.
    probabilities = custom_nn.predict(x[sample])
    # The most likely prediction based on the model's output probabilities.
    prediction = np.argmax(probabilities)
    # The label of the current sample.
    label = y[sample]

    # Check every sample until a wrong prediction is found.
    while sample < len(y) and prediction == label:
        if print_all_predictions:
            print(f"Sample {sample}: prediction is {prediction}, label is {label}.")
        # Move on to the next sample.
        sample += 1
        probabilities = custom_nn.predict(x[sample])
        prediction = np.argmax(probabilities)
        label = y[sample]

    # If a wrong prediction was found, print the results and its probabilities outputted by the model.
    if prediction != label:
        print(f"\nSample {sample}: prediction is {prediction}, label is {label}.")
        print(f"Here is the model's output for sample {sample}:")
        print(probabilities)
        return sample
    else:
        print("No wrong predictions were found.")
        return -1


# Runs the code of the intermediate first project.
def intermediate_project(x_train, y_train, x_test, y_test):
    # Calculate and print the train and test accuracy of the nearest class centroid classifier as well as the time it
    # took complete the calculations.
    print("Intermediate project:\n")
    print("NCC" + " calculations started...")
    start_time = time.time()
    [train_acc, test_acc] = ncc(x_train, y_train, x_test, y_test)
    finish_time = time.time()
    print("-Train accuracy: " + str(train_acc) + " (" + str(round(train_acc * 100, 2)) + "%).")
    print("-Test accuracy: " + str(test_acc) + " (" + str(round(test_acc * 100, 2)) + "%).")
    print("-Time elapsed for training and scoring train and test data: " + str(round(finish_time - start_time, 2))
          + " seconds.")

    # For each k value, calculate and print the train and test accuracy of the k nearest neighbor classifier as well as
    # the time it took to complete the calculations.
    for k in [1, 3]:
        print("\nKNN" + " calculations for " + "k" + "=" + str(k) + " started...")
        start_time = time.time()
        [train_acc, test_acc] = knn(x_train, y_train, x_test, y_test, k)
        finish_time = time.time()
        print("-k=" + str(k) + " train accuracy: " + str(train_acc) + " (" + str(
            round(train_acc * 100, 2)) + "%)")
        print("-k=" + str(k) + " test accuracy: " + str(test_acc) + " (" + str(
            round(test_acc * 100, 2)) + "%)")
        print("-Time elapsed for training and scoring train and test data: " + str(round(finish_time - start_time, 2))
              + " seconds.")

    print()


# Trains a neural network using the keras framework with a 64 neuron dense input layer with 784 inputs, a 256 neuron
# dense hidden layer and a 10 neuron output layer, one neuron per class/number. This code is taken from the lessons of
# mister Passalis.
def keras_nn(x_train, y_train, x_test, y_test):
    print("Keras neural network:\n")

    # Encode the labels.
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model = Sequential()

    # For the first layer we have to define the input dimensionality.
    model.add(Dense(64, activation='relu', input_dim=784))
    # Add a second hidden layer.
    model.add(Dense(256, activation='relu'))
    # Add an output layer (the number of neurons must match the number of classes).
    model.add(Dense(10, activation='softmax'))

    # Select an optimizer.
    adam = Adam(lr=0.0001)
    # Select the loss function and metrics that should be monitored.
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # Train the model with the test data.
    model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=2)

    print(f"\nTrain data results:")
    model.evaluate(x_train, y_train, batch_size=128)
    print(f"Test data results:")
    model.evaluate(x_test, y_test, batch_size=128)
    print()


# Trains and returns a custom neural network using the NeuralNetwork class with a 64 neuron dense input layer with 784
# inputs, a 256 neuron dense hidden layer and a 10 neuron output layer, one neuron per class/number.
def custom_neural_network(x_train, y_train, x_test, y_test):
    print("Custom neural network:\n")

    # Create the neural network using the NeuralNetwork class.
    custom_nn = NeuralNetwork(learning_rate=0.01, momentum=0.9, decay=0.01)

    # Add a dense input layer which takes 784 inputs, has 64 neurons and uses the ReLU activation function.
    custom_nn.add_layer(784, 64, "ReLU", "Dense")
    # Add a dense hidden layer which takes 64 inputs, has 256 neurons and uses the ReLU activation function.
    custom_nn.add_layer(64, 256, "ReLU", "Dense")
    # Add a dense output layer which takes 256 inputs, has 10 neurons/outputs (one for each class/number) and uses the
    # SoftMax activation function and the categorical cross entropy loss function.
    custom_nn.add_layer(256, 10, "SoftMaxCategoricalCrossEntropy", "Dense")

    # Train the model for 50 epochs with a batch size of 256 and save the total training time.
    training_time = custom_nn.fit(x_train, y_train, 50, batch_size=256)

    # Test the model with the train data and print the results.
    print("\nTrain data results:")
    custom_nn.evaluate(x_train, y_train, print_results=True)
    # Test the model with the test data and print the results.
    print("Test data results:")
    custom_nn.evaluate(x_test, y_test, print_results=True)
    # Print the training time.
    print(f"Total time elapsed for training: {round(training_time, 2)} seconds.")

    # Returns the neural network so that it can be used outside this function.
    return custom_nn


# Parent class that represents a general layer of neurons.
class Layer:
    # Constructor that initializes the layer's weights and biases.
    def __init__(self, input_num, neuron_num):
        # Biases initialized to 0.
        self.biases = np.zeros((1, neuron_num))
        # Weights initialized to random values using the standard normal distribution provided by numpy and multiplied
        # by a small factor to keep the initial weights small.
        self.weights = 0.01 * np.random.randn(input_num, neuron_num)
        # Saves the inputs of the layer.
        self.inputs = None
        # Saves the layer's output.
        self.output = None
        # Save the gradients with respect to the corresponding parameter.
        self.inputs_gradients = None
        self.biases_gradients = None
        self.weights_gradients = None
        # Initialize the biases momenta for the optimizer to zeroes.
        self.biases_momenta = np.zeros_like(self.biases)
        # Initialize the weights momenta for the optimizer to zeroes.
        self.weights_momenta = np.zeros_like(self.weights)

    # Method that executes a forward pass with the given inputs.
    def forward_pass(self, inputs):
        pass

    # Method that executes back propagation given the gradients returned by the next layer for its inputs.
    def back_propagate(self, next_layer_gradients):
        pass


# Class that implements a dense layer and inherits the Layer class.
class DenseLayer(Layer):
    def __init__(self, input_num, neuron_num):
        super().__init__(input_num, neuron_num)

    def forward_pass(self, inputs):
        # Save the inputs for back propagation.
        self.inputs = inputs
        # Calculate the output values based on the inputs, the weights and the biases.
        self.output = np.dot(inputs, self.weights) + self.biases

    def back_propagate(self, next_layer_gradients):
        # Calculate the gradients with respect to the layer's inputs. The weights' matrix must be transposed to make the
        # dimensions of the dot product fit.
        self.inputs_gradients = np.dot(next_layer_gradients, self.weights.T)
        # Calculate the gradients with respect to the layer's biases.
        self.biases_gradients = np.sum(next_layer_gradients, axis=0, keepdims=True)
        # Calculate the gradients with respect to the layer's weights. The inputs' matrix must be transposed to make the
        # dimensions of the dot product fit.
        self.weights_gradients = np.dot(self.inputs.T, next_layer_gradients)


# Parent class that represents an activation function.
class ActivationFunction:
    def __init__(self):
        # Saves the inputs of the layer.
        self.inputs = None
        # Saves the layer's output.
        self.output = None
        # Save the gradients with respect to the layer's inputs.
        self.inputs_gradients = None

    # Method that executes a forward pass with the given inputs.
    def forward_pass(self, inputs):
        pass

    # Method that executes back propagation given the gradients returned by the next layer with respect to its inputs.
    def back_propagate(self, next_layer_gradients):
        pass


# Class that implements the ReLU activation function and inherits the ActivationFunction class.
class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward_pass(self, inputs):
        # Save the inputs for back propagation.
        self.inputs = inputs
        # Calculate the output values by setting every input value which is negative to 0 and not touching the others.
        self.output = np.maximum(0, inputs)

    def back_propagate(self, next_layer_gradients):
        # Make a copy of the next layer's gradients so that it can be edited.
        self.inputs_gradients = next_layer_gradients.copy()
        # Turn every value of the next layer's gradients which is non-positive to 0 as the ReLU function wasn't
        # activated for those values.
        self.inputs_gradients[self.inputs <= 0] = 0


# Class that implements the SoftMax activation function and inherits the ActivationFunction class.
class SoftMax(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward_pass(self, inputs):
        # Save the inputs for back propagation.
        self.inputs = inputs
        # Calculate the output using the inputs based on the SoftMax formula.
        # First calculate the exponential inputs.
        exponential_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Then calculate the sum of the exponential inputs.
        exponential_inputs_sum = np.sum(exponential_inputs, axis=1, keepdims=True)
        # Lastly, normalize the exponential inputs.
        self.output = exponential_inputs/exponential_inputs_sum

    def back_propagate(self, next_layer_gradients):
        # Make an empty array with the dimensions of the next layer's gradients.
        self.inputs_gradients = np.empty_like(next_layer_gradients)

        # For loop which iterates each of the output and gradients.
        for index, (single_output, single_next_layer_gradient) in enumerate(zip(self.output, next_layer_gradients)):
            # Reshape the single input using reshape(-1, 1) to have only one column and as many rows as needed.
            reshaped_single_output = single_output.reshape(-1, 1)
            # Create a diagonal array whose diagonal values are the single output's values.
            diagonal_single_output = np.diagflat(reshaped_single_output)
            # Calculate the Jacobian matrix of the single output using the previous diagonal array and the reshaped
            # single output.
            jacobian_matrix = diagonal_single_output - np.dot(reshaped_single_output, reshaped_single_output.T)
            # Calculate the gradient and add it to the array of gradients.
            self.inputs_gradients[index] = np.dot(jacobian_matrix, single_next_layer_gradient)


# Class that implements the stochastic gradient descent optimizer.
class SgdOptimizer:
    # Constructor that initializes the optimizer's parameters. The default values of the parameters are the same as
    # keras' default values.
    def __init__(self, learning_rate=0.01, momentum=0.0, decay=0.0):
        # Learning rate, defaults to 0.01.
        self.learning_rate = learning_rate
        # Momentum, defaults to 0.0.
        self.momentum = momentum
        # Decay rate, defaults to 0.0.
        self.decay = decay
        # Current learning rate, is initialized to the given learning rate.
        self.current_learning_rate = learning_rate
        # Iteration counter.
        self.iteration_counter = 0

    # Updates the learning rate if the decay is enabled (it is different from 0.0). Must be called once before any
    # parameter update.
    def update_learning_rate(self):
        # If the decay is enabled, calculate the current learning rate using the decay formula.
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration_counter))

    # Updates the weights and biases of the given layer.
    def update_layer_parameters(self, layer):
        # Calculate the needed changes in biases and weights. If the momentum is not 0.0 (it is enabled), it is used in
        # the calculations, else if it is 0.0 (it is disabled), that part of the calculation becomes 0 and thus doesn't
        # affect it.
        bias_updates = self.momentum * layer.biases_momenta - self.current_learning_rate * layer.biases_gradients
        weight_updates = self.momentum * layer.weights_momenta - self.current_learning_rate * layer.weights_gradients

        # If the momentum is enabled (it is not 0.0), update the momenta for the biases and the weights.
        if self.momentum:
            layer.biases_momenta = bias_updates
            layer.weights_momenta = weight_updates

        # Update the layer's biases and weights using the previously calculated bias updates and weight updates.
        layer.biases += bias_updates
        layer.weights += weight_updates

    # Increments the iteration counter. Must be called once after any parameter update.
    def increment_iteration_counter(self):
        self.iteration_counter += 1


# Parent class that represents a loss function.
class Loss:
    # Constructor for the class.
    def __init__(self):
        # The gradients of the loss function with respect to the loss function's inputs.
        self.inputs_gradients = None

    # Calculates the average loss of all the give samples, given a model's output and the true values.
    def calculate_loss(self, output, y):
        # Calculate the loss for each sample.
        individual_losses = self.forward_pass(output, y)
        # Calculate and return the average loss.
        return np.mean(individual_losses)


# Class that represents the categorical cross entropy loss function and inherits the Loss function.
class CategoricalCrossEntropy(Loss):
    # Method that executes a forward pass given a model's predictions and the true values.
    def forward_pass(self, y_pred, y_true):
        # Calculate the number of samples in the given predictions.
        sample_num = len(y_pred)
        # Clip the predictions to prevent division by 0.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Save the probabilities for the true values in a 1d array, using different methods based on if the true values
        # are saved as categorical labels (1d array) or one-hot encoded labels (2d array).
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample_num), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Calculate and return the loss for each sample using the negative logarithm of the correct confidences.
        return -np.log(correct_confidences)

    # Method that executes back propagation given the gradients returned by the next layer and the true values.
    def back_propagate(self, next_layer_gradients, y_true):
        # Calculate the number of samples in the given gradients.
        sample_num = len(next_layer_gradients)
        # Calculate the number of labels in each sample using the first sample.
        label_num = len(next_layer_gradients[0])

        # If the true labels are sparse, turn them into a one-hot vector.
        if len(y_true.shape) == 1:
            y_true = np.eye(label_num)[y_true]

        # Calculate the gradient with respect to the inputs.
        self.inputs_gradients = - y_true / next_layer_gradients
        # Normalize the previously calculated gradient.
        self.inputs_gradients = self.inputs_gradients / sample_num


# Class that merges the SoftMax activation function and the categorical cross entropy loss function to avoid division by
# zero.
class SoftmaxCategoricalCrossEntropy:
    # Initializer that creates the SoftMax activation function and categorical cross entropy loss function objects.
    def __init__(self):
        self.activation_function = SoftMax()
        self.loss_function = CategoricalCrossEntropy()
        self.output = None
        self.inputs_gradients = None

    # Method that executes a forward pass given the inputs and the true values and returns the average loss value.
    def forward_pass(self, inputs, y_true):
        # Calculate the layer's activation function output.
        self.activation_function.forward_pass(inputs)
        # Save the previously calculated output.
        self.output = self.activation_function.output
        # Calculate and return the average loss value.
        return self.loss_function.calculate_loss(self.output, y_true)

    # Method that executes back propagation given the gradients returned by the next layer and the true values.
    def back_propagate(self, next_layer_gradients, y_true):
        # Calculate the number of samples in the given gradients.
        samples_num = len(next_layer_gradients)

        # If labels are one-hot encoded, turn them into categorical labels.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Make a copy of the next layer's gradients so that it can be edited.
        self.inputs_gradients = next_layer_gradients.copy()
        # Calculate and save the gradient.
        self.inputs_gradients[range(samples_num), y_true] -= 1
        # Normalize the previously calculated gradient.
        self.inputs_gradients = self.inputs_gradients / samples_num


# Class that represents a whole neural network.
class NeuralNetwork:
    # Initializer that takes as input the optimizer type and it's parameters and initializes the optimizer and the list
    # of layers and activation functions of the model.
    def __init__(self, optimizer="SGD", learning_rate=0.01, momentum=0.0, decay=0.0):
        # The list of layers of the model.
        self.layers = []
        # The list of activation functions of the model.
        self.activation_functions = []
        # If the chosen optimizer is the stochastic gradient descent, create it with the given parameters.
        if optimizer == "SGD":
            self.optimizer = SgdOptimizer(learning_rate, momentum, decay)

    # Method that adds a new layer to the mode taking it's input count, neuron count, activation function type and layer
    # type as input.
    def add_layer(self, input_num, neuron_num, activation_function, layer_type="Dense"):
        # Create and add the corresponding layer to the model's layers list based on its given type.
        if layer_type == "Dense":
            self.layers.append(DenseLayer(input_num, neuron_num))

        # Create and add the corresponding activation function to the model's activation functions list based on its
        # given type. The SoftMax activation function is bundled with the categorical cross entropy loss function using
        # the corresponding class because using the SoftMax class on its own produces zero division errors.
        if activation_function == "ReLU":
            self.activation_functions.append(ReLU())
        elif activation_function == "SoftMaxCategoricalCrossEntropy":
            self.activation_functions.append(SoftmaxCategoricalCrossEntropy())

    # Method that trains the model with the given input data (x) and labels (y). The parameters are the amount of
    # epochs, the epoch print rate which changes how often progress is printed and the batch size.
    def fit(self, x, y, epochs=1, epoch_print_rate=1, batch_size=32):
        # The fist dimension of the input data (the sample count).
        sample_num = len(x)
        # The amount of batches that will be created (or the amount of batches -1).
        batch_num = sample_num // batch_size
        # The total training time without counting the evaluation time when printing progress.
        training_time = 0

        # For loops which loops for the given amount of epochs.
        for epoch in range(epochs):
            # Save the start time of the epoch.
            epoch_start_time = time.time()

            # For loop which loops for all the batches.
            for batch_index in range(batch_num):
                # If this is the last batch, save the corresponding x and y values from the start of this batch to the
                # last sample and label.
                if batch_index == batch_num - 1:
                    current_input = x[range(batch_index * batch_size, batch_num * batch_size), :]
                    current_y = y[range(batch_index * batch_size, batch_num * batch_size)]

                # Else if this isn't the last batch, save the corresponding x and y values from the start of this batch
                # to the start of the next batch minus one.
                else:
                    current_input = x[range(batch_index * batch_size, (batch_index + 1) * batch_size), :]
                    current_y = y[range(batch_index * batch_size, (batch_index + 1) * batch_size)]

                # For loop which loops over each layer of the model as well as its corresponding activation function.
                for layer_index in range(len(self.layers)):
                    # Save the current layer and activation function.
                    layer = self.layers[layer_index]
                    activation_function = self.activation_functions[layer_index]

                    # Do a forward pass of the current input through the current layer.
                    layer.forward_pass(current_input)

                    # If the current layer's activation function is the SoftMax (which is bundled with the categorical
                    # cross entropy), do a forward pass of the current layer's output through the activation function.
                    if isinstance(activation_function, SoftmaxCategoricalCrossEntropy):
                        activation_function.forward_pass(layer.output, current_y)

                    # Else if the current layer's activation function is not the SoftMax , do a forward pass of the
                    # current layer's output through the activation function.
                    else:
                        activation_function.forward_pass(layer.output)

                    # Update the current input as the output of the current activation function so that it can be passed
                    # forward through the next layer if needed.
                    current_input = activation_function.output

                # Save the last activation function of the model (which is bundled with the loss function).
                final_function = self.activation_functions[len(self.activation_functions) - 1]

                # Initialize the gradients with respect to the inputs which will be used for back propagation.
                current_inputs_gradients = final_function.inputs_gradients
                # For loop which loops over all the model's layers in reverse order.
                for layer_index in range(len(self.layers) - 1, -1, -1):
                    # Save the current layer and activation function.
                    layer = self.layers[layer_index]
                    activation_function = self.activation_functions[layer_index]

                    # If the current layer's activation function is the SoftMax (which is bundled with the categorical
                    # cross entropy), do back propagation using the function's output and the true y values.
                    if isinstance(activation_function, SoftmaxCategoricalCrossEntropy):
                        activation_function.back_propagate(activation_function.output, current_y)

                    # Else if the current layer's activation function isn't the SoftMax, do back propagation using the
                    # gradients with respect to the inputs calculated in the previous loop.
                    else:
                        activation_function.back_propagate(current_inputs_gradients)

                    # Do back propagation through the current layer using the gradients with respect to inputs of the
                    # current activation function which were just calculated by the previous back propagation.
                    layer.back_propagate(activation_function.inputs_gradients)

                    # Update the gradients with respect to the inputs with the newly calculated gradients of the current
                    # layer.
                    current_inputs_gradients = layer.inputs_gradients

                # This is where the optimization is done.
                # Update the learning rate of the optimizer before optimizing.
                self.optimizer.update_learning_rate()

                # Optimize each layer of the model using its optimizer starting from the input layer.
                for layer_index in range(len(self.layers)):
                    self.optimizer.update_layer_parameters(self.layers[layer_index])

                # Increment the iteration counter of the optimizer.
                self.optimizer.increment_iteration_counter()

            # Save the finish time of the epoch and add the epoch's training time to the total training time.
            epoch_finish_time = time.time()
            training_time += epoch_finish_time - epoch_start_time

            # If it's time to print the progress given the epoch print rate, or it's the last epoch, print the progress.
            if (epoch + 1) % epoch_print_rate == 0 or epoch == epochs - 1:
                # Evaluate the whole data to get the average loss and accuracy.
                loss, accuracy = self.evaluate(x, y, print_results=False)

                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{math.ceil(sample_num / batch_size)}/{math.ceil(sample_num / batch_size)} - loss: {loss:.4f} "
                      f"- accuracy: {accuracy:.4f} - " + f"{(epoch_finish_time - epoch_start_time) * 1000:.0f}ms/epoch")

        return training_time

    # Method that evaluates the given input data (x) and labels (y) and returns the average loss and accuracy.
    def evaluate(self, x, y, print_results=False):
        # Initialize the current input to the give input data.
        current_input = x
        # Initialize the loss to a negative number to detect errors.
        loss = -1.0

        # For loop which loops over each layer of the model as well as its corresponding activation function.
        for layer_index in range(len(self.layers)):
            # Save the current layer and activation function.
            layer = self.layers[layer_index]
            activation_function = self.activation_functions[layer_index]

            # Do a forward pass of the current input through the current layer.
            layer.forward_pass(current_input)

            # If the current layer's activation function is the SoftMax (which is bundled with the categorical
            # cross entropy), do a forward pass of the current layer's output through the activation function
            # and save what it returns which is the average loss of the model.
            if isinstance(activation_function, SoftmaxCategoricalCrossEntropy):
                loss = activation_function.forward_pass(layer.output, y)

            # Else if the current layer's activation function is not the SoftMax , do a forward pass of the
            # current layer's output through the activation function.
            else:
                activation_function.forward_pass(layer.output)

            # Update the current input as the output of the current activation function so that it can be passed
            # forward through the next layer if needed.
            current_input = activation_function.output

        # Save the last activation function of the model (which is bundled with the loss function) and calculate
        # the model's average accuracy using the final activation/loss function's output.
        final_function = self.activation_functions[len(self.activation_functions) - 1]
        accuracy = calculate_accuracy(final_function.output, y)

        # If enabled, print the results.
        if print_results:
            print(f"loss: {loss:.4f} - accuracy: {accuracy:.4f}")

        # Return the calculated average loss and accuracy.
        return loss, accuracy

    # Method that returns the model's prediction for the given input x.
    def predict(self, x):
        # Initialize the current input to the give input data.
        current_input = x

        for layer_index in range(len(self.layers)):
            # Save the current layer and activation function.
            layer = self.layers[layer_index]
            activation_function = self.activation_functions[layer_index]

            # Do a forward pass of the current input through the current layer.
            layer.forward_pass(current_input)

            # If the current layer's activation function is the SoftMax (which is bundled with the categorical
            # cross entropy), do a forward pass of the current layer's output through the activation function with zero
            # as the real value (it is not needed in the calculations). Then return the output of the activation
            # function which is the model's prediction.
            if isinstance(activation_function, SoftmaxCategoricalCrossEntropy):
                activation_function.forward_pass(layer.output, np.array([0]))
                return activation_function.output

            # Else if the current layer's activation function is not the SoftMax , do a forward pass of the
            # current layer's output through the activation function.
            else:
                activation_function.forward_pass(layer.output)

            # Update the current input as the output of the current activation function so that it can be passed
            # forward through the next layer if needed.
            current_input = activation_function.output


# Read the flattened mnist dataset.
(X_train, Y_train), (X_test, Y_test) = read_flatten_mnist()

# Run the intermediate project's code with the three classifiers as described in the function's comments.
intermediate_project(X_train, Y_train, X_test, Y_test)

# Train and test a keras neural network as described in the function's comments. It is commented out by default.
# keras_nn(X_train, Y_train, X_test, Y_test)

# Train and test a custom neural network as described in the function's comments.
nn = custom_neural_network(X_train, Y_train, X_test, Y_test)

# Finds a wrong prediction by the model in the test data. It is commented out by default.
# wrong_sample_index = find_wrong_prediction(nn, X_test, Y_test, print_all_predictions=False)
