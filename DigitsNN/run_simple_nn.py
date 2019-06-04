import pickle
import numpy as np
import matplotlib.pyplot as plt
import NeuralNetworkClass

with open("D:/Python/MNIST/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

image_size = 28
image_pixel = image_size * image_size


print("Creating Neural Network.")
ANN = NeuralNetworkClass.NeuralNetwork(num_in_nodes = image_pixel, 
                                       num_out_nodes = 10, 
                                       num_hidden_nodes = 100, 
                                       learning_rate = 0.1)

print("Trainig is begining. Please wait.")
for i in range(len(train_images)):
    ANN.train(train_images[i], train_labels_one_hot[i])

print("Showing predictions for the first 20 images in the testing set.")
for i in range(20):
    res = ANN.run(test_images[i])
    image = test_images[i].reshape((28,28))
    plt.imshow(image, cmap="Greys")
    plt.show()
    print(test_labels[i], np.argmax(res), np.max(res))

print("Evaluationg Accuracy. Please wait.")    
corrects, wrongs = ANN.evaluate(train_images, train_labels)
print("Accruracy of train set: ", corrects / ( corrects + wrongs))

corrects, wrongs = ANN.evaluate(test_images, test_labels)
print("Accruracy of test set: ", corrects / ( corrects + wrongs))

print("Calculating Confusion Matrix. Please wait.")
cm = ANN.confusion_matrix(train_images, train_labels)
print(cm)

print("Calculating Precision and Recall. Please wait.")
for i in range(10):
    print("digit: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))