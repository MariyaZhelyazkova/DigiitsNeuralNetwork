import pickle
import numpy as np
import matplotlib.pyplot as plt
import NNMultyLeyersClass

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
ANN_ml = NNMultyLeyersClass.NeuralNetwork(network_structure =[image_pixel, 80, 80, 10], 
                                          learning_rate = 0.1,
                                          bias = None)

print("Training is begining. Please wait.")

epochs = 3
ANN_ml.train(train_images, train_labels_one_hot, epochs = epochs)
for i in range(20):
    res = ANN_ml.run(test_images[i])
    image = test_images[i].reshape((28,28))
    plt.imshow(image, cmap="Greys")
    plt.show()
    print(test_labels[i], np.argmax(res), np.max(res))
    
corrects, wrongs = ANN_ml.evaluate(train_images, train_labels)
print("Accruracy of train set: ", corrects / ( corrects + wrongs))

corrects, wrongs = ANN_ml.evaluate(test_images, test_labels)
print("Accruracy of test set: ", corrects / ( corrects + wrongs))