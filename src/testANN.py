import numpy as np
from ann import NeuralNetwork

new_network = NeuralNetwork()

training_file = '../config/training_data.csv'
data_set = new_network.read_data_sets(training_file)
print("Loaded training data...")

features, labels_norm = new_network.generate_train_test_data_sets(data_set)
print(features.shape)
labels = []
# convert label data to one_hot arrays
for k in range(0, len(labels_norm)):
    one_hot_label = np.zeros([3], dtype=float)
    one_hot_label[int(float(labels_norm[k]))] = 1.0
    labels = np.append(labels, one_hot_label)
labels = np.reshape(labels, [-1, new_network.n_labels])

print(labels.shape)

new_network.create_model()
new_network.predict(features)
