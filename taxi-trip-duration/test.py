import os.path
from sklearn import neighbors
import numpy as np
from six.moves import cPickle as pickle
import tensorflow as tf
from tensorflow.python.lib.io import file_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('input_train_data','train_data','Input Training Data File Name.')
pickle_file = os.path.join(FLAGS.input_dir, FLAGS.input_train_data)

def mydist(x, y):
    return np.dot((x - y) ** 2, weight)

with file_io.FileIO(pickle_file, 'r') as f:
    save = pickle.load(f)
    train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
        'valid_dataset'], save['valid_labels']

train_data = train_dataset[:1000]
train_label = train_labels[:1000]
test_data = valid_dataset[:100]
weight = [1.0]* len(train_dataset[1])
knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
knn.fit(train_data, train_label)
predict = knn.predict(test_data)
print(predict)
