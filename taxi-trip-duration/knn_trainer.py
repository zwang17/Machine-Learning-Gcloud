import os.path
from sklearn import neighbors
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('input_train_data','train_data','Input Training Data File Name.')
flags.DEFINE_integer('num_rounds', 101, 'Number of Rounds to Update Weight.')
flags.DEFINE_integer('learning_rate', 1, 'Training Learning Rate.')

def run_training(input_data):
    pickle_file = os.path.join(FLAGS.input_dir, input_data)

    def mydist(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return np.dot((x - y) ** 2, weight)

    def mse(predictions, labels):
        sum = 0.0
        for i in range(len(predictions)):
            sum = sum + (predictions[i][0] - labels[i][0]) ** 2
        return (sum / len(predictions)) ** 0.5

    def error(predictions, labels):
        sum = 0.0
        for x in range(len(predictions)):
            p = np.log(predictions[x][0] + 1)
            r = np.log(labels[x][0] + 1)
            sum = sum + (p - r) ** 2
        return (sum / len(predictions)) ** 0.5

    def batch_refresh():
        global test_data, test_label, train_data, train_label
        new_choice = np.random.choice(X.shape[0], mini_batch_size, replace=False)
        test_data, test_label = X[new_choice, :], Y[new_choice, :]
        train_choice = np.delete(range(X.shape[0]), new_choice)
        train_data, train_label = X[train_choice, :], Y[train_choice, :]

    def getLoss(type='error'):
        knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
        knn.fit(train_data, train_label)
        predict = knn.predict(test_data)
        if type == 'mse':
            return mse(predict, test_label)
        return error(predict, test_label)

    def weight_normalize():
        sum = 0
        for i in weight: sum+=i
        for i in range(len(weight)): weight[i] = weight[i]/sum * 100

    with file_io.FileIO(pickle_file, 'r') as f:
        save = pickle.load(f)
        train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'], save['train_labels'], save[
            'valid_dataset'], save['valid_labels']

    X = np.concatenate((train_dataset,valid_dataset))
    Y = np.concatenate((train_labels,valid_labels))
    print(X.shape)
    print(Y.shape)
    mini_batch_size = 80
    test_choice = np.random.choice(X.shape[0], mini_batch_size, replace=False)
    test_data, test_label = X[test_choice,:], Y[test_choice,:]
    train_choice = np.delete(range(X.shape[0]),test_choice)
    train_data, train_label = X[train_choice,:], Y[train_choice,:]
    print('Partitioned')
    print(train_data.shape)
    print(test_data.shape)

    weight = [1.000]*len(train_data[0])
    num_round = FLAGS.num_rounds
    step = 0.02
    learning_rate = FLAGS.learning_rate
    num_parameters = len(weight)
    round_list = []
    loss_list = []
    weight_list = []
    weight = np.asarray(weight)
    gradient = np.asarray([0.0]*weight.shape[0])
    print('Searching initialized!')
    weight_normalize()
    print('Initial weight:',['%.4f' % elem for elem in weight])
    for i in range(num_round):
        batch_refresh()
        # print('Initial loss: {}'.format(getLoss(type='mse')))
        for k in range(num_parameters):
            print("Looking for gradient on parameter {}".format(k))
            weight[k] = weight[k] + step
            right_loss = getLoss(type='mse')
            weight[k] = weight[k] - 2 * step
            left_loss = getLoss(type='mse')
            weight[k] = weight[k] + step
            gradient[k] = right_loss - left_loss
            print(gradient[k])
            if gradient[k] == 0.0:
                print('*flat')
        weight = weight - learning_rate * gradient
        for a in range(len(weight)):
            if weight[a] < 0:
                weight[a] = 0
        weight_normalize()
        # loss = getLoss(type='mse')
        # print('round:', i, ', current loss:', loss, ', current weight:',['%.4f' % elem for elem in weight])
        print('round:', i, ', current weight:', ['%.4f' % elem for elem in weight])
        if i % 10 == 0:
            knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
            knn.fit(train_dataset,train_labels)
            predict = knn.predict(valid_dataset)
            loss = error(predict,valid_labels)
            round_list.append(i)
            loss_list.append(loss)
            weight_list.append(weight)
            print('Test Loss at round {}: {}'.format(i,loss))
    print('final weight:',weight)

def main(_):
    run_training(FLAGS.input_train_data)

if __name__=='__main__':
    tf.app.run()