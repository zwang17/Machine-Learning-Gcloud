import os.path
from sklearn import neighbors
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import pandas as pd
from tensorflow.python.lib.io import file_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir','output','Output Directory.')

weight = [1.0]*13

def mydist(x,y):
    x,y = np.asarray(x),np.asarray(y)
    return np.dot((x-y)**2,weight)

def fetch_train(ven,day):
    pickle_file = os.path.join(FLAGS.input_dir, 'train_{}_{}.pickle'.format(ven,day))
    with file_io.FileIO(pickle_file, 'r') as f:
        save = pickle.load(f)
        train_dataset, train_labels, valid_dataset, valid_labels = save['train_dataset'],save['train_labels'],save['valid_dataset'],save['valid_labels']
    train_dataset = np.concatenate((train_dataset,valid_dataset))
    train_labels = np.concatenate((train_labels,valid_labels))
    return train_dataset,train_labels

def fetch_test(ven,day):
    pickle_file = os.path.join(FLAGS.input_dir, 'test_{}_{}.pickle'.format(ven, day))
    with file_io.FileIO(pickle_file, 'r') as f:
        save = pickle.load(f)
    return save['test_dataset']

def getPrediction(ven,day):
    train_dataset,train_labels = fetch_train(ven,day)
    test_dataset = fetch_test(ven,day)
    prediction = []
    for i in range(len(test_dataset)):
        prediction.append([])
        prediction[i].append(test_dataset[i][0])
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=20, metric=lambda x, y: mydist(x, y))
    knn.fit(train_dataset,train_labels)
    predict = knn.predict(test_dataset[:,1:])
    prediction = np.concatenate((prediction,predict),axis=1)
    return prediction

submission = np.array([['id','trip_duration']])
for v in [1,2]:
    for i in range(7):
        print('Predicting file test_{}_{}.pickle...'.format(v,i))
        submission = np.concatenate((submission,getPrediction(v,i)))
        print('vendor',v,' day',i,' completed!')

print(submission.shape)

df = pd.DataFrame(submission)
csv_file = os.path.join(FLAGS.output_dir,'submission.csv')
df.to_csv(csv_file,index=False,header=False)
