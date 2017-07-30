import os.path
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorflow.python.lib.io import file_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir','output','Output Directory.')
flags.DEFINE_integer('train_steps', 10000, 'Train Steps.')
pickle_file = os.path.join(FLAGS.input_dir, 'test.pickle')

def getTestDataset(file):
    with file_io.FileIO(file, 'r') as f:
      save = pickle.load(f)
      test_dataset = save['test_dataset']
      del save
      print('Test set', test_dataset.shape)
    return test_dataset

image_size = 28
num_labels = 10
num_channels = 1

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def run_training():
    session = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.output_dir, 'checkpoint-{}.meta'.format(FLAGS.train_steps)))
    saver.restore(session,os.path.join(FLAGS.output_dir, 'checkpoint-{}'.format(FLAGS.train_steps)))
    graph = tf.get_default_graph()

    test_prediction_one = graph.get_tensor_by_name('test_prediction_one:0')
    tf_test_one = graph.get_tensor_by_name('tf_test_one:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    test_dataset = getTestDataset(pickle_file).reshape(
        (-1, 1, image_size, image_size, num_channels)).astype(np.float32)

    submission = []
    for i in range(len(test_dataset)):
        result = session.run(test_prediction_one,feed_dict={tf_test_one:test_dataset[i],keep_prob: 1.0})
        if i % 1000 == 0:
            print(float(i)/len(test_dataset)*100,'%')
        submission.append(np.argmax(result))

    with file_io.FileIO(os.path.join(FLAGS.output_dir, 'submission.pickle'), 'w') as f:
        save = {'submission':submission}
        pickle.dump(save,f,protocol=2)

def main(_):
    run_training()

if __name__=='__main__':
    tf.app.run()