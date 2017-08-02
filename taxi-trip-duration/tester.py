import os.path
import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
from tensorflow.python.lib.io import file_io
### External Parameters:
### input_dir,output_dir,train_steps

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir','output','Output Directory.')
flags.DEFINE_string('input_test_file','input_test','Input File Name.')
flags.DEFINE_string('train_data_file','input_data','Corresponding Training Data File Name.')
flags.DEFINE_integer('train_steps', 100000, 'Train Steps.')

def run_testing(input_test_data):
    pickle_file = os.path.join(FLAGS.input_dir, input_test_data)
    with file_io.FileIO(pickle_file, 'r') as f:
      save = pickle.load(f)
      test_dataset = save['test_dataset']
      del save
      print('Test set', test_dataset.shape)

    input_size = len(test_dataset[1])-1

    submission = []
    for i in range(len(test_dataset)):
        submission.append([])
        submission[i].append(test_dataset[i][0])
    test_dataset = np.delete(test_dataset,0,1)

    session = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.output_dir, 'checkpoint-{}-{}.meta'.format(FLAGS.train_data_file,FLAGS.train_steps)))
    saver.restore(session,os.path.join(FLAGS.output_dir, 'checkpoint-{}-{}'.format(FLAGS.train_data_file,FLAGS.train_steps)))
    graph = tf.get_default_graph()

    test_prediction_one = graph.get_tensor_by_name('test_prediction_one:0')
    tf_test_one = graph.get_tensor_by_name('tf_test_one:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    test_dataset = test_dataset.reshape(
        (-1, 1, input_size)).astype(np.float32)

    for i in range(len(test_dataset)):
        result = session.run(test_prediction_one,feed_dict={tf_test_one:test_dataset[i],keep_prob: 1.0})
        if i % 500==0:
            print(float(i)/len(test_dataset)*100,'%')
        submission[i].append(result[0][0])

    with file_io.FileIO(os.path.join(FLAGS.output_dir, 'submission_from_{}.pickle'.format(input_test_data)), 'w') as f:
        save = {'submission':submission}
        pickle.dump(save,f,protocol=2)

def main(_):
    run_testing(FLAGS.input_test_file)

if __name__=='__main__':
    tf.app.run()