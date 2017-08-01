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
flags.DEFINE_string('input_train_data','train_data','Input Training Data File Name.')
flags.DEFINE_integer('train_steps', 100000, 'Train Steps.')
flags.DEFINE_integer('learning_rate', 0.00001, 'Training Learning Rate.')

def run_training(input_data):
    pickle_file = os.path.join(FLAGS.input_dir, input_data)
    with file_io.FileIO(pickle_file, 'r') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      del save  # hint to help gc free up memory
      print('Training set', train_dataset.shape, train_labels.shape)
      print('Validation set', valid_dataset.shape, valid_labels.shape)

    input_size = len(train_dataset[1])
    output_size = 1


    def reformat(dataset):
        dataset = dataset.reshape((-1, input_size)).astype(np.float32)
        return dataset

    def error(predictions, labels):
        sum = 0.0
        for x in range(len(predictions)):
            p = np.log(predictions[x][0] + 1)
            r = np.log(labels[x][0] + 1)
            sum = sum + (p - r) ** 2
        return (sum / len(predictions)) ** 0.5

    train_dataset = reformat(train_dataset)
    valid_dataset = reformat(valid_dataset)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)

    n_nodes_hl1 = 2048
    n_nodes_hl2 = 2048

    batch_size = 100
    learning_rate = FLAGS.learning_rate

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,input_size))
        tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))
        tf_valid_dataset = tf.placeholder(
            tf.float32, shape=(valid_dataset.shape[0],input_size))
        tf_test_one = tf.placeholder(tf.float32, shape = (1, input_size),name='tf_test_one')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        #Variables
        layer_1_weights = tf.Variable(tf.truncated_normal([input_size,n_nodes_hl1],stddev=0.1))
        layer_1_biases = tf.Variable(tf.zeros([n_nodes_hl1]))
        layer_2_weights = tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1))
        layer_2_biases = tf.Variable(tf.zeros([n_nodes_hl2]))
        output_layer_weights = tf.Variable(tf.truncated_normal([n_nodes_hl2, output_size],stddev=0.1))
        output_layer_biases = tf.Variable(tf.zeros([output_size]))

        def model(data):
            l1 = tf.add(tf.matmul(data,layer_1_weights), layer_1_biases)
            l1 = tf.nn.dropout(tf.nn.relu(l1),keep_prob=keep_prob)

            l2 = tf.add(tf.matmul(l1, layer_2_weights), layer_2_biases)
            l2 = tf.nn.dropout(tf.nn.relu(l2),keep_prob=keep_prob)

            return tf.add(tf.matmul(l2,output_layer_weights), output_layer_biases)

        # Training computation
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf_train_labels, model(tf_train_dataset)))))

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

        train_prediction = model(tf_train_dataset)
        valid_prediction = model(tf_valid_dataset)
        test_prediction_one = tf.add(model(tf_test_one),0,name='test_prediction_one')

    itera = []
    v_er_list = []
    num_steps = FLAGS.train_steps

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        print("Initialized")
        step = 0
        while step < num_steps:
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
            _, l, predictions = session.run(
                [optimizer,loss,train_prediction], feed_dict=feed_dict)
            if (step % 5000 == 0):
                v_e = error(
                    valid_prediction.eval(
                        {tf_valid_dataset: valid_dataset, keep_prob: 1.0}
                    ), valid_labels)
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Validation error: %.5f" % v_e)
                itera.append(step)
                v_er_list.append(v_e)
            step += 1
            if (step == num_steps):
                v_e = error(
                    valid_prediction.eval(
                        {tf_valid_dataset: valid_dataset, keep_prob: 1.0}
                    ), valid_labels)
                print("Final Validation error: %.5f" % v_e)

        saver = tf.train.Saver()
        checkpoint_file = os.path.join(FLAGS.output_dir, 'checkpoint-{}'.format(input_data))
        saver.save(session, checkpoint_file, global_step=step)


def main(_):
    run_training(FLAGS.input_train_data)

if __name__=='__main__':
    tf.app.run()