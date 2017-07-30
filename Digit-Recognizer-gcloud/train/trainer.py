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

pickle_file = os.path.join(FLAGS.input_dir, 'train.pickle');
with file_io.FileIO(pickle_file, 'r') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, 28, 28, 1)).astype(np.float32)
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def run_training():
    image_size = 28
    num_labels = 10
    num_channels = 1
    batch_size = 60
    patch_size = 5
    depth = 32
    num_hidden = 200

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.placeholder(
            tf.float32, shape=(valid_dataset.shape[0], image_size, image_size, num_channels),name='tf_valid_dataset')
        tf_test_one = tf.placeholder(
            tf.float32, shape = (1, image_size, image_size, num_channels),name='tf_test_one')
        tf_test_dataset = tf.placeholder(
            tf.float32, shape = (valid_dataset.shape[0], image_size, image_size, num_channels), name='tf_test_dataset')

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        # Model.
        def model(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            hidden = tf.nn.dropout(hidden,keep_prob=keep_prob)
            return tf.matmul(hidden, layer4_weights) + layer4_biases


        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset),name='valid_prediction')
        test_prediction_one = tf.nn.softmax(model(tf_test_one),name='test_prediction_one')
        test_prediction = tf.nn.softmax(model(tf_test_dataset),name='test_prediction')
    itera = []
    v_ac_list = []
    num_steps = FLAGS.train_steps

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      step = 0
      while step < num_steps:
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          a = accuracy(
            valid_prediction.eval({tf_valid_dataset: valid_dataset, keep_prob: 1.0}), valid_labels)
          print('Valid accuracy: %.1f%%' % a)
          itera.append(step)
          v_ac_list.append(a)
        step += 1
        # if (step % 10000 == 0):
        #     saver = tf.train.Saver()
        #     checkpoint_file = os.path.join(FLAGS.output_dir,'checkpoint')
        #     saver.save(session,checkpoint_file,global_step=step)
        if (step == num_steps):
            print('Valid accuracy: %.1f%%' % accuracy(
                valid_prediction.eval({tf_valid_dataset: valid_dataset, keep_prob: 1.0}), valid_labels))
            # if input("Optimization about to terminate. Do you want to proceed further? [Y/N] \n") == 'Y':
            #     inc = input("Increment by how many steps? \n")
            #     num_steps = num_steps + int(inc)
      saver = tf.train.Saver()
      checkpoint_file = os.path.join(FLAGS.output_dir, 'checkpoint')
      saver.save(session, checkpoint_file, global_step=step)


def main(_):
    run_training()

if __name__=='__main__':
    tf.app.run()