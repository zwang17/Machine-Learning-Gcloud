import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

### Global parameters
input_size = 8
output_size = 1
model1 = ''
model2 = ''
##########################################################
pickle_file = 'C:/Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\Data\\Taxi Trip Duration(Kaggle)\\test_1.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  del save
  print('Test set', test_dataset.shape)


submission = []
for i in range(len(test_dataset)):
    submission.append([])
    submission[i].append(test_dataset[i][0])
test_dataset = np.delete(test_dataset,0,1)

def reformat(dataset):
    dataset = dataset.reshape((-1, input_size)).astype(np.float32)
    return dataset

session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Taxi Trip Duration(Kaggle)\\{}\\Saved.meta'.format(model1))
saver.restore(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Taxi Trip Duration(Kaggle)\\{}\\Saved'.format(model1))
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

with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission_1.pickle','wb') as f:
    save = {'submission':submission}
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)

#########################################################

pickle_file = 'C:/Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\Data\\Taxi Trip Duration(Kaggle)\\test_2.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  test_dataset = save['test_dataset']
  del save
  print('Test set', test_dataset.shape)


submission = []
for i in range(len(test_dataset)):
    submission.append([])
    submission[i].append(test_dataset[i][0])
test_dataset = np.delete(test_dataset,0,1)

session = tf.Session()
saver = tf.train.import_meta_graph('C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Taxi Trip Duration(Kaggle)\\{}\\Saved.meta'.format(model2))
saver.restore(session,'C:\\Users\\zheye1218\\Google Drive\\Deep_Learning_Data\\model\\Taxi Trip Duration(Kaggle)\\{}\\Saved'.format(model2))
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

print(submission)
if input('proceed?') != 'Y':
    assert False

with open('C:\\Users\\alien\Desktop\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\submission_2.pickle','wb') as f:
    save = {'submission':submission}
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)

