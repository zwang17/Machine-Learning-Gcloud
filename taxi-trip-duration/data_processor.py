import os.path
import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir','output','Output Directory.')

def process():
    train_data = pd.read_csv(os.path.join(FLAGS.input_dir, 'train_processed.csv'))
    train_data = train_data.reset_index(drop=True)

    print('Augmenting data...')
    print(train_data.shape)

    def getDistance(lon1,lat1,lon2,lat2):
        # in kilometers
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r
    def getSpeed(index):
        distance = getDistance(train_data.at[index,'pickup_longitude'],train_data.at[index,'pickup_latitude'],
                                                             train_data.at[index,'dropoff_longitude'],train_data.at[index,'dropoff_latitude'])
        time = train_data.at[index,'trip_duration']/3600
        return distance/time

    drop_count = 0
    for i in range(len(train_data)):
        if i % 5000 == 0:
            print(i/len(train_data)*100,"%")
            print('dropped: ',drop_count)
        speed = getSpeed(i)
        if speed>140.0 or speed<0.9:
            train_data = train_data.drop(i)
            drop_count += 1

    print(drop_count)
    print(train_data.shape)

    train_data.to_csv(os.path.join(FLAGS.output_dir,'train_processed.csv'),index=False,header=True)

def main(_):
    process()

if __name__=='__main__':
    tf.app.run()