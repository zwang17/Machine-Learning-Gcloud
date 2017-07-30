import csv
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

with open('C:\\Users\zheye1218\Desktop\\temp\\submission.pickle','rb') as f:
    save = pickle.load(f)
    submission = save['submission']
    del save

for i in range(28000):
    submission.append(i+1)
submission = np.asarray(submission)
temp = []
for i in range(28000):
    temp.append(submission[i+28000])
    temp.append(submission[i])
submission = np.reshape(temp,(28000,2))
submission = np.asarray(submission,dtype=str)
submission = np.insert(submission,0,[['ImageId','Label']],axis=0)

print(submission)

if input('Proceed?') != 'Y':
    assert False

df = pd.DataFrame(submission)
df.to_csv('C:\\Users\zheye1218\Desktop\\temp\\submission.csv',index=False,header=False)


