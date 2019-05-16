#to save a model
# model.save('trained_model/name.model')

# This python file is used to generate the result files for every specific models

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

test_samples = pd.read_csv('data/test.csv')


def create_test_data(dataset,pic_root_dir, img_size):
    X=[] #contains imagedata for testing
    itemid=[] #corresponding itemid

    for i in range(len(dataset)):
        if dataset['image_path'][i][-4] != '.':
            img_path = pic_root_dir + str(dataset['image_path'][i]) + ".jpg"
        else:
            img_path = pic_root_dir + str(dataset['image_path'][i])
        img_array = cv2.imread(img_path)
        new_array = cv2.resize(img_array, (img_size, img_size))
        X.append(new_array)
        itemid.append(dataset['itemid'][i])
        
    X = np.array(X).reshape(-1, img_size, img_size, 3)

    return X, itemid



image_data, itemid= create_test_data(test_samples,'data/',128)

#to load a model
model = tf.keras.models.load_model('trained_models/kaggle.model')

submission_dict=dict(itemid=[],Category=[])


for i in range(len(itemid)):
    print("{}/{}".format(i, len(itemid)))
    submission_dict['itemid'].append(itemid[i])
    prediction = model.predict([test_data[i]])
    category = np.argmax(prediction[0])
    submission_dict['Category'].append(category)

submission = pd.DataFrame(data=submission_dict)

submission.to_csv('submission.csv', index=False)

