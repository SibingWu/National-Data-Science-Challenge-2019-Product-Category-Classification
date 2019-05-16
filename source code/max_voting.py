# This python files is used to use ensemble learning
# to combine the results of the two models with the best performance
# and generate the final result

import pandas as pd
from tensorflow import keras
import numpy as np
import tensorflow as tf
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

image_data = np.load("processed_data/test_image_data.npy")
itemid = np.load("processed_data/test_image_id.npy")
text_data = np.load("processed_data/test_text_data.npy")


#to load different models
model_image_1 = tf.keras.models.load_model('image_trained_models/VGG6.model')
model_image_2 = tf.keras.models.load_model('image_trained_models/VGG-class-weights.model')
#model_text_1 = tf.keras.models.load_model('text_trained_models/submission_1.model')


submission_dict=dict(itemid=[],Category=[])

for i in range(len(itemid)):
    print("{}/{}".format(i, len(itemid)))
    submission_dict['itemid'].append(itemid[i])

    image_data_corrected = np.array(image_data[i]).reshape(-1, 128, 128, 3)
    pred_1 = model_image_1 .predict([image_data_corrected])[0]
    pred_2 = model_image_2.predict([image_data_corrected])[0]
    #pred_3 = model_text_1.predict([text_data[i]])[0]

    weighted_pred = []

    for j in range(len(pred_2)):
        weighted_pred.append(pred_1[j]*0.5+pred_2[j]*0.5)

    category =np.argmax(weighted_pred)


    submission_dict['Category'].append(category)


submission = pd.DataFrame(data=submission_dict)

submission.to_csv('submission7.csv', index=False)

