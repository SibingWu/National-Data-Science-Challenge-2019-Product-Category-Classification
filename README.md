# National-Data-Science-Challenge-2019-Product-Category-Classification
This repo serves as an archive of the project for the Shopee Data Science Challenge 2019. This project is mainly about Image Classification. Although our team did not rank at a satisfying level, I still regard this challenge as a key to the door of Machine Learning and Deep Learning for me.

1. Introduction 
In our project, Convolutional Neural Network (CNN) was used to develop an image classiﬁer. The algorithm pipeline is shown below.

2. Algorithm Pipeline 
Image data was processed using CV2. All images were resized to 128*128. 
The dataset is highly unbalanced. Hence, image augmentation using tensorﬂow was used to balance the dataset. We utilized the pre-trained VGG model combined with fine-tuning.

3. Result Analysis 
The model was trained with learning rate 0.0001 in 35 epochs. The over-ﬁtting has occurred after 14 epochs.
