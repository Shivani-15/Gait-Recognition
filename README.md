# Gait-Recognition

### Dataset ###
CASIA -B dataset -> A large multiview gait database. There are 124 subjects, and the gait data was captured from 11 views with three variations, namely view angle, clothing and carrying condition changes, separately considered. All the videos are of different length.

### Problem Statement ###
Gait (walking style)  is a soft biometric. The purpose is to recognize people by their gait data.

### Solution ###
Our hypothesis challenged the traditional graph neural network approach, aiming to have the model itself learn unique relations through attention mechanisms for gait recognition. <br>
Two attention based architectures are implemented inspired from Video Vision Transformer. <br>
We have used model based approach instead of appearance based approach since it is better suited for gait recognition in wild. Body coordinates extracted from video frames using HRNet are used to generate input features.

### Steps to run ###
1. Use the coordinates dataset generated by passing CASIA-B dataset through HRNet as input for preprocess1.py.
2. Use the result csv of step 1 as input for preprocess2.py
3. Use the preprocessed dataset obtained in step 2 as input for any of the two models (model1 or model2).

### * Architecture 1: Factorized encoder * ###
<img src ="https://github.com/Shivani-15/Gait-Recognition/assets/58560161/1f83d35a-a46f-409b-9c5d-d2a70b0c95fd" width= 300 height = 400>

### * Architecture 2: Factorized self-attention * ###
<img src = "https://github.com/Shivani-15/Gait-Recognition/assets/58560161/7d73a6a7-0808-4c8c-beed-8dab2dd268ee" width= 300 height = 400>)
<br>

### Scope of improvement/ Future scope ###
1. Confidence value used as a input feature can be used in some other way, since it's nothing but a measure of accuracy of the model used to generate the coordinates from video frames.
2. Complexity of networks can be increased since the model is learning very complex relation all by itself (no information about natural connections in human body is fed).
