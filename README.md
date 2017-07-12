# Dual Attention Network
Tensorflow implementation of Dual Attention Network https://arxiv.org/abs/1611.00471

#### ./data
> this folder pocesses all the preprocessed training and testing data

#### ./vgg19
> this folder keeps thevgg19 network strucure and its pretrained weights

#### ./eval_result
> the default location of the prediction outcome after executing testing phase

#### pack_vqa_json.py
> split the VQA dataset into training and testing data. 
--split: 1: train on training set, test on validation set (default is 1)
         2: train on training and validation set, test on test set (test2015)

#### prepro.py
> preprocess the training and testing data; eliminate questions with uncommon answers, encode vocabularies in questions and answers, index training and testing data
--num_ans: consider only this amount of top frequent answers for the final classification; default is 1000
--max_length: truncate words of questions exceeding this length; default is 26
--word_count_threshold: encode words only which appear more than this threshold; default is 0

#### prepro_img.py
> use pretrained vgg19 to extract pool5 features of images and group V, Q, A together as Tensorflow Examples, and write them into TFRecord format

#### data_loader.py
> a class responsible for maintaining a queue reader to decode data TFRecord files and providing batched data for training and testing usage

#### action.py
> main program running training and testing phase of Dual Attention Network
--k: decide how many attention layers to use in this model; default is 2
--train: true for running training phase, false for running testing phase, default is true
--mc: only affect testing phase, choose between either answering OpenEnded type of question or MultipleChoice one; true for MultipleChoice, false for OpenEnded; default is false

#### Dual_Attention_Network.py
> the implementation of Dual Attention Network

!! the code is still dirty, please wait for more modification and refinement
