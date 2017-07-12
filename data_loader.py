import tensorflow as tf
import numpy as np
from vgg19 import vgg19_pretrain as vgg19_train 
from vgg19 import utils 
import h5py 
import codecs, json  
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
from scipy import ndimage
import scipy

IMG_TRAIN_TFRECORD_PATH = './data/prepro_img_feat_train/'
IMG_TEST_TFRECORD_PATH  = './data/prepro_img_feat_test/'
IMG_DIR_PATH = '../VQA_dataset_V1/images/'
MC_ANS_NUM = 18

class data_loader():
    def __init__( self, batch_size, dataset_path, qa_data_path, img_tfrecord_size, max_words_q, dim_img, is_training): 
        self.batch_size   = batch_size  
        self.dataset_path = dataset_path
        self.qa_data_path = qa_data_path 
        self.img_tfrecord_size = img_tfrecord_size 
        self.max_words_q  = max_words_q 
        self.dim_img      = dim_img 
        self.is_training  = is_training
        self.num_train = 0
        self.num_test  = 0
        self.dataset = {}
        self.qa_data = {}
        self.img_tfrecord_list = [] 
        self.img_buffer = {} 
        self.iterator_train = 0
        self.iterator_test = 0
        self.load_data()
        if self.is_training:
            self.idx_batch, self.ques_id_batch, self.question_batch, self.ques_len_batch, self.img_idx_batch, self.img_feat_batch, self.ans_batch = self.build_queue_train() 
        else:
            self.idx_batch, self.ques_id_batch, self.question_batch, self.ques_len_batch, self.img_idx_batch, self.img_feat_batch, self.MC_ans_batch = self.build_queue_test() 
        
    def load_data(self):  
        # load json file
        print('loading json file...')
        with open(self.dataset_path) as file: # data_prepro.json
            self.dataset = json.load(file)  
            self.vocabulary_size = len(self.dataset['ix_to_word'].keys()) # 12827
            self.unique_img_length_train = len(self.dataset['unique_img_train'])
            self.unique_img_length_test  = len(self.dataset['unique_img_test'])
            self.ans_vocab_size = len(self.dataset['ix_to_ans'])  
        # load h5 file
        print('loading h5 file...')
        with h5py.File(self.qa_data_path,'r') as hf: 
            # total number of training data is 225202
            self.qa_data['question'] = np.array(hf.get('ques_train')) # question (225202, 26)
            self.qa_data['ques_test'] = np.array(hf.get('ques_test'))  # 121512   
            self.num_train = self.qa_data['question'].shape[0]  
            self.num_test = self.qa_data['ques_test'].shape[0]
        
    def build_queue_train(self):
        print 'building queue to read img feat for training..'
        # generate filename
        for i in range(0, self.num_train, self.img_tfrecord_size):
            start = i
            end = start+self.img_tfrecord_size if start+self.img_tfrecord_size<self.num_train else self.num_train
            self.img_tfrecord_list.append( IMG_TRAIN_TFRECORD_PATH + 'train_%d_%d.tfrecord'%(start, end) )
    
        self.filename_queue = tf.train.string_input_producer(self.img_tfrecord_list, num_epochs=None, shuffle=True) 
        self.tfreader = tf.TFRecordReader()
        _, serialized_example = self.tfreader.read(self.filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={ 
                'i'      : tf.FixedLenFeature([], tf.int64),
                'q_id'   : tf.FixedLenFeature([], tf.int64),
                'ques'   : tf.FixedLenFeature([], tf.string),
                'q_len'  : tf.FixedLenFeature([], tf.int64),
                'img_idx': tf.FixedLenFeature([], tf.int64),
                'feats'  : tf.FixedLenFeature([], tf.string),
                'ans'    : tf.FixedLenFeature([], tf.int64)
            } 
        ) 
        i_out = features['i']  
        q_id_out = features['q_id']   
        ques_out  = tf.reshape( tf.decode_raw(features['ques'], tf.int32), [self.max_words_q]) 
        q_len_out = features['q_len']  
        img_idx_out = features['img_idx']  
        feats_out = tf.reshape( tf.decode_raw(features['feats'], tf.float32), self.dim_img) 
        ans_out = features['ans']  
        
        idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, ans_batch = tf.train.shuffle_batch(
                [i_out, q_id_out, ques_out, q_len_out, img_idx_out, feats_out, ans_out], 
                batch_size=self.batch_size, capacity=512, min_after_dequeue=256, num_threads=2)
         
        return idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, ans_batch
    
    def build_queue_test(self):
        print 'building queue to read img feat for testing..'
        for i in range(0, self.num_test, self.img_tfrecord_size):
            start = i
            end = start+self.img_tfrecord_size if start+self.img_tfrecord_size<self.num_test else self.num_test
            self.img_tfrecord_list.append( IMG_TEST_TFRECORD_PATH + 'test_%d_%d.tfrecord'%(start, end) )
        
        self.filename_queue = tf.train.string_input_producer(self.img_tfrecord_list, num_epochs=None, shuffle=False) 
        self.tfreader = tf.TFRecordReader()
        _, serialized_example = self.tfreader.read(self.filename_queue) 
                
        features = tf.parse_single_example(
            serialized_example,
            features={ 
                'i'      : tf.FixedLenFeature([], tf.int64),
                'q_id'   : tf.FixedLenFeature([], tf.int64),
                'ques'   : tf.FixedLenFeature([], tf.string),
                'q_len'  : tf.FixedLenFeature([], tf.int64),
                'img_idx': tf.FixedLenFeature([], tf.int64),
                'feats'  : tf.FixedLenFeature([], tf.string),
                'MC_ans' : tf.FixedLenFeature([], tf.string)
            } 
        ) 
        i_out = features['i']  
        q_id_out    = features['q_id']   
        ques_out     = tf.reshape( tf.decode_raw(features['ques'], tf.int32), [self.max_words_q])  ##### 
        q_len_out   = features['q_len']  
        img_idx_out = features['img_idx']  
        #feats_out   = tf.reshape( tf.decode_raw(features['feats'], tf.float32), self.dim_img) 
        feats_out   = tf.reshape( tf.decode_raw(features['feats'], tf.float32), [1, self.dim_img[0], self.dim_img[1], self.dim_img[2]]) 
        feats_out   = tf.reshape( feats_out, self.dim_img)
        MC_ans_out  = tf.reshape( tf.decode_raw(features['MC_ans'], tf.int32), [MC_ANS_NUM] )
        
        idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, MC_ans_batch = tf.train.batch(
                [i_out, q_id_out, ques_out, q_len_out, img_idx_out, feats_out, MC_ans_out], 
                batch_size=self.batch_size, capacity=512, num_threads=1)
        
        return idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, MC_ans_batch
    
    def get_train_length(self): 
        return self.num_train
    
    def get_test_length(self): 
        return self.num_test
    
    def get_answer_vocab_size(self):
        return self.ans_vocab_size
    
    def get_vocabulary_size(self):
        return self.vocabulary_size
     
    def get_next_train_batch( self, sess):
    # should only be called in training mode  
         
        idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, ans_batch = sess.run(
            [self.idx_batch, self.ques_id_batch, self.question_batch, self.ques_len_batch, self.img_idx_batch, self.img_feat_batch, self.ans_batch]
        ) 
        # img_idx_batch: the element of it is the index of 'unique_img_train'
         
        return idx_batch, img_idx_batch, img_feat_batch, ques_id_batch, question_batch, ques_len_batch, ans_batch
    
    def get_next_test_batch( self, sess):
        # should only be called in testing mode  
        
        idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, MC_ans_batch = sess.run(
            [self.idx_batch, self.ques_id_batch, self.question_batch, self.ques_len_batch, self.img_idx_batch, self.img_feat_batch, self.MC_ans_batch]
        )
        
        return idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, MC_ans_batch
    
    def get_answer(self, ans):
        #print ans
        return self.dataset['ix_to_ans'][str(ans)] 
    
    def get_prob(self, prob): 
        # print prob
        pred = np.argsort(prob)[::-1]
        
        # Get top1 label
        top1 = self.dataset['ix_to_ans'][str(pred[0])]
        #print "Top1: ", top1, prob[pred[0]]  
        # Get top5 label
        top5 = [(self.dataset['ix_to_ans'][str(pred[i])], prob[pred[i]]) for i in range(5)]
        #print "Top5: ", top5 
        return top1
 
    def get_question(self, q):
        #print q
        q_str = ''
        for i in range(len(q)):
            q_str = q_str+self.dataset['ix_to_word'][str(q[i])]+" "
        return q_str
 
    def get_train_image_path(self, img_idx):
        return IMG_DIR_PATH+self.dataset['unique_img_train'][img_idx]
    
    def get_test_image_path(self, img_idx):
        return IMG_DIR_PATH+self.dataset['unique_img_test'][img_idx]
    
    def print_accuracy(self, labels, predicts):
        # labels: [bs, 1]
        # predicts: [bs, 3000]
        num = len(labels)
        acc = 0
        for i in range(num):
            pred = np.argsort(predicts[i])[::-1]   
            if pred[0]==labels[i]: acc = acc + 1
        return acc*1.0/num
    
    def print_train_example(self, v, q, a): 
        print 'image: ', self.get_train_image_path(v)
        print 'question: ', self.get_question(q)
        print 'answer: ', self.get_prob(a), '\n' 
        
    def print_test_example(self, v, q, a): 
        print 'image: ', self.get_test_image_path(v)
        print 'question: ', self.get_question(q)
        print 'answer: ', self.get_prob(a), '\n' 
    
    def save_train_example(self, qid, img_idx, avn_list, ques, aut_list, ans):
        # Plot original image  
        img = utils.load_image( self.get_train_image_path(img_idx) )
        scipy.misc.imsave('./output/'+str(qid)+'.jpg', img) 
          
        for i in range(len(avn_list)):  
            alp_curr = avn_list[i].reshape(7,7)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
            scipy.misc.imsave('./output/'+str(qid)+'_'+str(i)+'.jpg', alp_img)
            
            img = Image.open('./output/'+str(qid)+'.jpg')
            heat = Image.open('./output/'+str(qid)+'_'+str(i)+'.jpg')
            img.paste( heat, (0,0), heat )
            img.save('./output/'+str(qid)+'_'+str(i)+'.jpg') 
    
    def save_test_example(self, qid, img_idx, avn_list, ques, aut_list, ans):
        # Plot original image  
        img = utils.load_image( self.get_test_image_path(img_idx) )
        scipy.misc.imsave('./output/'+str(qid)+'.jpg', img) 
          
        for i in range(len(avn_list)):  
            alp_curr = avn_list[i].reshape(7,7)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
            scipy.misc.imsave('./output/'+str(qid)+'_'+str(i)+'.jpg', alp_img)
            
            img = Image.open('./output/'+str(qid)+'.jpg')
            heat = Image.open('./output/'+str(qid)+'_'+str(i)+'.jpg')
            img.paste( heat, (0,0), heat )
            img.save('./output/'+str(qid)+'_'+str(i)+'.jpg') 