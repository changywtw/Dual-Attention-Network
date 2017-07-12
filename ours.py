#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, h5py, sys, argparse
import time
import math
import codecs, json  
import data_loader
import Dual_Attention_Network as DAN

# data_loader
IMG_TFRECORD_SIZE = 20000
PREPRO_DATASET_JSON = './data/prepro/data_prepro.json'  
PREPRO_QA_H5 = './data/prepro/data_prepro.h5' 

MAX_WORDS_Q = 26 
DIM_IMG = [7, 7, 512]

# DAN
SUMMARY_PATH = './log/' 
DAN_VAR_PATH='./data/dan_2_save.npy'
DAN_VAR_PATH=None

# Q
VOCAB_EMBED_SIZE = 300          # the encoding size of each token in the vocabulary
RNN_SIZE = 512                  # size of the rnn in number of hidden nodes in each layer 
HIDDEN_SIZE = 512 
 
# training settings
TOTAL_EPOCHS = 60  
LEARNING_RATE_STEP = [0.1, 0.05, 0.01] 
LEARNING_RATE_BOUNDARY = [20,  40] 
#TOTAL_EPOCHS = 60 
#LEARNING_RATE_STEP = [0.01, 0.005, 0.001, 0.0005, 0.0001] 
#LEARNING_RATE_BOUNDARY = [5,   10,   15,    20 ] 
BATCH_SIZE = 512
DROPOUT = 0.5 

PRINT_FREQ = 200 

def train( total_k_step ): 
    with tf.Graph().as_default():
        loader = data_loader.data_loader( 
                    batch_size=BATCH_SIZE, 
                    dataset_path=PREPRO_DATASET_JSON, 
                    qa_data_path=PREPRO_QA_H5, 
                    img_tfrecord_size=IMG_TFRECORD_SIZE, 
                    max_words_q=MAX_WORDS_Q, 
                    dim_img=DIM_IMG,
                    is_training=True)
        num_train = loader.get_train_length()
        vocab_size = loader.get_vocabulary_size() 
        answer_vocab_size = loader.get_answer_vocab_size() 
        print 'total questions: ' + str(num_train)
        print 'vocabulary_size : ' + str(vocab_size)
    
        with tf.Session() as sess:  
            print 'constructing model...'
            model = DAN.Dual_Attention_Network( 
                    scope='dan',
                    sess=sess, 
                    batch_size=BATCH_SIZE,  
                    k_step=total_k_step,  
                    dim_img=DIM_IMG, 
                    max_words_q=MAX_WORDS_Q, 
                    vocab_size=vocab_size, 
                    vocab_embed_size=VOCAB_EMBED_SIZE, 
                    rnn_size=RNN_SIZE, 
                    hidden_size=HIDDEN_SIZE,
                    ans_num=answer_vocab_size,
                    dan_npy_path=DAN_VAR_PATH)
            print 'model built'
            
            # writers
            print 'constructing writers...'
            summary_writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)
            #saver = tf.train.Saver() 
            
            print 'initialize global variables..'
            sess.run(tf.global_variables_initializer()) 
            tf.train.start_queue_runners(sess=sess)
            
            print 'start training...'
            lr_step = 0
            learning_rate = LEARNING_RATE_STEP[lr_step]  
            
            total_iteration = int( math.ceil( num_train*1.0 / BATCH_SIZE ) )
            print 'total_iteration per epoch: ', total_iteration
            
            global_step = 0
            tStart_total = time.time()
            for ep in range(TOTAL_EPOCHS): 
                  
                # init 
                tStart = time.time() 
                
                for itr in range(0, total_iteration):   
                    # fetch a batch
                    idx_batch, img_idx_batch, img_feat_batch, ques_id_batch, question_batch, ques_len_batch, ans_batch = loader.get_next_train_batch(sess)    
                    #print loader.get_train_image_path( img_idx_batch[0]) 
                    
                    # train model
                    batch_loss, sum, _  = model.train(img_feat_batch, question_batch, ques_len_batch, ans_batch, 1-DROPOUT, learning_rate)
                     
                    # write summary
                    summary_writer.add_summary( sum, global_step)
                    
                    # print loss & eval every print_freq
                    if np.mod(itr, PRINT_FREQ) == 0:
                        tStop = time.time()
                        
                        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                        
                        # print weight image    
                        avn_list = []
                        aut_list = []
                        for k in range(total_k_step):
                            alpha = model.get_avn(k+1, img_feat_batch, question_batch, ques_len_batch, 1.0)
                            print 'avn ', k+1, ': ', ['%.4f' % i for i in alpha[0].tolist()], '\n'
                            avn_list.append( alpha[0] )  
                        
                            alpha = model.get_aut(k+1, img_feat_batch, question_batch, ques_len_batch, 1.0)
                            print 'aut ', k+1, ': ', ['%.4f' % i for i in alpha[0].tolist()], '\n'
                            aut_list.append( alpha[0] )   
                        
                        print "Epoch: ", ep, " Iteration: ", itr, " Loss: ", batch_loss, ", Time Cost:", round(tStop - tStart,2), "s\n" 
                        
                        # eval
                        predicted = model.predict_answers( img_feat_batch, question_batch, ques_len_batch, 1.0)
                        loader.print_train_example( img_idx_batch[0], question_batch[0], predicted[0] )
                        loader.save_train_example( ques_id_batch[0], img_idx_batch[0], avn_list, question_batch[0], aut_list, predicted[0] )
                        print 'batch_accuracy: ', loader.print_accuracy( ans_batch, predicted ), '\n'
                         
                        tStart = time.time() 
                    # end if
                    
                    global_step = global_step + 1
                    
                # end itr loop
                print "Epoch ", ep, " is done. Saving model ..."
                #if (ep+1)%5==0: 
                #  saver.save(sess, CHECKPOINT_PATH + '%d.ckpt'%ep )
                model.save_npy()
                
                # adjust learning rate 
                lr_step = 0
                for b in LEARNING_RATE_BOUNDARY:
                    if ep>=b: lr_step += 1
                learning_rate = LEARNING_RATE_STEP[lr_step] 
                
            tStop_total = time.time()
            print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"
 
def test( total_k_step, is_mc ):
  with tf.Graph().as_default():
    loader = data_loader.data_loader( 
                    batch_size=BATCH_SIZE, 
                    dataset_path=PREPRO_DATASET_JSON, 
                    qa_data_path=PREPRO_QA_H5, 
                    img_tfrecord_size=IMG_TFRECORD_SIZE, 
                    max_words_q=MAX_WORDS_Q, 
                    dim_img=DIM_IMG,
                    is_training=False )
    
    num_test = loader.get_test_length() 
    vocab_size = loader.get_vocabulary_size() 
    answer_vocab_size = loader.get_answer_vocab_size() 
    fname = 'eval_result/MultipleChoice_mscoco_val2014_dan_results.json' if is_mc else 'eval_result/OpenEnded_mscoco_val2014_dan_results.json' 
    print 'total questions: ' + str(num_test)
    print 'vocabulary_size : ' + str(vocab_size)
    print 'file to write: ', fname
    
    with tf.Session() as sess:  
        print 'constructing model...'
        model = DAN.Dual_Attention_Network( 
                scope='dan',
                sess=sess, 
                batch_size=BATCH_SIZE,   
                k_step=total_k_step,   
                dim_img=DIM_IMG, 
                max_words_q=MAX_WORDS_Q, 
                vocab_size=vocab_size, 
                vocab_embed_size=VOCAB_EMBED_SIZE, 
                rnn_size=RNN_SIZE, 
                hidden_size=HIDDEN_SIZE,
                ans_num=answer_vocab_size,
                dan_npy_path=DAN_VAR_PATH)
        print 'model built'
    
        print 'initialize global variables..'
        sess.run(tf.global_variables_initializer())  
        tf.train.start_queue_runners(sess=sess)
        
        qa_dict = {}
        result = []
        tStart_total = time.time()
        
        for itr in range(0, num_test, BATCH_SIZE):    
            idx_batch, ques_id_batch, question_batch, ques_len_batch, img_idx_batch, img_feat_batch, MC_ans_batch  = loader.get_next_test_batch(sess)  
            
            predicted = model.predict_answers( img_feat_batch, question_batch, ques_len_batch, 1.0) 
            #loader.print_test_example( img_idx_batch[0], question_batch[0], predicted[0] ) 
             
            if is_mc is False:
                # for open-ended
                top_ans = np.argmax(predicted, axis=1) 
            else:
                # for multiple-choice  
                for b in range(BATCH_SIZE): 
                    predicted[b, MC_ans_batch[b]]  += 1 
                top_ans = np.argmax(predicted, axis=1)  
                 
            for i in range(0,BATCH_SIZE): 
                ans = loader.get_answer( top_ans[i] ) 
                if int(ques_id_batch[i]) not in qa_dict:
                    qa_dict[int(ques_id_batch[i])] = ans  
            
            if itr % 200*BATCH_SIZE==0:
            
                print "Testing batch: ", itr  
                avn_list = []
                aut_list = []
                for k in range(total_k_step):
                    alpha = model.get_avn(k+1, img_feat_batch, question_batch, ques_len_batch, 1.0)
                    avn_list.append( alpha[0] ) 
                    alpha = model.get_aut(k+1, img_feat_batch, question_batch, ques_len_batch, 1.0)
                    aut_list.append( alpha[0] )  
                loader.save_test_example( ques_id_batch[0], img_idx_batch[0], avn_list, question_batch[0], aut_list, predicted[0] )
                
        print "Testing done." 
        tStop_total = time.time()
        print "Total Time Cost:", round(tStop_total - tStart_total,2), "s" 
        
        print 'Saving result...' 
        for key in qa_dict.keys():
            result.append({ u'question_id': key, u'answer': qa_dict.pop(key) })
        print "total qa pairs: ", len(result)
        fname = 'eval_result/MultipleChoice_mscoco_val2014_dan_results.json' if is_mc else 'eval_result/OpenEnded_mscoco_val2014_dan_results.json' 
        print 'wirte %s'% fname
        json.dump(result,open(fname,'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=2, type=int, help='k-step attention, default is 2')
    parser.add_argument('--train', default='True', type=str, help='--train=True, or --train=False, default is True')
    parser.add_argument('--mc', default='False', type=str, help='--mc=True, or --mc=False, default is False (open-ended)')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    
    if tf.gfile.Exists(SUMMARY_PATH):
        tf.gfile.DeleteRecursively(SUMMARY_PATH)
        print 'deleting folder ', SUMMARY_PATH
    tf.gfile.MakeDirs(SUMMARY_PATH)  
    
    if params["train"].lower() in ['true', '1', 'y', 'yes']: 
        train( params["k"] )
    else: 
        test( params["k"], params["mc"].lower() in ['true', '1', 'y', 'yes'] )