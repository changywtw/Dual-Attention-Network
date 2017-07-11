import numpy as np
import tensorflow as tf 
from vgg19 import vgg19_pretrain as vgg19_train 
from vgg19 import utils 
import codecs, json  
import os, h5py, sys 

json_path =  './data/prepro/data_prepro.json'
input_ques_h5 = './data/prepro/data_prepro.h5'

dir_path = '../VQA_dataset_V1/images/'
tfrecord_save_path = './data/prepro_img_feat_%s/%s_%d_%d.tfrecord'
tfrecord_size = 20000
#h5_train_save_filename = './data/prepro_sorted/sorted_training_data.h5'
#h5_test_save_filename  = './data/prepro_sorted/sorted_testing_data.h5'

def load_data():

    dataset = {} 
    qa_data = {}
    
    with open(json_path) as data_file:
        dataset = json.load(data_file) 
     
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 225202 
        qa_data['question_id_train'] = np.array(hf.get('question_id_train')) 
        qa_data['ques_train'] = np.array(hf.get('ques_train')) # question is (225202, 26) 
        qa_data['ques_length_train'] = np.array(hf.get('ques_length_train')) 
        qa_data['img_pos_train'] = np.array(hf.get('img_pos_train'))  
        qa_data['answers'] = np.array(hf.get('answers')) # answer is 1~2000  
        
        qa_data['question_id_test'] = np.array(hf.get('question_id_test')) 
        qa_data['ques_test']        = np.array(hf.get('ques_test'))  
        qa_data['ques_length_test'] = np.array(hf.get('ques_length_test')) 
        qa_data['img_pos_test']     = np.array(hf.get('img_pos_test')) 
        qa_data['MC_ans_test']      = np.array(hf.get('MC_ans_test'))   
        
    return dataset, qa_data    
    
def write_h5_file_train( dataset, qa_data, write_filename ):
    
    sorted_images = [] # the elements are indexes of img_list
    sorted_questions = []
    sorted_length_q = []
    sorted_qids = []
    sorted_answers = []
     
    # order qa_data by unique_img_train
    for i in range( len(dataset['unique_img_train']) ): 
        img_idx = np.where(qa_data['img_pos_train'] == i)[0] # at most 3 elements 
        for idx in img_idx:
            sorted_images.append(i)
            sorted_questions.append( qa_data['ques_train'][idx] )
            sorted_length_q.append( qa_data['ques_length_train'][idx] )
            sorted_qids.append( qa_data['question_id_train'][idx] )
            sorted_answers.append( qa_data['answers'][idx] ) 
    
    print 'sorted_images len: ', len(sorted_images)
    print 'sorted_questions len: ', len(sorted_questions)
    print 'sorted_length_q len: ', len(sorted_length_q)
    print 'sorted_qids len: ', len(sorted_qids)
    print 'sorted_answers len: ', len(sorted_answers)
    
    # write h5 file
    # training 
    f = h5py.File(write_filename, "w") 
    f.create_dataset("ques_train",        dtype='uint32', data=np.array(sorted_questions) )  
    f.create_dataset("ques_length_train", dtype='uint32', data=np.array(sorted_length_q) )
    f.create_dataset("question_id_train", dtype='uint32', data=np.array(sorted_qids) )
    f.create_dataset("answers",           dtype='uint32', data=np.array(sorted_answers) )
    f.create_dataset("img_pos_train",     dtype='uint32', data=np.array(sorted_images))  
    f.close() 
    print 'wrote ', write_filename
    
def write_h5_file_test( dataset, qa_data, write_filename ):
    
    sorted_images_test = [] # the elements are indexes of img_list
    sorted_ques_test = []
    sorted_length_q_test = []
    sorted_qids_test = []
    sorted_MC_ans_test = []
     
    # order qa_data by unique_img_train
    for i in range( len(dataset['unique_img_test']) ): 
        img_idx = np.where(qa_data['img_pos_test'] == i)[0] # at most 3 elements 
        for idx in img_idx:
            sorted_images_test.append(i)
            sorted_ques_test.append( qa_data['ques_test'][idx] )
            sorted_length_q_test.append( qa_data['ques_length_test'][idx] )
            sorted_qids_test.append( qa_data['question_id_test'][idx] )
            sorted_MC_ans_test.append( qa_data['MC_ans_test'][idx] ) 
    
    # write h5 file
    # testing 
    f = h5py.File(write_filename, "w") 
    f.create_dataset("ques_test",        dtype='uint32', data=np.array(sorted_ques_test))
    f.create_dataset("ques_length_test", dtype='uint32', data=np.array(sorted_length_q_test))
    f.create_dataset("question_id_test", dtype='uint32', data=np.array(sorted_qids_test))
    f.create_dataset("MC_ans_test",      dtype='uint32', data=np.array(sorted_MC_ans_test))
    f.create_dataset("img_pos_test",     dtype='uint32', data=np.array(sorted_images_test))
    f.close() 
    
    print 'wrote ', write_filename


def get_answer(dataset, ans):
    #print ans
    return dataset['ix_to_ans'][str(ans)] 

def get_question(dataset, q):
    #print q
    q_str = ''
    for i in range(len(q)):
        q_str = q_str+ dataset['ix_to_word'][str(q[i])]+" "
    return q_str

def get_train_image_path(dataset, img_idx):
    return dir_path + dataset['unique_img_train'][img_idx]
    
def get_test_image_path(dataset, img_idx):
    return dir_path + dataset['unique_img_test'][img_idx]

#def print_example(dataset, v, q, a): 
#    print 'image: ',  get_image_path(dataset, v)
#    print 'question: ',  get_question(dataset, q)
#    #print 'answer: ', get_prob(a), '\n' 
    
    
def process_img_feat(dataset, qa_data, filename_prefix):
    # process vgg19
    images = tf.placeholder("float", [None, 224, 224, 3])  
    vgg = vgg19_train.Vgg19(vgg19_npy_path='./vgg19/vgg19.npy', conv_trainable=False, fc_trainable=False, mean=[103.939, 116.779, 123.68]) #bgr   
    pool5, _ = vgg.build(images) 
      
    with tf.Session() as sess:  
        if filename_prefix=='train':
            for i in range(0, len(qa_data['ques_train'])): 
                # decide which file to write
                if i%tfrecord_size==0: 
                    if i+tfrecord_size<len(qa_data['ques_train']):
                        file_name = tfrecord_save_path %(filename_prefix, filename_prefix, i, i+tfrecord_size) 
                    else:
                        file_name = tfrecord_save_path %(filename_prefix, filename_prefix, i, len(qa_data['ques_train']))  
                    writer_train = tf.python_io.TFRecordWriter( file_name ) 
                
                # prepare data for example
                ques_id  = qa_data['question_id_train'][i]
                question = qa_data['ques_train'][i]
                ques_len = qa_data['ques_length_train'][i]
                img_idx  = qa_data['img_pos_train'][i]
                answer   = qa_data['answers'][i]
                 
                img = utils.load_image( dir_path+dataset['unique_img_train'][img_idx] )
                imgr = img.reshape((1, 224, 224, 3)) 
                pool5_out = sess.run(pool5, feed_dict={images: imgr}) #(batch_size,7,7,512)
                example = tf.train.Example(
                    features = tf.train.Features(  
                        feature = { 
                            'i'      : tf.train.Feature(int64_list = tf.train.Int64List( value=[i])),
                            'q_id'   : tf.train.Feature(int64_list = tf.train.Int64List( value=[ques_id])),
                            'ques'   : tf.train.Feature(bytes_list = tf.train.BytesList( value=[question.tostring()])),
                            'q_len'  : tf.train.Feature(int64_list = tf.train.Int64List( value=[ques_len])),
                            'img_idx': tf.train.Feature(int64_list = tf.train.Int64List( value=[img_idx])),
                            'feats'  : tf.train.Feature(bytes_list = tf.train.BytesList( value=[pool5_out.tostring()])),
                            'ans'    : tf.train.Feature(int64_list = tf.train.Int64List( value=[answer]))}
                    )
                ) 
                #print 'writing record %d' % i 
                writer_train.write(example.SerializeToString())
                 
                if i % 500 == 0:
                    sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(qa_data['ques_train']), i*100.0/len(qa_data['ques_train'])) )
                    sys.stdout.flush()
            # end for
        else:
            for i in range(0, len(qa_data['ques_test'])): 
                # decide which file to write
                if i%tfrecord_size==0: 
                    if i+tfrecord_size<len(qa_data['ques_test']):
                        file_name = tfrecord_save_path %(filename_prefix, filename_prefix, i, i+tfrecord_size) 
                    else:
                        file_name = tfrecord_save_path %(filename_prefix, filename_prefix, i, len(qa_data['ques_test']))  
                    writer_train = tf.python_io.TFRecordWriter( file_name ) 
                
                # prepare data for example
                ques_id  = qa_data['question_id_test'][i]
                question = qa_data['ques_test'][i]
                ques_len = qa_data['ques_length_test'][i]
                img_idx  = qa_data['img_pos_test'][i]
                answers  = qa_data['MC_ans_test'][i]
                  
                img = utils.load_image( dir_path+dataset['unique_img_test'][img_idx] )
                imgr = img.reshape((1, 224, 224, 3)) 
                pool5_out = sess.run(pool5, feed_dict={images: imgr})  #(batch_size,7,7,512)
                example = tf.train.Example(
                    features = tf.train.Features(  
                        feature = { 
                            'i'      : tf.train.Feature(int64_list = tf.train.Int64List( value=[i])),
                            'q_id'   : tf.train.Feature(int64_list = tf.train.Int64List( value=[ques_id])),
                            'ques'   : tf.train.Feature(bytes_list = tf.train.BytesList( value=[question.tostring()])),
                            'q_len'  : tf.train.Feature(int64_list = tf.train.Int64List( value=[ques_len])),
                            'img_idx': tf.train.Feature(int64_list = tf.train.Int64List( value=[img_idx])),
                            'feats'  : tf.train.Feature(bytes_list = tf.train.BytesList( value=[pool5_out.tostring()])),
                            'MC_ans' : tf.train.Feature(bytes_list = tf.train.BytesList( value=[answers.tostring()]))}
                    )
                ) 
                #print 'writing record %d' % i 
                writer_train.write(example.SerializeToString())
                 
                if i % 500 == 0:
                    sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(qa_data['ques_test']), i*100.0/len(qa_data['ques_test'])) )
                    sys.stdout.flush()
             
        writer_train.close()  
        
def main(): 
    dataset, qa_data = load_data()
    
    print 'unique training image len: ', len(dataset['unique_img_train']) # total 82634 img, convert into 0~82633
    print 'unique testing image len: ', len(dataset['unique_img_test']) # validation: 40504
     
    #write_h5_file_train( dataset, qa_data, h5_train_save_filename ) ## deprecated
    #write_h5_file_test( dataset, qa_data, h5_test_save_filename )   ## deprecated
     
    process_img_feat( dataset, qa_data, 'train')
    process_img_feat( dataset, qa_data, 'test')
     
if __name__ == '__main__':    
    main()
    