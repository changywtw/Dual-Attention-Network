import tensorflow as tf
import numpy as np
 
WEIGHT_DECAY = 0.0005 
 
class Dual_Attention_Network():
    def __init__(self, scope, sess, batch_size, k_step, dim_img, max_words_q, vocab_size, vocab_embed_size, rnn_size, hidden_size, ans_num, dan_npy_path):
        self.scope = scope
        self.sess = sess
        self.batch_size = batch_size 
        self.k_step = k_step 
        self.dim_img = dim_img #[7,7,512] 
        self.max_words_q = max_words_q
        self.vocab_size = vocab_size
        self.vocab_embed_size = vocab_embed_size
        self.rnn_size = rnn_size 
        self.hidden_size = hidden_size 
        self.memory_size = self.rnn_size*2
        self.ans_num = ans_num
        self.var_dict = {} 
        self.memories = {}
        self.avn = {}
        self.aut = {} 
        if dan_npy_path is not None: # load pretrained weights
            self.data_dict = np.load(dan_npy_path, encoding='latin1').item()
            print "dan npy file loaded: ", dan_npy_path 
        else:
            self.data_dict = None
            print "using random init variables" 
            
        self.print_loading_info = False # print loading info, set False to disable it
        
        
        with tf.variable_scope(self.scope):
            # question-embedding 
            self.embed_ques_W = self.embedding_layer( self.vocab_size, self.vocab_embed_size, name='embed_ques_W')
            
            # compute visual features, textual features, init_memory
            self.input_images, self.input_questions, self.input_ques_lens, self.keep_prob, self.last_memory = self.build_network( ) 
            
            # compute ans 
            # memory: [bs, memory_size]
            self.ans_out = self.fc_layer(self.last_memory, self.memory_size, self.ans_num, "Wans" ) # Wans: (bs, 1000) <= (bs, 1024) * (1024, 1000)
            
            self.predicted_ans = tf.nn.softmax( self.ans_out )
            
            # compute loss  
            self.labels = tf.placeholder( tf.int64, [self.batch_size, ])  
            
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.ans_out) 
            self.loss = tf.reduce_mean(self.cross_entropy)
            
            # weight decay
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.total_loss = self.loss + l2*WEIGHT_DECAY
            
        #summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('total_loss', self.total_loss)
        
        #optimize 
        self.learning_rate = tf.placeholder(tf.float32)
        tf.summary.scalar('learning_rate', self.learning_rate)
        
        self.tvars = tf.trainable_variables() 
        self.opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)   
        self.gvs = self.opt.compute_gradients(self.total_loss, self.tvars) 
        self.clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in self.gvs]  
        self.train_op = self.opt.apply_gradients(self.clipped_gvs) 
            
        #summary
        #tf.summary.image('img_feature', tf.reshape(self.input_images[:,:,:,0],[-1,7,7,1]) )
        #for var in self.tvars:
        #    tf.summary.histogram(var.op.name, var)
        #for grad, var in self.gvs:
        #    if grad is not None: tf.summary.histogram(var.op.name + '/gradients', grad)
        
        self.merged = tf.summary.merge_all() 
        
    def build_network(self): 
        images    = tf.placeholder(tf.float32, [self.batch_size, self.dim_img[0], self.dim_img[1], self.dim_img[2]])
        questions = tf.placeholder(  tf.int32, [self.batch_size, self.max_words_q])  
        ques_lens = tf.placeholder(  tf.int32, [self.batch_size])
        keep_prob = tf.placeholder(tf.float32)
        
        # compute init memory
        # vn: (bs, 49, 512)
        # ut: (bs, 26, 1024)
        # m0: (bs, 1024)
        self.vn, self.ut, self.memories[0] = self.compute_init_memory( images, questions, ques_lens, keep_prob )
         
        # compute 1st-step attention
        for step in range(1, self.k_step+1):
            self.memories[step] = self.compute_attention( step, self.vn, self.ut, self.memories[step-1], keep_prob )  
         
        return images, questions, ques_lens, keep_prob, self.memories[self.k_step]
     
    def compute_init_memory(self, images, questions, ques_lens, keep_prob):  
        # ut    
        state_fw = tf.contrib.rnn.LSTMCell(self.rnn_size, use_peepholes=False).zero_state(self.batch_size, dtype=tf.float32)   
        state_bw = tf.contrib.rnn.LSTMCell(self.rnn_size, use_peepholes=False).zero_state(self.batch_size, dtype=tf.float32) 
            
        fw = [] 
        bw = [] 
        for i in range(self.max_words_q):
            with tf.variable_scope('forward_lstm') as scope: 
                if i>0: scope.reuse_variables()   
                ques_emb_linear_fw = tf.nn.embedding_lookup(self.embed_ques_W, questions[:,i])   
                _, state_fw = self.lstm_layer(self.rnn_size, ques_emb_linear_fw, state_fw, 'forward_lstm') 
                fw.append( tf.concat([state_fw.c,state_fw.h],1) ) 
                #fw.append( state_fw.h) 
            
            with tf.variable_scope('backward_lstm' ) as scope:
                if i>0: scope.reuse_variables() 
                ques_emb_linear_bw = tf.nn.embedding_lookup(self.embed_ques_W, questions[:,self.max_words_q-1-i]) 
                _, state_bw = self.lstm_layer(self.rnn_size, ques_emb_linear_bw, state_bw, 'backward_lstm') 
                bw.insert(0, tf.concat([state_bw.c,state_bw.h],1) )  
                #bw.insert( 0, state_bw.h)
                
        hf = tf.stack(fw, axis=1) # (bs, 26, 1024)
        hb = tf.stack(bw, axis=1) # (bs, 26, 1024) 
        ut = tf.add(hf, hb)       # (bs, 26, 1024) 
        u0 = tf.reduce_mean( ut, 1) # (bs, 1024)
         
        vn = tf.reshape( images, [self.batch_size, self.dim_img[0]*self.dim_img[1], self.dim_img[2]]) # (bs, 49, 512) 
        v_mean = tf.reduce_mean( vn, 1) # (bs, 512)
        with tf.variable_scope('P') as scope: 
            pv0_linear = self.fc_layer(v_mean, self.dim_img[2], self.memory_size, "P_0")  # (bs, 1024) <= (bs, 512) * (512, 1024)
        v0 = tf.nn.dropout( tf.tanh(pv0_linear), keep_prob)
        
        # m(0)  
        init_memory = tf.multiply( u0, v0) # (bs, 1024) <= (bs, 1024) (*) (bs, 1024)
        return vn, ut, init_memory
     
    def compute_attention(self, step, vfeats, qfeats, memory_in, keep_prob):   
        # vfeats: (bs, 49, 512)
        # qfeats: (bs, 26, 1024)
        # memory_in: (bs, 1024)
          
        # compute v(1)   
        # Wvm
        with tf.variable_scope("Wvm") as scope:
            Wvm_linear = self.fc_layer(memory_in, self.memory_size, self.hidden_size, "Wvm_%d" % step ) # (bs, 512) <= (bs, 1024) * (1024, 512)
            Wvm_linear = tf.reshape( Wvm_linear, [self.batch_size, 1, self.hidden_size]) # Wvm_linear: (bs, 1, 512)
            Wvm_out = tf.nn.dropout( tf.tanh( Wvm_linear ), keep_prob) # Wvm_out: (bs, 1, 512)
        
        # Wv
        with tf.variable_scope("Wv") as scope: 
            Wv_linear = self.fc_layer( vfeats, self.dim_img[2], self.hidden_size, "Wv_%d" % step ) # (bs*49, 512) <= (bs, 49, 512) * (512, 512)
            Wv_linear = tf.reshape(Wv_linear, [self.batch_size, -1, self.hidden_size]) # Wv_linear: (bs, 49, 512) <= (bs*49, 512)
            Wv_out = tf.nn.dropout( tf.tanh(Wv_linear), keep_prob) # Wv_out: (bs, 49, 512)
        
        # hvn
        hvn = tf.multiply( Wv_out, Wvm_out) # (bs, 49, 512) = (bs, 49, 512) (*) (bs, 1, 512)
        
        # avn
        with tf.variable_scope("Wvh") as scope: 
            Wvh_linear = self.fc_layer(hvn, self.hidden_size, 1, "Wvh_%d" % step ) # (bs*49, 1) <= (bs, 49, 512) * (512, 1)
            Wvh_linear = tf.reshape(Wvh_linear, [ self.batch_size, -1]) # (bs, 49)
            self.avn[step] = tf.nn.softmax( Wvh_linear ) # (bs, 49) 
         
        # v(1) 
        weighted_vfeats = tf.reduce_sum( tf.expand_dims(self.avn[step], 2)*vfeats, 1) # (bs, 512) <= (bs, 49, 512)  <= (bs, 49, 1) (*) (bs, 49, 512)
        with tf.variable_scope("P") as scope: 
            p_linear = self.fc_layer(weighted_vfeats, self.dim_img[2], self.memory_size, "P_%d" % step ) # (bs, 1024) <= (bs, 512) * (512, 1024)
            v = tf.nn.dropout( tf.tanh( p_linear ), keep_prob) # (bs, 1024)
        
        # compute u(1) 
        # Wum
        with tf.variable_scope("Wum") as scope:
            Wum_linear = self.fc_layer(memory_in, self.memory_size, self.hidden_size, "Wum_%d" % step ) # (bs, 512) <= (bs, 1024) * (1024, 512)
            Wum_linear = tf.reshape( Wum_linear, [self.batch_size, 1, self.hidden_size]) # (bs, 1, 512)
            Wum_out = tf.nn.dropout( tf.tanh(Wum_linear), keep_prob) # (bs, 1, 512) 
        
        # Wu
        with tf.variable_scope("Wu") as scope:
            Wu_linear = self.fc_layer( qfeats , self.rnn_size*2, self.hidden_size, "Wu_%d" % step ) # (bs*26, 512) <= (bs, 26, 1024) * (1024, 512)
            Wu_linear = tf.reshape(Wu_linear, [self.batch_size, -1, self.hidden_size]) # (bs, 26, 512) <= (bs*26, 512)
            Wu_out = tf.nn.dropout( tf.tanh(Wu_linear), keep_prob) # (bs, 26, 512) 
            
        # hut 
        hut = tf.multiply( Wu_out, Wum_out) # (bs, 26, 512) <= (bs, 26, 512) (*) (bs, 1, 512) 
        
        # aut
        with tf.variable_scope("Wuh") as scope: 
            Wuh_linear = self.fc_layer(hut, self.hidden_size, 1, "Wuh_%d" % step ) # (bs*26, 1) <= (bs, 26, 512) * (512, 1)
            Wuh_linear = tf.reshape(Wuh_linear, [self.batch_size, -1]) # (bs, 26) <= (bs*26, 1)
            self.aut[step] = tf.nn.softmax(Wuh_linear) # (bs, 26)  
        
        # weighted ut 
        u = tf.reduce_sum(tf.expand_dims(self.aut[step], 2)*qfeats, 1) # (bs, 1024) <= (bs, 26, 1024) <= (bs, 26, 1) (*) (bs, 26, 1024)
         
        # m(1)
        memory_out = tf.add( memory_in, tf.multiply(v, u) ) # (bs, 1024) <= (bs, 1024) + (bs, 1024)(*)(bs, 1024)
        
        return memory_out

    def get_init_memory(self, input_images, input_questions, input_ques_lens, keep_prob):
        return self.sess.run(self.memories[0], feed_dict={
                                self.input_images: input_images, 
                                self.input_questions: input_questions, 
                                self.input_ques_lens: input_ques_lens,
                                self.keep_prob: keep_prob })
    
    def get_avn(self, step, input_images, input_questions, input_ques_lens, keep_prob):
        return self.sess.run(self.avn[step], feed_dict={
                                self.input_images: input_images, 
                                self.input_questions: input_questions, 
                                self.input_ques_lens: input_ques_lens,
                                self.keep_prob: keep_prob })
    
    def get_aut(self, step, input_images, input_questions, input_ques_lens, keep_prob):
        return self.sess.run(self.aut[step], feed_dict={
                                self.input_images: input_images, 
                                self.input_questions: input_questions, 
                                self.input_ques_lens: input_ques_lens,
                                self.keep_prob: keep_prob })
    
    def get_loss(self, input_images, input_questions, input_ques_lens, labels, keep_prob):
        # added by Hui-Po
        return self.sess.run(self.loss,
                                feed_dict={
                                    self.input_images: input_images,
                                    self.input_questions: input_questions,
                                    self.input_ques_lens: input_ques_lens,
                                    self.keep_prob: keep_prob,
                                    self.labels: labels})

    def predict_answers(self, input_images, input_questions, input_ques_lens, keep_prob): 
        # without softmax!!
        return self.sess.run(self.predicted_ans, 
                                feed_dict={ 
                                    self.input_images: input_images, 
                                    self.input_questions: input_questions, 
                                    self.input_ques_lens: input_ques_lens,
                                    self.keep_prob: keep_prob }) 
         
    def train(self, input_images, input_questions, input_ques_lens, labels, keep_prob, learning_rate): 
        return self.sess.run([self.loss, self.merged, self.train_op],
                            feed_dict={ 
                                self.input_images:input_images, 
                                self.input_questions:input_questions, 
                                self.input_ques_lens:input_ques_lens, 
                                self.labels: labels, 
                                self.keep_prob: keep_prob,
                                self.learning_rate: learning_rate
                            })
    
    def lstm_layer(self, size, input, state, name):  
        with tf.variable_scope(name) as scope: 
            if self.data_dict is not None and name in self.data_dict: 
                w = tf.get_variable(name='lstm_cell/weights', initializer=self.data_dict[name][0] ) 
                b = tf.get_variable(name='lstm_cell/biases',  initializer=self.data_dict[name][1] ) 
                scope.reuse_variables() 
                
                self.print_info('using pretrain var ' + name )
            else: 
                self.print_info('using randon init var ' + name )  
            
            lstm = tf.contrib.rnn.LSTMCell(size, initializer=None ) 
            out, new_state = lstm(input, state)
            lstm_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]   
            for idx in range(len(lstm_variables)): 
                self.var_dict[(name, idx)] = lstm_variables[idx]
            
            return out, new_state
     
    def embedding_layer(self, vocab_size, vocab_embed_size, name):  
        with tf.variable_scope(name):
            idx = 0  
            if self.data_dict is not None and name in self.data_dict:
                value = self.data_dict[name][idx] 
                self.print_info('using pretrain var ' + name + " " + str(idx) )
            else:
                value = tf.random_uniform([vocab_size, vocab_embed_size], -0.08, 0.08)
                self.print_info('using randon init var ' + name ) 
                
            weights = tf.get_variable( name=name+'_weights', initializer=value)  
            self.var_dict[(name, idx)] = weights  
             
            return weights
            
    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):  
            if self.data_dict is not None and name in self.data_dict:
                weight_value = tf.constant_initializer(self.data_dict[name][0])
                weight_value = self.data_dict[name][0] 
                biases_value = tf.constant_initializer(self.data_dict[name][1])
                biases_value = self.data_dict[name][1] 
                
                self.print_info('using pretrain var ' + name + " " + str(0) )
                self.print_info('using pretrain var ' + name + " " + str(1) )
            else:
                weight_value = tf.random_normal([in_size, out_size], stddev=0.01)
                biases_value = tf.random_normal([out_size], stddev=0.01)
                
                self.print_info('using randon init var ' + name+"_weights" ) 
                self.print_info('using randon init var ' + name+"_biases" ) 
            
            weights = tf.get_variable( name=name+"_weights", initializer=weight_value) 
            biases  = tf.get_variable( name=name+"_biases",  initializer=biases_value) 
            
            self.var_dict[(name, 0)] = weights
            self.var_dict[(name, 1)] = biases
             
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases) 
            
            return fc 
    
    def save_npy(self):
        assert isinstance(self.sess, tf.Session)
        data_dict = {}
        
        npy_path = "./%s_%d_save.npy" % (self.scope, self.k_step)
        
        for (name, idx), var in list(self.var_dict.items()): 
            var_out = self.sess.run(var) 
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print "file saved ", npy_path 
        return npy_path 
    
    # for debug
    def print_info( self, words ):
        if self.print_loading_info: print words
    
    def print_var_dict(self):
        print self.var_dict
        print len(self.var_dict)
        print self.var_dict.keys()