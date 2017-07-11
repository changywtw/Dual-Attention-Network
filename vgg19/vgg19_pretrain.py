import tensorflow as tf 
import numpy as np 

VGG_MEAN = [103.939, 116.779, 123.68] # bgr 

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, conv_trainable=True, fc_trainable=True, mean=[103.939, 116.779, 123.68]): #bgr
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
            print("vgg npy file loaded")
        else:
            self.data_dict = None

        self.var_dict = {}
        self.mean = mean
        self.conv_trainable = conv_trainable 
        self.fc_trainable = fc_trainable 

    def build(self, rgb): 
        """
        load variable from npy to build the VGG 
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """  
        rgb_scaled = rgb * 255.0
        
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled) 
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[ blue - self.mean[0], green - self.mean[1], red - self.mean[2] ])
         
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3') 
        
        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4') 

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")  
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        print 'self.pool5.shape:', self.pool5.shape
        
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6") # edited: 2048 = 2*2*512 
        self.relu6 = tf.nn.relu(self.fc6) 

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")  
        self.relu7 = tf.nn.relu(self.fc7) 
            
        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8") # edited 
 
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        
        return self.pool5, self.prob
    
    def batch_norm( self, x ):
        return tf.contrib.layers.batch_norm( inputs=x, trainable=True, updates_collections=None, decay=0.99, is_training=True )
      
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu( bias )
            #relu = tf.nn.relu( self.batch_norm(bias) )

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name): 
        initial_value = tf.random_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.03)
        filters = self.get_var(initial_value, name, 0, name + "_filters", 'conv')
 
        initial_value = tf.random_normal([out_channels], stddev=0.03)
        biases = self.get_var(initial_value, name, 1, name + "_biases", 'conv')

        return filters, biases

    def get_fc_var(self, in_size, out_size, name): 
        initial_value = tf.random_normal([in_size, out_size], stddev=0.01)
        weights = self.get_var(initial_value, name, 0, name + "_weights", 'fc')
 
        initial_value = tf.random_normal([out_size], stddev=0.01)
        biases = self.get_var(initial_value, name, 1, name + "_biases", 'fc')

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, layer):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            #print ('using pretrain var ' + name + " " + str(idx) )
        else:
            value = initial_value
            #print ('using randon init var ' + var_name )

        if layer=='conv':
          if self.conv_trainable:
            var = tf.Variable(value, name=var_name)
          else:
            var = tf.constant(value, dtype=tf.float32, name=var_name) 
        elif layer=='fc':
          if  self.fc_trainable:
            var = tf.Variable(value, name=var_name)
          else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
     
        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path 
