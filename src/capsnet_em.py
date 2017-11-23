import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np

def spread_loss(predict, y,margin):
    
    y_hot = tf.one_hot(y, 10)
    mask_t = tf.equal(y_hot,1)
    mask_i = tf.equal(y_hot,0)
    a_t = tf.boolean_mask(predict, mask_t)    
    a_i = tf.boolean_mask(predict, mask_i)
    a_t = tf.reshape(a_t, [-1,1])
    a_i = tf.reshape(a_i, [-1,9])    
    
    loss = tf.square(tf.maximum(0.0, margin - (a_t - a_i)))    
    return tf.reduce_mean(loss) 
    
def cross_entropy_loss(predict, y):
    return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=predict))


def add_scaled_coordinate(vote):
    vote = tf.reshape(vote, [-1,3,3,cfg.D,cfg.E,4*4])
        
    coordinate = tf.constant((np.arange(3)+0.5)/ 3, tf.float32) 
    coordinate_y = tf.reshape(coordinate,[3,1,1])
    coordinate_x = tf.reshape(coordinate,[1,3,1])
    coord_y = tf.tile(coordinate_y,[1,3,1])
    coord_x = tf.tile(coordinate_x,[3,1,1])    
    coord_zero = tf.zeros([3,3,14])
    
    coord_xyz = tf.concat([coord_y,coord_x,coord_zero],-1)#(3, 3, 16)
    coord_reshape = tf.reshape(coord_xyz, [1,3,3,1,1,4*4])
    coord_vote = tf.tile(coord_reshape,   [1,1,1,cfg.D,cfg.E,1]) 
    vote_coord = vote + coord_vote
    vote_coord = tf.reshape(vote_coord, [-1,1,3*3,cfg.D,cfg.E,4*4])
    return vote_coord 

def tile_recpetive_field(capsules_5d, kernel, stride):
    shape = capsules_5d.get_shape()   
    capsules_4d = tf.reshape(capsules_5d, [int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])*int(shape[4])])
    input_shape = capsules_4d.get_shape()
    weight = np.zeros(shape=[kernel, kernel, input_shape[3], kernel*kernel])
    for i in range(kernel):
        for j in range(kernel):
            weight[i, j, :, i * kernel + j] = 1

    tile_weight = tf.constant(weight, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(capsules_4d, tile_weight, strides=[1, stride, stride, 1], padding='VALID')
    print (stride,'tile_weight ',tile_weight.shape)#(3, 3, 544, 9)
    print ('    tile input',capsules_4d)#(128, 12, 12, 544)
    print ('    tile output0', output)#(128, 5, 5, 4896)
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel*kernel])
    print ('    tile output1', output)
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    print ('    tile output2', output)
    output = tf.reshape(output,[-1,16+1])
    activation = output[:, 0]
    pose = output[:, 1:]
    return activation,pose 

# input should be a tensor with size as [batch_size, caps_num_i, 16]
def mat_transform(pose, caps_num_i, caps_num_c):
    
    output = tf.reshape(pose, shape=[cfg.batch_size,-1, 3*3,caps_num_i, 1, 4, 4])
    b = int(output.get_shape()[0])
    wh = int(output.get_shape()[1])
    
    w = slim.variable('w'+str(caps_num_c), shape=[1,1, 3*3, caps_num_i, caps_num_c, 4, 4], dtype=tf.float32)
    print ('    mat_transform input0',pose)
    print ('    mat_transform input1',output)
    print ('    mat_transform  w',w)
    w = tf.tile(w, [b,wh, 1, 1, 1, 1, 1])
    output = tf.tile(output, [1, 1, 1, 1, caps_num_c, 1, 1])
    print ('    mat_transform tile a',output)
    print ('    mat_transform tile b',w)
    votes = tf.matmul(output, w)
    print ('    mat_transform tile a*b',votes)
    votes = tf.reshape(votes, [b, wh, 3*3,caps_num_i, caps_num_c, 16])
    print ('    mat_transform votes ',votes )
    return votes

def build_arch(X):    
    # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
    b_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01) #tf.constant_initializer(0.0)
    w_init = tf.contrib.layers.xavier_initializer()
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    w_reg = tf.contrib.layers.l2_regularizer(1e-5)

    with slim.arg_scope([slim.conv2d], padding='VALID', weights_initializer=w_init,biases_initializer=b_init, weights_regularizer=w_reg):
        with tf.variable_scope('relu_conv1'):
            kernel = int(5)
            stride = int(2)
            height = width = int( (int(X.get_shape()[1])-int(kernel/2)*2)/stride) # 12  
            print ('relu_conv1 height, width',height,width)
            feature_map = slim.conv2d(X, num_outputs=cfg.A, kernel_size=[kernel, kernel], stride=stride)
            assert feature_map.get_shape() == [cfg.batch_size, height, width, 32]

        with tf.variable_scope('primary_caps'):
            kernel = int(1)
            stride = int(1)
            height = width = int( (int(feature_map.get_shape()[1])-int(kernel/2)*2)/stride) # 12
            print ('primary_caps height, width',height,width)  
            pose = slim.conv2d(feature_map, num_outputs=cfg.B*16, kernel_size=[kernel, kernel])
            activation = slim.conv2d(feature_map, num_outputs=cfg.B, kernel_size=[1, 1],activation_fn=tf.nn.sigmoid)            
            capsules_4d = tf.concat([activation, pose],-1)
            capsules = tf.reshape(capsules_4d, [-1,height, width,cfg.B,16+1]) 
            assert capsules.get_shape() == [cfg.batch_size, height, width, cfg.B,16+1]

        with tf.variable_scope('conv_caps1'):
            kernel = int(3)
            stride = int(2)
            height = width = int( (int(capsules.get_shape()[1])-int(kernel/2)*2)/stride) # 5
            print ('conv_caps1 height, width',height,width)
            activation, pose = tile_recpetive_field(capsules, kernel, stride)
            votes = mat_transform(pose, cfg.B,cfg.C)                        
            capsules,check = em_routing(votes, activation, kernel*kernel, cfg.B, cfg.C)
            capsules = tf.reshape(capsules, [-1, height ,width, cfg.C,16+1])
            assert capsules.get_shape() == [cfg.batch_size, height, width, cfg.C,16+1]

        with tf.variable_scope('conv_caps2'):
            kernel = int(3)
            stride = int(1)
            height = width = int( (int(capsules.get_shape()[1])-int(kernel/2)*2)/stride) # 3
            print ('conv_caps2 height, width',height,width)
            activation, pose = tile_recpetive_field(capsules, kernel, stride)
            votes = mat_transform(pose, cfg.C, cfg.D)                     
            capsules,check = em_routing(votes, activation, kernel*kernel, cfg.C, cfg.D)
            capsules = tf.reshape(capsules, [-1, height ,width, cfg.D,16+1])
            assert capsules.get_shape() == [cfg.batch_size, height,width, cfg.D,16+1]

        with tf.variable_scope('class_caps'):
            kernel = int(3)
            stride = int(1)
            height = width = int((int(capsules.get_shape()[1])-int(kernel/2)*2)/stride) # 1
            print ('class_caps height, width',height,width)
            activation,pose = tile_recpetive_field(capsules, kernel, 1)                   
            votes = mat_transform(pose, cfg.D, cfg.E)            
            print (' final votes',votes )            
            votes = add_scaled_coordinate(votes)             
            capsules,check = em_routing(votes, activation, kernel*kernel, cfg.D,cfg.E)
            capsules = tf.reshape(capsules, [-1, height,width,cfg.E,16+1])
            assert capsules.get_shape() == [cfg.batch_size, height,width, cfg.E,16+1]
        
        final_activation = tf.reshape(capsules[:,:,:,:,0], shape=[-1, 10])

    return final_activation, check

def m_step(temp_lambda,r,activation,votes,caps_num_c):
    print ('#temp_lambda',temp_lambda)
    print ('routing r',caps_num_c, r)#(100, 1, 9, 8, 10, 1)
    print ('routing a',caps_num_c, activation) 
     
    ra = r * activation#(10, 25, 9, 8, 1)
    print ('routing ra',ra)#
    
    rv_sum = tf.reduce_mean(ra * votes, [2,3], keep_dims=True)  
    ra_sum = tf.reduce_mean(ra, [2,3], keep_dims=True)
    print ('rv_sum',rv_sum)
    print ('ra_sum',ra_sum)#(100, 1, 1, 1, 10, 1)
    ra_sum = tf.clip_by_value(ra_sum, cfg.clip_min, cfg.clip_max)
    mean = rv_sum / ra_sum    
    print ('mean',mean)
                                 
    v_minus_mu = ra * tf.square(votes - mean)
    print ('v_minus_mu',v_minus_mu)#(10, 25, 9, 8, 8, 16)

    sigma_square = tf.reduce_mean(v_minus_mu,[2,3],keep_dims=True)/ ra_sum
    sigma = tf.sqrt(sigma_square)    
    print ('sigma_square',sigma_square)    
    
    beta_v = tf.Variable(tf.zeros([1,1,1,1,caps_num_c,16]))    
    beta_a = tf.Variable(tf.zeros([1,1,1,1,caps_num_c,1]))    
    
    cost_h = (beta_v + tf.log(sigma)) * ra_sum
    print ('cost_h',cost_h)
    activation = tf.nn.sigmoid(temp_lambda * (beta_a - tf.reduce_mean(cost_h, axis=-1,keep_dims=True)))
    #activation = tf.nn.relu(beta_a - tf.reduce_sum(cost_h, axis=-1,keep_dims=True))
    return mean, sigma, activation

def e_step(activation,sigma,mean,votes):    
    sigma_square = tf.square(sigma)
    norm = tf.square(votes-mean)/(2*sigma_square)
    norm_sum = -tf.reduce_sum(norm,-1, keep_dims=True)
    prod = tf.reduce_prod(2*np.pi*sigma_square,-1,keep_dims=True)
    denominator = tf.sqrt(prod+1e-8) 
    print ('* e_step norm',norm)#(100, 1, 9, 8, 10, 16)
    
    denominator = tf.clip_by_value(denominator, cfg.clip_min, cfg.clip_max)
    p = tf.exp(norm_sum)/denominator

    ap = activation * p
    print ('* e_step ap',ap)#(100, 1, 9, 8, 10, 1)
    ap_sum = tf.reduce_sum(ap, axis=-2, keep_dims=True)
    ap_sum = tf.clip_by_value(ap_sum, cfg.clip_min, cfg.clip_max)    
    r = ap / ap_sum
    return r,norm

def em_routing(votes, activation, kxk, caps_num_i,caps_num_c):
    #(20, 1, 9, 32, 10, 16)
    b = cfg.batch_size
    print ('routing votes in',caps_num_i,caps_num_c,votes)
    votes = tf.reshape(votes, [b ,-1, kxk,caps_num_i,caps_num_c,16])    
    activation = tf.reshape(activation, [b ,-1,kxk, caps_num_i,1,1])
    print ('routing votes',votes)
    print ('routing a',caps_num_c, activation)
        
    r = tf.ones([b, 1, kxk, caps_num_i, caps_num_c,1], dtype=np.float32) / caps_num_c    
    print ('routing r',caps_num_c, r)
    temperature_lambda = 0.33 # temp
    for i in range(cfg.iter_routing):
        print ('\n routing',i)    
        mean, sigma, activation = m_step(temperature_lambda, r,activation,votes,caps_num_c)
        r,check = e_step(activation,sigma,mean,votes)
        temperature_lambda+=0.33 # temp              
    
    capsules = tf.concat([activation,mean], axis=-1)    
    return capsules, r

