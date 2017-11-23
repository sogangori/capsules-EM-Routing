import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from config import cfg
import time
from time import localtime, strftime
import numpy as np
import capsnet_em as net

def get_mnist():
    mnist = input_data.read_data_sets('/mnist')
    train_x = np.reshape(mnist.train.images,[-1,28,28,1])
    train_y = np.array(mnist.train.labels).astype(np.int32)   
    test_x = np.reshape(mnist.test.images,[-1,28,28,1])
    test_y = np.array(mnist.test.labels).astype(np.int32)
    print ('train',train_x.shape,train_y.shape, np.max(train_x))
    print ('testy',test_x.shape,test_y.shape)
    return train_x, train_y,test_x,test_y

isNewTrain =  True

def main(_):      

    train_x, train_y, test_x,test_y = get_mnist()
    
    X = tf.placeholder(tf.float32, [cfg.batch_size, 28,28,1])
    Y = tf.placeholder(tf.int32, [cfg.batch_size])
    M = tf.placeholder(tf.float32, ())      

    predict,check = net.build_arch(X)
    
    predict_class = tf.cast(tf.argmax(predict,-1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_class,Y),tf.float32))
    
    loss_spread = net.spread_loss(predict, Y, M)
    loss_regular = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = loss_spread + loss_regular        
    train_op = tf.train.AdamOptimizer(0.001,0.5).minimize(loss)
    
    sess = tf.Session()        
    saver = tf.train.Saver()
    if isNewTrain: 
        sess.run(tf.global_variables_initializer())
        print('Initialized!')
    else :        
        saver.restore(sess, cfg.modelName)
        print("Model restored")
    
    margin = 0.8
    start_sec = time.time()
    iteration_train = int(np.minimum(cfg.max_count,len(train_x))/cfg.batch_size)
    iteration_test = int(np.minimum(cfg.max_count,len(test_x))/cfg.batch_size) 
    num_batches = cfg.batch_size
    for step in range(cfg.epoch+1):
        acc_sum_train = 0.0
        acc_sum_test = 0.0
        for i in range(iteration_train):
            tic = time.time()
            start = i*num_batches
            end = start+num_batches                
            dic = {X:train_x[start:end],Y:train_y[start:end],M:margin}
            _, loss_,acc,check_ = sess.run([train_op, loss,accuracy,check],dic)
            
            acc_sum_train+=acc/iteration_train
            now = strftime("%H:%M:%S", localtime())
            if i%(cfg.epoch/10)==0:         
                print('%d/%d %d/%d acc:%.3f, loss:%.5f margin:%.2f sec:%.2f check:%.3f' 
                      % (step,cfg.epoch,i,iteration_train, acc,loss_, margin,time.time()-tic,np.std(check_)))

            assert not np.isnan(loss_) 
        
        for i in range(iteration_test):
            start = i * num_batches
            end = start + num_batches                
            acc_te = sess.run(accuracy, {X:test_x[start:end],Y:test_y[start:end],M:margin})
            acc_sum_test+=acc_te/iteration_test
            
        print ('%d/%d train:%.3f  test:%.3f' %(step, cfg.epoch, acc_sum_train, acc_sum_test))
                    
        this_sec = time.time()
        if margin < 0.9 and margin < acc: margin = np.minimum(margin+0.001, 0.9)
        if acc_sum_train>=1 or i==cfg.epoch or this_sec - start_sec > 60 * 5 :
            start_sec = this_sec
            save_path = saver.save(sess, cfg.modelName)            
            print("Model Saved, time:%s, %s" %(now, save_path))                
            if acc_sum_train>=1: break   
        
    print ('training finish',acc_sum_train,acc_sum_test)
            
if __name__ == "__main__":
    np.set_printoptions(2)
    tf.app.run()