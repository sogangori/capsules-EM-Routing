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

modelName = './weights/mnist_s.pd'
isPadding = not True
isNewTrain = not True

def main(_):      
    height = width = 28
    if isPadding: height = width = 40
    
    train_x, train_y, test_x,test_y = get_mnist()
    
    X = tf.placeholder(tf.float32, [cfg.batch_size, height, width,1])
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
        saver.restore(sess, modelName)
        print("Model restored")
    
    margin = 0.5
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
            batch_x = train_x[start:end]
            if isPadding:batch_x = padding(batch_x, 40)                
            dic = {X:batch_x,Y:train_y[start:end],M:margin}
            _, loss_,acc,check_ = sess.run([train_op, loss,accuracy,check],dic)
            
            acc_sum_train+=acc/iteration_train
            now = strftime("%H:%M:%S", localtime())
            if i%int(iteration_train/2)==0:         
                print('%d/%d %d/%d batch acc:%.3f, loss:%.5f margin:%.2f sec:%.2f check:%.3f' 
                      % (step,cfg.epoch,i,iteration_train, acc,loss_, margin,time.time()-tic,np.std(check_)))

            assert not np.isnan(loss_) 
        
        for i in range(iteration_test):
            start = i * num_batches
            end = start + num_batches   
            batch_x = test_x[start:end]             
            if isPadding:batch_x = padding(batch_x, 40)
            acc_te = sess.run(accuracy, {X:batch_x,Y:test_y[start:end],M:margin})
            acc_sum_test+=acc_te/iteration_test
            
        print ('%d/%d train:%.3f  test:%.3f' %(step, cfg.epoch, acc_sum_train, acc_sum_test))
                    
        this_sec = time.time()
        if margin < 0.9 and margin < acc: margin = np.minimum(margin+0.01, 0.9)
        if acc_sum_train>=1 or step==cfg.epoch or this_sec - start_sec > 60 * 5 :
            start_sec = this_sec
            save_path = saver.save(sess, modelName)            
            print("Model Saved, time:%s, %s" %(now, save_path))                
            if acc_sum_train>=1: break   
        
    print ('training finish',acc_sum_train,acc_sum_test)

def padding(x, dstHeight=40):    
    bz = x.shape[0]
    if np.ndim(x)==2:
        wh = x.shape[1]
        w = h = (int)(np.sqrt(wh))
    else:
        w = h = x.shape[1]

    x = np.reshape(x, [-1,h,w])
    bg = np.zeros([bz,dstHeight,dstHeight])
    max_offset = dstHeight-h
    offsets = np.random.randint(0,max_offset,2)
    #offsets = [6,6]
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w] = x
    bg = np.expand_dims(bg,-1)
    return bg
            
if __name__ == "__main__":
    np.set_printoptions(2)
    tf.app.run()