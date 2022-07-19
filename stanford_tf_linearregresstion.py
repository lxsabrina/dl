import numpy as np
import tensorflow as tf
import matplotlib as plt

tf.__version__      #2.4.0
def  generate_dataset():
    # y = 2x +e ,where e is sampled from a normal distribution
    x_batch = np.linspace( -1,1, 101)
    y_batch = 2 * x_batch +np.random.randn(*x_batch.shape) * 0.3    # *x_batch.shape 表示和x_batch相同size
    return x_batch, y_batch     #返回结构是 numpy array


def linear_regression(): #定义 placeholder， 定义variable，定义ops，定义prediction，定义loss
    x = tf.placeholder(tf.float32, shape=(None, ), name ="x")   #2.4版本的tensorflow没有placeholder，需要修改
    y = tf.placeholder(tf.float32, shape=(None, ), name ="y")

    with tf.variable_scope("linear_regression") as scope:
        weights = tf.get_variable(np.random.normal(), name="weights")
        y_predict = tf.mul(weights, x)

        loss = tf.reduce_mean(tf.square(y_predict -y))  #机器学习的任务就是最小化损失函数，loss是一个ops
    return x, y, y_predict, loss

def run():
    x_batch, y_batch = generate_dataset()

    x,y, y_predict, loss = linear_regression()  #bug here

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #在模型训练的时候，添加一个操作API（GradientDescentOptimizer）这个对象

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        feed_dict = {x:x_batch, y:y_batch }     #session.run自动将numpy array数据结构转化成tensor
        #训练
        for i in range(30):
            loss_val, _ = session.run([loss, optimizer], feed_dict)  #loss_value对应loss,_ 对应optimizer，可以理解为用optimizer去优化loss，并且返回每一次迭代loss数值。
            print("loss:", loss_val.mean())
        
        #推理
        y_predict_batch = session.run(y_predict, {x:x_batch})  # fetches: y_predict
    
    plt.figure(1)
    plt.scatter(x_batch,y_batch)
    plt.plot(x_batch,y_predict_batch)
    plt.savefig("plot.png")

if __name__ == "__main__":
    run()
