
import tensorflow as tf
import csv

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(1234)

### define layers ###
tf.reset_default_graph()
z_dim = 10

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

def encoder(x):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        h1 = tf.layers.Dense(units=200, activation=tf.nn.relu)(x)
        h2 = tf.layers.Dense(units=200, activation=tf.nn.relu)(h1)
        mean = tf.layers.Dense(units=z_dim)(h2)
        var = tf.layers.Dense(units=z_dim, activation=tf.nn.softplus)(h2)
    return mean, var

def sampling_z(mean, var):
    epsilon = tf.random_normal(shape=tf.shape(mean))
    z = mean + tf.sqrt(var) * epsilon
    return z

def decoder(z):
    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
        h3 = tf.layers.Dense(units=200, activation=tf.nn.relu)(z)
        h4 = tf.layers.Dense(units=200, activation=tf.nn.relu)(h3)
        y = tf.layers.Dense(units=784, activation=tf.nn.sigmoid)(h4)
    return y

def lower_bound(x):
    #Encode
    mean, var = encoder(x)
    KL = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + tf_log(var) - mean**2 - var, axis=1))
    
    #Z
    z = sampling_z(mean, var)
    
    #Decode
    y = decoder(z)
    reconstruction = tf.reduce_mean(tf.reduce_sum(x * tf_log(y) + (1 - x) * tf_log(1 - y), axis=1))
    
    lower_bound = [-KL, reconstruction]
    
    return lower_bound


### training ###
#学習データと検証データに分割
x_train, x_valid = train_test_split(x_train, test_size=0.1)

x = tf.placeholder(tf.float32, [None, 784])
lower_bound = lower_bound(x)

cost = -tf.reduce_sum(lower_bound)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(cost)

valid = tf.reduce_sum(lower_bound)

batch_size =100

n_batches = x_train.shape[0] // batch_size
n_epochs = 15

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(n_epochs):
    rng.shuffle(x_train)
    lower_bound_all = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        _, lowerbound = sess.run([train, lower_bound], feed_dict={x: x_train[start:end]})
        lower_bound_all.append(lowerbound)
    lower_bound_all = np.mean(lower_bound_all, axis=0)
    lower_bound_valid = sess.run(valid, feed_dict={x: x_valid[0:100]})
    print('EPOCH:%d, Train Lower Bound:%lf, (%lf, %lf), Valid Lower Bound:%lf' %
          (epoch+1, np.sum(lower_bound_all), lower_bound_all[0], lower_bound_all[1], lower_bound_valid))
    
    
### sampling ###
x = tf.placeholder(tf.float32, [None, 784])
sample_z_func = encoder(x)

z = tf.placeholder(tf.float32, [None, z_dim])
sample_x_func = decoder(z)

# Encode
mean, var = sess.run(sample_z_func, feed_dict={x: x_test})
sample_z = mean

# Decode
sample_x = sess.run(sample_x_func, feed_dict={z: sample_z})


### to_csv ###
with open('submission.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(sample_x.reshape(-1, 28*28).tolist())