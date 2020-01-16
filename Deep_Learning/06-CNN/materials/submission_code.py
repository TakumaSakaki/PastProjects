
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
random_state = 42

### レイヤー定義 ###

class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='SAME'):
        # Heの初期値
        # filter_shape: (縦の次元数)x(横の次元数)x(入力チャンネル数)x(出力チャンネル数)
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low= - np.sqrt(6 / fan_in),
                        high= np.sqrt(6 / fan_in),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力チャンネル数と同じ次元数
        self.function = function
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding = self.padding) + self.b
        return self.function(u)

class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
    
    def __call__(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides = self.strides, padding = self.padding)
    
    
class Flatten:
    def __call__(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    
    
class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # He Initialization
        # in_dim: 入力の次元数、out_dim: 出力の次元数
        self.W = tf.Variable(rng.uniform(
                        low= - np.sqrt(6 / in_dim),
                        high= np.sqrt(6 / in_dim),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def __call__(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

class Dropout:
    def __init__(self, dropout_keep_prob=1.0):
        self.dropout_keep_prob = dropout_keep_prob
        self.params = []
    
    def __call__(self, x):
        # 訓練時のみdropoutを適用
        return tf.cond(
            pred=is_training,
            true_fn=lambda: tf.nn.dropout(x, keep_prob=self.dropout_keep_prob),
            false_fn=lambda: x
        )
    
def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))
    
### ネットワーク ###

x_train, x_test, t_train = load_mnist()
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.1, random_state=random_state)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

h = Conv((7, 7, 1, 50), tf.nn.relu)(x) # 28*28*1 -> 28*28*20
h = Conv((3, 3, 50, 20), tf.nn.relu)(h) 
h = Conv((6, 6, 20, 20), tf.nn.relu)(h)
h = Conv((6, 6, 20, 10), tf.nn.relu)(h) #28*28*20 -> 28*28*10
h = Pooling((1, 2, 2, 1))(h)            # 28*28*10 -> 14*14*10
h = Conv((5, 5, 10, 50), tf.nn.relu, padding = 'VALID')(h) # 14*14*10 -> 10*10*50
h = Pooling((1, 2, 2, 1))(h)                           #10*10*50 -> 5*5*50
h = Flatten()(h)
h = Dense(5*5*50, 500, tf.nn.relu)(h)
h = Dropout(0.5)(h)
h = Dense(500, 100, tf.nn.relu)(h)
h = Dropout(0.5)(h)
y = Dense(100, 10, tf.nn.softmax)(h)

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

### 学習 ###
n_epochs = 100
batch_size = 50
n_batches = x_train.shape[0]//batch_size

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        x_train, t_train = shuffle(x_train, t_train, random_state=random_state)
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: x_train[start:end], t: t_train[start:end], is_training: True})
        y_pred, cost_valid = sess.run([y, cost], feed_dict={x: x_valid, t: t_valid, is_training: False})
        print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
            epoch,
            cost_valid,
            accuracy_score(t_valid.argmax(axis=1), y_pred.argmax(axis=1))
        ))
### 出力 ###
    y_pred = sess.run(y, feed_dict={x: x_test, is_training: False})
    y_pred = np.argmax(y_pred, axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission_pred.csv', header=True, index_label='id')