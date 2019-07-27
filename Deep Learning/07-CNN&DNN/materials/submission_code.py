
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
random_state = 42

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

### ネットワーク ###
tf.reset_default_graph()
is_training = tf.placeholder(tf.bool, shape=())
x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
t = tf.placeholder(tf.float32, [None, 10])

h = tf.layers.Conv2D(filters=64, kernel_size= [3, 3])(x) # 32x32x3 -> 30x30x32 # conv2dの初期設定: strides=(1, 1), padding='valid'
h = tf.layers.BatchNormalization()(h, training=is_training)
h = tf.nn.relu(h)
h = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(h) # 30x30x32 -> 15x15x32

h = tf.layers.Conv2D(filters=128, kernel_size= [3, 3])(h) # 15x15x32 -> 13x13x64
h = tf.layers.BatchNormalization()(h, training=is_training)
h = tf.nn.relu(h)
h = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(h) # 13x13x64 -> 6x6x64

h = tf.layers.Conv2D(filters=128, kernel_size= [3, 3])(h) # 6x6x64 -> 4x4x128
h = tf.layers.BatchNormalization()(h, training=is_training)
h = tf.nn.relu(h)
h = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(h) # 4x4x128 -> 2x2x128

h = tf.layers.Flatten()(h)
h = tf.layers.Dense(units=256, activation=tf.nn.relu)(h)
y = tf.layers.Dense(units=10, activation=tf.nn.softmax)(h)

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

### 前処理 ###
def gcn(x):
    mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    std = np.std(x, axis=(1, 2, 3), keepdims=True)
    return (x - mean)/std

class ZCAWhitening:
    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.mean = None
        self.ZCA_matrix = None
    
    def fit(self, x):
        x = x.reshape(x.shape[0], -1)
        self.mean = np.mean(x, axis=0)
        x -= self.mean
        cov_matrix = np.dot(x.T, x) / x.shape[0]
        A, d, _ = np.linalg.svd(cov_matrix)
        self.ZCA_matrix = np.dot(np.dot(A, np.diag(1. / np.sqrt(d + self.epsilon))), A.T)
    
    def transform(self, x):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x -= self.mean
        x = np.dot(x, self.ZCA_matrix.T)
        return x.reshape(shape)

print("Started Preprocess...")
x_train, x_test, t_train = load_cifar10()
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.05, random_state=random_state)
x_train = np.concatenate([x_train, x_train[:, :, ::-1, :]], axis=0)
padded = np.pad(x_train, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant')
crops = rng.randint(8, size=(len(x_train), 2))
x_train_cropped = np.array([padded[i, c[0]:(c[0]+32), c[1]:(c[1]+32), :] for i, c in enumerate(crops)])
x_train = np.concatenate([x_train, x_train_cropped], axis=0)
t_train = np.concatenate([t_train, t_train, t_train, t_train], axis=0)
zca = ZCAWhitening()
zca.fit(x_train)
x_train_zca = zca.transform(gcn(x_train))
t_train_zca = t_train[:]
x_valid_zca = zca.transform(gcn(x_valid))
t_valid_zca = t_valid[:]
x_test_zca = zca.transform(gcn(x_test))
print("Done.")

### 学習 ###
print("Started Learning...")
n_epochs = 10
batch_size = 100
n_batches = x_train.shape[0]//batch_size

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(n_epochs):
    x_train_zca, t_train_zca = shuffle(x_train_zca, t_train_zca, random_state=random_state)
    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        sess.run(optimizer, feed_dict={x: x_train_zca[start:end], t: t_train_zca[start:end], is_training: True})

    y_inpred, cost_train = sess.run([y, cost], feed_dict={x: x_train_zca[:10000], t : t_train_zca[:10000], is_training: False})
    y_pred, cost_valid = sess.run([y, cost], feed_dict={x: x_valid_zca, t: t_valid_zca, is_training: False})

    print('EPOCH: {}, Train Cost : {:.3f}, Valid Cost: {:.3f}, Train Accuracy : {:.3f}, Valid Accuracy: {:.3f}'.format(
                                                                                                                       epoch,
                                                                                                                       cost_train,
                                                                                                                       cost_valid,
                                                                                                                       accuracy_score(t_train_zca[:10000].argmax(axis=1), y_inpred.argmax(axis=1)),
                                                                                                                       accuracy_score(t_valid_zca.argmax(axis=1), y_pred.argmax(axis=1))
                                                                                                                       ))

y_pred = sess.run(y, feed_dict={x :x_test_zca, is_training: False})
y_pred = np.argmax(y_pred, axis=1)

sess.close()

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission_pred.csv', header=True, index_label='id')
