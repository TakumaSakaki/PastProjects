
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
random_state = 42

### レイヤー定義 ###

class Conv:
    # WRITE ME
    
    
class Pooling:
    # WRITE ME
    
    
class Flatten:
    # WRITE ME
    
    
class Dense:
    # WRITE ME
    
    
def tf_log(x):
    # WRITE ME
    
### ネットワーク ###

x_train, x_test, t_train = load_mnist()
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.1, random_state=random_state)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])

# WRITE ME

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

### 学習 ###

n_epochs = 10
batch_size = 100
n_batches = x_train.shape[0]//batch_size

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # WRITE ME
    
y_pred = # WRITE ME
submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission_pred.csv', header=True, index_label='id')