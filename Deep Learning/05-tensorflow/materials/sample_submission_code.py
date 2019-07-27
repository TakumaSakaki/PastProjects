
import math
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tf.reset_default_graph() # グラフのリセット

x_train, y_train, x_test = load_fashionmnist()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1000)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])

class Dense:
    # WRITE ME

def sgd(cost, params, eps=0.1):
    # WRITE ME

# tf.log(0)によるnanを防ぐ
def tf_log(x):
    # WRITE ME

layers = [
    # WRITE ME
]

params = []
h = x
for layer in layers:
    h = layer(h)
    params += layer.params
y = h

cost = # WRITE ME

updates = sgd(cost, params)
train = tf.group(*updates)

n_epochs = 1
batch_size = 100
n_batches = math.ceil(len(x_train) / batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    # WRITE ME

y_pred = # WRITE ME

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/chap05/materials/submission_pred.csv', header=True, index_label='id')
