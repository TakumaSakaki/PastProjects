import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences

### レイヤー定義 ###
class Embedding:
    # WRITE ME
    
class RNN:
    # WRITE ME

### グラフ構築 ###
tf.reset_default_graph()

emb_dim = 1
hid_dim = 1
num_words = max([max(s) for s in np.hstack((x_train, x_test))])
pad_index = 0

x = tf.placeholder(tf.int32, [None, None], name='x')
t = tf.placeholder(tf.float32, [None, None], name='t')

# WRITE ME

test = tf.round(y)

### データの準備 ###
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train)

### 学習 ###
n_epochs = 1
batch_size = 1 # バッチサイズが大きいと、ResourceExhaustedErrorになることがあります

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # WRITE ME
    
submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/chap08/materials/submission_pred.csv', header=True, index_label='id')