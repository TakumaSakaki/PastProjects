import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import csv

### レイヤー定義 ###
class Embedding:
    # WRITE ME

# WRITE ME

### グラフ構築 ###
tf.reset_default_graph()

emb_dim = 1
hid_dim = 1
pad_index = 0

x = tf.placeholder(tf.int32, [None, None], name='x')
t = tf.placeholder(tf.int32, [None, None], name='t')

# WRITE ME

### データの準備 ###
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train)

### 学習 ###
n_epochs = 1
batch_size = 1

# WRITE ME

### 生成用グラフ構築 ###
max_len = tf.placeholder(tf.int32, name='max_len') # iterationの繰り返し回数の限度

def cond():
    # WRITE ME

def body():
    # WRITE ME

# WRITE ME

### 生成 ###
bos_id_ja, eos_id_ja = tokenizer_ja.texts_to_sequences(['<s> </s>'])[0]
y_pred = sess.run(
    # WRITE ME
)

### 出力 ###
def get_raw_contents(dataset, num, bos_id, eos_id):
    result = []
    for index in dataset[num]:
        if index == eos_id:
            break
            
        result.append(index)
        
        if index == bos_id:
            result = []
            
    return result

output = [get_raw_contents(y_pred, i, bos_id_ja, eos_id_ja) for i in range(len(y_pred))]

with open('/root/userspace/chap09/materials/submission_gen.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(output)