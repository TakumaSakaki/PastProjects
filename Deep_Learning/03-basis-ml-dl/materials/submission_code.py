#modules
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(10)


def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))

def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def train(x, t, eps=1.0):
    
    global W, b

    batch_size = x.shape[0]
    
    # 順伝播
    y = softmax(np.matmul(x, W) + b)
    
    # パラメータの更新
    delta = y - t
    dW = np.matmul(x.T, delta) / batch_size
    db = np.matmul(np.ones(batch_size), delta) / batch_size
    W -= eps * dW
    b -= eps * db
    
    return

def valid(x, t):
    y = softmax(np.matmul(x, W) + b)
    cost = (- t * np_log(y)).sum(axis=1).mean()
    return cost, y


#data
x_train, y_train, x_test = load_fashionmnist()

#initial values
W = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32')
b = np.zeros(shape=(10,)).astype('float32')
batch_size = 50
learning_rate = 0.1

#split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

#learning
for epoch in range(10):
    x_train, y_train = shuffle(x_train, y_train)
    
    for i in range(int(len(x_train) / batch_size)):
        x = x_train[i * batch_size : (i+1) * batch_size]
        y = y_train[i * batch_size : (i+1) * batch_size]
        cost = train(x, y, eps = learning_rate)
    cost, y_pred_val = valid(x_valid, y_valid)
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost,
        accuracy_score(y_valid.argmax(axis=1), y_pred_val.argmax(axis=1))
    ))

#prediction
y_pred = softmax(np.matmul(x_test, W) + b).argmax(axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('submission_pred.csv', header=True, index_label='id')