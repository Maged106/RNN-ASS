text = "The cat is fluffy"
words = text.lower().split()
vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}

def one_hot(word):
    vec = [0] * 4
    vec[word_to_idx[word]] = 1
    return vec

X = [one_hot(w) for w in words[:3]]
y = one_hot(words[3])

hidden_size = 4
Wxh = [[0.1] * 4 for _ in range(hidden_size)]
Whh = [[0.1] * hidden_size for _ in range(hidden_size)]
Why = [[0.1] * hidden_size for _ in range(4)]
bh = [0.0] * hidden_size
by = [0.0] * 4
lr = 0.1

def sigmoid(x):
    return 1 / (1 + 2.71828 ** -x)

def softmax(x):
    ex = [2.71828 ** xi for xi in x]
    s = sum(ex)
    return [e / s for e in ex]

def matmul(A, B):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(B)):
            result[i] += A[i][j] * B[j]
    return result

def add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

for _ in range(100):
    h = [0.0] * hidden_size
    for x in X:
        xh = matmul(Wxh, x)
        hh = matmul(Whh, h)
        h = [sigmoid(xh[i] + hh[i] + bh[i]) for i in range(hidden_size)]
    
    hy = matmul(Why, h)
    y_pred = softmax(add(hy, by))
    
    loss = -sum(yt * (2.71828 ** yp) for yt, yp in zip(y, y_pred))
    grad_y = [yp - yt for yp, yt in zip(y_pred, y)]
    
    for i in range(4):
        for j in range(hidden_size):
            Why[i][j] -= lr * grad_y[i] * h[j]
        by[i] -= lr * grad_y[i]
    
    for i in range(hidden_size):
        for j in range(4):
            Wxh[i][j] -= lr * sum(grad_y) * x[j]

h = [0.0] * hidden_size
for x in X:
    xh = matmul(Wxh, x)
    hh = matmul(Whh, h)
    h = [sigmoid(xh[i] + hh[i] + bh[i]) for i in range(hidden_size)]
hy = matmul(Why, h)
y_pred = softmax(add(hy, by))
pred_idx = y_pred.index(max(y_pred))
print(f"Predicted 4th word: {vocab[pred_idx]}")