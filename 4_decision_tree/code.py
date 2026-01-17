import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

class Node:# định nghĩa lớp Node (một nút trong cây quyết định)
    def __init__(self, label=None, attr=None, children=None):
        self.label = label
        self.attr = attr
        self.children = children or {}

def entropy(labels): # hàm tính entropy của danh sách nhãn labels
    m = len(labels)
    if m == 0:
        return 0.0
    counts = Counter(labels)# đếm số lần xuất hiện của từng nhãn, ví dụ {"Yes":3,"No":2}
    return -sum((c/m) * math.log2(c/m) for c in counts.values())# công thức entropy: -∑ p*log2(p)


def info_gain(X, y, attr):# tính information gain khi tách theo thuộc tính attr
    m = len(y)
    ent_before = entropy(y)

    groups = defaultdict(list)   # groups: value -> list chỉ số (index) của các mẫu có value đó
    for i, row in enumerate(X):
        groups[row[attr]].append(i)# lấy giá trị row[attr] rồi thêm index i vào nhóm đó

    ent_after = 0.0
    for idxs in groups.values():# idxs là danh sách index của từng nhóm (mỗi value một nhóm)
        ys = [y[i] for i in idxs]
        ent_after += len(idxs) * entropy(ys) / m

    return ent_before - ent_after, groups

def most_common(y): # hàm lấy nhãn xuất hiện nhiều nhất trong y
    return Counter(y).most_common(1)[0][0]

def id3(X, y, attrs):
    if len(set(y)) == 1:
        return Node(label=y[0])

    if not attrs:
        return Node(label=most_common(y))

    best_attr, best_gain, best_groups = None, -1.0, None
    for a in attrs:
        g, groups = info_gain(X, y, a)
        if g > best_gain:
            best_attr, best_gain, best_groups = a, g, groups

    if best_attr is None or best_gain <= 1e-12:
        return Node(label=most_common(y))

    node = Node(attr=best_attr)
    remaining = [a for a in attrs if a != best_attr]

    for val, idxs in best_groups.items():
        childX = [X[i] for i in idxs]
        childY = [y[i] for i in idxs]
        node.children[val] = id3(childX, childY, remaining)

    return node

def predict_one(node, x):
    while node.label is None:
        v = x.get(node.attr)
        if v not in node.children:
            return None
        node = node.children[v]
    return node.label

def predict(node, X):
    return [predict_one(node, x) for x in X]

def accuracy(y_true, y_pred):
    correct = 0
    total = 0
    for yt, yp in zip(y_true, y_pred):
        if yp is None:
            continue
        total += 1
        correct += (yt == yp)
    return correct / total if total else 0.0


if __name__ == "__main__":

    df = pd.read_csv("mushrooms.csv")

    # để chạy nhanh hơn nếu dataset lớn (tuỳ chọn)
    df = df.sample(n=min(3000, len(df)), random_state=0).reset_index(drop=True)

    target_col = "class"  # với mushrooms.csv thường là 'class' 
    attrs = [c for c in df.columns if c != target_col]

    df = df.fillna("NA").astype(str) # đưa sang hết string

    X = df[attrs].to_dict(orient="records")   # list[dict]
    y = df[target_col].tolist()               # list

    
    np.random.seed(0)
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = idx[:split]
    test_idx  = idx[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]

    tree = id3(X_train, y_train, attrs)
    y_pred = predict(tree, X_test)

    print("Test accuracy:", accuracy(y_test, y_pred))
