import numpy as np
from sklearn.decomposition import PCA

# ===== 1. MNISTのロード =====
# keras.datasetsから直接読み込める
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")

# 28x28 → 784次元にreshape & 正規化（0~1に）
x_train = x_train.reshape(len(x_train), -1) / 255.0
x_test  = x_test.reshape(len(x_test), -1) / 255.0

# ===== 2. PCAで次元削減 =====
pca = PCA(n_components=100, random_state=42)
x_train_pca = pca.fit_transform(x_train)
x_test_pca  = pca.transform(x_test)

print("元の次元:", x_train.shape[1])   # 784
print("圧縮後:", x_train_pca.shape[1]) # 100

# ===== 3. npz形式で保存 =====
np.savez_compressed(
    "mnist_pca_100.npz",
    x_train=x_train_pca, y_train=y_train,
    x_test=x_test_pca,   y_test=y_test
)

print("mnist_pca_100.npz を保存しました！")
