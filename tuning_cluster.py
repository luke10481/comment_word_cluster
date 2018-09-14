import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
with open('./store/low_dim_word_embending', 'rb') as f:
    low_dim_embs = pickle.load(f)
x = low_dim_embs[:, 0]
y = low_dim_embs[:, 1]
for i in range(100):
    for j in range(10):
        plt.clf()
        eps_num = 1.5 + (i * 0.1)
        min_samples = 10 + (j * 1)
        plt.title('eps:' + str(eps_num) + '  ' + 'min_samples:' + str(min_samples))
        db = DBSCAN(eps=eps_num, min_samples=8).fit_predict(low_dim_embs)
        plt.scatter(x, y, c=db)

        plt.draw()
        plt.pause(0.01)
