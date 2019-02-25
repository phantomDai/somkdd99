
from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('data_train.csv', delimiter=',', usecols=(range(0, 41)))

# data normalization
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

# Initialization and training
som = MiniSom(7, 7, len(data[0]), sigma=3, learning_rate=0.5,
              neighborhood_function='triangle', random_seed=10)
#som.random_weights_init(data)
som.pca_weights_init(data)
print("Training...")
som.train_random(data, 4000)  # random training
print("\n...ready!")

plt.figure(figsize=(7, 7))
# Plotting the response for each pattern in the dataset
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#plt.colorbar()

target = np.genfromtxt('data_train.csv', delimiter=',', usecols=(len(data[0] + 1)), dtype=str)
t = np.zeros(len(target), dtype=int)
t[target == '0'] = 0
t[target == '4'] = 1
t[target == '5'] = 2

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for cnt, xx in enumerate(data):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
plt.savefig('som_99.png')
plt.show()