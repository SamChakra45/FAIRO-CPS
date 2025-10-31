import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

L1 = np.load('L1_trace.npy')
L2 = np.load('L2_trace.npy')
L3 = np.load('L3_trace.npy')
timesteps = np.arange(len(L1))

plt.figure(figsize=(8, 4))
plt.title("Fairness state $s_t$")
plt.plot(timesteps, L1, label="Household1")
plt.plot(timesteps, L2, label="Household2")
plt.plot(timesteps, L3, label="Household3")
plt.xlabel("time step")
plt.ylabel("fairness state ($L$)")
plt.legend()
plt.ylim(0, 1.01)
plt.grid(True)
plt.tight_layout()
plt.savefig("fairness_state_all.png")
plt.show()

max_L = np.maximum.reduce([L1, L2, L3])
min_L = np.minimum.reduce([L1, L2, L3])

plt.figure(figsize=(8, 4))
plt.title("Maximum fairness: $\\max L_{st}$")
plt.plot(timesteps, max_L, color='k')
plt.xlabel("time step")
plt.ylabel("maximum fairness")
plt.ylim(0, 1.01)
plt.grid(True)
plt.tight_layout()
plt.savefig("fairness_state_max.png")
plt.show()

# Minimum Fairness
plt.figure(figsize=(8, 4))
plt.title("Minimum fairness: $\\min L_{st}$")
plt.plot(timesteps, min_L, color='r')
plt.xlabel("time step")
plt.ylabel("minimum fairness")
plt.ylim(0, 1.01)
plt.grid(True)
plt.tight_layout()
plt.savefig("fairness_state_min.png")
plt.show()

argmax = np.argmax([L1, L2, L3], axis=0)  
argmin = np.argmin([L1, L2, L3], axis=0)
counts_max = Counter(argmax)
counts_min = Counter(argmin)

labels = ['Household1', 'Household2', 'Household3']
sizes_max = [counts_max[i]/len(argmax)*100 for i in range(3)]
sizes_min = [counts_min[i]/len(argmin)*100 for i in range(3)]

plt.figure()
plt.pie(sizes_max, labels=labels, autopct='%1.1f%%')
plt.title("Max Fairness Distribution")
plt.savefig("pie_max_fairness.png")
plt.show()

plt.figure()
plt.pie(sizes_min, labels=labels, autopct='%1.1f%%')
plt.title("Min Fairness Distribution")
plt.savefig("pie_min_fairness.png")
plt.show()
