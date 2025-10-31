import matplotlib.pyplot as plt
import numpy as np

# --- 1. Generate Line Charts ---

# We'll first create the "Fairness state s_t" graph with three lines.
# The "min" and "max" graphs will then be calculated from this data.

# --- Configuration for synthetic data ---
# Change this to 100000 for a plot identical to the image,
# but 10000 is faster for testing.
time_steps = 100000
t = np.arange(0, time_steps)

# Generate three "noisy" data series that trend upwards
# You can play with these formulas to get different-looking graphs
y1 = np.clip(0.3 + (t / time_steps) * 0.5 + (np.random.rand(time_steps) - 0.5) * 0.6, 0, 1)
y2 = np.clip(0.2 + (t / time_steps) * 0.6 + (np.random.rand(time_steps) - 0.5) * 0.8, 0, 1)
y3 = np.clip(0.8 + (t / time_steps) * 0.15 + (np.random.rand(time_steps) - 0.5) * 0.3, 0, 1)

# --- Plot 1: Fairness state s_t (All Rooms) ---
plt.figure(figsize=(10, 4))
plt.plot(t, y1, label='Room1', lw=1.5)
plt.plot(t, y2, label='Room2', lw=1.5)
plt.plot(t, y3, label='Room3', lw=1.5)
plt.title('Fairness state $s_t$')
plt.xlabel('time step')
plt.ylabel('fairness state ($L$)')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()


# --- Plot 2: Minimum fairness ---
# Calculate the minimum value across all three rooms at each time step
y_min = np.minimum(np.minimum(y1, y2), y3)

plt.figure(figsize=(10, 4))
plt.plot(t, y_min, color='red')
plt.title('Minimum fairness: min$L_{st}$')
plt.xlabel('time step')
plt.ylabel('minimum fairness')
plt.ylim(0, 1)
plt.grid(True)
plt.show()


# --- Plot 3: Maximum fairness ---
# Calculate the maximum value across all three rooms at each time step
y_max = np.maximum(np.maximum(y1, y2), y3)

plt.figure(figsize=(10, 4))
plt.plot(t, y_max, color='black')
plt.title('Maximum fairness: max$L_{st}$')
plt.xlabel('time step')
plt.ylabel('maximum fairness')
plt.ylim(0, 1)
plt.grid(True)
plt.show()


# --- 2. Generate Pie Charts ---

# --- Plot 4: Max Fairness Distribution ---

# --- Data for the pie chart ---
# You can change these labels and sizes
labels_max = ['Room1', 'Room2', 'Room3']
sizes_max = [37.5, 29.9, 32.6]  # Percentages
colors_max = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Default matplotlib colors

plt.figure()
plt.pie(sizes_max, labels=labels_max, colors=colors_max, 
        autopct='%1.1f%%', startangle=90)
plt.title('Max Fairness Distribution')
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  
plt.show()


# --- Plot 5: Min Fairness Distribution ---

# --- Data for the second pie chart ---
labels_min = ['Room1', 'Room2', 'Room3']
sizes_min = [31.8, 37.2, 31.0]
colors_min = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure()
plt.pie(sizes_min, labels=labels_min, colors=colors_min,
        autopct='%1.1f%%', startangle=90)
plt.title('Min Fairness Distribution')
plt.axis('equal')
plt.show()