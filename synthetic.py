import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_cameras = 20000
n_servers = 800
random_seed = 42

np.random.seed(random_seed)

priorities = np.random.choice([3, 2, 1], size=n_cameras, p=[0.15, 0.25, 0.6])

loads = np.zeros(n_cameras)
mask_p3 = priorities == 3
mask_p2 = priorities == 2
mask_p1 = priorities == 1

loads[mask_p3] = np.random.uniform(8, 15, np.sum(mask_p3))
loads[mask_p2] = np.random.uniform(4, 8, np.sum(mask_p2))
loads[mask_p1] = np.random.uniform(1, 3, np.sum(mask_p1))

camera_x = np.random.uniform(0, 1000, n_cameras)
camera_y = np.random.uniform(0, 1000, n_cameras)

cameras_df = pd.DataFrame({
    'camera_id': range(n_cameras),
    'priority': priorities,
    'load_GFLOPS': loads,
    'x': camera_x,
    'y': camera_y
})

server_x = np.random.uniform(0, 1000, n_servers)
server_y = np.random.uniform(0, 1000, n_servers)

server_types = np.random.choice([3, 2, 1], size=n_servers, p=[0.1, 0.3, 0.6])
capacities = np.zeros(n_servers)

capacities[server_types == 3] = np.random.uniform(800, 1000, np.sum(server_types == 3))
capacities[server_types == 2] = np.random.uniform(400, 800, np.sum(server_types == 2))
capacities[server_types == 1] = np.random.uniform(200, 400, np.sum(server_types == 1))

servers_df = pd.DataFrame({
    'server_id': range(n_servers),
    'capacity_GFLOPS': capacities,
    'x': server_x,
    'y': server_y
})

scale_to_km = 10  # 1000 единиц → 100 км
cameras_df['x_km'] = cameras_df['x'] / scale_to_km
cameras_df['y_km'] = cameras_df['y'] / scale_to_km
servers_df['x_km'] = servers_df['x'] / scale_to_km
servers_df['y_km'] = servers_df['y'] / scale_to_km

color_map = {
    1: ('green',    'light'),
    2: ('orange',  'medium'),
    3: ('blue',    'heavy')
}

plt.figure(figsize=(10, 8))
plt.title('Cameras and Servers (by type)', fontsize=30, fontname='Times New Roman')

for priority, (color, label) in color_map.items():
    subset = cameras_df[cameras_df['priority'] == priority]
    plt.scatter(subset['x_km'], subset['y_km'],
                c=color, s=8, alpha=0.7, label=label)

plt.scatter(servers_df['x_km'], servers_df['y_km'],
            c='red', marker='^', s=60, edgecolors='darkred', label='servers')

plt.xlabel('x, km', fontsize=20, fontname='Times New Roman')
plt.ylabel('y, km', fontsize=20, fontname='Times New Roman')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='lower right', markerscale=1.5)

plt.legend(
    loc='lower right',
    markerscale=1.5,
    fontsize=35,
    prop={'family': 'Times New Roman', 'size': 25},
    # title='VSN Priority Levels',
    title_fontsize=18
    # fontname='Times New Roman'
)

plt.tight_layout()
plt.show()