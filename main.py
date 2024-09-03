import pykitti
import matplotlib.pyplot as plt
import numpy as np

basedir = '/path/to/KITTI/raw'
date = '2011_09_26'
drive = '0001'

# Load the raw dataset
dataset = pykitti.raw(basedir, date, drive)

# Access images and LIDAR
image = next(iter(dataset.cam2))  # Left color camera
point_cloud = next(iter(dataset.velo))  # LIDAR point cloud

# Display the image
plt.imshow(image)
plt.show()

# Display a simple scatter plot of the point cloud
plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1)
plt.show()
