import numpy as np

speed=np.load(r"speed_rec.npy")

import matplotlib.pyplot as plt
plt.figure (figsize=(10,8))
plt.title ("temp_speed_error",fontsize=20)
plt.plot(speed-(-56),label="error")
plt.plot([0]*len(speed),'r--',label="zero")
plt.legend (fontsize=15)
plt.show()