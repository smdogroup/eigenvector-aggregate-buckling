import matplotlib.pylab as plt
import numpy as np
from icecream import ic

dir = "result/Fancy/volume_frac=0.4/vtk/it_300.vtk"
dir_result = "result/Fancy/volume_frac=0.4/"
phi_i = "phi2"
nnodes = []

# find string "phi0" in file
with open(dir) as f:
    for num, line in enumerate(f, 1):
        if "POINTS" in line:
            nnodes = int(line.split()[1])
        if phi_i in line:
            # save the line of data from this line to line+40402
            phi = np.loadtxt( dir, skiprows=num, max_rows=40401 )

n = np.sqrt(nnodes).astype(int)
          
Z = np.sqrt(phi[:,0]**2 + phi[:,1]**2).reshape(n,n)
X, Y = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))

plt.contourf(X, Y, Z, levels=50, alpha=0.75, cmap=plt.cm.coolwarm, antialiased=True)
plt.colorbar()
# save the figure into dir
plt.savefig(dir_result + phi_i + ".png")

# plt as a 3d surface
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(dir_result + phi_i + "_3d.png")
plt.show()

