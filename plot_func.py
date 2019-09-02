import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 

def Func01(x,y):
	#eq = (x-50)**2 + (y-100)**2 #optimals x=50 y=100
	eq = 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))# Rastrigin x=0;y=0;v=0
	#eq = x**2 + y**2 # sphere x=0;y=0;v=0
	return(eq) 
 
X = np.linspace(-5, 5, 100)     
Y = np.linspace(-5, 5, 100)  
#X = Y = np.linspace(0, 200, 100) #test   
X, Y = np.meshgrid(X, Y) 

Z = Func01(X,Y)
 
fig = plt.figure() 
ax = fig.gca(projection='3d') 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)    
plt.savefig('test_graph.png')
plt.show()
