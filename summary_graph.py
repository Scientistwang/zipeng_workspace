from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
import numpy as np


# the main axes is subplot(111) by default
fig, ax = plt.subplots()
plt.axis([-1,2,-1,2])
plt.axvline(x=1)
plt.axhline(y=1)
plt.axis([0, 2, 0, 2])
plt.xlabel(r'$M_{ab}$',fontsize=20)
plt.ylabel(r'$M_{ba}$',fontsize=20)
plt.xticks([])
plt.yticks([])

#lower left
a = plt.axes([0.15, 0.15, 0.3, 0.3],frameon= False)
plt.axis([-1,2,-1,2])
plt.axvline(x=0,color='g')
plt.axhline(y=0,color='g')
plt.scatter([1],[0],facecolor = 'none',edgecolor = 'k')
plt.scatter([0],[1],facecolor = 'none',edgecolor = 'k')
plt.scatter([(2/3)],[(2/3)],facecolor = 'k',edgecolor = 'k')
plt.xticks([])
plt.yticks([])

#lower right
a = plt.axes([0.55, 0.15, 0.3, 0.3],frameon= False)
plt.axis([-1,2,-1,2])
plt.axvline(x=0,color='g')
plt.axhline(y=0,color='g')
plt.scatter([1],[0],facecolor = 'none',edgecolor = 'k')
plt.scatter([0],[1],facecolor = 'k',edgecolor = 'k')
plt.scatter([1.3333],[-0.6667],facecolor = 'none',edgecolor = 'k')
plt.xticks([])
plt.yticks([])

#upper left
a = plt.axes([0.15, 0.55, 0.3, 0.3],frameon= False)
plt.axis([-1,2,-1,2])
plt.axvline(x=0,color='g')
plt.axhline(y=0,color='g')
plt.scatter([1],[0],facecolor = 'k',edgecolor = 'k')
plt.scatter([0],[1],facecolor = 'none',edgecolor = 'k')
plt.scatter([-0.6667],[1.3333],facecolor = 'none',edgecolor = 'k')
plt.xticks([])
plt.yticks([])

#upper right
a = plt.axes([0.55, 0.55, 0.3, 0.3],frameon= False)
plt.axis([-1,2,-1,2])
plt.axvline(x=0,color='g')
plt.axhline(y=0,color='g')
plt.scatter([1],[0],facecolor = 'k',edgecolor = 'k')
plt.scatter([0],[1],facecolor = 'k',edgecolor = 'k')
plt.scatter([0.4444],[0.444],facecolor = 'none',edgecolor = 'k')
plt.xticks([])
plt.yticks([])

#middle
a = plt.axes([0.44, 0.42, 0.15, 0.15])
plt.axis([-1,2,-1,2])
plt.axvline(x=0,color='g')
plt.axhline(y=0,color='g')
plt.scatter([1],[0],facecolor = (.5,.5,.5),edgecolor = 'none')
plt.scatter([0],[1],facecolor = (.6,.6,.6),edgecolor = 'none')
plt.plot([0.5],[0.5],'x')
plt.xticks([])
plt.yticks([])

plt.text(0, 9, r' $\frac{1}{\mu_b}$', fontsize=20)
plt.text(8.5, 0, r' $\mu_b$', fontsize=20)

plt.show()