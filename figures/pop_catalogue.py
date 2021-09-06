import matplotlib.pyplot as plt
x= np.arange(0,len(popularity))
y = np.asarray(popularity)
fig,ax=plt.subplots()
ax.plot(x,y)
ax.set(xlabel='Tracks',ylabel='Popularity',title='Popularity trend across catalogue')
ax.grid()
fig.savefig("/Users/pantera/MSc/figures/popularity_catalogue.png")
