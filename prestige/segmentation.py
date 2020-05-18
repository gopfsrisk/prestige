# functions for segmentation/cluster analysis
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# define function to get inertia by n_clusters
def plot_inertia(df_X, n_max_clusters, tpl_figsize, title_fontsize, axis_fontsize, str_figname):
	# instatiate list_inertia
	list_inertia = []
	# iterate through a range of clusters
	for n_clusters in np.arange(1,n_max_clusters+1):
	    # instantiate model
	    model = KMeans(n_clusters=n_clusters)
	    # fit model to df_cluster
	    model.fit(df_X)
	    # get inertia
	    inertia = model.inertia_
	    # append to list
	    list_inertia.append(inertia)

	# create axis
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# generate title
	ax.set_title('Inertia by n Clusters', fontsize=title_fontsize)
	# x label
	ax.set_xlabel('n Clusters', fontsize=axis_fontsize)
	# y label
	ax.set_ylabel('Inertia', fontsize=axis_fontsize)
	# plot inertia by n_clusters
	ax.plot(list(np.arange(1, n_max_clusters+1)), list_inertia)
	# xticks
	ax.set_xticks(list(np.arange(1, n_max_clusters+1)))
	# save figures
	plt.savefig(f'{str_figname}', bbox_inches='tight')
	# return fig
	return fig
