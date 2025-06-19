
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from kneed import KneeLocator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler,RobustScaler


import time
from numpy import log as ln

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from kneed import KneeLocator



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go





def twod(input_data):
    
    # If you have more than two features, reduce dimensions using PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    reduced_data = pca.fit_transform(input_data)

    # Create a scatter plot
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, c='blue', alpha=0.7)
    plt.title("Data Points Before Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
    
def threeD(input_data):
    # Reduce to 3 dimensions
    pca = PCA(n_components=3)
    reduced_data_3d = pca.fit_transform(input_data)

    # 3D Scatter Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_data_3d[:, 0], reduced_data_3d[:, 1], reduced_data_3d[:, 2], s=50, c='blue', alpha=0.7)
    ax.set_title("Data Points Before Clustering")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.show()

def twoD_tsne(scaled_data):
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000,random_state=0)
        tsne_results = tsne.fit_transform(scaled_data)
        tsne_df = pd.DataFrame(data = tsne_results,columns=['tsne1','tsne2'])
        # tsne_df = pd.concat([tsne_df, pd.DataFrame({'cluster':labels})], axis=1)
        fig = px.scatter(tsne_df,x="tsne1", y="tsne2",
                         color_discrete_map="identity", width=800, height=600)
        fig.show()    
    
def twoD_data_ceation(labels,scaled_data):
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000,random_state=0)
        tsne_results = tsne.fit_transform(scaled_data)
        tsne_df = pd.DataFrame(data = tsne_results,columns=['tsne1','tsne2'])
        tsne_df = pd.concat([tsne_df, pd.DataFrame({'cluster':labels})], axis=1)
        fig = px.scatter(tsne_df,x="tsne1", y="tsne2", color='cluster',symbol='cluster',
                         color_discrete_map="identity", width=800, height=600)
        fig.show()
        
        

def twoD_data_creation_with_boundaries(labels, scaled_data):
    # Perform t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, random_state=0)
    tsne_results = tsne.fit_transform(scaled_data)
    
    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne1', 'tsne2'])
    tsne_df['cluster'] = labels
    tsne_df=tsne_df[tsne_df['cluster'].isin([0,1,2])]
    
    # Plot points with cluster colors
    fig = px.scatter(tsne_df, x="tsne1", y="tsne2", color='cluster', symbol='cluster',
                     width=800, height=600)
    fig.update_traces(marker=dict(size=10)) 
    # Add convex hull boundaries
    for cluster_id in tsne_df['cluster'].unique():
        cluster_points = tsne_df[tsne_df['cluster'] == cluster_id][['tsne1', 'tsne2']].values
        if len(cluster_points) > 2:  # ConvexHull needs at least 3 points
            hull = ConvexHull(cluster_points)
            hull_vertices = hull.vertices
            hull_points = cluster_points[hull_vertices]
            hull_points = list(hull_points) + [hull_points[0]]  # Close the loop
            
            # Add boundary line to the figure
            fig.add_trace(go.Scatter(
                x=[point[0] for point in hull_points],
                y=[point[1] for point in hull_points],
                mode='lines',
                line=dict(color='black'),
                name=f'Boundary Cluster {cluster_id}'
            ))
    
    # Show the figure
    fig.show()
def tsne_3d(labels,scaled_data):
    # 4. Visualize the clusters in 3D
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=10000)
    reduced_data_3d = tsne.fit_transform(scaled_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster with a different color
    scatter = ax.scatter(
        reduced_data_3d[:, 0],
        reduced_data_3d[:, 1],
        reduced_data_3d[:, 2],
        c=labels,  # Cluster labels as colors
        s=50, cmap='viridis', alpha=0.7
    )

    # Add labels
    ax.set_title("K-Means Clustering on 3D t-SNE Data (k=4)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")

    # Add color bar to indicate clusters
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    # Optional: Inspect cluster centers
    print("Cluster Centers (in t-SNE space):")
    
def dbscan_on_multiple_matrics(df,dist_matrics,min_sample_k):
    for min_sample,list_k in min_sample_k.items():
        k=list_k[0]
        
        for matric in dist_matrics:
            print(f'dist_matrics: {matric}\n min_sample_k:{list_k[1]}\nk:{list_k[-1]}')

            
            nearest_neighbors = NearestNeighbors(n_neighbors=k,metric=matric)
            neighbors = nearest_neighbors.fit(df)
            distances, indices = neighbors.kneighbors(df)
            distances = np.sort(distances[:,-1], axis=0)

            i = np.arange(len(distances))
            knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
            # fig = plt.figure(figsize=(5, 5))
            # knee.plot_knee()
            # plt.xlabel("Points")
            # plt.ylabel("Distance")
            # print(distances[knee.knee])
            eps=distances[knee.knee]
            print(f'epsilon value for metric: {matric} is {eps} ')
    
            clus_data=pd.DataFrame()

            model = DBSCAN(eps=eps, min_samples=min_sample, metric=matric)
            yhat = model.fit_predict(df)


            clus_data['cluster']=yhat 
            print(clus_data['cluster'].value_counts())

            # print('the clusters formed are:',clusters)
            # Calculate cluster goodness metrics

            if len(set(yhat))==1:
                pass
            else:
                score_dbsacn_s = silhouette_score(df, yhat)
                score_dbsacn_c = calinski_harabasz_score(df, yhat)
                score_dbsacn_d = davies_bouldin_score(df, yhat)
                print(f'Silhouette Score on basis of {matric}: %.2f' % score_dbsacn_s)
                print(f'Calinski Harabasz Score on basis of {matric}: %.2f' % score_dbsacn_c)
                print(f'Davies Bouldin Score on basis of {matric}: %.2f' % score_dbsacn_d)
                print('_______________________________________________________\n')
                
               