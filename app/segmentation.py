import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

cfg_file_nm = "../config/segmentation_config.json"
with open(cfg_file_nm, "r") as f:
    cfg_dict = json.load(f)

file_nm = cfg_dict.get('input_file')
excl_var_list = cfg_dict.get('eda_excl_list')
num_of_clusters = cfg_dict.get('num_of_clusters')
cluster_incl_list = cfg_dict.get('cluster_incl_list')


def disp_corr_matrix(data):
    correlation_mat = data.corr()
    plt.rcParams["figure.figsize"] = (8, 8)
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

def run_pca(scaled_features):
    pca = PCA(n_components=3).fit(scaled_features)
    features_3d = pca.transform(scaled_features)

    ax = plt.axes(projection='3d')
    plt.rcParams["figure.figsize"] = (15, 15)
    # Data for three-dimensional scattered points
    zdata = features_3d[:, 0]
    xdata = features_3d[:, 1]
    ydata = features_3d[:, 2]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()

    pca = PCA(n_components=2).fit(scaled_features)
    features_2d = pca.transform(scaled_features)
    features_2d[0:10]

    plt.scatter(features_2d[:, 0], features_2d[:, 1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Data')
    plt.show()


def run_elbow_method(features):
    features = pd.DataFrame(features)
    #print(features.head(10))
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        # Fit the data points
        kmeans.fit(features.values)
        # Get the WCSS (inertia) value
        wcss.append(kmeans.inertia_)

    # Plot the WCSS values onto a line graph
    plt.plot(range(1, 11), wcss)
    # plt.title('Inertia by Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def cluster_data(data, num_clusters):
    model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=100, max_iter=1000)
    # Fit to the data and predict the cluster assignments for each data point
    km_clusters = model.fit_predict(data.values)
    # View the cluster assignments
    return km_clusters

def plot_clusters(samples, clusters):
    col_dic = {0: 'blue', 1: 'green', 2: 'orange'}
    mrk_dic = {0: '*', 1: 'x', 2: '+'}
    col_dic = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    mrk_dic = {0: '*', 1: 'x', 2: '+', 3: '_'}
    col_dic = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'yellow'}
    mrk_dic = {0: '*', 1: 'x', 2: '+', 3: '_', 4: '^'}

    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color=colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()




if __name__ == '__main__':

    data = pd.read_csv(file_nm)

    data = data.loc[:,~data.columns.isin(excl_var_list)]

    #disp_corr_matrix(data)

    #data = data[['TOTAL_USAGE_MEAN', 'PROFIT_MEAN', 'MEAN_TM_RPR', 'NBR_AR_UPGR', 'NBR_CPLN_OPN', 'NBR_CPLN_CLS']]


    # Normalize the numeric features so they're on the same scale
    scaled_features = MinMaxScaler().fit_transform(data)

    #print(scaled_features[:5])


    #run_pca(scaled_features)


    #run_elbow_method(data)

    pca = PCA(n_components=2).fit(scaled_features)

    features_2d = pca.transform(scaled_features)




    km_clusters = cluster_data(pd.DataFrame(scaled_features), num_of_clusters)
    # View the cluster assignments
    km_clusters

    plot_clusters(features_2d, km_clusters)
