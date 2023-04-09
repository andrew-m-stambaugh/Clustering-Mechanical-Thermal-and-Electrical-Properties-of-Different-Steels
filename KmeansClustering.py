# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing



#%% Import the data and preprocess

# Read in the data and remove second column
data = pd.read_csv("Steel_data_for_clustering.csv")
data = data.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

# Split data into categories of properties
mech = data.iloc[:, 1:8]
therm = data.iloc[:, 8:11]
elec = data.iloc[:, 11]

# Normalize mechanical and thermal properties
mech_norm = preprocessing.scale(mech)
therm_norm = preprocessing.scale(therm)
elec_norm = preprocessing.scale(elec)
mech_norm = pd.DataFrame(mech_norm, columns=mech.columns)
therm_norm = pd.DataFrame(therm_norm, columns=therm.columns)
elec_norm = pd.DataFrame(elec_norm, columns=['Electrical Resistivity'])

# Reduce weight of Poisson Ratio
mech_norm['Poissons Ratio'] *= 0.5

#%% Mechanical data clusters and plots

# Define function for plotting scatter plot
def plot_scatter(groups, title, labels):
    plt.title(title)
    for name, group in groups:
        plt.scatter(group['Tensile Strength Ultimate'], group['Elongation at Break'], label=name)
    plt.xlabel('Tensile Strength Ultimate')
    plt.ylabel('Elongation at Break')
    plt.legend(labels)
    plt.show()

# Define function for plotting cluster centers
def plot_centers(centers, title):
    plt.figure(figsize=(8,9))
    for i in range(7):
        plt.plot(range(len(centers)), centers[:,i], '--', marker='o')
    plt.title(title, size=16)
    plt.legend(list(data.columns[1:8]), loc='best')
    plt.show()

# Perform KMeans clustering with different number of clusters
inertia = []
for i in range(1, 12):
    Kmech = KMeans(n_clusters=i, tol=1e-6, n_init=100, random_state=12345)
    Kmech.fit(mech_norm)
    inertia.append(Kmech.inertia_)

# Plot objective cost of KMeans vs number of clusters
plt.title('Mech Objective Cost of Kmeans vs # of clusters')
plt.plot(range(1,12), inertia)
plt.show()

# Perform clustering with 2 to 5 clusters and plot scatter plots and cluster centers
for k in range(2, 6):
    Kmech_k = KMeans(n_clusters=k, tol=1e-6, n_init=100, random_state=12345)
    Kmech_k.fit(mech_norm)
    mech['clusters{}'.format(k)] = Kmech_k.labels_
    data['mech clusters{}'.format(k)] = Kmech_k.labels_
    groups = data.groupby(f'mech clusters{k}')
    labels = ['Cluster {}'.format(i+1) for i in range(k)]
    plot_scatter(groups, f'Elongation at Break vs Tensile Strength, k = {k}', labels)
    plot_centers(Kmech_k.cluster_centers_, f'Mech Normalized Cluster Means, k = {k}')
    
    
#%% Thermal data clusters and plots

# Calculate inertia for different number of clusters
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, tol=1e-6, n_init=100, random_state=12345)
    kmeans.fit(therm_norm)
    inertia.append(kmeans.inertia_)

# Plot the objective cost vs. number of clusters
plt.plot(range(1, 10), inertia)
plt.title('Therm Objective Cost of Kmeans vs # of clusters')
plt.show()

# Perform clustering for 2 to 5 clusters
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, tol=1e-6, n_init=100, random_state=12345)
    kmeans.fit(therm_norm)
    therm[f'clusters{k}'] = kmeans.labels_

    # Create a 3D scatter plot for each number of clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Thermal Conductivity vs. Heat Capacity vs. Coeff of Thermal Expansion, k = {k}')

    # Group data points by their cluster label
    groups = therm.groupby(f'clusters{k}')

    # Create scatter plot for each cluster
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for idx, (name, group) in enumerate(groups):
        ax.scatter(group['Coefficient of Thermal Expansion'], 
                   group['Thermal Conductivity'], 
                   group['Specific Heat Capacity'], 
                   label=f'Cluster {name}', 
                   color=colors[idx])
        
    ax.set_xlabel('Coeff of Thermal Exp.')
    ax.set_ylabel('Therm. Cond.')
    ax.set_zlabel('Spec. Heat Cap.')
    ax.set_xlim(10, 17.5)
    ax.set_ylim(16, 53)
    ax.set_zlim(0.45, 0.51)
    plt.legend(loc = 'best')
    plt.show()
    
#%% Electrical Data Clusters

# Compute KMeans objective cost for different number of clusters
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, tol=1e-6, n_init=100, random_state=12345)
    kmeans.fit(elec_norm)
    inertia.append(kmeans.inertia_)

# Plot KMeans objective cost vs number of clusters
plt.plot(range(1, 10), inertia)
plt.title('Electrical Resistivity: KMeans Objective Cost vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Cost')
plt.show()
    
# Perform clustering with 2 to 5 clusters and plot scatter plots and cluster centers
for k in range(2, 4):
    Kelec_k = KMeans(n_clusters=k, tol=1e-6, n_init=100, random_state=12345)
    Kelec_k.fit(elec_norm)
    elec_norm['clusters{}'.format(k)] = Kelec_k.labels_
    groups = elec_norm.groupby('clusters{}'.format(k))
    labels = ['Cluster {}'.format(i+1) for i in range(k)]
    for name, group in groups:
        plt.plot(group['Electrical Resistivity'], len(group) * [1], "o", label = labels[name])
        plt.xlabel('Electrical Resistivity')
    plt.legend()
    plt.title(f"Electrical Resistivity per Cluster, k = {k}")
    plt.yticks([])
    plt.show()
