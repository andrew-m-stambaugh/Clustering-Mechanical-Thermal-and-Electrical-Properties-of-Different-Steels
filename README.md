# Clustering-Mechanical-Thermal-and-Electrical-Properties-of-Different-Steels

This repository contains the files as part of a longer homework in my Mechanistic Data Science course, a course which seeks to take a deep dive into the intersection between physics/mathematics and data science. In this assignment, we were given data on 34 different types of steel and asked to cluster them by 1.) mechanical properties 2.) thermal properties and 3.) electrical resistivity. The intention of this assignment was to see how the Kmeans clustering algorithm could be used in material science to classify materials by different properties. 

Files in this repository:

KmeansClustering.py: This is Python file contains all of the code written to import, preprocess, and cluster all of the data. For each of the three types of properties, I utilized a plot of Objective Cost vs. Number of Clusters to get an idea of how many clusters would be ideal. Then, I tasted a range of clusters and relied on my interpretation skills to decide the best outcome.

Steel_data_for_clustering.csv: This csv file contains all of the data used for the assignment. There are 12 total features, with 11 being numeric, and 34 different types of steel.
