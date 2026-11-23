# Lecture 8: Unsupervised Learning -- all code from the slides
# Only small made-up data sets: no files to read.

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from scipy.cluster.hierarchy import linkage

# ---------------------------------------------------------------- Part 2: K-means by hand
# 8 customers: (Fresh $K, Grocery $K)
P = np.array([(2, 10), (1, 7), (2, 11), (3, 8), (7, 4), (9, 0), (9, 5), (11, 3)], float)
names = list("ABCDEFGH")

centers = P[[4, 5]].copy()          # start: centers on E and F (a bad start)
for round_no in range(1, 10):
    # Assign: every customer joins the nearest center
    dist = np.linalg.norm(P[:, None] - centers[None], axis=2)
    group = dist.argmin(axis=1)
    print("Round", round_no, "groups:", [names[i] + str(group[i] + 1) for i in range(8)])
    # Move: each center goes to the average of its group
    new_centers = np.array([P[group == k].mean(axis=0) for k in range(2)])
    print("   new centers:", new_centers.tolist())
    if np.allclose(new_centers, centers):   # nothing changed -> stop
        break
    centers = new_centers

# Silhouette of every customer: final groups vs. round-1 groups
final = np.array([0, 0, 0, 0, 1, 1, 1, 1])
round1 = np.array([0, 0, 0, 0, 0, 1, 0, 1])
print("final  s:", dict(zip(names, silhouette_samples(P, final).round(2).tolist())), " average:", round(silhouette_score(P, final), 2))
print("round1 s:", dict(zip(names, silhouette_samples(P, round1).round(2).tolist())), " average:", round(silhouette_score(P, round1), 2))

# Choosing K: elbow (inertia) and silhouette
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(P)
    print("K =", k, " inertia =", round(km.inertia_, 1), " silhouette =", round(silhouette_score(P, km.labels_), 2))

# The code slide
km = KMeans(n_clusters=2, n_init=10, random_state=0)
labels = km.fit_predict(P)            # no y: only X
print(labels)                         # [1 1 1 1 0 0 0 0]
print(km.cluster_centers_)            # [[9. 3.]  [2. 9.]]
print(silhouette_score(P, labels))    # 0.66

# ---------------------------------------------------------------- Part 3: spectral clustering
Xm, ym = make_moons(300, noise=0.05, random_state=0)     # two moon shapes

def share_right(labels):           # share of points in the right moon (labels can be swapped)
    return round(max((labels == ym).mean(), (labels != ym).mean()), 2)

moons_km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xm)
print("K-means on the moons:", share_right(moons_km))            # 0.75

# Optional, not on the slides: how the computer finds the numbers in Step 2
# (9-student network: W = links, L = D - W, the 2nd eigenvector = each person's number)
people = list("ABCDEFGHI")
links = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E"),
         ("E", "F"), ("E", "G"), ("F", "G"), ("F", "H"), ("G", "H"), ("H", "I")]
W = np.zeros((9, 9))
for a, b in links:
    i, j = people.index(a), people.index(b)
    W[i, j] = W[j, i] = 1
L = np.diag(W.sum(axis=1)) - W               # D - W
values, vectors = np.linalg.eigh(L)          # eigenvalues sorted from small to large
second = vectors[:, 1]                       # the 2nd eigenvector = each person's number
print("2nd eigenvector:", dict(zip(people, second.round(2).tolist())))   # signs split the network (may be flipped)

# Steps 1-2 on the moons: network of 10 nearest neighbors, then project
A = kneighbors_graph(Xm, 10, include_self=True).toarray()
A = 0.5 * (A + A.T)                                            # make the links two-way
new_coord = spectral_embedding(A, n_components=1, drop_first=True, random_state=0)
print("new coordinates:", np.unique(new_coord.round(3)))       # two values: two piles
# Step 3: K-means on the new coordinate
print("K-means on the new coordinate:", share_right(KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(new_coord)))

# All three steps in one call
sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=0)
print("Spectral clustering on the moons:", share_right(sc.fit_predict(Xm)))   # 1.0

# A network you already have: affinity="precomputed"
print(SpectralClustering(n_clusters=2, affinity="precomputed", random_state=0).fit_predict(W))

# ---------------------------------------------------------------- Part 4: hierarchical clustering and DBSCAN
moons_ward = AgglomerativeClustering(n_clusters=2).fit_predict(Xm)                     # Ward (default)
moons_single = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(Xm) # follows chains
print("Hierarchical on the moons -- Ward:", share_right(moons_ward), " single:", share_right(moons_single))

# the tree for the 8 customers (average linkage): each row = one join
print(linkage(P, method="average").round(2))

for eps in [0.05, 0.2, 0.6]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(Xm)     # -1 = noise
    print("DBSCAN eps =", eps, " groups:", len(set(labels) - {-1}), " noise points:", (labels == -1).sum())
