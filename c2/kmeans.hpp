#include <vector>
#include <unordered_map>

#include "dataobj.hpp"

struct ClusterResults
{
  // The number of clusters
  int k;

   // The SSE results
  float sse;

};

class Cluster
{
public:
  Cluster(std::vector<int> dims); 

  // Randomly initialize the center
  void random_init();

  // Update the center (or mean) of this cluster
  void update();

  // Add DataObject to cluster
  void add(DataObj* obj);

  // Remove all DataObjs from cluster
  void clear();

  // calculate the euclidean distance between the center and DataObj 
  float distance(DataObj& obj);

  // Calculate the sum of squared errors
  float sse();

private:
  // The DataObjs belonging to this cluster
  std::vector<DataObj*> objs;

  std::vector<int> dims;
  std::vector<float> center;
};

class Kmeans
{
public:
  Kmeans(int min_k, int max_k);

  // Create a cluster with a random initialized center
  Cluster create_cluster(std::vector<int>& dims);

  // Cluster the DataObjs on the specified dimensions
  std::vector<ClusterResults> cluster(
    std::vector<DataObj>& objs,
    std::vector<int>& dims,
    int max_kmean_iter);

private:
  int min_k;
  int max_k;
  //std::vector<Cluster> clusters;
};