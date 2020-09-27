#include <random>
#include <ctime>
#include <cmath>
#include <limits>

#include "kmeans.hpp"


Cluster::Cluster(std::vector<int> dims) : dims(dims), center(dims.size())
{
    random_init();
}

void Cluster::random_init()
{
    for (auto& el : center)
    {
        el = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void Cluster::update()
{
    if (objs.size() > 0)
    {
        float sum[dims.size()] = {};
        //int i = 0;
        // Get the sum of each dimension over all DataObjs
        for (DataObj* obj : objs)
        {
            for (int i = 0; i < dims.size(); i++)
            {
                sum[i] = obj->attrs[dims[i]];
            }
        }
        // Take the mean to get center
        for (int i = 0; i < center.size(); i++)
        {
            center[i] = sum[i] / objs.size();
        }
    }
}

void Cluster::add(DataObj* obj)
{
    objs.push_back(obj);
}

void Cluster::clear()
{
    objs.clear();
}

float Cluster::distance(DataObj& obj)
{
    // Sum of squared differences
    float ssd = 0.0;
    for (int i = 0; i < dims.size(); ++i)
    {
        ssd += std::pow(center[i] - obj.attrs[dims[i]], 2);
    }
    return std::pow(ssd, 0.5);
}

float Cluster::sse()
{
    for (int i = 0; i < dims.size(); ++i)
    {
        std::pow(objs[i]->attrs[dims[i]] - center[i], 2);
    }    
}

Kmeans::Kmeans(int min_k, int max_k) : min_k(min_k), max_k(max_k)
{
    srand(static_cast<unsigned>(time(0)));
}

std::vector<ClusterResults> Kmeans::cluster(
    std::vector<DataObj>& objs,
    std::vector<int>& dims,
    int max_kmean_iter)
{
    for (int k = min_k; k <= max_k; ++k)
    {
        // Create the initial clusters
        std::vector<Cluster> clusters;
        for (int i = 0; i < k; ++i)
        {
            Cluster cluster(dims);
            clusters.push_back(cluster);
        }

        // Run for max_kmean_iter iterations
        for (int i = 0; i < max_kmean_iter; ++i)
        {
            
            // Remove DataObjs from all the clusters
            // This is to prevent duplicates
            if (i > 0)
            {
                for(auto& cluster : clusters)
                {
                    cluster.clear();
                }
            }
            
            // Assignment Step (Assign every DataObj to a Cluster)
            for (auto& obj : objs)
            {
                float min_dist = std::numeric_limits<float>::max();
                float cur_dist;
                int closest_index;

                // Find which cluster this DataObj is closest to
                for (int j = 0; j < clusters.size(); ++j)
                {
                    cur_dist = clusters[j].distance(obj);
                    if (cur_dist < min_dist)
                    {
                        closest_index = j;
                        min_dist = cur_dist;
                    }
                }
                clusters[closest_index].add(&obj);
            }

            // Update Step (Update cluster centers)
            for (auto& cluster : clusters)
            {
                cluster.update();
            }
        }
    }


}