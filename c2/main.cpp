#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <limits>

#include "kmeans.hpp"
#include "entropy_subspace.hpp"

const unsigned int MAX_KMEANS_ITER = 20;
const unsigned int DIVIDE_SIZE = 10;

KMeanData read_datafile(std::string& data_filename)
{
    std::ifstream datafile;
    KMeanData kmean_data;
    datafile.open(data_filename);
    if (datafile.is_open())
    {
        
        std::string line;
        
        // Get the first line 
        getline(datafile, line);
        size_t delim_pos; 
        size_t last_pos = 0;

        int num_objs, num_attrs;
        for (int i = 0; i < 3; ++i)
        {
            delim_pos = line.find(" ", last_pos);
            int val = std::stoi(line.substr(last_pos, delim_pos - last_pos));
            if (i == 0)
            {
                num_objs = val; 
            }
            else if (i == 1)
            {
                num_attrs = val;
            }
            else
            {
                kmean_data.min_k = val;
            }
            last_pos = delim_pos + 1;
        }
        // Get the last number
        kmean_data.max_k = std::stoi(line.substr(last_pos));
        
        // Add Dim objects
        for (int i = 0; i < num_attrs; i++)
        {
            kmean_data.dims.push_back(Dim(
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                i));
        }

        // Read all the objects and there attributes
        for (int i = 0; i < num_objs; ++i)
        {
            // Reset the positions
            delim_pos = 0;
            last_pos = 0;
            //float attr[nums[1]];
            getline(datafile, line);
            DataObj obj;
            obj.id = i;
            float dim_val;
            for (int j = 0; j < num_attrs; j++)
            {   
                float dim_val;
                if (j == num_attrs - 1)
                {
                    dim_val = std::stof(line.substr(last_pos));
                    
                }
                else
                {
                    delim_pos = line.find(" ", last_pos);
                    dim_val = std::stof(line.substr(last_pos, delim_pos - last_pos));
                    last_pos = delim_pos + 1;    
                }
                obj.attrs.push_back(dim_val);
    
                // Update value range for dimensions
                if (kmean_data.dims[j].max_val < dim_val)
                {
                    kmean_data.dims[j].max_val = dim_val;
                }
                if (kmean_data.dims[j].min_val > dim_val)
                {
                    kmean_data.dims[j].min_val = dim_val;
                }
            }
            kmean_data.objs.push_back(obj);
        }
    }
    return kmean_data;
}

int main()
{
    std::string data_filename =  "data/A2-small-test.dat";
    //std::string data_filename = "data/temp.dat";
    KMeanData kmean_data = read_datafile(data_filename);
    Kmeans k_means(kmean_data.min_k, kmean_data.max_k);
    std::vector<int> dims = {0, 1, 2,3};
    
    
    std::vector<ClusterResults> results = k_means.cluster(kmean_data.objs, dims, MAX_KMEANS_ITER, 5);

    for (auto& result : results)
    {
        std::cout << "k: " << result.k << " with sse " << result.sse << std::endl;
    }

    EntropySubspace entropy_subspace(DIVIDE_SIZE);
    entropy_subspace.setup(kmean_data.dims);
    //entropy_subspace.setup();
    
    //std::ifstream datafile;
    //datafile.open();
    // if (datafile.is_open())
    // {
    //     std::string line;
    //     while(std::getline(datafile, line))
    //     {
    //         std::cout << line << std::endl;
    //     }
    // }
    
    
    //std::cout << "Hello World!" << std::endl;
    //std::ifstream  
}