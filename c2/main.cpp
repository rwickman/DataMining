#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <limits>

#include "kmeans.hpp"
#include "entropy_subspace.hpp"

// The max number of iteration to run K-Means
const unsigned int MAX_KMEANS_ITER = 20;
// The number of equal parts to divide each dimension for subspace selection
const unsigned int DIVIDE_SIZE = 10;
// The maximum number of dimensions
const unsigned int MAX_NUM_SUBSPACES = 3;
// The number of times to repeat K-Means with a different random initialization
const unsigned int NUM_KMEAN_REPEAT = 25;
// The minmum entropy threshold required for a subspaces 
// This could be set to 0.2 as in slides, but could risk removing all
// possible subspaces
const unsigned int ENTROPY_THRESHOLD = 1.0; 

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

void write_results(std::vector<ClusterResults> results, std::string output_filename)
{
    std::ofstream result_file;
    result_file.open(output_filename);
    if (result_file.is_open())
    {
        for(int i = 0; i < results.size(); ++i)
        {
            result_file << results[i].k << " " << results[i].sse;
            if (i < results.size() -1)
            {
                result_file << "\n";
            }
        }
        
    }
    result_file.close();
}

int main()
{
    //std::string data_filename =  "data/A2-small-test.dat";
    std::string data_filename =  "test.dat";
    std::string output_filename = "test.res";
    
    //std::string data_filename = "data/temp.dat";
    KMeanData kmean_data = read_datafile(data_filename);
    Kmeans k_means(kmean_data.min_k, kmean_data.max_k);

    EntropySubspace entropy_subspace(DIVIDE_SIZE);
    entropy_subspace.setup(kmean_data.dims);
    
    std::vector<int> dims_vec;
    if (kmean_data.dims.size() > MAX_NUM_SUBSPACES)
    {
        std::cout << "Applying entropy-based subspace selection method to get top " << MAX_NUM_SUBSPACES << " dimensions" << std::endl;
        auto best_subspace = entropy_subspace.find_best_subspaces(
            kmean_data.objs,
            MAX_NUM_SUBSPACES,
            ENTROPY_THRESHOLD);
        // for (auto& el : best_subspace)
        // {
        //     std::cout << el << " ";
        // }
        // std::cout << std::endl;
        std::copy(best_subspace.begin(), best_subspace.end(), std::back_inserter(dims_vec));
    }
    else
    {
        for(int i = 0; i < kmean_data.dims.size(); ++i)
        {
            dims_vec.push_back(i);
        }
    }
    
    std::cout << "Running K-Means" << std::endl;
    std::vector<ClusterResults> results = k_means.cluster(
        kmean_data.objs,
        dims_vec,
        MAX_KMEANS_ITER,
        NUM_KMEAN_REPEAT);

    // for (auto& result : results)
    // {
    //     //std::cout << "k: " << result.k << " with sse " << result.sse << std::endl;
    //     std::cout << result.k << " " << result.sse << std::endl;
    // }
    // Write out the results
    write_results(results, output_filename);
    std::cout << "Wrote results to " << output_filename << std::endl;
}