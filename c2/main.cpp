#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "kmeans.hpp"

const int KMEANS_ITERATIONS = 20;

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
            for (int j = 0; j < num_attrs; j++)
            {   
                if (j == num_attrs - 1)
                {
                    obj.attrs.push_back(std::stof(line.substr(last_pos)));
                    std::cout << obj.attrs[obj.attrs.size() - 1];
                }
                else
                {
                    delim_pos = line.find(" ", last_pos);
                    obj.attrs.push_back(std::stof(line.substr(last_pos, delim_pos - last_pos)));
                    std::cout << obj.attrs[obj.attrs.size() - 1] << " ";
                    last_pos = delim_pos + 1;    
                }
            }
            kmean_data.objs.push_back(obj);
            std::cout << "\n";
        }
    }
    return kmean_data;
}

int main()
{
    std::string data_filename =  "data/A2-small-test.dat";
    KMeanData kmean_data = read_datafile(data_filename);
    Kmeans k_means(kmean_data.min_k, kmean_data.max_k);
    
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