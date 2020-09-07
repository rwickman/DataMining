#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <experimental/filesystem>

#include "fptree.h"

namespace fs = std::experimental::filesystem;

//const int NUM_TOPICS = 5;
//const int MIN_SUPPORT = 400;
const int NUM_TOPICS = 2;
const int MIN_SUPPORT = 1;

// The suport and support order idx
struct support_order
{
    int support;
    int idx;  
};

std::unordered_map<int, std::string> read_vocab(std::fstream& vocab_file)
 {
     std::unordered_map<int, std::string> vocab_map;
     if(vocab_file.is_open())
     {
        std::string line;
        while(std::getline(vocab_file, line))
        {
            // Remove extra \r create by Windows file
            if(line.back() == '\r')
                line.pop_back();

            // Split the line
            size_t pos;
            if ((pos = line.find("\t")) != std::string::npos)
            {
                vocab_map.insert(std::make_pair<int, std::string>(std::stoi(line.substr(0, pos)), line.substr(pos+1) ));
            }
        }
     }
     return vocab_map;
 }

std::vector<std::vector<int>> read_topic(std::fstream& topic_file)
{
    std::vector<std::vector<int>> topic;
    if (topic_file.is_open())
    {
        std::string line;
        while(std::getline(topic_file, line))
        {
            // Remove extra \r create by Windows file
            if(line.back() == '\r')
                line.pop_back();

            //std::cout << line << std::endl;
            std::vector<int> trans;
            size_t pos = 0;
            size_t prev_pos = 0;
            std::string sub;
            // Get all integers in this line
            while ((pos = line.find(" ",  prev_pos)) != std::string::npos)
            {
                sub = line.substr(prev_pos, pos-prev_pos);
                //std::cout << "Item Index: " << sub << std::endl;
                trans.push_back(std::stoi(sub));
                prev_pos = pos + 1;
            }
            // Get the last integer
            topic.push_back(trans);
        }
    }
    return topic;
}

void get_support(std::vector<std::vector<int>>& topic, 
                std::map<std::string, int>& support_map,
                std::unordered_map<int, std::string>& vocab_map)
{
    for (auto& itemset : topic)
    {
        for (int& el: itemset)
        {
            std::string word = vocab_map[el];
            //std::map<std::string, int>::const_iterator got = support_map.find(word);
            if (support_map.find(word) == support_map.end())
            {
                support_map[word] = 0;
            }
            support_map[word] += 1;
            //std::cout << word << support_map[word] << std::endl;
        }
    }
}


// Create map for sort order index
std::unordered_map<std::string, support_order> create_sort_idx_map(
    std::vector<std::pair<std::string, int>> sorted_items)
{
    std::unordered_map<std::string, support_order> sort_idx_map;
    for (int i = 0; i < sorted_items.size(); ++i)
    {
        std::pair<std::string, int> item_pair = sorted_items[i];
        sort_idx_map[item_pair.first] = {item_pair.second, i};
    }
    return sort_idx_map;
}


std::vector<std::vector<std::string>> create_freq_trans(
    std::vector<std::vector<int>>& topic,
    std::unordered_map<int, std::string> vocab_map,
    std::unordered_map<std::string, support_order> sort_idx_map)
{
    std::vector<std::vector<std::string>> freq_trans;
    for (auto& itemset : topic)
    {
        std::vector<std::pair<int, std::string>> trans;
        for (int& el: itemset)
        {
            if(sort_idx_map[vocab_map[el]].support >= MIN_SUPPORT)
            {
                trans.push_back({sort_idx_map[vocab_map[el]].idx, vocab_map[el]});
            }
        }
        if (trans.size() > 0)
        {
            std::sort(trans.begin(), trans.end());
            std::vector<std::string> sorted_trans;
            for (auto& item : trans)
            {
                sorted_trans.push_back(item.second);
            }
            freq_trans.push_back(sorted_trans);
        }
    }
    return freq_trans;
}

int main()
{
    fs::path data_dir = fs::current_path() / "test_data";
    //fs::path data_dir = fs::current_path() / "data";
    std::cout << data_dir << std::endl;
    std::fstream datafile;
    std::vector<std::vector<int>> topics[NUM_TOPICS];
    std::unordered_map<int, std::string> vocab_map;
    for (const auto& entry : fs::directory_iterator(data_dir))
    {
        std::cout << entry.path().string() << std::endl;
        
        // Read contents of file 
        std::string line;
        std::string filename = entry.path().string();
        datafile.open(filename);

        if (filename.find("vocab") != std::string::npos)
        {
            vocab_map = read_vocab(datafile);
            // for (auto& pair : vocab_map)
            // {
            //     std::cout << pair.first << ":" << pair.second << std::endl; 
            // }
        }
        else
        {
            size_t pos;
            if ((pos = filename.find("-")) != std::string::npos)
            {
                std::cout << "NUMBER: " << std::stoi(filename.substr(pos+1, 1)) << std::endl;
                topics[std::stoi(filename.substr(pos+1, 1))] = read_topic(datafile);
                //for (auto& )
            }
        }
        datafile.close();
    }

    // Iterate over all the topics
    for (unsigned int i = 0; i < NUM_TOPICS; i++)
    {
        std::map<std::string, int> support_map;
        get_support(topics[i], support_map, vocab_map);

        std::vector<std::pair<std::string, int>> vec;
        
        // copy key-value pairs from support map to vector
        std::copy(support_map.begin(),
                support_map.end(),
                std::back_inserter<std::vector<std::pair<std::string, int>>>(vec));
        
        // Sort in descending order
        std::sort(vec.begin(), vec.end(),
                [](const std::pair<std::string,int>& l, const std::pair<std::string,int>& r) 
                {
                    if (l.second != r.second)
                        return l.second > r.second;
                    return l.first < r.first; 
                });

        std::unordered_map<std::string, support_order> sort_idx_map = create_sort_idx_map(vec);
        std::vector<std::vector<std::string>> freq_trans = create_freq_trans(topics[i], vocab_map, sort_idx_map);
        FPTree fptree;
        for (std::vector<std::string>& trans : freq_trans)
        {
            //std::cout << "Inserting item" << std::endl;
            fptree.InsertItemset(trans);
        }
        fptree.PrintTree();
        //break;
        // Get the 
        // int k = 0;
        // for (std::pair<std::string, int>& pair: vec) {
        //     std::cout << pair.second << " " << pair.first << " asdf" << std::endl;
        //     k++;
        //     if (k > 100)
        //     {
        //         break;
        //     }
        // }
        break;

    }
}