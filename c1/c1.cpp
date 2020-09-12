#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <experimental/filesystem>
#include <queue>
#include <unordered_set>
#include <algorithm>

#include "fptree.h"

namespace fs = std::experimental::filesystem;


//const int NUM_TOPICS = 5;
//const int MIN_SUPPORT = 400;
const int NUM_TOPICS = 2;
const int MIN_SUPPORT = 2;

// The suport and support order idx
struct support_order
{
    int support;
    int idx;  
};

struct FreqItemset
{
    std::vector<std::string> items;
    int support;
    bool operator<(const FreqItemset& itemset) const
    {
        return support < itemset.support;
    }
};

using Pattern = std::vector<FreqItemset>; 

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
    // Create a map from every item to its' support order and support
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
    /* Create set of frequent items from every transactions */
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
        // If there was at least one item that is frequent in this transaction
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

std::queue<FPNode*> get_leaf_nodes(FPTree& fptree,  std::unordered_set<std::string>& nodes_seen)
{
    // Get the leaf nodes from an FPTree
    std::queue<FPNode*> leaf_nodes;
    for (std::pair<std::string, FPNode*> node_pair : fptree.GetNodeLinks())
    {
        if (node_pair.second->GetChildren().size() <= 0)
        {
            leaf_nodes.push(node_pair.second);
            nodes_seen.insert(node_pair.first);
        }
    }
    return leaf_nodes;
}

Pattern create_patterns(FPNode* cur_node, std::vector<std::string> cur_prefix)
{

    std::string pref;
    for(auto& s : cur_prefix) pref += s + " ";
    std::cout <<"PREFIX: " << pref << std::endl;

    // Recursively create patterns
    std::string prefix;
    for (const auto& p : cur_prefix) prefix += p + " ";
    
    // The conditional support
    std::unordered_map<std::string, int> cond_sup;
 
    // All the conditional transactions
    std::vector<std::vector<std::string>> cond_trans;
    std::vector<int> cond_trans_counts;


    // Get the prefix paths
    FPNode* cur_leaf_node = cur_node;
    while (cur_leaf_node)
    {
        std::vector<std::string> cur_trans;
        std::vector<int> cur_trans_counts;
        FPNode* cur_prefix_node = cur_leaf_node->GetParent();
        // Traverse the tree from this node to root, but also skipping root
        while(cur_prefix_node && cur_prefix_node->GetParent())
        { 
            cur_trans.push_back(cur_prefix_node->GetName());
            cond_sup[cur_prefix_node->GetName()] += cur_leaf_node->GetCount();
            cur_prefix_node = cur_prefix_node->GetParent();
        }
        if (!cur_trans.empty())
            cond_trans.push_back(cur_trans);
            cond_trans_counts.push_back(cur_leaf_node->GetCount());
        cur_leaf_node = cur_leaf_node->GetSibling();
    }

    // The conditionally frequent items
    std::vector<std::string> cond_freq_items;

    // Remove nodes that do not have a minimum support
    for(auto& cond_sup_pair: cond_sup)
    {

        std::cout << cond_sup_pair.first << ": " << cond_sup_pair.second << std::endl; 
        // Check if item meets minimum support
        if(cond_sup_pair.second < MIN_SUPPORT)
        {
            // Remove this item from all the conditional transactions
            for(int i = 0; i < cond_trans.size(); ++i)
            {
                cond_trans[i].erase(std::remove(cond_trans[i].begin(), cond_trans[i].end(), cond_sup_pair.first));
                // If this removal made the transcation empty remove this as well
                if (cond_trans[i].empty())
                {
                    cond_trans.erase(cond_trans.begin() + i);
                    cond_trans_counts.erase(cond_trans_counts.begin() + i);
                }
            }
        }
        else
        {
            cond_freq_items.push_back(cond_sup_pair.first);
        }
    }
    

    // The total conditional patterns including this prefix
    Pattern cond_patterns;
    
    // Get total current node frequency
    FPNode* sib_node = cur_node;
    int total_freq = 0;
    while(sib_node)
    {
        total_freq += sib_node->GetCount();
        sib_node = sib_node->GetSibling();
    }
    std::cout << "CUR NODE: " << cur_node->GetName() << " " << total_freq << " VS " << cur_node->GetCount()<< std::endl;

    FreqItemset freq_itemset = {cur_prefix, total_freq};
    cond_patterns.push_back(freq_itemset);

    // The conditional patterns
    if (!cond_trans.empty())
    {
        for(std::string& item_name : cond_freq_items)
        {
            // Copy the current prefix
            std::vector<std::string> cond_prefix(cur_prefix);
            
            // Prepend the item name to prefix
            cond_prefix.insert(cond_prefix.begin(), item_name);

            // Create FP-Tree from all the conditional transactions, conditioned on cond_prefix
            FPTree cond_fptree;
            for(int i = 0; i < cond_trans.size(); ++i)
            {
                // Add subset of transaction if it contains element
                std::vector<std::string>::iterator it = std::find(cond_trans[i].begin(), cond_trans[i].end(), item_name);
                if (it != cond_trans[i].end())
                {
                    std::vector<std::string> cond_prefix_subset(it, cond_trans[i].end());
                    // Reverse so they are added in the correct order
                    std::reverse(cond_prefix_subset.begin(), cond_prefix_subset.end());
                    
                    // PROBLEM IS RIGHT HERE, YOU ARE NOT SETTING THE COUNTS CORRECT (count defaults to 1)
                    // NEED TO UPDATE THE COUNTS TO BE CORRECT HERE
                    cond_fptree.InsertItemset(cond_prefix_subset, cond_trans_counts[i]);
                }
            }
            // TODO: NEED TO FIX COST NOT CORRECT IN COND TREE
            cond_fptree.PrintTree();
            if (FPNode* next_node = cond_fptree.GetNodeLink(item_name))
            {
                Pattern cond_prefix_pattern = create_patterns(next_node, cond_prefix);
                cond_patterns.insert(cond_patterns.end(), cond_prefix_pattern.begin(), cond_prefix_pattern.end());
            }
        }
    }
    return cond_patterns;
}

Pattern fp_growth(FPTree& fptree)
{
    // Nodes previously added to the queue 
    std::unordered_set<std::string> nodes_seen;
    
    // Nodes to be visited
    std::queue<FPNode*> node_queue = get_leaf_nodes(fptree, nodes_seen);
    
    Pattern total_freq_itemsets;

    while(node_queue.size() > 0)
    {
        FPNode* cur_node = node_queue.front();
        node_queue.pop();

        // Add all the parents of this node (and its siblings) to node_queue
        FPNode* sib_node = cur_node;
        while(sib_node)
        {
            // If the parent of this nodes is not the root
            if (sib_node->GetParent()->GetParent())
            {
                std::string parent_name = sib_node->GetParent()->GetName();
                // Only add if not added to the queue before and not the root node
                if (nodes_seen.find(parent_name) == nodes_seen.end())
                {
                    nodes_seen.insert(parent_name);
                    node_queue.push(sib_node->GetParent());
                }
            }
            sib_node = sib_node->GetSibling();
        }
        Pattern patterns = create_patterns(cur_node, {cur_node->GetName()});
        total_freq_itemsets.insert(total_freq_itemsets.end(), patterns.begin(), patterns.end());
    }
    return total_freq_itemsets;
}

void write_patterns(Pattern total_freq_itemsets, std::string out_filename)
{
    std::ofstream out_file;
    out_file.open(out_filename);
    for (int i = total_freq_itemsets.size()-1; i >= 0 ; --i)
    {
        FreqItemset is = total_freq_itemsets[i];
        out_file << is.support << " ";
        for (int j = 0; j < is.items.size(); ++j)
        {
            out_file << is.items[j];
            if (j < is.items.size() - 1)
            {
                out_file << " ";
            }
        }
        if (i != 0)
            out_file << "\n";
    } 
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
        
        // Construct the FP-tree
        FPTree fptree;
        for (std::vector<std::string>& trans : freq_trans)
        {
            fptree.InsertItemset(trans);
        }

        // Print out the FP-tree
        //fptree.PrintTree();
        
        Pattern total_freq_itemsets = fp_growth(fptree);
        std::cout << total_freq_itemsets.size() << std::endl;

        // Sort the itemsets by support count
        std::sort(total_freq_itemsets.begin(), total_freq_itemsets.end());
        // for (auto& is: total_freq_itemsets)
        // {
        //     std::string item_str = std::to_string(is.support);
        //     for(auto& item : is.items)
        //     {
        //         item_str += " " + item;
        //     }
        //     std::cout << item_str << std::endl;
        // }
        
        write_patterns(total_freq_itemsets, fs::current_path() / ("output/pattern-" + std::to_string(i) + ".txt"));
    }
}