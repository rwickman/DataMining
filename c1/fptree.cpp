#include <iostream>
#include <stack>

#include "fptree.h"

void FPTree::InsertItemset(std::vector<std::string>& itemset, int init_count)
{
    // Need to iterate over tree and incrementing the count until
    // a node is not found. Then, a new path of FPNodes needs to be created
    if (itemset.size() <= 0)
    {
        return;
    }

    int cur_idx = 0;
    FPNode* cur_node = &root_node;
    FPNode* next_node = root_node.GetChild(itemset[cur_idx]);;
    
    // Increment counts of nodes that already exist in the tree
    while (next_node)
    {
        next_node->AddCount(init_count);
        if (++cur_idx < itemset.size())
        {
            cur_node = next_node;
            next_node = cur_node->GetChild(itemset[cur_idx]);
        }
        else 
            return;
    }

    // Add new nodes to the tree
    while(cur_idx < itemset.size())
    {
        cur_node = cur_node->AddChild(itemset[cur_idx++], init_count);

        // Check if not link exists, and if it doesn't create
        // If it does, add sibling link
        AddNodeLink(cur_node);
    }
}

FPNode& FPTree::GetRoot()
{
    return root_node;
}

void FPTree::AddNodeLink(FPNode* node)
{
    std::unordered_map<std::string, FPNode*>::const_iterator it = node_links.find(node->GetName());
    if (it == node_links.end())
    {
        //std::cout << "Setting parent node link " << node->GetName() << std::endl;
        node_links[node->GetName()] = node;
    }
    else
    {
        FPNode* prev = it->second;
        FPNode* sibling;
        bool printed;
        while ((sibling = prev->GetSibling())) {
            prev = sibling;
        }
        prev->SetSibling(node);
    }
}

std::unordered_map<std::string, FPNode*>& FPTree::GetNodeLinks()
{
    return node_links;
}

FPNode* FPTree::GetNodeLink(std::string node_name)
{
    return node_links[node_name];
}

void FPTree::PrintTree()
{
    // Uses DFS to print out the FP-tree
    std::stack<std::pair<FPNode*, std::string>> s;
    s.push({&root_node, ""});
    std::pair<FPNode*, std::string> cur;
    while(s.size() > 0)
    {
        cur = s.top();
        s.pop();
        for (FPNode* child : cur.first->GetChildren())
        {
            //std::cout << "Parent: " << cur.first->GetName() << " Child: " << child->GetName() << std::endl; 
            s.push({child, cur.second + " "});
        }
        std::cout <<  cur.second  << cur.first->GetName() << ":" << cur.first->GetCount() << std::endl;
    }
}