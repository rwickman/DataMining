#include <iostream>
#include <stack>

#include "fptree.h"

// FPTree::FPTree() 
// {

// }

void FPTree::InsertItemset(std::vector<std::string>& itemset)
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
        next_node->IncCount();
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
        cur_node = cur_node->AddChild(itemset[cur_idx++]);
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
        
        if (prev == node)
        {
            std::cout << "prev is equal to node!!!" << std::endl;
        }else
        {
            std::cout << "prev is NOT equal to node!!!" << std::endl;
        }
        
        prev->SetSibling(node);
    }
}

void FPTree::PrintTree()
{
    // Uses DFS to print out the FPTree
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
        std::string temp = cur.second + cur.first->GetName();
        temp += ": ";

        std::cout <<  cur.second << cur.first->GetCount() << ":" << cur.first->GetName() << std::endl;
    }
}