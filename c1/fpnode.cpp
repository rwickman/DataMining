#include "fpnode.h"


FPNode::FPNode(std::string name) : name(name), sibling(nullptr), count(0) {}

std::string FPNode::GetName()
{
    return name;
}

void FPNode::IncCount()
{
    count++;
}

int FPNode::GetCount()
{
    return count;
}

FPNode* FPNode::GetSibling()
{
    return sibling;
}

void FPNode::SetSibling(FPNode* to_sibling)
{
    sibling = to_sibling;
}

std::vector<FPNode*>& FPNode::GetChildren()
{
    return children;
}

FPNode* FPNode::AddChild(std::string child_name)
{
    FPNode* child_node = new FPNode(child_name);
    
    child_node->IncCount();
    children.push_back(child_node);

    // TODO: Your problem is creating a pointer to an element in a vector
    return child_node;
}

FPNode* FPNode::GetChild(std::string child_name)
{
    for (auto& node : children)
    {
        if (child_name.compare(node->GetName()) == 0)
        {
            return node;
        }
    }
    return nullptr;
}

