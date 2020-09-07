#include <vector>
#include <string>

class FPNode
{
public:
    FPNode() : sibling(nullptr), count(0) { }
    FPNode(std::string name);

    // Get the name of this node
    std::string GetName();

    // Increment the count of this node
    void IncCount();
    
    // Get the count of the node
    int GetCount();

    // Get sibling
    FPNode* GetSibling();

    // Set the sibling of this node
    void SetSibling(FPNode* sibling);
    
    // Get the children of this node
    std::vector<FPNode*>& GetChildren();
    
    // Add child node
    FPNode* AddChild(std::string child_name);

    // Get the child with name if it exists
    FPNode* GetChild(std::string child_name);
    
private:
    std::string name;
    unsigned int count;
    FPNode* sibling;
    std::vector<FPNode*> children;
};