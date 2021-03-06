#include <vector>
#include <string>

class FPNode
{
public:
    FPNode() : sibling(nullptr), parent(nullptr), count(0) { }
    FPNode(std::string name);

    // Get the name of this node
    std::string GetName();

    // Increment the count of this node
    void IncCount();
    
    // Add frequency to count
    void AddCount(int freq);

    // Get the count (i.e., relative frequency) of thjs node
    int GetCount();

    // Get sibling of this node
    FPNode* GetSibling();

    // Set the sibling of this node
    void SetSibling(FPNode* sibling);
    
    // Get the children of this node
    std::vector<FPNode*>& GetChildren();
    
    // Add child node
    FPNode* AddChild(std::string child_name, int init_count);

    // Get the child with name if it exists
    FPNode* GetChild(std::string child_name);
    
    // Get the parent node
    FPNode* GetParent();


private:
    std::string name;
    unsigned int count;
    FPNode* sibling;
    FPNode* parent;
    std::vector<FPNode*> children;

};