#include <vector>
#include <unordered_map>
#include <string>

#include "fpnode.h"

class FPTree
{
public:
    FPTree() {};

    // Insert itemset
    void InsertItemset(std::vector<std::string>& itemset, int init_count=1);

    FPNode& GetRoot();

    void AddNodeLink(FPNode* node);

    std::unordered_map<std::string, FPNode*>&  GetNodeLinks();
    
    FPNode* GetNodeLink(std::string);

    void PrintTree();


private:
    // The root node of the FPTree
    // NOTE: this does not actually represent an item
    FPNode root_node;

    // The root of the link of siblings for every item
    std::unordered_map<std::string, FPNode*> node_links;
};