#include <vector>
#include <unordered_set>

#include "dataobj.hpp"

class Cell
{
public:
  Cell(std::vector<Dim> cell_dims);

  // Should the DataObj be included in this cell based on only dimensions in  
  bool in_range(DataObj& obj, std::vector<int> cur_dims);

  // Add DataObj to this cell
  void add(DataObj* obj);

  // Remove all DataObjs
  void clear();

private:
  // The dim of range this cell occupies
  std::vector<Dim> cell_dims;
  
  // The DataObj in this cell
  std::vector<DataObj*> objs;

};

class EntropySubspace
{
public:
  EntropySubspace(int divide_size);

  void setup(std::vector<Dim>& dims);

  std::vector<int> find_best_subspaces(std::vector<DataObj>& objs, int num_subspaces = 3);


private:
  // Divide every dimension into equal parts
  std::vector<std::vector<Dim>> divide_dims(std::vector<Dim>& dims);

  std::vector<Cell> create_cells(std::vector<std::vector<Dim>>& divided_dims);

  void assign_objs(
    std::vector<DataObj>& objs,
    std::vector<Cell> cells,
    std::vector<int> cur_dims);

  // Prune sets from dim_sets that are not in old_dim_sets
  std::vector<std::unordered_set<int>> 
  prune(
    std::vector<std::unordered_set<int>>& dim_sets,
    std::vector<std::unordered_set<int>>& old_dim_sets);

private:
  std::vector<Cell> cells;
  int divide_size;
  int num_dims;


};