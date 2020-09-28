#include <vector>

#ifndef DATAOBJ
#define DATAOBJ
struct DataObj
{
  // Vector of attributes
  std::vector<float> attrs;

  // ID
  unsigned int id;
};

// A dimension
struct Dim
{
  Dim() {}
  Dim(float min_val, float max_val, int id) : 
    min_val(min_val),
    max_val(max_val),
    id(id) {}
  
  // The index of this dimension
  unsigned int id;

  // The min value found in a dimension
  float min_val;

  // The max value found in a dimension
  float max_val;
  
  // Should the min be included in the range
  bool include_min;

  // Should the max be included in the range
  bool include_max;
};

struct KMeanData
{
  int min_k;
  int max_k;
  std::vector<DataObj> objs;
  std::vector<Dim> dims;
};
#endif