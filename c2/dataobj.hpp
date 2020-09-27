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

struct KMeanData
{
  int min_k;
  int max_k;
  std::vector<DataObj> objs;
};

#endif