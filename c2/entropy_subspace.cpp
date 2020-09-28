#include <iostream>

#include "entropy_subspace.hpp"

// Create every permutation of indices
void indices_perm(std::vector<std::vector<int>>& perms, std::vector<int> cur, int n, int r)
{
    if (cur.size() >= r)
    {
        perms.push_back(cur);
        return;
    }
  
    for (int i = 0; i < n; i++)
    {
        cur.push_back(i);
        indices_perm(perms, cur, n, r);
        cur.pop_back();
    }
}


// Create every combination of n choose r
void indices_comb(std::vector<std::unordered_set<int>>& nums, std::vector<int> cur, int start, int r, int n)
{
    if (cur.size() == r)
    {
        nums.push_back(std::unordered_set<int>(cur.begin(), cur.end()));
        return;
    }
    
    for (int i = start; i < n; ++i)
    {
        cur.push_back(i);
        indices_comb(nums, cur, i+1, r, n);
        cur.pop_back();
    }
}

Cell::Cell(std::vector<Dim> cell_dims) : cell_dims(cell_dims) {}

bool Cell::in_range(DataObj& obj, std::vector<int> cur_dims)
{
    for (int& dim : cur_dims)
    {
        // Check if this object should be in this cell
        if (obj.attrs[dim] < cell_dims[dim].min_val || 
        obj.attrs[dim] > cell_dims[dim].max_val || 
        (obj.attrs[dim] == cell_dims[dim].max_val && !cell_dims[dim].include_max))
        {
            return false;
        }
    }
    return true;
}

void Cell::add(DataObj* obj)
{
    objs.push_back(obj);
}

void Cell::clear()
{
    objs.clear();
}

EntropySubspace::EntropySubspace(int divide_size) : divide_size(divide_size) {}

void EntropySubspace::setup(std::vector<Dim>& dims)
{
    std::cout << "Setting Up Subspace Cells" << std::endl;
    //Create Cells
    //Divide the dimension into equal ranges
    num_dims = dims.size();
    std::cout << "num_dims " << num_dims << std::endl;
    std::vector<std::vector<Dim>> divided_dims = divide_dims(dims);
    std::cout << "divided_dims size " << divided_dims.size() << " divided_dims[0].size() " << divided_dims[0].size() << std::endl; 
    
    // Cells wil be created from every permutation of divided_dims
    cells = create_cells(divided_dims);
    std::cout << "cells.size()" << cells.size() << std::endl;
    std::cout << "Done Setting Up Subspace Cells" << std::endl;
}

std::vector<int> EntropySubspace::find_best_subspaces(std::vector<DataObj>& objs, int num_subspaces)
{
    if (cells.size() == 0)
    {
        throw "Must call EntropySubspace::setup before finding best subspace!";
    }

    // Use Aprori approach to find best subspaces
    std::vector<int> best_subspaces;

    std::vector<std::unordered_set<int>> old_dim_sets;
    for(int i = 0; i < num_subspaces; ++i)
    {
        // 1. Join (Create DimSets)
        std::vector<std::unordered_set<int>> dim_sets;
        indices_comb(dim_sets, {}, 0, i+1, num_dims);

        // 2. Prune
        if (i > 0)
        {
            // Check if every subset of dim_set is in old_dim_sets
            dim_sets = prune(dim_sets, old_dim_sets);
        }

        // 3. 
    }
    return best_subspaces;
}


std::vector<std::vector<Dim>> EntropySubspace::divide_dims(std::vector<Dim>& dims)
{
    std::vector<std::vector<Dim>> divided_dims;
    for (auto& dim : dims)
    {
        std::vector<Dim> cur_dim_range;
        float val_range = dim.max_val - dim.min_val;
        float part_range = val_range * (1/divide_size); 
        for(int i = 0; i < divide_size; ++i)
        {
            Dim dim_part;
            dim_part.min_val = part_range * i + dim.min_val;
            dim_part.max_val = part_range * (i+1) + dim.max_val;
            if (i == divide_size - 1)
            {
                dim_part.include_max = true;
            }
            else
            {
                dim_part.include_max = false;
            }
            cur_dim_range.push_back(dim_part);
        }
        divided_dims.push_back(cur_dim_range);
    }
    return divided_dims;    
}


std::vector<Cell> EntropySubspace::create_cells(std::vector<std::vector<Dim>>& divided_dims)
{
    std::vector<std::vector<int>> perms;

    // Create permutation of indices
    indices_perm(perms, {}, divide_size, divided_dims.size());
    std::vector<Cell> cells;

    std::cout << "perm.size() " << perms.size() << "perm[0].size() " << perms[0].size() << std::endl;
    // Create the cells
    for(auto& perm : perms)
    {
        // Create a cell using a unique permutation of the dimensions 
        std::vector<Dim> cell_dims;
        for(int i = 0; i < divided_dims.size(); i++)
        {
            cell_dims.push_back(divided_dims[i][perm[i]]);
        }
        cells.push_back(Cell(cell_dims));
    }
    return cells;
}

void EntropySubspace::assign_objs(
    std::vector<DataObj>& objs,
    std::vector<Cell> cells,
    std::vector<int> cur_dims)
{
    int count = 0;
    for (auto& obj : objs)
    {
        for (auto& cell : cells)
        {
            if(cell.in_range(obj, cur_dims))
            {
                // TODO: VERIFY ALL THE DOs ARE ADDED
                cell.add(&obj);
                count += 1;
                break;
            }
        }
    }
    std::cout << "COUNT " << count << " NUM OBJS " << objs.size() << std::endl; 
}

std::vector<std::unordered_set<int>> 
EntropySubspace::prune(
    std::vector<std::unordered_set<int>>& dim_sets,
    std::vector<std::unordered_set<int>>& old_dim_sets)
{
    // The set of dim_sets has all n-1 subsets in old_dim_sets
    std::vector<std::unordered_set<int>> pruned_dim_sets;
    for (auto& dim_set : dim_sets)
    {
        // The number of subsets in old_dim_sets
        int num_subsets = 0;
        for(auto& old_dim_set : old_dim_sets)
        {
            // Get the intersect betewen the two sets
            int intersect_size = 0;
            for (const auto& el : old_dim_set)
            {
                if (dim_set.find(el) != dim_set.end())
                {
                    intersect_size += 1;
                }
            }
            
            // Check if every element of old_dim_set is in dim_set
            if (intersect_size == old_dim_set.size())
            {
                num_subsets += 1;
            }
        }

        if (num_subsets == dim_set.size())
        {
            pruned_dim_sets.push_back(dim_set);
        }
    }
    return pruned_dim_sets;
}