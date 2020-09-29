#include <iostream>
#include <cmath>
#include <stdexcept>

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

bool Cell::in_range(DataObj& obj, std::unordered_set<int> cur_dims)
{
    for (const int& dim : cur_dims)
    {
        // Check if this object should be in this cell
        if (obj.attrs[dim] < cell_dims[dim].min_val || 
        obj.attrs[dim] > cell_dims[dim].max_val || 
        (obj.attrs[dim] == cell_dims[dim].max_val && !cell_dims[dim].include_max))
        {
            return false;
        }
    }

    std::string cell_mins = "";
    std::string cell_maxs = "";
    std::string obj_attrs = "";
    for (const int& dim : cur_dims)
    {
        cell_mins += " " + std::to_string(cell_dims[dim].min_val);
        cell_maxs += " " + std::to_string(cell_dims[dim].max_val);
        obj_attrs += " " + std::to_string(obj.attrs[dim]); 
    }
    // std::cout << "CELL MINS " << cell_mins << std::endl;
    // std::cout << "CELL MAXS " << cell_maxs << std::endl;
    // std::cout << "OBJ ATTRIBUTES " << obj_attrs << std::endl; 
    return true;
}

void Cell::add(DataObj* obj)
{
    objs.push_back(obj);
}

size_t Cell::size()
{
    return objs.size();
}

void Cell::clear()
{
    objs.clear();
}

EntropySubspace::EntropySubspace(int divide_size) : divide_size(divide_size) {}

void EntropySubspace::setup(std::vector<Dim>& dims)
{
    //std::cout << "Setting Up Subspace Cells" << std::endl;
    //Create Cells
    //Divide the dimension into equal ranges
    num_dims = dims.size();
    //std::cout << "num_dims " << num_dims << std::endl;
    std::vector<std::vector<Dim>> divided_dims = divide_dims(dims);
    //std::cout << "divided_dims size " << divided_dims.size() << " divided_dims[0].size() " << divided_dims[0].size() << std::endl; 
    
    // Cells wil be created from every permutation of divided_dims
    cells = create_cells(divided_dims);
    //std::cout << "cells.size()" << cells.size() << std::endl;
    //std::cout << "Done Setting Up Subspace Cells" << std::endl;
}

std::unordered_set<int>
EntropySubspace::find_best_subspaces(
    std::vector<DataObj>& objs,
    int num_subspaces,
    float entropy_threshold)
{
    if (cells.size() == 0)
    {
        throw std::runtime_error("Must call EntropySubspace::setup before finding best subspace!");
    }

    // Use Aprori to find best subspaces
    std::vector<std::unordered_set<int>> old_dim_sets;
    std::vector<float> entropy_scores;
    for(int i = 0; i < num_subspaces; ++i)
    {
        //std::cout << "SUBSPACE " << i << std::endl;
        // 1. Join (Create DimSets)
        std::vector<std::unordered_set<int>> dim_sets;
        indices_comb(dim_sets, {}, 0, i+1, num_dims);
        for (auto dim_set : dim_sets)
        {
            for (auto el : dim_set)
            {
                std::cout << el << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "dim_sets.size() " << dim_sets.size() << " dim_sets[0].size() " << dim_sets[0].size() << std::endl;
        //std::cout << "COMB" << std::endl;

        // 2. Prune
        if (i > 0)
        {
            //std::cout << "PRUNE" << std::endl;
            // Check if every subset of dim_set is in old_dim_sets
            dim_sets = prune(dim_sets, old_dim_sets);
            //std::cout << "PRUNE DONE" << std::endl;
        }

        // 3. Counting
        //std::cout << "COUNTING" << std::endl;
        auto dim_entropy_pair = counting(objs, dim_sets, entropy_threshold);
        //std::cout << "COUNTING DONE" << std::endl;
        old_dim_sets = dim_entropy_pair.first;
        entropy_scores = dim_entropy_pair.second;
        
        for (auto score : entropy_scores)
        {
            std::cout << score << " ";
        }
        std::cout << std::endl;
    
    }
    std::unordered_set<int> best_subspace;
    float best_entropy_score;
    if (old_dim_sets.size() > 0)
    {
        best_subspace = old_dim_sets[0];
        best_entropy_score = entropy_scores[0];
        for (int i = 1; i < old_dim_sets.size(); ++i)
        {
            //std::cout << entropy_scores[i] << std::endl;
            if (entropy_scores[i] < best_entropy_score)
            {
                best_entropy_score = entropy_scores[i];
                best_subspace = old_dim_sets[i]; 
            }
        }
    }
    else
    {
        throw std::runtime_error("Did not find any best subspaces!");
    }
    std::cout << "BEST ENTROPY: " << best_entropy_score << std::endl;

    return best_subspace;
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

    //std::cout << "perm.size() " << perms.size() << "perm[0].size() " << perms[0].size() << std::endl;
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
    std::unordered_set<int> cur_dims)
{
    int count = 0;
    for (auto& obj : objs)
    {
        int i = 0;
        for (auto& cell : cells)
        {
            if(cell.in_range(obj, cur_dims))
            {
                //std::cout << "IN CELL " << i << std::endl;
                // TODO: VERIFY ALL THE DOs ARE ADDED
                cell.add(&obj);
                count += 1;
                break;
            }
            ++i;
        }
    }
    //std::cout << "COUNT " << count << " NUM OBJS " << objs.size() << std::endl; 
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

std::pair<std::vector<std::unordered_set<int>>, std::vector<float>>
EntropySubspace::counting(
    std::vector<DataObj>& objs,
    std::vector<std::unordered_set<int>>& dim_sets,
    float entropy_threshold)
{
    //assign_objs(objs, dim_sets);
    std::vector<std::unordered_set<int>> updated_dim_sets;
    std::vector<float> entropy_scores;
    for (auto& dim_set : dim_sets)
    {
        // Assign every DataObj to a cell
        assign_objs(objs, dim_set);

        // Calculate total entropy
        float total_entropy = calc_total_entropy(objs.size());
        
        // See if it has a minimal entropy
        if (total_entropy <= entropy_threshold)
        {
            updated_dim_sets.push_back(dim_set);
            entropy_scores.push_back(total_entropy);
        }
        reset();
    }
    return {updated_dim_sets, entropy_scores};
}

float EntropySubspace::calc_total_entropy(int num_objs)
{
    float total_entropy = 0;
    //std::cout << "ITERATING OVER CELLS" << std::endl;
    for (auto& cell : cells)
    {
        // Calculate cell density
        float cell_density = static_cast<float>(cell.size()) / static_cast<float>(num_objs);
        if (cell.size() > 0)
        {
            //std::cout << "CELL DENSITY " << cell_density << " size: " << cell.size() << " NUM OBJ " << num_objs<< std::endl;
        }
        
        if (cell_density > 0)
        {
            //std::cout << "CUR ENTROPY " << cell_density * std::log2(cell_density) << std::endl;
            total_entropy += cell_density * std::log2(cell_density);
        }
        
    }
    //std::cout << "TOTAL ENTROPY " << -total_entropy << std::endl;
    return -total_entropy;
}

void EntropySubspace::reset()
{
    //std::cout << "Clearing Cells" << std::endl;
    for (auto& cell : cells)
    {
        cell.clear();
        if (cell.size() > 0)
        {
            throw std::runtime_error("STILL ELEMENTS LEFT IN CELL");
        }
    }
}