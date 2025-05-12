## Only graph coarsening, without sparsification

# using Laplacian matrix as input]

include("HyperNodes.jl")
include("Clique_sm.jl")
include("Unmapping.jl")
include("Filter_fast.jl")
include("Mapping_fast.jl")
include("decomposition.jl")
include("sparsification.jl")
include("StarW.jl")
include("Star.jl")
include("h_score3.jl")
include("mx_func.jl")
include("INC3.jl")
include("Filter_fast.jl")

using SparseArrays
using LinearAlgebra
using Clustering
using NearestNeighbors
using Distances
using Laplacians
using Arpack
using Statistics
using DelimitedFiles
using StatsBase
using Laplacians#master
using Random
using MatrixMarket


## spL is the sparsification level;
# the larger spL results in more aggressive sparsification
# you can start with spL = 4 and increase it if the results are good

## idx is the output node indices
# L is the Laplacian matrix corresponding to the weighted adjacency matrix


function HyperEF1(L, spL)
#=
    if grass == 0
        m, n = L.shape
        colPtr = Int[i+1 for i in PyArray(L."indptr")]
        rowVal = Int[i+1 for i in PyArray(L."indices")]
        nzVal = Vector{Float64}(PyArray(L."data"))
        L = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    else
        rowval = Int[i+1 for i in PyArray(L."row")]
        colval = Int[i+1 for i in PyArray(L."col")]
        Val = Vector{Float64}(PyArray(L."data"))
        L = sparse(rowval, colval, Val)

    end
=#
    ## Decomposition
    println("------------Decomposition Time -----------")
    @time ar, ar_mat, idx_mat, ar_org = decomposition(L, spL)

    #mapper
    idx = 1:maximum(idx_mat[end])

    for ii = spL:-1:1

        idx = idx[idx_mat[ii]]

    end # for ii

    ## detecting inter-cluster edges

    rr = findnz(triu(L,1))[1]
    cc = findnz(triu(L,1))[2]
    mx = max(maximum(rr), maximum(cc))

    dictCL = Dict{Array{Int64,1}, Array{Int64,1}}()

    ICE = falses(length(rr))

    for ii = 1:length(rr)

        key = sort([idx[rr[ii]], idx[cc[ii]]])

        if key[1] != key[2]
            vals = get!(Array{Int64,1}, dictCL, key)
            push!(vals, ii)
            ICE[ii] = 1
        end

    end # for ii

    fdICE = findall(x->x==1, ICE)

    EGS = hcat(rr[fdICE], cc[fdICE])


    R = append!(copy(rr[fdICE]), copy(cc[fdICE]))

    C = append!(copy(cc[fdICE]), copy(rr[fdICE]))

    V = ones(Int, length(R))

    A_ICE = sparse(R, C, V, mx, mx)



    return idx, A_ICE


end # function
