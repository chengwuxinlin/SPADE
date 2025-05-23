using LinearAlgebra
using LinearMaps
using MAT
using SparseArrays
using Arpack

using PyCall, SparseArrays

function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    return PyCall.pyjlwrap_new(B)
end


function main(PyX, PyY, k::Int64)
 
    rowval = Int[i+1 for i in PyArray(PyX."row")]
    colval = Int[i+1 for i in PyArray(PyX."col")]
    Val = Vector{Float64}(PyArray(PyX."data"))
    X = sparse(rowval, colval, Val)

    rowval = Int[i+1 for i in PyArray(PyY."row")]
    colval = Int[i+1 for i in PyArray(PyY."col")]
    Val = Vector{Float64}(PyArray(PyY."data"))
    Y = sparse(rowval, colval, Val)



    (Λ, V) = eigs(X, Y, nev=k, tol=1e-6, which=:LM)

    return Λ, V
end


function main_copy(PyX, PyY, k::Int64)
    m, n = PyX.shape
    colPtr = Int[i+1 for i in PyArray(PyX."indptr")]
    rowVal = Int[i+1 for i in PyArray(PyX."indices")]
    nzVal = Vector{Float64}(PyArray(PyX."data"))
    X = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)

    colPtr = Int[i+1 for i in PyArray(PyY."indptr")]
    rowVal = Int[i+1 for i in PyArray(PyY."indices")]
    nzVal = Vector{Float64}(PyArray(PyY."data"))
    Y = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)



    (Λ, V) = eigs(X, Y, nev=k, tol=1e-6, which=:LM)

    return Λ, V
end


function not_main(PyX, k::Int64)
 
    rowval = Int[i+1 for i in PyArray(PyX."row")]
    colval = Int[i+1 for i in PyArray(PyX."col")]
    Val = Vector{Float64}(PyArray(PyX."data"))
    X = sparse(rowval, colval, Val)

    (Λ, V) = eigs(X, nev=k, which=:SM,maxiter=500000)

    return Λ, V
end

