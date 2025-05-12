

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


## SW is the switch to enable the sparsification;
# if SW=0, there is no sparsification (edge elimination) technique


## spL is the sparsification level;
# the larger spL results in more aggressive sparsification
# you can start with spL = 4 and increase it if the results are good


## scL is the spectral coarsening ratio
# larger scL means more node reduction


## idx is the output node indices
# A is the weighted adjacency matrix


function HyperEF_SP(A, SW, spL, scL)

    if SW == 1

        fdnz = findnz(triu(A,1))
        W = fdnz[3]

        ## Decomposition
        println("------------Decomposition Time -----------")
        @time ar, ar_mat, idx_mat_SP, ar_org = decomposition(A, spL)
        Lmat = length(ar_mat)

        NN = [1, Lmat]


        ## sparsification using spanning tree
        println("------------Sparsification Time -----------")
        @time arF, V, Eidx = sparsification(NN, ar, idx_mat_SP, ar_mat)



        ## Adding Inter-cluster edges
        dict = Dict{Any, Any}()
        println("------------ICE Time -----------")
        count = 0
        #Eidx = zeros(Int, 0)
        @time for jj =1:length(ar_org)

            nd = ar_org[jj]

            if V[nd[1]] != V[nd[2]]

                count+=1

                push!(arF, sort(nd))

                append!(Eidx, jj)

            end

        end

        fdU = unique(z -> arF[z], 1:length(arF))

        arF = arF[fdU]

        W_new = W[Eidx[fdU]]

        ms = length(arF)

        println("--------------------- Sparsification----------------------")
        println("Size of each cluster = ", round.(Int, mx_func(arF) / maximum(V)))

        println("Percentage of eliminated edges = ", round.(Int,(mg - ms)/mg*100),"%")

    else

        ## Coarsening

        #arF, W_new = mtx2arW(Inp)
        fdnz = findnz(triu(A,1))
        arF = Any[]
        rr = fdnz[1]
        cc = fdnz[2]
        W_new = fdnz[3]

        for ii = 1:length(rr)
            nd1 = [rr[ii], cc[ii]]
            push!(arF, sort(nd1))
        end

    end

    ar = arF

    W = W_new

    ar_new = Any[]

    mx = mx_func(ar)

    idx_mat = Any[]

    Neff = zeros(Float64, mx)
    println("------------Coarsening Time -----------")

    @time @inbounds for loop = 1:scL

        mx = mx_func(ar)

        ## star expansion
        A = StarW(ar, W)

        ## computing the smoothed vectors
        initial = 0

        SmS = 100

        interval = 1

        Nrv = 1

        RedR = 1

        Nsm = Int((SmS - initial) / interval)

        Ntot = Nrv * Nsm

        Qvec = zeros(Float64, 0)

        Eratio = zeros(Float64, length(ar), Ntot)

        SV = zeros(Float64, mx, Ntot)

        for ii = 1:Nrv

            sm = zeros(mx, Nsm)

            Random.seed!(1); randstring()

            rv = (rand(Float64, size(A, 1), 1) .- 0.5).*2

            sm = Filter_fast(rv, SmS, A, mx, initial, interval, Nsm)

            SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm

        end

        ## Make all the smoothed vectors orthogonal to each other
        QR = qr(SV)

        SV = Matrix(QR.Q)

        ## Computing the ratios using all the smoothed vectors
        for jj = 1:size(SV, 2)

            hscore = h_score3(ar, SV[:, jj])

            Eratio[:, jj] = hscore ./ sum(hscore)

        end #for jj

        ## Approximating the effective resistance of hyperedges by selecting the top ratio
        Evec = sum(Eratio, dims=2) ./ size(SV,2)

        # Adding the effective resistance of super nodes from previous levels
        @inbounds for kk = 1:length(ar)

            nd2 = ar[kk]

            Evec[kk] = Evec[kk] + sum(Neff[nd2])

        end

        ## Normalizing the ERs
        P = Evec ./ maximum(Evec)

        ## Choosing a ratio of all the hyperedges
        Nsample = round(Int, RedR * length(ar))

        PosP = sortperm(P[:,1])

        ## Increasing the weight of the hyperedges with small ERs
        W[PosP[1:Nsample]] = W[PosP[1:Nsample]] .* (1 .+  1 ./ P[PosP[1:Nsample]])

        ## Selecting the hyperedges with higher weights for contraction
        Pos = sortperm(W, rev=true)

        ## Hyperedge contraction
        flag = falses(mx)

        val = 1

        idx = zeros(Int, mx)

        Hcard = zeros(Int, 0)

        Neff_new = zeros(Float64, 0)

        @inbounds for ii = 1:Nsample

            nd = ar[Pos[ii]]

            fg = flag[nd]

            fd1 = findall(x->x==0, fg)

            if length(fd1) > 1

                nd = nd[fd1]

                idx[nd] .= val

                flag[nd] .= 1

                append!(Hcard, length(ar[ii]))

                val +=1

                new_val = Evec[Pos[ii]] + sum(Neff[nd])

                append!(Neff_new, new_val)

            end # endof if

        end #end of for ii

        ## indexing the isolated nodes
        fdz = findall(x-> x==0, idx)

        fdnz = findall(x-> x!=0, idx)

        V = vec(val:val+length(fdz)-1)

        idx[fdz] = V

        ## Adding the weight od isolated nodes
        append!(Neff_new, Neff[fdz])

        push!(idx_mat, idx)

        ## generating the coarse hypergraph
        ar_new = Any[]

        @inbounds for ii = 1:length(ar)

            nd = ar[ii]

            nd_new = unique(idx[nd])

            push!(ar_new, sort(nd_new))

        end #end of for ii


        ## removing the repeated hyperedges
        fdU = unique(z -> ar_new[z], 1:length(ar_new))

        ar_new = ar_new[fdU]
        W_new = W[fdU]

        #ar_new123 = unique(ar_new)


        ### removing hyperedges with cardinality of 1
        HH = INC3(ar_new)
        ss = sum(HH, dims=2)
        fd1 = findall(x->x==1, ss[:,1])
        deleteat!(ar_new, fd1)
        deleteat!(W_new, fd1)

        W = W_new

        ar = ar_new

        Neff = Neff_new


    end #end for loop

    #mapper
    idx = 1:maximum(idx_mat[end])

    for ii = scL:-1:1

        idx = idx[idx_mat[ii]]

    end # for ii

    return idx


end # function
