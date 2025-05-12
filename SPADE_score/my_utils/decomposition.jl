function decomposition(Lp, L)

    ar_mat = Any[]

    fdnz = findnz(triu(Lp,1))

    r1 = fdnz[1]
    c1 = fdnz[2]

    ar = Any[]

    for ii = 1:length(r1)
        nd1 = [r1[ii], c1[ii]]
        push!(ar, sort(nd1))
    end

    ar_org = copy(ar)

    W = fdnz[3].*-1.0

    ar_new = Any[]

    idx_mat = Any[]

    Neff = zeros(Float64, mx_func(ar))

    #W = ones(Float64, length(ar))

    @inbounds for loop = 1:L

        mx = mx_func(ar)

        ## star expansion
        A = StarW(ar, W)

        ## computing the smoothed vectors
        initial = 0

        SmS = 100

        interval = 20

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
        #global Evec = sum(Eratio, dims=2) ./ size(SV,2)
        E2 = sort(Eratio, dims=2, rev=true)
        Evec = E2[:, 1]

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
        #global Pos = [4,3,2,1]

        ## Hyperedge contraction
        flag = falses(mx)

        flagE = falses(length(ar))

        val = 1

        idx = zeros(Int, mx)

        Neff_new = zeros(Float64, 0)

        @inbounds for ii = 1:Nsample

            nd = ar[Pos[ii]]

            fg = flag[nd]

            fd1 = findall(x->x==0, fg)

            if length(fd1) > 1

                nd = nd[fd1]

                flagE[Pos[ii]] = 1

                idx[nd] .= val

                flag[nd] .= 1

                val +=1

                ## creating the super node weights
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

        ## Keeping the edge weights of non unique elements
        fdnu = unique(z -> ar_new[z], 1:length(ar_new))
        W2 = W[fdnu]


        ## removing the repeated hyperedges
        ar_new = ar_new[fdnu]

        ### removing hyperedges with cardinality of 1
        HH = INC3(ar_new)
        ss = sum(HH, dims=2)
        fd1 = findall(x->x==1, ss[:,1])
        deleteat!(ar_new, fd1)
        deleteat!(W2,fd1)

        ar = ar_new

        Neff = Neff_new

        W = W2

    end #end for loop


    return ar, ar_mat, idx_mat, ar_org
end
