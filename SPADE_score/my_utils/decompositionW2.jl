function decompositionW2(ar, L, RedR, W)

    ar_new = Any[]

    idx_mat = Any[]

    Neff = zeros(Float64, mx_func(ar))

    #W = ones(Float64, length(ar))

    ar_mat = Any[]

    Emat = Any[]

    ## scale edge weights W
    W = W ./ maximum(W)
    W = 100 .* (W)

    @inbounds for loop = 1:L

        push!(ar_mat, ar)

        mx = mx_func(ar)

        ## star expansion
        A = StarW(ar, W)

        ## computing the smoothed vectors
        initial = 0

        SmS = 300

        interval = 20

        Nrv = 1

        Nsm = Int((SmS - initial) / interval)

        Ntot = Nrv * Nsm

        Qvec = zeros(Float64, 0)

        Eratio = zeros(Float64, length(ar), Ntot)

        SV = zeros(Float64, mx, Ntot)

        for ii = 1:Nrv

            sm = zeros(mx, Nsm)

            Random.seed!(1); randstring()

            rv = (rand(Float64, size(A, 1), 1) .- 0.5).*2

            sm = Filter_fast2(rv, SmS, A, mx, initial, interval, Nsm)

            SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm

        end

        ## Make all the smoothed vectors orthogonal to each other
        QR = qr(SV)

        SV = Matrix(QR.Q)

        ## Computing the ratios using all the smoothed vectors
        for jj = 1:size(SV, 2)

            hscore = h_scoreW(ar, SV[:, jj], W)

            Eratio[:, jj] = hscore ./ sum(hscore)

        end #for jj

        ## Approximating the effective resistance of hyperedges
        ## by selecting the top ratio
        E2 = sort(Eratio, dims=2, rev=true)
        Evec = E2[:, 1]

        #Har = hcat(ar...)

        #MS = sparse(Har[1,:], Har[2,:], Evec)

        push!(Emat, Evec)

        # Adding the effective resistance of super nodes from previous levels
        @inbounds for kk = 1:length(ar)

            nd2 = ar[kk]

            Evec[kk] = Evec[kk] + sum(Neff[nd2])

        end

        ## Normalizing the ERs
        P = Evec ./ maximum(Evec)

        # Scaling Evec
        Evec = Evec ./ maximum(Evec)
        Evec = Evec * 1 * loop

        # scale the P values
        #P = 0.6*P .+ 0.3
        #println("MIN P = ", minimum(P))

        ## Choosing a ratio of the hyperedges for contraction
        Nsample = round(Int, RedR * length(ar))

        PosP = sortperm(P[:,1])

        ## Increasing the weight of the hyperedges with small ERs
        W[PosP[1:Nsample]] = W[PosP[1:Nsample]] .* (1 .+  1 ./ P[PosP[1:Nsample]])

        ## Selecting the hyperedges with higher weights for contraction
        Pos = sortperm(W, rev=true)

        ## low-ER diameter clustering which starts by contracting
        # the hyperedges with low ER diameter
        flag = falses(mx)

        val = 1

        idx = zeros(Int, mx)

        Neff_new = zeros(Float64, 0)

        @inbounds for ii = 1:Nsample

            nd = ar[Pos[ii]]

            fg = flag[nd]

            fd1 = findall(x->x==0, fg)

            if length(fd1) > 1

                nd = nd[fd1]

                idx[nd] .= val

                # flag the discovered nodes
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
        ## Adding the weight of isolated nodes
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

        # scaling W
        W = W ./ maximum(W)
        W = 100 *(loop+1) .* (W)

    end #end for loop


    return ar, idx_mat, ar_mat, Emat
end
