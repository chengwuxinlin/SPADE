function sparsification(NN, ar, idx_mat_SP, ar_mat)

    mx = mx_func(ar)

    V = collect(1:mx)

    arF = Array{Int32}[]

    ## selected edges
    ES = zeros(Int, 0)

    for ii = 1:length(NN)-1

        N2 = NN[end - ii + 1]

        mx = mx_func(ar)

        V = collect(1:mx)

        if ii == length(NN) - 1

            LB = 1

        else

            LB = NN[end - ii]+1

        end

        for jj = N2: -1:LB

            idx1 = idx_mat_SP[jj]

            V = V[idx1]

        end # for jj

        #V = [5,5,4,4,3,3,2,2,1,1]

        ar = ar_mat[LB]

        NH = HyperNodes(ar)


        dict2 = Dict{Any, Any}()

        for jj =1:length(V)

            vals = get!(Vector{Int}, dict2, V[jj])

            push!(vals, jj)

        end # for jj

        KS = collect(keys(dict2))

        VL = collect(values(dict2))

        LN = zeros(Int, 0)
        for ii = 1:length(VL)
            append!(LN, length(VL[ii]))
        end

        KSp = sortperm(KS)

        VL = VL[KSp]

        for kk =1:length(VL)

            fd1 = VL[kk]

            if length(fd1) > 1

                NH2 = NH[fd1]

                EG = unique(vcat(NH2...))

                Eidx = zeros(Int, 0)

                for mm = 1:length(EG)

                    nd = ar[EG[mm]]

                    Vnd = V[nd]

                    if (Vnd[1]==kk && Vnd[2]==kk)

                        append!(Eidx, EG[mm])

                    end # if

                end # for mm

                ## spanning tree
                ar_sub = ar[Eidx]

                ar_map, V2 = Mapping_fast(ar_sub)

                Msub = Clique_sm(ar_map)

                Tree = akpwU(Msub)

                arF = Unmapping(Tree, V2, arF)

                ## finding the selected edges
                fds = findall(x->in(x, arF), ar_sub)
                append!(ES, Eidx[fds])

            else

                nh1 = NH[fd1]

                for loop = 1:length(nh1)

                    egs = nh1[loop]
                    append!(ES, egs)

                    for ll = 1:length(egs)

                        push!(arF, sort(ar[egs[ll]]))

                    end # for ll

                end # for loop

            end # end of if

        end # end of for kk

    end # for ii

    #println("arF = ", arF)

    return arF, V, ES


end # end of function
