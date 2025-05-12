# selecting a few vectors among all the given vectors for the output
function Filter_fast2(rv, k, AD, mx, initial, interval, Ntot)


    sz = size(AD, 1)

    V = similar(zeros(mx, Ntot));

    sm_vec = similar(zeros(mx, k));

    AD = AD .* 1.0

    AD[diagind(AD, 0)] = AD[diagind(AD, 0)] .+ 0.001

    dg = sum(AD, dims = 1) .^ (-.5)

    I2 = 1:sz

    D = sparse(I2, I2, sparsevec(dg))

    on = ones(Int, length(rv))

    #sm_ot = rv - ((dot(rv, on) / dot(on, on)) * on)
     sm_ot = rv .- ((dot(rv, on) / dot(on, on)) .* on)

    sm = sm_ot ./ norm(sm_ot);

    count = 1

    for loop in 1:k

        sm = D * sm

        sm = AD * sm

        sm = D * sm

        #sm_ot = sm - ((dot(sm, on) / dot(on, on)) * on)

        #sm_norm = sm_ot ./ norm(sm_ot);

        #sm_vec[:, loop] = sm_norm[1:mx]

        sm_ot .-= (dot(sm, on) / dot(on, on)) .* on
        sm_ot ./= norm(sm_ot)
        sm_vec[:, loop] = sm_ot[1:mx]



        #sm_vec[:, loop] = sm[1:mx]

    end # for loop

    V = sm_vec[:, interval:interval:end]

    return V

end #end of function
