cd("/Users/aliaghdaei/Desktop/my_utils/")

include("HyperEF1.jl")
include("SPF.jl")

using LightGraphs
using TickTock
using SparseArrays

Inp = "LF1.mtx"
Inp = "fe_4elt2.mtx"
Inp = "M2W.mtx"

io = open(Inp, "r")
rr = zeros(Int, 0)
cc = zeros(Int, 0)
vv = zeros(Float64, 0)
while !eof(io)
    ln = readline(io)
    sp = split(ln)
    r = parse(Int, sp[1])
    c = parse(Int, sp[2])
    #v = parse(Int, sp[3])
    append!(rr, r)
    append!(cc, c)
    #append!(vv, v)
end

vv = ones(Float64, length(rr))

W = copy(vv)

mg = length(rr)

R = append!(copy(rr), copy(cc))

C = append!(copy(cc), copy(rr))

V = append!(copy(vv), copy(vv))

A = sparse(R, C, V)

L = lap(A)




L = 4
ICr = 1
AS, A_ICE = SPF(A, L, ICr)



#=
L = lap(A)


EV = eigs(L; nev=10, which =:SM)

EVs = EV[2]

ev2 = EVs[:,2]

ev3 = EVs[:,3]

ev9 = EVs[:,end]

# kn = kmeans(ev9', 9)
kn = kmeans(EVs', 9)

V = kn.assignments






fd1 = VL[1]

e2 = ev2[fd1]

e3 = ev3[fd1]

#plot(ev2, ev3, seriestype = :scatter)

PP = plot(e2, e3, seriestype = :scatter, markersize=1,legend=false, markercolor=RGBA(rand(1)[1], rand(1)[1], rand(1)[1], rand(1)[1]))

PP

for ii = 2:length(VL)

    fd1 = VL[ii] #findall(x->x==ii, V)

    e2 = ev2[fd1]

    e3 = ev3[fd1]

    PP = plot!(e2, e3, seriestype = :scatter, legend=false, markercolor=RGBA(rand(1)[1], rand(1)[1], rand(1)[1], rand(1)[1]))


end

PP
=#

#G = SimpleGraph(13,15)

#A = adjacency_matrix(G)

#fdnz = findnz(triu(A,1))
#W = fdnz[3] .* 1.0
## SW is the switch to enable the sparsification;
# if SW=0, there is no sparsification (edge elimination) technique
SW = 0
## spL is the sparsification level;
# the larger spL results in more aggressive sparsification
# you can start with spL = 4 and increase it if the results are good
spL = 4

## scL is the spectral coarsening ratio
# larger scL means more node reduction
scL = 2
## idx is the output node indices
# L is the Laplacian matrix corresponding to the weighted adjacency matrix
@time idx, A_ICE = HyperEF1(L, spL)


#=
rr = findnz(triu(L,1))[1]
cc = findnz(triu(L,1))[2]
mx = max(maximum(rr),maximum(cc))

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

A = sparse(R, C, V, mx, mx)
=#
