include("HyperEF.jl")
ar = [[1,2], [1,3], [2,3], [2,4]]
L = 1

test = [[0, 2],[0, 4],[0, 6],[0, 7],[1, 4],[1, 6],[1, 7],[2, 3],[2, 4],[2, 5],[2, 8],[3, 4],[3, 6],[3, 7],[3, 8],[3, 9],[4, 5],[5, 6],[6, 7],[7, 8],[7, 9]]

list = HyperEF(test, 2)
println(list)