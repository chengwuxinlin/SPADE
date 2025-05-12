import hnswlib
import numpy as np
from scipy.sparse import coo_matrix, diags, identity, csr_matrix, find, triu
from julia.api import Julia
import scipy.sparse.linalg as sla
#import networkx as nx
#import grass_mtx as mtx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from torch_sparse import SparseTensor
import torch
import random
import copy
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from torch.nn.functional import softmax


def adj2laplacian(A):
    D = diags(np.squeeze(np.asarray(A.sum(axis=1))), 0)
    L = D - A + identity(A.shape[0]).multiply(1e-6)

    return L

def laplacian2adj(L):
    A = copy.copy(L)
    A = np.absolute(A)
    A.setdiag(0)
    A.eliminate_zeros()

    return A

def julia_eigs(l_in, l_out, num_eigs):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./my_utils/eigen.jl")
    print('Generate eigenpairs')
    eigenvalues, eigenvectors = Main.main(l_in, l_out, num_eigs)

    return eigenvalues.real, eigenvectors.real

def py_eigs(l_in, l_out, num_eigs):
    Dxy, Uxy = sla.eigs(l_in, num_eigs, l_out)
    return Dxy.real, Uxy.real

def GetRiemannianDist(Gx, Gy, Lx, Ly, num_eigs): 
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = julia_eigs(Lx, Ly, num_eigs)
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()
    num_edge_tot=len(Gx.edges) # number of total edges  
    Zpq=np.zeros((num_edge_tot,));# edge embedding distance
    p = np.array(Gx.edges)[:,0];# one end node of each edge
    q = np.array(Gx.edges)[:,1];# another end node of each edge
    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]
    Zpq = Zpq/max(Zpq)

    node_score=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)

    TopNodeList = np.flip(node_score.argsort(axis=0))
    TopEdgeList=np.column_stack((p,q))[np.flip(Zpq.argsort(axis=0)),:]

    return TopEig, TopEdgeList, TopNodeList, node_score


def construct_adj(neighs, weight):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    adj.data[:] = 1
    #G = nx.from_scipy_sparse_matrix(adj)
    #lap = nx.laplacian_matrix(G, weight=None)

    return adj#, lap, G


def construct_weighted_adj(neighs, distances):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1
    weights = np.exp(-distances)

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    # calculate weights for each edge
    edge_weights = weights[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    all_data = np.concatenate((edge_weights, edge_weights), axis=0)  # use weights instead of ones
    adj = csr_matrix((all_data, (all_row, all_col)), shape=(dim, dim))
    #G = nx.from_scipy_sparse_array(adj)
    # construct a graph from the adjacency matrix
    #lap = laplacian(adj, normed=False)

    return adj#, lap, G



def hnsw(features, k=10, ef=100, M=48):
    num_samples, dim = features.shape

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, weight = p.knn_query(features, k+1)
  
    return neighs, weight

def spade(adj_in, data_output, k=10, num_eigs=2,weighted=True,sparse=True): 

    G_in = nx.from_scipy_sparse_matrix(adj_in)
    neighs, distance = hnsw(data_output, k)
    if weighted:
        adj_out, _, G_out = construct_weighted_adj(neighs, distance)
    else:
        adj_out, _, G_out = construct_adj(neighs, distance)

    assert nx.is_connected(G_in), "input graph is not connected"
    assert nx.is_connected(G_out), "output graph is not connected"

    #adj_in = SPF(adj_in, 4)
    L_in = laplacian(adj_in, normed=False)#.tocsr()#adj2laplacian(adj_in)
    #L_in = L_in + identity(adj_in.shape[0]).multiply(1e-6)
    #L_in = coo_matrix(L_in)

    if sparse:
        adj_out = SPF(adj_out, 4)
    
    L_out = laplacian(adj_out, normed=False)#.tocsr()#adj2laplacian(adj_out)
  
    TopEig, TopEdgeList, TopNodeList, node_score = GetRiemannianDist(G_in, G_out, L_in, L_out, num_eigs)# full function
    return TopEig, TopEdgeList, TopNodeList, node_score, L_in, L_out



def spade2(data_input, data_output, k=15, weighted=True, sparse=True): 

    #neighs, distance = hnsw(data_input, k)
    neighs1, distance1 = hnsw(data_output, k)
    if weighted:
        #adj_in, _, G_in = construct_weighted_adj(neighs, distance)#construct_adj, construct_weighted_adj
        adj_out, _, G_out = construct_weighted_adj(neighs1, distance1)#construct_adj, construct_weighted_adj
    else:
        #adj_in, _, G_in = construct_adj(neighs, distance)#construct_adj, construct_weighted_adj
        adj_out, _, G_out = construct_adj(neighs1, distance1)#construct_adj, construct_weighted_adj
    #assert nx.is_connected(G_in), "input graph is not connected"
    assert nx.is_connected(G_out), "output graph is not connected"
    if sparse:
        #adj_in = SPF(adj_in, 4)
        adj_out = SPF(adj_out, 4)
    adj_in = data_input
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./SPADE2/SPF.jl")
    py_dict = Main.SPF(adj_in, adj_out)
    # Initialize NodeScore array with zeros
    NodeScore = np.zeros(adj_in.shape[0])
    # Iterate over all nodes
    for node in range(adj_in.shape[0]):
        # Get the indices of the non-zero elements in the row
        # These indices represent the neighbors of the node
        _, neighbors, _ = find(adj_in[node:node+1, :])
        # Initialize node score to 0
        node_score = 0
        # Iterate over the neighbors of the node
        for neighbor in neighbors:
            # Attempt to retrieve the score from the dictionary using both possible key orders
            # If a key is not found in the dictionary, the get method returns 0
            node_score += py_dict.get((node, neighbor), py_dict.get((neighbor, node), 0))
        # Store the total score of the node in the NodeScore array
        NodeScore[node] = node_score/len(neighbors)
        sorted_nodes_ID = np.argsort(NodeScore)[::-1]
    return sorted_nodes_ID,adj_in,adj_out


def CL_kmean(adj_in, data_output, k=10,weighted=True,sparse=True,num_eigs=2,kmean = 2):
    neighs, distance = hnsw(data_output, k)
    if weighted:
        adj_out, _, G_out = construct_weighted_adj(neighs, distance)
    else:
        adj_out, _, G_out = construct_adj(neighs, distance)
    if sparse:
        adj_out = SPF(adj_out, 4)

    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./CL_kmean/decompositionW2.jl")
    cluster_adj_in, mapper_dic_in = Main.decompositionW2(adj_in, kmean)
    cluster_adj_out, mapper_dic_out = Main.decompositionW2(adj_out, kmean)

    G_in = nx.from_scipy_sparse_matrix(cluster_adj_in)
    L_in = laplacian(cluster_adj_in, normed=False)
    L_out = laplacian(cluster_adj_out, normed=False)

    _, _, TopNodeList, _ = GetRiemannianDist(G_in, G_in, L_in, L_out, num_eigs)

    # Create a dictionary mapping each cluster to its nodes
    cluster_to_nodes = {}
    for node, cluster in enumerate(mapper_dic_in-1, start=0):
        if cluster not in cluster_to_nodes:
            cluster_to_nodes[cluster] = []
        cluster_to_nodes[cluster].append(node)

    # Use the cluster ranking to get the node ranking
    node_ranking = []
    for cluster in TopNodeList:
        node_ranking.extend(cluster_to_nodes[cluster])

    return node_ranking



def embedding_normalize(embedding, norm):
    if norm == "unit_vector":
        return normalize(embedding, axis=1)
    elif norm == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(embedding)
    elif norm == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(embedding)
    else:
        return embedding
    
def normal_adj(adj):
    adj = SparseTensor.from_scipy(adj)
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)

    return DAD.to_scipy(layout='csr')

def hyperEF(L, level, grass):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./HyperEF1.jl")
    idx  = Main.HyperEF1( L, level, grass)

def spectral_embedding(adj_mtx,features,use_feature=True,embedding_norm=None,adj_norm=True):
    adj_mtx = adj_mtx.asfptype()
    num_nodes = adj_mtx.shape[0]
    if adj_norm:
        adj_mtx = normal_adj(adj_mtx)
    U, S, Vt = svds(adj_mtx, 50)

    spec_embed = np.sqrt(S.reshape(1,-1))*U
    spec_embed = embedding_normalize(spec_embed, embedding_norm)
    if use_feature:
        feat_embed = adj_mtx @ (adj_mtx @ features)/2
        feat_embed = embedding_normalize(feat_embed, embedding_norm)
        spec_embed = np.concatenate((spec_embed, feat_embed), axis=1)
    return spec_embed

def spectral_embedding_eig(adj_mtx,features,use_feature=True,embedding_norm=None,adj_norm=True,eig_julia=False):
    adj_mtx = adj_mtx.asfptype()
    num_nodes = adj_mtx.shape[0]
    if adj_norm:
        adj_mtx = normal_adj(adj_mtx)
    #U, S, Vt = svds(adj_mtx, 50)
    L_mtx = adj2laplacian(adj_mtx)
    
    if not eig_julia:
        S, U = eigsh(L_mtx,k=50,which='SM', maxiter=500000)
    else:
        jl = Julia(compiled_modules=False)
        from julia import Main
        Main.include("./my_utils/eigen.jl")
        S, U = Main.not_main(L_mtx.tocoo(), 50)
    
    spec_embed = U[:, 1:]
    spec_embed = embedding_normalize(spec_embed, embedding_norm)
    if use_feature:
        feat_embed = adj_mtx @ (adj_mtx @ features)/2
        feat_embed = embedding_normalize(feat_embed, embedding_norm)
        spec_embed = np.concatenate((spec_embed, feat_embed), axis=1)
    return spec_embed


def adj2graph(adj):
    G = nx.from_scipy_sparse_matrix(adj)
    return G


def SPF(adj, L, ICr=0.11):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./my_utils/SPF.jl")
    agj_c, inter_edge_adj = Main.SPF(adj, L, ICr)

    return agj_c, inter_edge_adj

def heterophily_score(adj_matrix: csr_matrix, model_output: torch.Tensor):
    num_nodes = model_output.shape[0]
    labels = torch.argmax(model_output, dim=1)
    assert adj_matrix.shape[0] == len(labels), "Number of nodes in adjacency matrix and labels must match"
    heterophily_scores = []

    for node_idx in range(num_nodes):
        neighbors = adj_matrix[node_idx].indices
        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            heterophily_scores.append(0)
            continue
        node_label = labels[node_idx].item()
        diff_labels_count = sum([1 for neighbor in neighbors if labels[neighbor].item() != node_label])
        heterophily_scores.append(diff_labels_count / num_neighbors)

    heteroNodeList = sorted(range(len(heterophily_scores)), key=lambda i: heterophily_scores[i], reverse=True)

    return np.array(heteroNodeList),heterophily_scores


def jaccard_similarity(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def edge2adj(edge_index):
    # Find the number of nodes (assuming the node indices are 0-based)
    num_nodes = edge_index.max().item() + 1
    # Convert the edge index tensor to a NumPy array
    edge_index_np = edge_index.numpy()
    # Create the sparse adjacency matrix using scipy.sparse.coo_matrix
    coo_adj_matrix = coo_matrix((torch.ones(edge_index.shape[1]), (edge_index_np[0], edge_index_np[1])), shape=(num_nodes, num_nodes))
    # Convert the COO matrix to CSR format
    csr_adj_matrix = coo_adj_matrix.tocsr()
    return csr_adj_matrix

def adj2edge(adj):
    # Convert the CSR matrix to COO format
    coo_adj_matrix = adj.tocoo()
    # Get the row (source nodes) and col (target nodes) attributes of the COO matrix
    source_nodes = coo_adj_matrix.row
    target_nodes = coo_adj_matrix.col
    # Combine source and target nodes as edge index
    edge_index_np = np.vstack((source_nodes, target_nodes))
    # Convert the NumPy array back to a PyTorch tensor
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    return edge_index

def featurePT(x,beta):
    samples_num,dimen = x.shape
    std_dev = torch.std(x)
    noise = torch.randn(samples_num, dimen) * std_dev
    x += noise * beta
    return x



def random_edgePT(orig_adj_mtx, the_p,label,node_index):
    graph = adj2graph(orig_adj_mtx)
    perturbed_graph = graph.copy()
    for node in node_index:
        other_nodes = [n for n in perturbed_graph.nodes() if n != node and label[n] != label[node]]
        nodes_to_connect = random.sample(other_nodes, the_p)
        edges = [(node, n) for n in nodes_to_connect]
        perturbed_graph.add_edges_from(edges)
        # randomly select x edges to remove between nodes with the same label as node
        same_label_neighbors = [n for n in perturbed_graph.neighbors(node) if label[n] == label[node]]
        edges_to_remove = random.sample(same_label_neighbors, min(the_p, len(same_label_neighbors)))
        # remove the selected edges
        perturbed_graph.remove_edges_from([(node, n) for n in edges_to_remove])
    pt_adj = csr_matrix(nx.adjacency_matrix(perturbed_graph))
    return pt_adj

def find_edge_index_dif(edge_index1,edge_index2):
    # Transpose the edge_index tensors so that the shape is (num_edges, 2)
    edge_index1_t = edge_index1.t()
    edge_index2_t = edge_index2.t()
    # Find the unique rows in both tensors and their indices
    unique_edge_index1, indices1 = torch.unique(edge_index1_t, dim=0, return_inverse=True)
    unique_edge_index2, indices2 = torch.unique(edge_index2_t, dim=0, return_inverse=True)
    # Find the difference between the unique edge indices
    only_in_edge_index1 = unique_edge_index1[torch.isin(indices1, indices2, invert=True)]
    only_in_edge_index2 = unique_edge_index2[torch.isin(indices2, indices1, invert=True)]
    combined_edges = torch.cat((only_in_edge_index1, only_in_edge_index2), dim=0)
    # Find the node rankings for both sets of unique edges
    ranked_combined = node_ranking(combined_edges)
    combined_node_ids_rank = [node for node, _ in ranked_combined]

    return np.array(combined_node_ids_rank), ranked_combined

def node_ranking(unique_edges):
    node_counts = {}
    for edge in unique_edges:
        for node in edge:
            if node.item() in node_counts:
                node_counts[node.item()] += 1
            else:
                node_counts[node.item()] = 1       
    # Sort the nodes by their frequency in descending order
    sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes


def spade_nonetworkx(adj_in, data_output, k=10, num_eigs=2,weighted=True,sparse=True): 

    neighs, distance = hnsw(data_output, k)
    if weighted:
        adj_out = construct_weighted_adj(neighs, distance)
    else:
        adj_out = construct_adj(neighs, distance)
    L_in = laplacian(adj_in, normed=False)#.tocsr()#adj2laplacian(adj_in)
    if sparse:
        adj_out,_ = SPF(adj_out, 4)
    adj_out.data = np.ones_like(adj_out.data)#weighted to unweighted
    L_out = laplacian(adj_out, normed=False)#.tocsr()#adj2laplacian(adj_out)
  
    TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy = GetRiemannianDist_nonetworkx(L_in, L_out, num_eigs)# full function
    return TopEig, TopEdgeList, TopNodeList, node_score, L_in, L_out


def SAGMAN_V2(data_input, data_output, k=10, num_eigs=2,weighted=True,sparse=True, weighted_spade=False, SPF_cors=3): 

    neighs_in, distance_in = hnsw(data_input, k)
    neighs, distance = hnsw(data_output, k)
    if weighted:
        adj_in = construct_weighted_adj(neighs_in, distance_in)
        adj_out = construct_weighted_adj(neighs, distance)
    else:
        adj_in = construct_adj(neighs_in, distance_in)
        adj_out = construct_adj(neighs, distance)

    if sparse:
        adj_in, internal_edge = SPF(adj_in, SPF_cors)
        adj_out,_ = SPF(adj_out, SPF_cors)
    else:
        internal_edge = None

    if not weighted_spade:
        adj_in.data = np.ones_like(adj_in.data)#weighted to unweighted
        adj_out.data = np.ones_like(adj_out.data)#weighted to unweighted

    L_in = laplacian(adj_in, normed=False)#.tocsr()#adj2laplacian(adj_in)
    L_out = laplacian(adj_out, normed=False)#.tocsr()#adj2laplacian(adj_out)
  
    TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy = GetRiemannianDist_nonetworkx(L_in, L_out, num_eigs)# full function

    return TopEig, TopEdgeList, TopNodeList, node_score, internal_edge


def GetRiemannianDist_nonetworkx(Lx, Ly, num_eigs): 
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = py_eigs(Lx, Ly, num_eigs)# py_eigs, julia_eigs 
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()

    #num_edge_tot=len(Gx.edges) # number of total edges  
    #Zpq=np.zeros((num_edge_tot,));# edge embedding distance
    #p = np.array(Gx.edges)[:,0];# one end node of each edge
    #q = np.array(Gx.edges)[:,1];# another end node of each edge

    lap_matrix_upper = triu(Lx, k=1)# k=1 excludes the diagonal
    rows, cols, _ = find(lap_matrix_upper)  # Find the indices of non-zero elements
    num_edge_tot = len(rows)  # Number of total edges
    Zpq = np.zeros((num_edge_tot,))  # edge embedding distance
    p = rows  # one end node of each edge
    q = cols  # another end node of each edge

    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]
    Zpq = Zpq/max(Zpq)

    node_score=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)

    TopNodeList = np.flip(node_score.argsort(axis=0))
    TopEdgeList=np.column_stack((p,q))[np.flip(Zpq.argsort(axis=0)),:]

    return TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy



def SAGMAN_V2_bi_lip(data_input, data_output, k=10, num_eigs=2,weighted=True,sparse=True, weighted_spade=False, SPF_cors=3): 

    neighs_in, distance_in = hnsw(data_input, k)
    neighs, distance = hnsw(data_output, k)
    if weighted:
        adj_in = construct_weighted_adj(neighs_in, distance_in)
        adj_out = construct_weighted_adj(neighs, distance)
    else:
        adj_in = construct_adj(neighs_in, distance_in)
        adj_out = construct_adj(neighs, distance)

    if sparse:
        adj_in, internal_edge = SPF(adj_in, SPF_cors)
        adj_out,_ = SPF(adj_out, SPF_cors)
    else:
        internal_edge = None

    if not weighted_spade:
        adj_in.data = np.ones_like(adj_in.data)#weighted to unweighted
        adj_out.data = np.ones_like(adj_out.data)#weighted to unweighted

    L_in = laplacian(adj_in, normed=False)#.tocsr()#adj2laplacian(adj_in)
    L_out = laplacian(adj_out, normed=False)#.tocsr()#adj2laplacian(adj_out)
  
    TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy = GetRiemannianDist_nonetworkx_bi_lip(L_in, L_out, num_eigs)# full function

    return TopEig, TopEdgeList, TopNodeList, node_score, internal_edge


def GetRiemannianDist_nonetworkx_bi_lip(Lx, Ly, num_eigs): 
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = julia_eigs(Lx, Ly, num_eigs)
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()

    lap_matrix_upper = triu(Lx, k=1)# k=1 excludes the diagonal
    rows, cols, _ = find(lap_matrix_upper)  # Find the indices of non-zero elements
    num_edge_tot = len(rows)  # Number of total edges
    Zpq = np.zeros((num_edge_tot,))  # edge embedding distance
    p = rows  # one end node of each edge
    q = cols  # another end node of each edge

    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]
    Zpq = Zpq/max(Zpq)

    node_score=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)


    Dxy1, Uxy1 = julia_eigs(Ly, Lx, num_eigs)
    num_node_tot1 = Uxy1.shape[0]
    TopEig1=max(Dxy1)
    Zpq1 = np.zeros((num_edge_tot,))
    for i in np.arange(0,num_eigs):
        Zpq1 = Zpq1 + np.power(Uxy1[p,i]-Uxy1[q,i], 2)*Dxy1[i]
    Zpq1 = Zpq1/max(Zpq1)

    node_score1=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score1[p[i]]=node_score1[p[i]]+Zpq1[i]
        node_score1[q[i]]=node_score1[q[i]]+Zpq1[i]
    node_score1=node_score1/NodeDegree
    node_score1=node_score1/np.amax(node_score1)

    total_score = node_score + node_score1

    # TopNodeList = np.flip(node_score.argsort(axis=0))
    TopNodeList = np.flip(total_score.argsort(axis=0))
    TopEdgeList=np.column_stack((p,q))[np.flip(Zpq.argsort(axis=0)),:]

    return TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy


def extract_embeddings_advanced_v5(
    model, 
    tokenizer, 
    texts, 
    labels, 
    batch_size=4, 
    device=None,
    input_layer=0,
    output_layer=-1,
    attention_layer=-1,
    use_cls_token_attention=True
):
    """
    Extracts sentence-level input (X) and output (Y) embeddings from a Transformer model using
    attention-based pooling.

    Parameters
    ----------
    model : PreTrainedModel
        A Hugging Face Transformer model.
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer for the model.
    texts : list of str
        The input sentences.
    labels : list
        The labels corresponding to each text (not used in embeddings, can be used for downstream tasks).
    batch_size : int
        Batch size for embedding extraction.
    device : torch.device or None
        Device to run the model on. If None, will use CUDA if available, else CPU.
    input_layer : int
        Index of the hidden state layer to use for input embeddings. 
        Typically 0 corresponds to token embeddings right after the embedding layer.
    output_layer : int
        Index of the hidden state layer to use for output embeddings. 
        -1 corresponds to the last layer.
    attention_layer : int
        Index of the attention layer to use for pooling.
        -1 corresponds to the last layer.
    use_cls_token_attention : bool
        If True, use attention weights from the [CLS] token to pool token embeddings. 
        If False, can implement another pooling strategy.

    Returns
    -------
    input_embeddings : np.ndarray, shape (N, d)
        Sentence-level input embeddings.
    output_embeddings : np.ndarray, shape (N, d)
        Sentence-level output embeddings.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_embeddings_list = []
    output_embeddings_list = []

    # Ensure input_layer and output_layer indices are valid
    # (This depends on the model, but typically model returns all layers including embeddings)
    # For a model with L+1 hidden_states: 
    # hidden_states[0] is embedding layer output
    # hidden_states[1] is output of first transformer layer, etc.
    # Negative indexing works as in Python lists.

    # Similarly for attentions, ensure indexing is valid
    # attentions[l] corresponds to attention after layer l (0-based),
    # so attentions[-1] is last layer's attention.
    # No explicit check here, but could be added for robustness.

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize and move to device
        encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # Verify that we have attentions
        if attentions is None or len(attentions) == 0:
            # Fallback: If no attentions, just use a simple average pooling
            # This fallback should be rare if output_attentions=True is supported by the model.
            # Otherwise, consider raising an error or use another strategy.
            token_input_embs = hidden_states[input_layer]   # [N, T, d]
            token_output_embs = hidden_states[output_layer] # [N, T, d]

            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1)    # [N, T, 1]
            sum_input_emb = torch.sum(token_input_embs * mask_expanded, dim=1)
            sum_output_emb = torch.sum(token_output_embs * mask_expanded, dim=1)
            valid_counts = torch.sum(attention_mask, dim=1, keepdim=True).float()
            z_X = sum_input_emb / valid_counts
            z_Y = sum_output_emb / valid_counts
        else:
            # Use attention-based pooling
            token_input_embs = hidden_states[input_layer]   # [N, T, d]
            token_output_embs = hidden_states[output_layer] # [N, T, d]

            # Select the attention layer to use
            selected_attention = attentions[attention_layer]  # [N, num_heads, T, T]
            # Average over heads
            avg_attention = selected_attention.mean(dim=1)    # [N, T, T]

            # Use CLS token attention for pooling if requested
            if use_cls_token_attention:
                # CLS typically at index 0
                cls_attention_weights = avg_attention[:, 0, :]  # [N, T]
            else:
                # Another strategy: e.g. average over all tokens for equal weighting
                # or use a different token as reference. For now, just sum over tokens.
                cls_attention_weights = avg_attention.mean(dim=1)  # [N, T]

            # Now we have attention weights from CLS to every token. 
            # Ensure normalization (though they should already be normalized per token)
            # We'll explicitly apply a softmax over the token dimension:
            alpha_input = softmax(cls_attention_weights, dim=1)  # [N, T]
            # Optionally, we could pick a different set of attentions for output embeddings,
            # but here we reuse the same alpha for simplicity:
            alpha_output = alpha_input

            # Mask out padding tokens if necessary
            # Although attention should account for masks, let's be safe:
            alpha_input = alpha_input * attention_mask
            alpha_output = alpha_output * attention_mask

            # Renormalize after masking
            alpha_input = alpha_input / (alpha_input.sum(dim=1, keepdim=True) + 1e-9)
            alpha_output = alpha_output / (alpha_output.sum(dim=1, keepdim=True) + 1e-9)

            # Weighted sum for input embeddings
            alpha_input_expanded = alpha_input.unsqueeze(-1)    # [N, T, 1]
            z_X = torch.sum(alpha_input_expanded * token_input_embs, dim=1)  # [N, d]

            # Weighted sum for output embeddings
            alpha_output_expanded = alpha_output.unsqueeze(-1)  # [N, T, 1]
            z_Y = torch.sum(alpha_output_expanded * token_output_embs, dim=1)  # [N, d]

        # Move to CPU and convert to numpy
        z_X = z_X.detach().cpu().numpy()
        z_Y = z_Y.detach().cpu().numpy()

        input_embeddings_list.append(z_X)
        output_embeddings_list.append(z_Y)

    # Concatenate all batches
    input_embeddings = np.vstack(input_embeddings_list)
    output_embeddings = np.vstack(output_embeddings_list)

    return input_embeddings, output_embeddings





def Euclidean_distance_distortion(data_input, data_output, k=10, weighted=True): 

    neighs_in, distance_in = hnsw(data_input, k)
    neighs, distance = hnsw(data_output, k)
    if weighted:
        adj_in = construct_weighted_adj(neighs_in, distance_in)
        adj_out = construct_weighted_adj(neighs, distance)
    else:
        adj_in = construct_adj(neighs_in, distance_in)
        adj_out = construct_adj(neighs, distance)

    # L_in = laplacian(adj_in, normed=False)#.tocsr()#adj2laplacian(adj_in)
    # L_out = laplacian(adj_out, normed=False)#.tocsr()#adj2laplacian(adj_out)

    TopNodeList, node_score = GetEuclideanDist_nonetworkx_bi_lip(adj_in, adj_out)

    return TopNodeList, node_score



def GetEuclideanDist_nonetworkx_bi_lip(adj_in, adj_out): 
    # Gy not updated 
    # Lx = Lx.asfptype()
    # Ly = Ly.asfptype()

    # Lx = Lx.toarray()  
    # Ly = Ly.toarray() 

    # num_node_tot = Lx.shape[0]
    # NodeDegree=Lx.diagonal()

    # lap_matrix_upper = triu(Lx, k=1)# k=1 excludes the diagonal
    # rows, cols, _ = find(lap_matrix_upper)  # Find the indices of non-zero elements
    # num_edge_tot = len(rows)  # Number of total edges
    # Zpq = np.zeros((num_edge_tot,))  # edge embedding distance
    # p = rows  # one end node of each edge    
    # q = cols  # another end node of each edge 

    # Zpq = np.sqrt(np.sum(((Ly[p, q]**2)/(Lx[p, q]**2)), axis=1)) 
    # Zpq = Zpq/max(Zpq)

    # node_score=np.zeros((num_node_tot,))        
    # for i in np.arange(0,num_edge_tot):
    #     node_score[p[i]]=node_score[p[i]]+Zpq[i]
    #     node_score[q[i]]=node_score[q[i]]+Zpq[i]
    # node_score=node_score/NodeDegree

    node_score = compute_node_scores(adj_in, adj_out)
    node_score=node_score/np.amax(node_score)

    # TopNodeList = np.flip(node_score.argsort(axis=0))
    TopNodeList = np.flip(node_score.argsort(axis=0))

    return TopNodeList, node_score



def compute_node_scores(adj_in, adj_out):
    L_in  = laplacian(adj_in,  normed=False)
    L_out = laplacian(adj_out, normed=False)
    L_in  = L_in.toarray()  if hasattr(L_in, "toarray")  else np.asarray(L_in)
    L_out = L_out.toarray() if hasattr(L_out, "toarray") else np.asarray(L_out)
    
    n = L_in.shape[0]
    node_scores = np.zeros(n, dtype=float)
    
    if hasattr(adj_in, "toarray"):
        adj_in  = adj_in.toarray()
        adj_out = adj_out.toarray()
    
    for p in range(n):
        neighbors = np.where((adj_in[p] + adj_out[p]) > 0)[0]
        ratios = []
        for q in neighbors:
            dist_in  = np.linalg.norm(L_in[p]  - L_in[q])
            dist_out = np.linalg.norm(L_out[p] - L_out[q])
            if dist_in == 0.0:
                ratio = 0.0
            else:
                ratio = dist_out / dist_in
            
            ratios.append(ratio)
        
        if len(ratios) > 0:
            node_scores[p] = np.mean(ratios)
        else:
            node_scores[p] = 0.0
    
    return node_scores