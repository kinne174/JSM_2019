import dgl

'''
create the graph network, possibly create the sharing functions here as well so in the next step can just pass on
requested word vectors, don't know about this though because of how parameter updating works, may be better just to pass
on the graph network 
'''

def build_graph(edge_tuples, embedding_dict):
    g = dgl.DGLGraph()

    g.add_nodes(len(embedding_dict))

    dst, src = tuple(zip(*edge_tuples))
    g.add_edges(src, dst)
    g.add_edges(dst, src)

    embedding_list = list(embedding_dict.items())
    embedding_list.sort(key=lambda t: t[0])
    embeddings = [t[1] for t in embedding_list]
    g.ndata['embeddings'] = embeddings

    return g


