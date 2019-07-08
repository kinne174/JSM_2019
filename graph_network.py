import dgl
import torch

'''
create the graph network, possibly create the sharing functions here as well so in the next step can just pass on
requested word vectors, don't know about this though because of how parameter updating works, may be better just to pass
on the graph network 
'''

def build_graph(edge_tuples, unique_ids, embedding_dict):
    assert isinstance(unique_ids, list)

    g = dgl.DGLGraph()

    g.add_nodes(len(embedding_dict))

    edge_tuples = [(unique_ids.index(e1), unique_ids.index(e2)) for e1, e2 in edge_tuples]

    dst, src = tuple(zip(*edge_tuples))
    g.add_edges(src, dst)
    g.add_edges(dst, src)

    embedding_list = list(embedding_dict.items())
    embedding_list.sort(key=lambda t: t[0])
    embeddings = torch.tensor([t[1] for t in embedding_list])
    g.ndata['embeddings'] = embeddings

    return g

if __name__ == '__main__':
    from sentence_stream import get_idx

    unique_word_idx, word_vectors_dict, _, co_occ_list = get_idx(sentences_filename='ARC/visualization/test_dataset.txt', spacy_language='en_core_web_md', threshold=0.9)

    G = build_graph(edge_tuples=co_occ_list, unique_ids=unique_word_idx, embedding_dict=word_vectors_dict)