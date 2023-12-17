import networkx as nx
import stellargraph
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from stellargraph.mapper import GraphWaveGenerator
from stellargraph import StellarGraph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
query_graph = nx.DiGraph()
doc_graph = nx.DiGraph()
nodes = {
    "Woodside Energy": {"type": "Parent", "text": "Woodside Energy"},
    "Spot": {"type": "Child", "text": "Spot"},
    "Spector": {"type": "Child", "text": "Spector"},
    "Pluto LNG": {"type": "Child", "text": "Pluto LNG"},
    "Boston Dynamics": {"type": "Child", "text": "Boston Dynamics"},
    "Shawn Fernando": {"type": "Child", "text": "Shawn Fernando"},
    "DroneDeploy": {"type": "Child", "text": "DroneDeploy"},
    "FUSE": {"type": "Child", "text": "FUSE"},
    "David Inggs": {"type": "Child", "text": "David Inggs"},
    "Safety Payload": {"type": "Child", "text": "Safety Payload"},
    "Bruce Hill": {"type": "Child", "text": "Bruce Hill"}
}

# Adding nodes to graph
for node, attr in nodes.items():
    doc_graph.add_node(node, **attr)

# Edges
edges = [
    ("Woodside Energy", "Pluto LNG"),
    ("Pluto LNG", "Spector"),
    ("Spector", "Spot"),
    ("Spot", "Boston Dynamics"),
    ("Woodside Energy", "FUSE"),
    ("Woodside Energy", "Shawn Fernando"),
    ("Woodside Energy", "David Inggs"),
    ("Spector", "Safety Payload"),
    ("Woodside Energy", "Bruce Hill"),
    ("Spot", "DroneDeploy")
]

# Adding edges to graph
doc_graph.add_edges_from(edges)

nodes_solution_query = {
    "Solution": {"type": "Parent", "text": "Solution"},
    "Boston Dynamics": {"type": "Child", "text": "Boston Dynamics"},
    "Provided": {"type": "Child", "text": "Provided"}
}


# Adding nodes to the solution graph
for node, attr in nodes_solution_query.items():
    query_graph.add_node(node, **attr)

# Edges for the solution graph
edges_solution_query = [
    ("Solution", "Boston Dynamics"),
    ("Solution", "Provided")
]
#What is the solution provided by Boston Dynamics
query_graph.add_edges_from(edges_solution_query)


from transformers import BertTokenizer, BertModel
import torch

class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def get_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = self.model(**inputs)
            # Using the [CLS] token embedding as the representation of the text
            return outputs.last_hidden_state[:, 0, :].numpy()

bert_embedder = BERTEmbedder()

def get_textual_embedding(node, graph):
    node = graph.nodes[node]['node']
    return node.embedding


def compute_graphwave_embeddings(graph, scales=(1, 2, 3), num_samples=50, sample_points=range(0, 100)):
    # Convert NetworkX graph to StellarGraph
    G = StellarGraph.from_networkx(graph)

    # Create the GraphWave generator
    generator = GraphWaveGenerator(G, scales=scales)

    # Convert sample_points to a numpy array
    sample_points_array = np.array(list(sample_points))

    # Generate the embeddings
    embeddings_dataset = generator.flow(
        node_ids=G.nodes(),
        sample_points=sample_points_array, # use the numpy array here
        batch_size=1,
        repeat=False,
        shuffle=False
    )

    # Retrieve the embeddings from the generator
    graphwave_embeddings = []
    for batch in embeddings_dataset:
        graphwave_embeddings.append(batch[0])

    graphwave_embeddings = np.vstack(graphwave_embeddings)

    return graphwave_embeddings, list(G.nodes())
from gensim.models.poincare import PoincareModel
def compute_struct(graph):
    from GraphEmbedding.ge.models import Struc2Vec

    model = Struc2Vec(graph, walk_length=10, num_walks=80, workers=4, verbose=40)

    # Train the model
    model.train()

    # Get the embeddings
    embeddings = model.get_embeddings()
    return embeddings, list(graph.nodes())

def unify_embeddings(textual_embeddings, structural_embeddings, embedding_dim):
    # Normalize both embedding sets
    norm_textual = normalize(textual_embeddings)
    norm_structural = normalize(structural_embeddings)

    # Dimensionality alignment
    pca = PCA(n_components=embedding_dim)
    aligned_textual = pca.fit_transform(norm_textual)
    aligned_structural = pca.fit_transform(norm_structural)

    # Concatenate embeddings
    unified_embeddings = np.concatenate([aligned_textual, aligned_structural], axis=1)

    # Optional: Normalize the concatenated embeddings
    unified_embeddings = normalize(unified_embeddings)

    return unified_embeddings
def get_unified_embedding(node, graph, precomputed_embeddings, node_ids):
    # Find the position of the node in the node_ids list
    node_position = node_ids.index(node)

    # Use this position to get the corresponding structural embedding
    structural_embedding = precomputed_embeddings[node]
    textual_embedding = get_textual_embedding(node, graph)

    # structural_embedding = normalize(structural_embedding.reshape(1, -1))
    #textual_embedding = normalize(textual_embedding.reshape(1, -1))

    # Concatenate the normalized embeddings
    return np.concatenate([structural_embedding, structural_embedding])
    # # Dimensionality alignment
    # pca = PCA(n_components=100)
    # aligned_textual = pca.fit_transform(norm_textual)
    # aligned_structural = pca.fit_transform(norm_structural)
    #
    # # Concatenate embeddings
    # unified_embeddings = np.concatenate([aligned_textual, aligned_structural], axis=1)

def get_struct(precomputed_embeddings, node):
    return precomputed_embeddings[node]

def get_text(node, graph):
    return get_textual_embedding(node, graph)

# 4. Node alignment based on GraphWave embeddings
# def align_nodes(query_graph, doc_graph, query_embeddings, query_node_ids, doc_embeddings, doc_node_ids, top_n=5):
#     alignments = []
#     for query_node in query_graph.nodes():
#         similarities = []
#         for doc_node in doc_graph.nodes():
#
#             struct_sim =  get_struct(precomputed_embeddings, node)
#             similarity = cosine_similarity([get_unified_embedding(query_node, query_graph, query_embeddings, query_node_ids)],
#                                            [get_unified_embedding(doc_node, doc_graph, doc_embeddings, doc_node_ids)])[0][0]
#             if similarity > .5:
#                 similarities.append((similarity, doc_node))
#
#         # Sort the nodes by similarity and select the top n nodes
#         sorted_nodes = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
#         best_matches = [node[1] for node in sorted_nodes]
#         alignments.append((query_node, best_matches))
#
#     return alignments
def align_nodes(query_graph, doc_graph, query_embeddings, query_node_ids, doc_embeddings, doc_node_ids, top_n=4):
    alignments = []
    for query_node in query_node_ids:
        struct_similarities = []
        text_similarities = []

        for doc_node in doc_node_ids:
            # Structural similarity
            query_struct = get_struct(query_embeddings, query_node)
            doc_struct = get_struct(doc_embeddings, doc_node)
            struct_similarity = cosine_similarity([query_struct], [doc_struct])[0][0]
            if struct_similarity > .5:
                struct_similarities.append((struct_similarity, doc_node))

            # Textual similarity
            query_text = get_text(query_node, query_graph)
            doc_text = get_text(doc_node, doc_graph)
            text_similarity = cosine_similarity([query_text], [doc_text])[0][0]
            if text_similarity > .70:
                text_similarities.append((text_similarity, doc_node))

        # Sort and select top n for structural similarities
        sorted_struct_nodes = sorted(struct_similarities, key=lambda x: x[0], reverse=True)[:top_n]
        best_struct_matches = [node[1] for node in sorted_struct_nodes]

        # Sort and select top n for textual similarities
        sorted_text_nodes = sorted(text_similarities, key=lambda x: x[0], reverse=True)[:top_n]
        best_text_matches = [node[1] for node in sorted_text_nodes]

        alignments.append((query_node, {'struct': best_struct_matches, 'text': best_text_matches}))

    return alignments


# Get alignments
# query_embeddings, query_node_ids = compute_graphwave_embeddings(query_graph)
# doc_embeddings, doc_node_ids = compute_graphwave_embeddings(doc_graph)
#
# alignments = align_nodes(query_graph, doc_graph, query_embeddings, query_node_ids, doc_embeddings, doc_node_ids)
#
# print(alignments)

def get_alignments(q_graph, d_graph):
    query_embeddings, query_node_ids = compute_struct(q_graph)
    doc_embeddings, doc_node_ids = compute_struct(d_graph)

    alignments = align_nodes(q_graph, d_graph, query_embeddings, query_node_ids, doc_embeddings, doc_node_ids)

    return alignments

