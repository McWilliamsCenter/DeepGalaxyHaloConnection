import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import radius_neighbors_graph
from multiprocessing import Pool
import jraph
import jax
import jax.numpy as jnp

def _process_graph(args):
    """
    Function that preprocesses the dataset for fast training
    """
    gid, group_ids, Xsp, x_vec, x_scal, x_cont, graph_radius = args
    g = np.where(group_ids == gid)[0]
    xsp = Xsp[g]
    x = x_vec[g]
    y = x_scal[g]
    z = x_cont[g].mean(axis=0)
    # Compute adjacency matrix for each entry
    adj = radius_neighbors_graph(xsp, 
                                 graph_radius, 
                                 mode='connectivity',
                                 include_self=False).tocoo()
    graph = jraph.GraphsTuple(
                        nodes={'pos': xsp,
                               'vectors': jnp.stack([np.stack([x[...,i], 
                                                    x[...,i+1], 
                                                    x[...,i+2]], axis=-1) for i in range(x.shape[-1]//3)],axis=1),
                               'scalars': y},
                        senders=adj.row,
                        receivers=adj.col,
                        edges=adj.data,
                        n_node=np.array([len(x)]),
                        n_edge=np.array([len(adj.data)]),
                        globals=z,
                    )
    # Padding the graph to make our life much simpler
    return jraph.pad_with_graphs(graph, 
                                 n_node=70, # 66 sats is the largest halo
                                 n_edge=70*70, # to avoid issues
                                )

def get_batch_fn(catalog, 
                   vector_keys=[],
                   scalar_keys=[],
                   context_keys=[],
                   pos_key=['x', 'y', 'z'],
                   group_key='halo_hostid',
                   graph_radius=1000.,
                   poolsize=12,
                   batch_size=32
                   ):
    """ Returns a function that will draw random batches of graphs
    """
    
    # It takes a minute but we precompute all the graphs and data
    # Identify the individual groups and pre-extract the relevant data
    group_ids = catalog[group_key]
    gids, idx = np.unique(group_ids, return_index=True)
    
    # Extracts columns of interest into memory first
    Xsp = np.stack([np.array(catalog[k].astype('float32')) for k in pos_key],axis=-1)
    x_vec = np.stack([np.array(catalog[k].astype('float32')) for k in vector_keys],axis=-1)
    x_scl = np.stack([np.array(catalog[k].astype('float32')) for k in scalar_keys],axis=-1)
    x_ctx = np.stack([np.array(catalog[k].astype('float32')) for k in context_keys],axis=-1) 

    print("Precomputing dataset")
    with Pool(poolsize) as p:
        cache = p.map(_process_graph, [(gid, group_ids, Xsp, x_vec, x_scl, x_ctx, graph_radius) for gid in gids])
    print("Done")
    
    # Creating a graph with batch dimension
    graph = jraph.GraphsTuple(
                        nodes={'pos': jnp.stack([g.nodes['pos'] for g in cache],axis=0),
                               'vectors': jnp.stack([g.nodes['vectors'] for g in cache],axis=0),
                               'scalars': jnp.stack([g.nodes['scalars'] for g in cache],axis=0)},
                        senders=jnp.stack([g.senders for g in cache],axis=0),
                        receivers=jnp.stack([g.receivers for g in cache],axis=0),
                        edges=jnp.stack([g.edges for g in cache],axis=0),
                        n_node=jnp.stack([g.n_node for g in cache],axis=0),
                        n_edge=jnp.stack([g.n_edge for g in cache],axis=0),
                        globals=jnp.stack([g.globals for g in cache],axis=0),
                    )
    @jax.jit
    def batch_fn(key):
        inds = jax.random.choice(key, len(graph.globals), shape=[batch_size], replace=False)
        return jax.tree_util.tree_map(lambda x: x[inds], graph)

    return batch_fn