from __future__ import annotations
from typing import TypedDict
from collections.abc import Iterator
from dataclasses import dataclass, field, KW_ONLY
from DevAgent.Utils.NatLang.embed import EmbeddingInterface, Embedding
import networkx as nx, math, logging

logger = logging.getLogger(__name__)
  
class SemanticNode(TypedDict):
  """Semantic Information that is a constituent of a SemanticLatent"""
  kind: str
  """Some Classifcation of the content"""
  content: str
  """The Raw Textual Content"""
  latent: Embedding
  """The Embedding Vector representing the Node's location in cognitive space"""

@dataclass
class SemanticLatent:
  """
  A Semantic Latent represents a thought; it start
  """
  embedding_interface: EmbeddingInterface
  """The Embedding Interface to use"""
  _: KW_ONLY
  data: dict[int, SemanticNode] = field(default_factory=dict)
  """The Semantic Data Constituting the Latent; keyed off it's hash value"""
  graph: nx.DiGraph = field(default_factory=nx.DiGraph)
  """The Semantic Nodes as a Directed Graph"""

  def __str__(self) -> str:
    if len(self.data) > 0: return self.render()
    else: return '<EMPTY>'

  @property
  def root_id(self) -> int:
    if len(self.data) == 0: raise RuntimeError('Latent Graph is Empty')
    elif len(self.data) == 1: return next(iter(self.data))
    else: return next(n for n, d in self.graph.nodes(data='is_root', default=False) if d)

  def walk(self) -> Iterator[SemanticNode]:
    """Walk the Latent's Hierarchical Layout Top Down"""
    raise NotImplementedError

  def _get_most_similiar_node(self, latent: Embedding) -> int:
    """Get the Node (by id) whose latent is most similiar"""
    if len(self.data) == 0: raise RuntimeError('Latent Graph is Empty')
    elif len(self.data) == 1: return self.root_id
    max_sim: tuple[float, int] = (-1, -1)
    v1 = Embedding.unpack(latent)[0]
    for n_id, n in self.data.items():
      v2 = Embedding.unpack(n['latent'])[0]
      if (sim := math.sumprod(v1, v2)) > max_sim[0]:
        max_sim = (sim, n_id)
    assert max_sim != (-1, -1)
    return max_sim[1]

  def add_semantics(self, content: str, kind: str):
    """
    Adds semantic information of a certain kind to the Latent Hierarchy.
    
    This method creates a new SemanticNode with the given content,
    generates an embedding for it, and adds it to the graph.
    The first node added to the graph is considered the root.
    Otherwise attaches it as a child of the most similiar node.

    Args:
      content (str): The textual content to be added as semantic information.

    Returns:
      int: The index of the newly added node.
    """
    if (content_id := hash(content)) in self.data: raise ValueError('Can\'t add the same semantic data more than once')
    node: SemanticNode = {
      'kind': kind,
      'content': content,
      'latent': self.embedding_interface.embed(content),
    }
    assert node['latent']['shape'][0] == 1
    if len(self.graph) > 0: # This is additional semantics
      if len(self.graph) > 1: most_similar_node_id = self._get_most_similiar_node(node['latent'])
      else: most_similar_node_id = self.root_id # The Root Node
      self.data[content_id] = node
      self.graph.add_node(content_id)
      self.graph.add_edge(most_similar_node_id, content_id)
    else: # This is the root of the Semantic Latent Hierarchy
      self.data[content_id] = node
      self.graph.add_node(content_id, is_root=True)

  def organize(self):
    """Calculate the Hierarchy of the Semantic Information; replaces all current edges"""
    raise NotImplementedError()
  
  def compute_centroid(self) -> float:
    """Computes the Centroid of the Latent
    
    For now we just take the mean similarity of the latent's nodes.

    # TODO: Take a weighted centroid based on the hierarchy
    """
    if len(self.data) == 0: raise RuntimeError('Latent Graph is Empty')
    elif len(self.data) == 1: return 1.0
    root_id = self.root_id
    v1 = Embedding.unpack(self.data[root_id])[0]
    mean_sim = 0
    for n_id, n_latent in self.data.items():
      if n_id == root_id: continue
      v2 = Embedding.unpack(n_latent)[0]
      mean_sim += math.sumprod(v1, v2)
    return mean_sim / len(self.data)

  def render(self) -> str:
    """Renders the Semantic Latent as a Nested Element rooted at the original semantic"""

    """NOTE
    It should look like this (sans the newlines):
    ```
    <meta kind=root>
    {content}
    <meta kind=child1>
    {content}
    ...
    </meta>
    <meta kind=child2>
    {content}
    ...
    </meta>
    </meta>
    ```

    So walk the Tree, Depth First Post-Order Merging children renders with the parent's.
    """

    renders: dict[int, str] = {}
    for n_id in nx.dfs_postorder_nodes(self.graph, source=self.root_id):
      assert n_id not in renders
      n = self.data[n_id]
      # Merge the child renders
      children_render = ''.join(
        renders.pop(c_id) for c_id in self.graph[n_id]
      )
      renders[n_id] = f'<meta kind={n["kind"]}>{n["content"]}{children_render}</meta>'
    # At the end of things, we should only be left with the root render
    assert len(renders) == 1 and next(iter(renders)) == self.root_id, renders
    return next(iter(renders.values()))
