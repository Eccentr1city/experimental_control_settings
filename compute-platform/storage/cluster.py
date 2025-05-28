"""
Cluster storage implementation
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from models.cluster import Cluster, Node, NodeStatus, ClusterState

logger = __import__('logging').getLogger(__name__)


class ClusterStore:
    """Abstract base class for cluster storage"""
    
    async def create_cluster(self, cluster: Cluster) -> str:
        """Create a new cluster"""
        raise NotImplementedError
    
    async def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID"""
        raise NotImplementedError
    
    async def list_clusters(self) -> List[Cluster]:
        """List all clusters"""
        raise NotImplementedError
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        raise NotImplementedError
    
    async def list_nodes(self, cluster_id: Optional[str] = None) -> List[Node]:
        """List nodes, optionally filtered by cluster"""
        raise NotImplementedError


class MockClusterStore(ClusterStore):
    """Mock cluster store for development"""
    
    def __init__(self):
        self.clusters: Dict[str, Cluster] = {}
        self.nodes: Dict[str, Node] = {}
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize mock cluster and node data"""
        # Create a default cluster
        cluster = Cluster(
            id="default-cluster",
            name="Default Cluster",
            region="us-west-2",
            state=ClusterState.ACTIVE,
            created_at=datetime.utcnow()
        )
        self.clusters[cluster.id] = cluster
        
        # Create mock nodes
        for i in range(1, 11):
            node_id = f"node-{i:03d}"
            node = Node(
                id=node_id,
                cluster_id="default-cluster",
                hostname=f"worker-{i:03d}.cluster.local",
                status=NodeStatus.READY,
                labels={
                    "node.kubernetes.io/instance-type": f"m5.{['xlarge', 'large', 'medium'][i % 3]}",
                    "zone": f"us-west-2{chr(ord('a') + (i % 3))}"
                }
            )
            self.nodes[node_id] = node
    
    async def create_cluster(self, cluster: Cluster) -> str:
        """Create a new cluster"""
        self.clusters[cluster.id] = cluster
        logger.info(f"Created cluster {cluster.id}")
        return cluster.id
    
    async def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID"""
        return self.clusters.get(cluster_id)
    
    async def list_clusters(self) -> List[Cluster]:
        """List all clusters"""
        return list(self.clusters.values())
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    async def list_nodes(self, cluster_id: Optional[str] = None) -> List[Node]:
        """List nodes, optionally filtered by cluster"""
        nodes = list(self.nodes.values())
        if cluster_id:
            nodes = [n for n in nodes if n.cluster_id == cluster_id]
        return nodes 