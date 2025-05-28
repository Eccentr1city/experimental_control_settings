"""
Resource management service - tracks and allocates resources
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from models.resource import ResourceType, ResourceQuantity
from models.cluster import Node, NodeStatus

logger = __import__('logging').getLogger(__name__)


class ResourceManager:
    """Mock resource manager for tracking allocations"""
    
    def __init__(self):
        self.allocations: Dict[str, Dict[str, Any]] = {}  # job_id -> allocation info
        self.nodes: Dict[str, Dict[str, Any]] = {}  # Using Dict for mock data
        self.node_resources: Dict[str, Dict[str, float]] = {}  # node_id -> resource usage
        self._initialize_mock_cluster()
    
    def _initialize_mock_cluster(self):
        """Initialize mock cluster with nodes and resources"""
        # Create mock nodes with different resource profiles
        node_configs = [
            {"cpu": 32, "memory": 128, "gpu": 4, "storage": 1000},  # High-end GPU nodes
            {"cpu": 16, "memory": 64, "gpu": 2, "storage": 500},   # Medium GPU nodes
            {"cpu": 8, "memory": 32, "gpu": 0, "storage": 200},    # CPU-only nodes
        ]
        
        for i in range(1, 11):
            node_id = f"node-{i:03d}"
            config = node_configs[i % len(node_configs)]
            
            # Create mock node (simplified structure)
            self.nodes[node_id] = {
                "id": node_id,
                "hostname": f"worker-{i:03d}.cluster.local",
                "status": "ready",
                "total_resources": config,
                "labels": {
                    "node.kubernetes.io/instance-type": f"m5.{['xlarge', 'large', 'medium'][i % 3]}",
                    "zone": f"us-west-2{chr(ord('a') + (i % 3))}"
                }
            }
            
            # Initialize resource tracking
            self.node_resources[node_id] = {
                "cpu_used": 0.0,
                "memory_used": 0.0,
                "gpu_used": 0.0,
                "storage_used": 0.0
            }
    
    async def reserve_resources(self, node_id: str, job_id: str, 
                               resources: Dict[ResourceType, ResourceQuantity]) -> bool:
        """Reserve resources for a job on a specific node"""
        try:
            if node_id not in self.nodes:
                logger.error(f"Node {node_id} not found")
                return False
            
            node = self.nodes[node_id]
            node_usage = self.node_resources[node_id]
            
            # Check if resources are available
            required_cpu = 0
            required_memory = 0
            required_gpu = 0
            required_storage = 0
            
            for resource_type, quantity in resources.items():
                if resource_type == ResourceType.CPU:
                    required_cpu = quantity.value
                elif resource_type == ResourceType.MEMORY:
                    required_memory = quantity.value
                elif resource_type == ResourceType.GPU:
                    required_gpu = quantity.value
                elif resource_type == ResourceType.STORAGE:
                    required_storage = quantity.value
            
            # Check availability
            total_cpu = node["total_resources"]["cpu"]
            total_memory = node["total_resources"]["memory"]
            total_gpu = node["total_resources"]["gpu"]
            total_storage = node["total_resources"]["storage"]
            
            if (node_usage["cpu_used"] + required_cpu > total_cpu or
                node_usage["memory_used"] + required_memory > total_memory or
                node_usage["gpu_used"] + required_gpu > total_gpu or
                node_usage["storage_used"] + required_storage > total_storage):
                logger.warning(f"Insufficient resources on node {node_id}")
                return False
            
            # Reserve resources
            node_usage["cpu_used"] += required_cpu
            node_usage["memory_used"] += required_memory
            node_usage["gpu_used"] += required_gpu
            node_usage["storage_used"] += required_storage
            
            # Track allocation
            self.allocations[job_id] = {
                "node_id": node_id,
                "resources": {
                    "cpu": required_cpu,
                    "memory": required_memory,
                    "gpu": required_gpu,
                    "storage": required_storage
                },
                "allocated_at": datetime.utcnow()
            }
            
            logger.info(f"Reserved resources for job {job_id} on node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reserve resources for job {job_id}: {str(e)}")
            return False
    
    async def release_resources(self, node_id: str, job_id: str, 
                               resources: Dict[ResourceType, ResourceQuantity]) -> bool:
        """Release resources allocated to a job"""
        try:
            if job_id not in self.allocations:
                logger.warning(f"No allocation found for job {job_id}")
                return False
            
            allocation = self.allocations[job_id]
            if allocation["node_id"] != node_id:
                logger.error(f"Job {job_id} not allocated to node {node_id}")
                return False
            
            # Release resources
            node_usage = self.node_resources[node_id]
            allocated_resources = allocation["resources"]
            
            node_usage["cpu_used"] -= allocated_resources["cpu"]
            node_usage["memory_used"] -= allocated_resources["memory"]
            node_usage["gpu_used"] -= allocated_resources["gpu"]
            node_usage["storage_used"] -= allocated_resources["storage"]
            
            # Ensure no negative values
            for key in node_usage:
                node_usage[key] = max(0, node_usage[key])
            
            # Remove allocation
            del self.allocations[job_id]
            
            logger.info(f"Released resources for job {job_id} from node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to release resources for job {job_id}: {str(e)}")
            return False
    
    async def get_node_availability(self, node_id: str) -> Optional[Dict[str, float]]:
        """Get available resources on a node"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        usage = self.node_resources[node_id]
        total = node["total_resources"]
        
        return {
            "cpu_available": total["cpu"] - usage["cpu_used"],
            "memory_available": total["memory"] - usage["memory_used"],
            "gpu_available": total["gpu"] - usage["gpu_used"],
            "storage_available": total["storage"] - usage["storage_used"]
        }
    
    async def list_nodes(self) -> List[Dict[str, Any]]:
        """List all nodes with their resource status"""
        nodes_info = []
        for node_id, node in self.nodes.items():
            availability = await self.get_node_availability(node_id)
            nodes_info.append({
                "id": node_id,
                "hostname": node["hostname"],
                "status": node["status"],
                "total_resources": node["total_resources"],
                "available_resources": availability,
                "used_resources": self.node_resources[node_id],
                "labels": node["labels"]
            })
        return nodes_info
    
    async def get_cluster_summary(self) -> Dict[str, Any]:
        """Get overall cluster resource summary"""
        total_cpu = sum(node["total_resources"]["cpu"] for node in self.nodes.values())
        total_memory = sum(node["total_resources"]["memory"] for node in self.nodes.values())
        total_gpu = sum(node["total_resources"]["gpu"] for node in self.nodes.values())
        total_storage = sum(node["total_resources"]["storage"] for node in self.nodes.values())
        
        used_cpu = sum(usage["cpu_used"] for usage in self.node_resources.values())
        used_memory = sum(usage["memory_used"] for usage in self.node_resources.values())
        used_gpu = sum(usage["gpu_used"] for usage in self.node_resources.values())
        used_storage = sum(usage["storage_used"] for usage in self.node_resources.values())
        
        return {
            "total_nodes": len(self.nodes),
            "active_jobs": len(self.allocations),
            "resources": {
                "cpu": {"total": total_cpu, "used": used_cpu, "available": total_cpu - used_cpu},
                "memory": {"total": total_memory, "used": used_memory, "available": total_memory - used_memory},
                "gpu": {"total": total_gpu, "used": used_gpu, "available": total_gpu - used_gpu},
                "storage": {"total": total_storage, "used": used_storage, "available": total_storage - used_storage}
            }
        } 