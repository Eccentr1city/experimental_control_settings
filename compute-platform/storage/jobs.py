"""
Job storage implementation
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import aiofiles
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient

from ..models.job import Job, JobStatus

logger = logging.getLogger(__name__)


class JobStore:
    """Abstract base class for job storage"""
    
    async def create_job(self, job: Job) -> str:
        """Create a new job"""
        raise NotImplementedError
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        raise NotImplementedError
    
    async def update_job(self, job: Job) -> bool:
        """Update an existing job"""
        raise NotImplementedError
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        raise NotImplementedError
    
    async def list_jobs(self, filters: Dict[str, Any] = None, 
                       page: int = 1, page_size: int = 20,
                       sort_by: str = "created_at", 
                       sort_order: str = "desc") -> Tuple[List[Job], int]:
        """List jobs with filtering and pagination"""
        raise NotImplementedError
    
    async def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        """Get all jobs with a specific status"""
        raise NotImplementedError
    
    async def get_jobs_by_user(self, user_id: str) -> List[Job]:
        """Get all jobs for a user"""
        raise NotImplementedError
    
    async def get_jobs_by_project(self, project_id: str) -> List[Job]:
        """Get all jobs for a project"""
        raise NotImplementedError


class MongoJobStore(JobStore):
    """MongoDB implementation of job storage"""
    
    def __init__(self, connection_string: str, database: str = "compute_platform"):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db.jobs
        
        # Create indexes
        asyncio.create_task(self._create_indexes())
    
    async def _create_indexes(self):
        """Create database indexes"""
        await self.collection.create_index("id", unique=True)
        await self.collection.create_index("status")
        await self.collection.create_index("metadata.owner_id")
        await self.collection.create_index("metadata.project_id")
        await self.collection.create_index("created_at")
        await self.collection.create_index([("status", 1), ("priority", -1)])
    
    async def create_job(self, job: Job) -> str:
        """Create a new job"""
        try:
            job_dict = self._job_to_dict(job)
            result = await self.collection.insert_one(job_dict)
            logger.info(f"Created job {job.id}")
            return job.id
        except Exception as e:
            logger.error(f"Failed to create job {job.id}: {str(e)}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        try:
            job_dict = await self.collection.find_one({"id": job_id})
            if job_dict:
                return self._dict_to_job(job_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {str(e)}")
            raise
    
    async def update_job(self, job: Job) -> bool:
        """Update an existing job"""
        try:
            job_dict = self._job_to_dict(job)
            result = await self.collection.replace_one(
                {"id": job.id}, 
                job_dict
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update job {job.id}: {str(e)}")
            raise
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        try:
            result = await self.collection.delete_one({"id": job_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {str(e)}")
            raise
    
    async def list_jobs(self, filters: Dict[str, Any] = None, 
                       page: int = 1, page_size: int = 20,
                       sort_by: str = "created_at", 
                       sort_order: str = "desc") -> Tuple[List[Job], int]:
        """List jobs with filtering and pagination"""
        try:
            query = filters or {}
            
            # Convert status enum to string if present
            if "status" in query and hasattr(query["status"], "value"):
                query["status"] = query["status"].value
            
            # Count total
            total = await self.collection.count_documents(query)
            
            # Get paginated results
            skip = (page - 1) * page_size
            sort_direction = -1 if sort_order == "desc" else 1
            
            cursor = self.collection.find(query).sort(
                sort_by, sort_direction
            ).skip(skip).limit(page_size)
            
            jobs = []
            async for job_dict in cursor:
                jobs.append(self._dict_to_job(job_dict))
            
            return jobs, total
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            raise
    
    async def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        """Get all jobs with a specific status"""
        try:
            cursor = self.collection.find({"status": status.value})
            jobs = []
            async for job_dict in cursor:
                jobs.append(self._dict_to_job(job_dict))
            return jobs
        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {str(e)}")
            raise
    
    async def get_jobs_by_user(self, user_id: str) -> List[Job]:
        """Get all jobs for a user"""
        try:
            cursor = self.collection.find({"metadata.owner_id": user_id})
            jobs = []
            async for job_dict in cursor:
                jobs.append(self._dict_to_job(job_dict))
            return jobs
        except Exception as e:
            logger.error(f"Failed to get jobs for user {user_id}: {str(e)}")
            raise
    
    async def get_jobs_by_project(self, project_id: str) -> List[Job]:
        """Get all jobs for a project"""
        try:
            cursor = self.collection.find({"metadata.project_id": project_id})
            jobs = []
            async for job_dict in cursor:
                jobs.append(self._dict_to_job(job_dict))
            return jobs
        except Exception as e:
            logger.error(f"Failed to get jobs for project {project_id}: {str(e)}")
            raise
    
    def _job_to_dict(self, job: Job) -> Dict[str, Any]:
        """Convert Job object to dictionary for storage"""
        # Convert dataclass to dict
        job_dict = {
            "id": job.id,
            "status": job.status.value,
            "priority": job.priority,
            "queue_name": job.queue_name,
            "command": job.command,
            "args": job.args,
            "entrypoint": job.entrypoint,
            "debug_mode": job.debug_mode,
            "dry_run": job.dry_run,
            
            # Nested objects
            "metadata": {
                "name": job.metadata.name,
                "namespace": job.metadata.namespace,
                "owner_id": job.metadata.owner_id,
                "project_id": job.metadata.project_id,
                "team_id": job.metadata.team_id,
                "labels": job.metadata.labels,
                "annotations": job.metadata.annotations,
                "tags": job.metadata.tags,
                "experiment_id": job.metadata.experiment_id,
                "run_id": job.metadata.run_id,
                "parent_job_id": job.metadata.parent_job_id,
                "workflow_id": job.metadata.workflow_id,
                "description": job.metadata.description,
                "dataset_ids": job.metadata.dataset_ids,
            },
            
            "resources": {
                "cpu_count": job.resources.cpu_count,
                "gpu_count": job.resources.gpu_count,
                "gpu_type": job.resources.gpu_type,
                "memory_gb": job.resources.memory_gb,
                "storage_gb": job.resources.storage_gb,
                "ephemeral_storage_gb": job.resources.ephemeral_storage_gb,
                "network_bandwidth_gbps": job.resources.network_bandwidth_gbps,
                "exclusive_node": job.resources.exclusive_node,
            },
            
            "environment": {
                "docker_image": job.environment.docker_image,
                "docker_registry": job.environment.docker_registry,
                "environment_variables": job.environment.environment_variables,
                "secrets": job.environment.secrets,
                "volumes": job.environment.volumes,
                "working_directory": job.environment.working_directory,
            },
            
            # Timestamps
            "created_at": job.created_at,
            "submitted_at": job.submitted_at,
            "queued_at": job.queued_at,
            "scheduled_at": job.scheduled_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "last_updated_at": job.last_updated_at,
            
            # Runtime info
            "assigned_node_id": job.assigned_node_id,
            "assigned_resources": job.assigned_resources,
            "container_id": job.container_id,
            "exit_code": job.exit_code,
            "failure_reason": job.failure_reason,
            
            # Metrics
            "cpu_seconds_used": job.cpu_seconds_used,
            "gpu_seconds_used": job.gpu_seconds_used,
            "memory_gb_hours": job.memory_gb_hours,
            "network_gb_transferred": job.network_gb_transferred,
            "storage_gb_hours": job.storage_gb_hours,
            "estimated_cost": job.estimated_cost,
            
            # Features
            "experimental_features": job.experimental_features,
            "feature_flags": job.feature_flags,
        }
        
        # Remove None values to save space
        return {k: v for k, v in job_dict.items() if v is not None}
    
    def _dict_to_job(self, job_dict: Dict[str, Any]) -> Job:
        """Convert dictionary from storage to Job object"""
        # This is a simplified version - in production would need full reconstruction
        from ..models.job import JobMetadata, JobEnvironment, JobResources
        
        job = Job(
            id=job_dict["id"],
            metadata=JobMetadata(
                name=job_dict["metadata"]["name"],
                owner_id=job_dict["metadata"]["owner_id"],
                project_id=job_dict["metadata"]["project_id"],
                **{k: v for k, v in job_dict["metadata"].items() 
                   if k not in ["name", "owner_id", "project_id"]}
            ),
            environment=JobEnvironment(
                docker_image=job_dict["environment"]["docker_image"],
                **{k: v for k, v in job_dict["environment"].items() 
                   if k != "docker_image"}
            ),
            resources=JobResources(**job_dict["resources"]),
            status=JobStatus(job_dict["status"]),
            priority=job_dict["priority"],
            queue_name=job_dict["queue_name"],
            command=job_dict.get("command", []),
            args=job_dict.get("args", []),
        )
        
        # Set timestamps
        for field in ["created_at", "submitted_at", "queued_at", "scheduled_at", 
                     "started_at", "completed_at", "last_updated_at"]:
            if field in job_dict and job_dict[field]:
                setattr(job, field, job_dict[field])
        
        # Set runtime info
        for field in ["assigned_node_id", "assigned_resources", "container_id",
                     "exit_code", "failure_reason"]:
            if field in job_dict:
                setattr(job, field, job_dict[field])
        
        return job


class PostgresJobStore(JobStore):
    """PostgreSQL implementation of job storage"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        
    async def initialize(self):
        """Initialize connection pool and create tables"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        await self._create_tables()
    
    async def _create_tables(self):
        """Create database tables"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(36) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL,
                    priority INTEGER NOT NULL,
                    queue_name VARCHAR(50) NOT NULL,
                    
                    -- Metadata
                    name VARCHAR(255) NOT NULL,
                    owner_id VARCHAR(36) NOT NULL,
                    project_id VARCHAR(36) NOT NULL,
                    
                    -- Serialized data
                    data JSONB NOT NULL,
                    
                    -- Timestamps
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    
                    -- Indexes
                    INDEX idx_status (status),
                    INDEX idx_owner (owner_id),
                    INDEX idx_project (project_id),
                    INDEX idx_created (created_at),
                    INDEX idx_priority (status, priority DESC)
                )
            ''')
    
    # Implementation would follow similar pattern to MongoDB


class FileJobStore(JobStore):
    """File-based job storage for development/testing"""
    
    def __init__(self, base_path: str = "/tmp/compute-platform/jobs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def create_job(self, job: Job) -> str:
        """Create a new job"""
        async with self._lock:
            job_path = self.base_path / f"{job.id}.json"
            
            if job_path.exists():
                raise ValueError(f"Job {job.id} already exists")
            
            job_dict = self._job_to_dict(job)
            
            async with aiofiles.open(job_path, 'w') as f:
                await f.write(json.dumps(job_dict, default=str, indent=2))
            
            logger.info(f"Created job {job.id}")
            return job.id
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        job_path = self.base_path / f"{job_id}.json"
        
        if not job_path.exists():
            return None
        
        async with aiofiles.open(job_path, 'r') as f:
            content = await f.read()
            job_dict = json.loads(content)
            return self._dict_to_job(job_dict)
    
    async def update_job(self, job: Job) -> bool:
        """Update an existing job"""
        async with self._lock:
            job_path = self.base_path / f"{job.id}.json"
            
            if not job_path.exists():
                return False
            
            job_dict = self._job_to_dict(job)
            
            async with aiofiles.open(job_path, 'w') as f:
                await f.write(json.dumps(job_dict, default=str, indent=2))
            
            return True
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        async with self._lock:
            job_path = self.base_path / f"{job_id}.json"
            
            if job_path.exists():
                job_path.unlink()
                return True
            
            return False
    
    # Simplified implementations of other methods
    def _job_to_dict(self, job: Job) -> Dict[str, Any]:
        """Convert job to dictionary"""
        # Simplified - would need proper serialization
        return {
            "id": job.id,
            "status": job.status.value,
            "priority": job.priority,
            "metadata": {
                "name": job.metadata.name,
                "owner_id": job.metadata.owner_id,
                "project_id": job.metadata.project_id,
            },
            "created_at": job.created_at.isoformat(),
        }
    
    def _dict_to_job(self, job_dict: Dict[str, Any]) -> Job:
        """Convert dictionary to job"""
        # Simplified - would need proper deserialization
        from ..models.job import JobMetadata, JobEnvironment, JobResources
        
        return Job(
            id=job_dict["id"],
            metadata=JobMetadata(name=job_dict["metadata"]["name"]),
            environment=JobEnvironment(docker_image="placeholder"),
            status=JobStatus(job_dict["status"]),
            priority=job_dict["priority"],
        ) 