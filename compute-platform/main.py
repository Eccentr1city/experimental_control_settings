"""
Main application entry point for the compute platform
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from storage.jobs import MongoJobStore
from storage.cluster import MockClusterStore
from services.resource_manager import ResourceManager
from services.monitoring import MonitoringService
from scheduler.core import Scheduler
from models.job import Job, JobMetadata, JobEnvironment, JobResources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ComputePlatform:
    """Main compute platform application"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize storage
        self.job_store = MongoJobStore("mongodb://localhost:27017")
        self.cluster_store = MockClusterStore()
        
        # Initialize services
        self.resource_manager = ResourceManager()
        self.monitoring_service = MonitoringService()
        
        # Initialize scheduler
        self.scheduler = Scheduler(
            job_store=self.job_store,
            cluster_store=self.cluster_store,
            resource_manager=self.resource_manager,
            monitoring_service=self.monitoring_service,
            config=self.config.get('scheduler', {})
        )
    
    async def start(self):
        """Start the compute platform"""
        logger.info("Starting Compute Platform")
        
        # Start monitoring
        await self.monitoring_service.start_monitoring()
        
        # Start scheduler
        await self.scheduler.start()
        
        logger.info("Compute Platform started successfully")
    
    async def stop(self):
        """Stop the compute platform"""
        logger.info("Stopping Compute Platform")
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # Stop monitoring
        await self.monitoring_service.stop_monitoring()
        
        logger.info("Compute Platform stopped")
    
    async def submit_test_job(self) -> str:
        """Submit a test job for demonstration"""
        job = Job(
            metadata=JobMetadata(
                name="test-simulation-job",
                owner_id="test-user",
                project_id="test-project",
                description="Test job for simulation"
            ),
            environment=JobEnvironment(
                docker_image="python:3.9",
                environment_variables={"TEST_VAR": "test_value"}
            ),
            resources=JobResources(
                cpu_count=2,
                memory_gb=4,
                gpu_count=1,
                gpu_type="A100"
            ),
            command=["python", "-c", "print('Hello from simulated job!')"],
            priority=1
        )
        
        # Submit job through scheduler
        job_id = await self.scheduler.submit_job(job)
        logger.info(f"Submitted test job: {job_id}")
        return job_id


async def main():
    """Main application entry point"""
    platform = ComputePlatform()
    
    try:
        # Start the platform
        await platform.start()
        
        # Submit a test job
        job_id = await platform.submit_test_job()
        
        # Wait a bit to see the simulation in action
        logger.info("Waiting for job simulation to complete...")
        await asyncio.sleep(60)  # Wait 1 minute
        
        # Check job status
        job = await platform.job_store.get_job(job_id)
        if job:
            logger.info(f"Job {job_id} final status: {job.status}")
            logger.info(f"Job metrics - CPU: {job.cpu_seconds_used:.2f}s, "
                       f"GPU: {job.gpu_seconds_used:.2f}s, "
                       f"Cost: ${job.estimated_cost:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await platform.stop()


if __name__ == "__main__":
    asyncio.run(main()) 