"""
Job execution simulator - simulates jobs running without actual execution
"""
import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from models.job import Job, JobStatus
from storage.jobs import JobStore

logger = logging.getLogger(__name__)


class JobSimulator:
    """Simulates job execution with realistic timing and state transitions"""
    
    def __init__(self, job_store: JobStore):
        self.job_store = job_store
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(__name__)
    
    async def start_job_simulation(self, job: Job) -> bool:
        """Start simulating a job execution"""
        if job.id in self.running_jobs:
            return False
            
        # Update job to running state
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.assigned_node_id = job.assigned_node_id or f"node-{random.randint(1, 10):03d}"
        job.container_id = f"container-{job.id[:8]}"
        await self.job_store.update_job(job)
        
        # Start simulation task
        task = asyncio.create_task(self._simulate_job_execution(job))
        self.running_jobs[job.id] = task
        self.logger.info(f"Started simulation for job {job.id}")
        return True
    
    async def _simulate_job_execution(self, job: Job):
        """Simulate the job execution process"""
        try:
            # Simulate startup time (1-5 seconds)
            await asyncio.sleep(random.uniform(1, 5))
            
            # Calculate simulated duration based on resources and job type
            base_duration = 30  # 30 seconds base
            gpu_factor = job.resources.gpu_count * 0.5 if job.resources.gpu_count else 0
            memory_factor = job.resources.memory_gb * 0.1 if job.resources.memory_gb else 0
            duration = base_duration + random.uniform(10, 60) + gpu_factor + memory_factor
            
            # Simulate execution with periodic updates
            start_time = datetime.utcnow()
            update_interval = 5  # Update every 5 seconds
            
            while True:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed >= duration:
                    break
                    
                # Update job metrics
                await self._update_job_metrics(job, elapsed, duration)
                await asyncio.sleep(update_interval)
            
            # Job completion (90% success rate)
            await self._complete_job(job, success=random.random() > 0.1)
            
        except asyncio.CancelledError:
            # Job was cancelled
            await self._complete_job(job, success=False, cancelled=True)
        except Exception as e:
            self.logger.error(f"Job simulation error for {job.id}: {e}")
            await self._complete_job(job, success=False)
        finally:
            self.running_jobs.pop(job.id, None)
    
    async def _update_job_metrics(self, job: Job, elapsed: float, total_duration: float):
        """Update job with realistic metrics"""
        progress = elapsed / total_duration
        
        # Generate realistic metrics
        cpu_count = job.resources.cpu_count if job.resources.cpu_count else 1
        gpu_count = job.resources.gpu_count if job.resources.gpu_count else 0
        memory_gb = job.resources.memory_gb if job.resources.memory_gb else 1
        
        # Simulate realistic resource usage patterns
        cpu_utilization = random.uniform(0.5, 0.9)
        gpu_utilization = random.uniform(0.7, 0.95) if gpu_count > 0 else 0
        memory_utilization = random.uniform(0.6, 0.8)
        
        job.cpu_seconds_used = elapsed * cpu_count * cpu_utilization
        job.gpu_seconds_used = elapsed * gpu_count * gpu_utilization
        job.memory_gb_hours = (elapsed / 3600) * memory_gb * memory_utilization
        job.network_gb_transferred = progress * random.uniform(0.1, 5.0)
        
        # Simple cost calculation (this could be more sophisticated)
        cpu_cost = job.cpu_seconds_used * 0.0001  # $0.0001 per CPU second
        gpu_cost = job.gpu_seconds_used * 0.001   # $0.001 per GPU second
        memory_cost = job.memory_gb_hours * 0.01  # $0.01 per GB hour
        job.estimated_cost = cpu_cost + gpu_cost + memory_cost
        
        job.last_updated_at = datetime.utcnow()
        await self.job_store.update_job(job)
    
    async def _complete_job(self, job: Job, success: bool, cancelled: bool = False):
        """Complete job simulation"""
        job.completed_at = datetime.utcnow()
        
        if cancelled:
            job.status = JobStatus.CANCELLED
            job.exit_code = 130  # SIGINT
            job.failure_reason = "Job was cancelled by user"
        elif success:
            job.status = JobStatus.COMPLETED
            job.exit_code = 0
        else:
            job.status = JobStatus.FAILED
            job.exit_code = random.choice([1, 2, 125, 126])
            job.failure_reason = random.choice([
                "Out of memory error",
                "CUDA out of memory", 
                "Network timeout",
                "Container image pull failed",
                "Permission denied",
                "Disk space exhausted",
                "Process killed by OOM killer"
            ])
        
        await self.job_store.update_job(job)
        self.logger.info(f"Job {job.id} completed with status {job.status}")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job simulation"""
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            self.logger.info(f"Cancelled job simulation for {job_id}")
            return True
        return False
    
    async def get_running_jobs(self) -> List[str]:
        """Get list of currently running job IDs"""
        return list(self.running_jobs.keys())
    
    async def is_job_running(self, job_id: str) -> bool:
        """Check if a job is currently running"""
        return job_id in self.running_jobs 