"""
Core scheduler implementation for job scheduling and resource allocation
"""
import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import threading
from concurrent.futures import ThreadPoolExecutor

from ..models.job import Job, JobStatus, JobPriority
from ..models.resource import (
    ResourceType, ResourceQuantity, Allocation, 
    AllocationStatus, ResourcePool
)
from ..models.cluster import Cluster, Node, NodeStatus
from ..models.schedule import (
    Schedule, SchedulingPolicy, QueueEntry, 
    SchedulingContext, SchedulingResult, SchedulingDecision
)
from ..models.user import User, Project, Quota
from ..storage.jobs import JobStore
from ..storage.cluster import ClusterStore
from ..services.resource_manager import ResourceManager
from ..services.monitoring import MonitoringService
from ..utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class SchedulerState:
    """Maintains scheduler state"""
    def __init__(self):
        # Queues by priority
        self.pending_queue: List[QueueEntry] = []  # Min heap by priority
        self.running_jobs: Dict[str, Job] = {}
        self.job_allocations: Dict[str, Allocation] = {}
        
        # Resource tracking
        self.cluster_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.node_allocations: Dict[str, Set[str]] = defaultdict(set)  # node_id -> job_ids
        
        # Fair share tracking
        self.user_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.project_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.historical_usage: deque = deque(maxlen=10000)  # Rolling window
        
        # Gang scheduling
        self.gang_groups: Dict[str, Set[str]] = defaultdict(set)  # gang_id -> job_ids
        self.gang_ready: Dict[str, bool] = {}
        
        # Preemption tracking
        self.preemption_history: Dict[str, List[datetime]] = defaultdict(list)
        self.preempted_jobs: Set[str] = set()
        
        # Backfill
        self.backfill_windows: List[Tuple[datetime, datetime, Dict[str, float]]] = []
        
        # Locks
        self.state_lock = threading.RLock()
        self.scheduling_lock = threading.Lock()


class Scheduler:
    """Main scheduler implementation"""
    
    def __init__(self, 
                 job_store: JobStore,
                 cluster_store: ClusterStore,
                 resource_manager: ResourceManager,
                 monitoring_service: MonitoringService,
                 config: Dict[str, Any] = None):
        self.job_store = job_store
        self.cluster_store = cluster_store
        self.resource_manager = resource_manager
        self.monitoring = monitoring_service
        self.config = config or {}
        
        # Scheduler state
        self.state = SchedulerState()
        
        # Scheduling configuration
        self.schedule_config = Schedule(
            primary_policy=SchedulingPolicy(self.config.get('primary_policy', 'priority')),
            secondary_policy=SchedulingPolicy(self.config.get('secondary_policy', 'fair_share')),
            enable_preemption=self.config.get('enable_preemption', True),
            enable_backfill=self.config.get('enable_backfill', True),
            enable_gang_scheduling=self.config.get('enable_gang_scheduling', False)
        )
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Scheduling loop control
        self.running = False
        self.scheduling_interval = self.config.get('scheduling_interval', 5)  # seconds
        self.scheduling_thread = None
        
        # Metrics
        self.metrics = MetricsCollector('scheduler')
        
    async def start(self):
        """Start the scheduler"""
        logger.info("Starting scheduler")
        self.running = True
        
        # Load existing state
        await self._load_state()
        
        # Start scheduling loop
        self.scheduling_thread = threading.Thread(target=self._scheduling_loop)
        self.scheduling_thread.start()
        
        # Start background tasks
        asyncio.create_task(self._monitor_jobs())
        asyncio.create_task(self._update_fair_share())
        asyncio.create_task(self._cleanup_completed_jobs())
        
    async def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler")
        self.running = False
        
        if self.scheduling_thread:
            self.scheduling_thread.join()
        
        self.executor.shutdown(wait=True)
        
    async def submit_job(self, job: Job) -> str:
        """Submit a job to the scheduler"""
        start_time = time.time()
        
        try:
            with self.state.state_lock:
                # Create queue entry
                queue_entry = QueueEntry(
                    job_id=job.id,
                    queue_name=job.queue_name,
                    priority=job.priority,
                    user_id=job.metadata.owner_id,
                    project_id=job.metadata.project_id,
                    resource_requirements=self._get_resource_requirements(job),
                    preemptible=job.scheduling_constraints.preemptible,
                    estimated_runtime=self._estimate_runtime(job)
                )
                
                # Handle gang scheduling
                if job.distributed.is_distributed and self.schedule_config.enable_gang_scheduling:
                    gang_id = job.distributed.gang_id or f"gang-{job.id}"
                    self.state.gang_groups[gang_id].add(job.id)
                    queue_entry.gang_id = gang_id
                    queue_entry.gang_size = job.distributed.world_size
                
                # Add to pending queue
                heapq.heappush(self.state.pending_queue, 
                             (-queue_entry.effective_priority, queue_entry))
                
                # Update job status
                job.status = JobStatus.QUEUED
                job.queued_at = datetime.utcnow()
                await self.job_store.update_job(job)
                
                logger.info(f"Job {job.id} submitted to queue {job.queue_name}")
                
                # Trigger immediate scheduling attempt
                self._trigger_scheduling()
                
                # Metrics
                self.metrics.increment('jobs_submitted')
                self.metrics.histogram('job_submission_latency', 
                                     time.time() - start_time)
                
                return job.id
                
        except Exception as e:
            logger.error(f"Failed to submit job {job.id}: {str(e)}")
            self.metrics.increment('job_submission_failures')
            raise
    
    def _scheduling_loop(self):
        """Main scheduling loop"""
        while self.running:
            try:
                with self.state.scheduling_lock:
                    asyncio.run(self._schedule_jobs())
            except Exception as e:
                logger.error(f"Scheduling loop error: {str(e)}")
                self.metrics.increment('scheduling_errors')
            
            time.sleep(self.scheduling_interval)
    
    async def _schedule_jobs(self):
        """Core scheduling logic"""
        start_time = time.time()
        scheduled_count = 0
        
        try:
            # Get current cluster state
            clusters = await self.cluster_store.list_clusters()
            if not clusters:
                logger.warning("No clusters available for scheduling")
                return
            
            # Build scheduling context
            context = await self._build_scheduling_context(clusters)
            
            # Get jobs to schedule
            jobs_to_schedule = self._get_schedulable_jobs(context)
            
            for queue_entry in jobs_to_schedule:
                job = await self.job_store.get_job(queue_entry.job_id)
                if not job:
                    continue
                
                # Try to schedule job
                result = await self._try_schedule_job(job, queue_entry, context)
                
                if result.decision == SchedulingDecision.SCHEDULED:
                    scheduled_count += 1
                    await self._execute_scheduling_decision(job, result)
                elif result.decision == SchedulingDecision.PREEMPTED:
                    # Handle preemption
                    await self._handle_preemption(job, result)
                elif result.decision == SchedulingDecision.DEFERRED:
                    # Update queue entry for next attempt
                    queue_entry.scheduling_attempts += 1
                    queue_entry.last_scheduling_attempt = datetime.utcnow()
                    queue_entry.last_failure_reason = result.reasons[0] if result.reasons else None
            
            # Backfill scheduling
            if self.schedule_config.enable_backfill:
                backfilled = await self._backfill_schedule(context)
                scheduled_count += backfilled
            
            # Update metrics
            self.metrics.gauge('pending_jobs', len(self.state.pending_queue))
            self.metrics.gauge('running_jobs', len(self.state.running_jobs))
            self.metrics.histogram('scheduling_cycle_duration', time.time() - start_time)
            self.metrics.increment('jobs_scheduled', scheduled_count)
            
            if scheduled_count > 0:
                logger.info(f"Scheduled {scheduled_count} jobs in this cycle")
                
        except Exception as e:
            logger.error(f"Error in scheduling cycle: {str(e)}")
            self.metrics.increment('scheduling_cycle_errors')
    
    async def _try_schedule_job(self, job: Job, queue_entry: QueueEntry, 
                               context: SchedulingContext) -> SchedulingResult:
        """Try to schedule a single job"""
        result = SchedulingResult(
            job_id=job.id,
            decision=SchedulingDecision.QUEUED
        )
        
        try:
            # Check dependencies
            if not await self._check_dependencies(job):
                result.decision = SchedulingDecision.DEFERRED
                result.reasons.append("Dependencies not satisfied")
                return result
            
            # Check quotas
            if not await self._check_quotas(job):
                result.decision = SchedulingDecision.REJECTED
                result.reasons.append("Quota exceeded")
                return result
            
            # Gang scheduling check
            if queue_entry.gang_id and not self._is_gang_ready(queue_entry.gang_id):
                result.decision = SchedulingDecision.DEFERRED
                result.reasons.append("Waiting for gang members")
                return result
            
            # Find suitable nodes
            suitable_nodes = await self._find_suitable_nodes(job, context)
            
            if not suitable_nodes:
                # Try preemption if enabled
                if self.schedule_config.enable_preemption:
                    preemption_result = await self._try_preemption(job, context)
                    if preemption_result:
                        result.decision = SchedulingDecision.PREEMPTED
                        result.preempted_jobs = preemption_result['preempted_jobs']
                        result.assigned_node = preemption_result['node_id']
                        result.assigned_resources = preemption_result['resources']
                        return result
                
                result.decision = SchedulingDecision.DEFERRED
                result.reasons.append("No suitable nodes available")
                return result
            
            # Score and rank nodes
            node_scores = await self._score_nodes(job, suitable_nodes, context)
            best_node = max(node_scores, key=node_scores.get)
            
            # Allocate resources
            allocation = await self._allocate_resources(job, best_node, context)
            
            if allocation:
                result.decision = SchedulingDecision.SCHEDULED
                result.assigned_node = best_node
                result.assigned_resources = allocation.resources
                result.score = node_scores[best_node]
                result.satisfied_constraints = self._get_satisfied_constraints(job, best_node)
            else:
                result.decision = SchedulingDecision.DEFERRED
                result.reasons.append("Resource allocation failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scheduling job {job.id}: {str(e)}")
            result.decision = SchedulingDecision.DEFERRED
            result.reasons.append(f"Scheduling error: {str(e)}")
            return result
    
    async def _find_suitable_nodes(self, job: Job, context: SchedulingContext) -> List[str]:
        """Find nodes that can potentially run the job"""
        suitable_nodes = []
        requirements = self._get_resource_requirements(job)
        
        for node_id in context.available_nodes:
            node = await self.cluster_store.get_node(node_id)
            if not node or node.status != NodeStatus.READY:
                continue
            
            # Check basic resource availability
            if not self._check_node_resources(node, requirements, context):
                continue
            
            # Check node selector
            if job.scheduling_constraints.node_selector:
                if not all(node.labels.get(k) == v 
                          for k, v in job.scheduling_constraints.node_selector.items()):
                    continue
            
            # Check taints and tolerations
            if node.taints:
                if not self._check_tolerations(job, node):
                    continue
            
            # Check affinity rules
            if not self._check_affinity(job, node, context):
                continue
            
            suitable_nodes.append(node_id)
        
        return suitable_nodes
    
    async def _score_nodes(self, job: Job, nodes: List[str], 
                          context: SchedulingContext) -> Dict[str, float]:
        """Score nodes for job placement"""
        scores = {}
        
        for node_id in nodes:
            score = 0.0
            node = await self.cluster_store.get_node(node_id)
            
            # Resource fit score (prefer nodes with just enough resources)
            resource_score = self._calculate_resource_fit_score(job, node, context)
            score += resource_score * 0.3
            
            # Affinity score
            affinity_score = self._calculate_affinity_score(job, node, context)
            score += affinity_score * 0.2
            
            # Load balancing score (prefer less loaded nodes)
            load_score = self._calculate_load_score(node, context)
            score += load_score * 0.2
            
            # Data locality score
            locality_score = await self._calculate_locality_score(job, node)
            score += locality_score * 0.2
            
            # Cost score (prefer cheaper nodes)
            cost_score = self._calculate_cost_score(job, node)
            score += cost_score * 0.1
            
            scores[node_id] = score
        
        return scores
    
    async def _allocate_resources(self, job: Job, node_id: str, 
                                 context: SchedulingContext) -> Optional[Allocation]:
        """Allocate resources for a job on a node"""
        try:
            node = await self.cluster_store.get_node(node_id)
            requirements = self._get_resource_requirements(job)
            
            # Create allocation request
            allocation = Allocation(
                job_id=job.id,
                node_id=node_id,
                resources=requirements,
                status=AllocationStatus.PENDING
            )
            
            # Reserve resources on node
            success = await self.resource_manager.reserve_resources(
                node_id=node_id,
                job_id=job.id,
                resources=requirements
            )
            
            if success:
                allocation.status = AllocationStatus.ALLOCATED
                allocation.allocated_at = datetime.utcnow()
                
                # Update state
                with self.state.state_lock:
                    self.state.job_allocations[job.id] = allocation
                    self.state.node_allocations[node_id].add(job.id)
                
                # Update resource tracking
                self._update_resource_usage(node_id, requirements, allocate=True)
                
                return allocation
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to allocate resources for job {job.id}: {str(e)}")
            return None
    
    async def _execute_scheduling_decision(self, job: Job, result: SchedulingResult):
        """Execute a scheduling decision"""
        try:
            # Update job with allocation info
            job.status = JobStatus.SCHEDULED
            job.scheduled_at = datetime.utcnow()
            job.assigned_node_id = result.assigned_node
            job.assigned_resources = result.assigned_resources
            
            await self.job_store.update_job(job)
            
            # Move to running state
            with self.state.state_lock:
                self.state.running_jobs[job.id] = job
                # Remove from pending queue
                self.state.pending_queue = [
                    (p, e) for p, e in self.state.pending_queue 
                    if e.job_id != job.id
                ]
                heapq.heapify(self.state.pending_queue)
            
            # Start job execution
            asyncio.create_task(self._start_job_execution(job))
            
            # Update metrics
            wait_time = (job.scheduled_at - job.queued_at).total_seconds()
            self.metrics.histogram('job_wait_time', wait_time)
            
            logger.info(f"Job {job.id} scheduled on node {result.assigned_node}")
            
        except Exception as e:
            logger.error(f"Failed to execute scheduling decision for job {job.id}: {str(e)}")
            # Rollback allocation
            await self._rollback_allocation(job.id, result.assigned_node)
    
    async def _try_preemption(self, job: Job, context: SchedulingContext) -> Optional[Dict[str, Any]]:
        """Try to preempt lower priority jobs to make room"""
        if job.priority < self.schedule_config.preemption_priority_threshold:
            return None
        
        requirements = self._get_resource_requirements(job)
        
        # Find preemption candidates
        candidates = []
        
        for node_id in context.available_nodes:
            node_jobs = self.state.node_allocations.get(node_id, set())
            preemptible_jobs = []
            freed_resources = defaultdict(float)
            
            for job_id in node_jobs:
                running_job = self.state.running_jobs.get(job_id)
                if not running_job:
                    continue
                
                # Check if job can be preempted
                if (running_job.scheduling_constraints.preemptible and
                    running_job.priority < job.priority and
                    self._can_preempt_job(running_job)):
                    
                    job_resources = self.state.job_allocations[job_id].resources
                    preemptible_jobs.append(job_id)
                    
                    for resource_type, quantity in job_resources.items():
                        freed_resources[resource_type] += quantity.value
                    
                    # Check if we have enough resources
                    if all(freed_resources.get(rt, 0) >= req.value 
                          for rt, req in requirements.items()):
                        candidates.append({
                            'node_id': node_id,
                            'preempted_jobs': preemptible_jobs[:],
                            'freed_resources': dict(freed_resources),
                            'cost': len(preemptible_jobs)  # Minimize preemptions
                        })
                        break
        
        if not candidates:
            return None
        
        # Select best preemption option (minimize disruption)
        best_option = min(candidates, key=lambda x: x['cost'])
        
        return best_option
    
    def _can_preempt_job(self, job: Job) -> bool:
        """Check if a job can be preempted"""
        # Check preemption count
        preemption_count = len(self.state.preemption_history.get(job.id, []))
        if preemption_count >= job.scheduling_constraints.max_restart_count:
            return False
        
        # Check minimum runtime
        if job.started_at:
            runtime = (datetime.utcnow() - job.started_at).total_seconds()
            min_runtime = job.scheduling_constraints.min_runtime_seconds or 300
            if runtime < min_runtime:
                return False
        
        # Check if job is checkpointing
        if job.status == JobStatus.CHECKPOINTING:
            return False
        
        return True
    
    async def _handle_preemption(self, new_job: Job, result: SchedulingResult):
        """Handle job preemption"""
        try:
            # Preempt jobs
            for job_id in result.preempted_jobs:
                await self._preempt_job(job_id, reason=f"Preempted by higher priority job {new_job.id}")
            
            # Wait for preemptions to complete
            await asyncio.sleep(self.schedule_config.preemption_grace_period)
            
            # Schedule new job
            await self._execute_scheduling_decision(new_job, result)
            
        except Exception as e:
            logger.error(f"Failed to handle preemption for job {new_job.id}: {str(e)}")
    
    async def _preempt_job(self, job_id: str, reason: str):
        """Preempt a running job"""
        try:
            job = self.state.running_jobs.get(job_id)
            if not job:
                return
            
            logger.info(f"Preempting job {job_id}: {reason}")
            
            # Update preemption history
            self.state.preemption_history[job_id].append(datetime.utcnow())
            self.state.preempted_jobs.add(job_id)
            
            # Checkpoint if enabled
            if job.checkpointing.checkpoint_on_preemption:
                job.status = JobStatus.CHECKPOINTING
                await self.job_store.update_job(job)
                
                # Trigger checkpoint
                await self._checkpoint_job(job)
            
            # Stop job execution
            await self._stop_job_execution(job, preempted=True)
            
            # Update job status
            job.status = JobStatus.PREEMPTED
            job.failure_reason = reason
            await self.job_store.update_job(job)
            
            # Release resources
            await self._release_job_resources(job)
            
            # Re-queue job
            if job.scheduling_constraints.restart_policy != "never":
                await self.submit_job(job)
            
            self.metrics.increment('jobs_preempted')
            
        except Exception as e:
            logger.error(f"Failed to preempt job {job_id}: {str(e)}")
    
    async def _backfill_schedule(self, context: SchedulingContext) -> int:
        """Perform backfill scheduling for better utilization"""
        if not self.schedule_config.enable_backfill:
            return 0
        
        scheduled_count = 0
        
        try:
            # Calculate resource availability windows
            windows = await self._calculate_availability_windows(context)
            
            # Sort pending jobs by estimated runtime
            backfill_candidates = []
            for _, entry in self.state.pending_queue:
                if entry.backfill_eligible and entry.estimated_runtime:
                    backfill_candidates.append(entry)
            
            backfill_candidates.sort(key=lambda x: x.estimated_runtime)
            
            # Try to fit jobs in windows
            for window_start, window_end, available_resources in windows:
                window_duration = (window_end - window_start).total_seconds()
                
                for entry in backfill_candidates[:]:
                    if entry.estimated_runtime and entry.estimated_runtime <= window_duration:
                        job = await self.job_store.get_job(entry.job_id)
                        if not job:
                            continue
                        
                        requirements = self._get_resource_requirements(job)
                        
                        # Check if job fits in window
                        if all(available_resources.get(rt, 0) >= req.value 
                              for rt, req in requirements.items()):
                            
                            # Schedule job with time limit
                            job.scheduling_constraints.max_runtime_seconds = int(window_duration)
                            
                            result = await self._try_schedule_job(job, entry, context)
                            if result.decision == SchedulingDecision.SCHEDULED:
                                await self._execute_scheduling_decision(job, result)
                                scheduled_count += 1
                                backfill_candidates.remove(entry)
                                
                                # Update available resources
                                for rt, qty in requirements.items():
                                    available_resources[rt] -= qty.value
            
            if scheduled_count > 0:
                logger.info(f"Backfilled {scheduled_count} jobs")
                self.metrics.increment('jobs_backfilled', scheduled_count)
            
            return scheduled_count
            
        except Exception as e:
            logger.error(f"Backfill scheduling error: {str(e)}")
            return 0
    
    async def cancel_job(self, job_id: str, reason: str = None, force: bool = False):
        """Cancel a job"""
        try:
            job = await self.job_store.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            logger.info(f"Cancelling job {job_id}: {reason}")
            
            if job.status in [JobStatus.RUNNING, JobStatus.CHECKPOINTING]:
                # Stop execution
                await self._stop_job_execution(job, cancelled=True)
                
                # Release resources
                await self._release_job_resources(job)
            
            # Update job status
            job.status = JobStatus.CANCELLED
            job.failure_reason = reason or "Cancelled by user"
            job.completed_at = datetime.utcnow()
            await self.job_store.update_job(job)
            
            # Remove from queues
            with self.state.state_lock:
                if job_id in self.state.running_jobs:
                    del self.state.running_jobs[job_id]
                
                self.state.pending_queue = [
                    (p, e) for p, e in self.state.pending_queue 
                    if e.job_id != job_id
                ]
                heapq.heapify(self.state.pending_queue)
            
            self.metrics.increment('jobs_cancelled')
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            raise
    
    def _get_resource_requirements(self, job: Job) -> Dict[ResourceType, ResourceQuantity]:
        """Get resource requirements from job"""
        requirements = {}
        
        if job.resources.cpu_count > 0:
            requirements[ResourceType.CPU] = ResourceQuantity(
                value=job.resources.cpu_count,
                unit="cores"
            )
        
        if job.resources.memory_gb > 0:
            requirements[ResourceType.MEMORY] = ResourceQuantity(
                value=job.resources.memory_gb,
                unit="GB"
            )
        
        if job.resources.gpu_count > 0:
            requirements[ResourceType.GPU] = ResourceQuantity(
                value=job.resources.gpu_count,
                unit="devices"
            )
        
        if job.resources.storage_gb > 0:
            requirements[ResourceType.STORAGE] = ResourceQuantity(
                value=job.resources.storage_gb,
                unit="GB"
            )
        
        return requirements
    
    def _estimate_runtime(self, job: Job) -> Optional[int]:
        """Estimate job runtime in seconds"""
        # Use user-provided estimate if available
        if hasattr(job, 'estimated_runtime'):
            return job.estimated_runtime
        
        # Use historical data for similar jobs
        # This is a simplified implementation
        similar_jobs = self._find_similar_completed_jobs(job)
        if similar_jobs:
            runtimes = [j.runtime_seconds for j in similar_jobs if j.runtime_seconds]
            if runtimes:
                return int(sum(runtimes) / len(runtimes))
        
        # Default estimates based on resource requirements
        if job.resources.gpu_count > 0:
            return 3600 * 4  # 4 hours for GPU jobs
        else:
            return 3600  # 1 hour for CPU jobs
    
    def _trigger_scheduling(self):
        """Trigger immediate scheduling attempt"""
        # Signal scheduling thread to wake up
        # In a real implementation, this would use a condition variable
        pass
    
    async def _monitor_jobs(self):
        """Monitor running jobs"""
        while self.running:
            try:
                for job_id, job in list(self.state.running_jobs.items()):
                    # Check job health
                    health = await self.monitoring.check_job_health(job_id)
                    
                    if not health['healthy']:
                        logger.warning(f"Job {job_id} unhealthy: {health['reason']}")
                        
                        # Handle based on restart policy
                        if job.scheduling_constraints.restart_policy == "always":
                            await self._restart_job(job)
                        elif job.scheduling_constraints.restart_policy == "on_failure":
                            if health['exit_code'] != 0:
                                await self._restart_job(job)
                    
                    # Update metrics
                    if health.get('metrics'):
                        await self._update_job_metrics(job_id, health['metrics'])
                
            except Exception as e:
                logger.error(f"Job monitoring error: {str(e)}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _update_fair_share(self):
        """Update fair share calculations"""
        while self.running:
            try:
                # Calculate usage over time window
                window_start = datetime.utcnow() - timedelta(
                    hours=self.schedule_config.fair_share_window_hours
                )
                
                # Reset usage counters
                self.state.user_usage.clear()
                self.state.project_usage.clear()
                
                # Calculate from historical data
                for record in self.state.historical_usage:
                    if record['timestamp'] > window_start:
                        user_id = record['user_id']
                        project_id = record['project_id']
                        
                        for resource_type, amount in record['resources'].items():
                            self.state.user_usage[user_id][resource_type] += amount
                            self.state.project_usage[project_id][resource_type] += amount
                
                # Update effective priorities based on fair share
                for _, entry in self.state.pending_queue:
                    entry.effective_priority = self._calculate_fair_share_priority(entry)
                
                # Re-heapify queue
                heapq.heapify(self.state.pending_queue)
                
            except Exception as e:
                logger.error(f"Fair share update error: {str(e)}")
            
            await asyncio.sleep(300)  # Update every 5 minutes
    
    def _calculate_fair_share_priority(self, entry: QueueEntry) -> float:
        """Calculate effective priority with fair share"""
        base_priority = entry.priority
        
        if self.schedule_config.secondary_policy != SchedulingPolicy.FAIR_SHARE:
            return base_priority
        
        # Get user's resource usage
        user_usage = sum(self.state.user_usage[entry.user_id].values())
        
        # Get target share
        target_share = self.schedule_config.target_share_ratios.get(
            entry.user_id, 1.0 / len(self.state.user_usage)
        )
        
        # Calculate actual share
        total_usage = sum(sum(usage.values()) for usage in self.state.user_usage.values())
        actual_share = user_usage / total_usage if total_usage > 0 else 0
        
        # Adjust priority based on share difference
        share_ratio = target_share / (actual_share + 0.001)  # Avoid division by zero
        
        # Apply decay factor for age-based priority boost
        age_minutes = (datetime.utcnow() - entry.submission_time).total_seconds() / 60
        age_boost = 1.0 + (age_minutes / self.schedule_config.priority_boost_age_minutes) * \
                    (self.schedule_config.priority_boost_factor - 1.0)
        
        return base_priority * share_ratio * age_boost 