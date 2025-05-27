"""
Model registry service
"""
from typing import Dict, List, Optional, Any, Tuple

try:
    from ..models.experiment import ModelVersion, ModelRegistry
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.experiment import ModelVersion, ModelRegistry


class ModelRegistryService:
    """Service for managing model registry"""
    
    def __init__(self):
        self.registries = {}
        self.models = {}
    
    async def register_model(self, model_version: ModelVersion, make_public: bool = False, allowed_users: Optional[List[str]] = None) -> ModelRegistry:
        """Register a model version in the registry"""
        registry_id = f"registry-{model_version.model_name}"
        
        if registry_id not in self.registries:
            registry = ModelRegistry(
                id=registry_id,
                name=f"{model_version.model_name} Registry"
            )
            self.registries[registry_id] = registry
        else:
            registry = self.registries[registry_id]
        
        # Add model to registry
        if model_version.model_name not in registry.models:
            registry.models[model_version.model_name] = []
        
        registry.models[model_version.model_name].append(model_version)
        
        # Set access control
        if make_public:
            registry.public_models.add(model_version.version_id)
        elif allowed_users:
            registry.restricted_models[model_version.version_id] = allowed_users
        
        self.models[model_version.version_id] = model_version
        
        return registry
    
    async def search_models(self, query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, 
                          user_id: str = "", page: int = 1, page_size: int = 20) -> Tuple[List[ModelVersion], int]:
        """Search for models in the registry"""
        # Simple search implementation
        results = []
        
        for model in self.models.values():
            # Apply filters
            if filters:
                if 'framework' in filters and model.framework.value != filters['framework']:
                    continue
                if 'min_accuracy' in filters:
                    accuracy = model.validation_metrics.get('accuracy', 0)
                    if accuracy < filters['min_accuracy']:
                        continue
            
            # Apply query
            if query and query.lower() not in model.model_name.lower():
                continue
            
            results.append(model)
        
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        
        return results[start:end], len(results) 