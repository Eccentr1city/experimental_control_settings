"""
Mock MongoDB implementation with realistic behavior
"""
import asyncio
import copy
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


class MockCollection:
    """Mock MongoDB collection with realistic async behavior"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents: Dict[str, Dict] = {}
        self.indexes: List[str] = []
    
    async def insert_one(self, document: Dict[str, Any]) -> 'MockInsertResult':
        # Simulate network delay
        await asyncio.sleep(0.001)
        
        doc_id = document.get('_id') or document.get('id')
        if not doc_id:
            # Auto-generate ID if not provided
            doc_id = str(uuid.uuid4())
        
        if doc_id in self.documents:
            raise Exception(f"Duplicate key error: {doc_id}")
        
        # Deep copy to simulate database storage
        stored_doc = copy.deepcopy(document)
        stored_doc['_id'] = doc_id
        self.documents[doc_id] = stored_doc
        
        return MockInsertResult(doc_id)
    
    async def find_one(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await asyncio.sleep(0.001)
        
        for doc in self.documents.values():
            if self._matches_filter(doc, filter_dict):
                return copy.deepcopy(doc)
        return None
    
    async def replace_one(self, filter_dict: Dict[str, Any], 
                         replacement: Dict[str, Any]) -> 'MockUpdateResult':
        await asyncio.sleep(0.001)
        
        for doc_id, doc in self.documents.items():
            if self._matches_filter(doc, filter_dict):
                new_doc = copy.deepcopy(replacement)
                new_doc['_id'] = doc_id
                self.documents[doc_id] = new_doc
                return MockUpdateResult(1, doc_id)
        
        return MockUpdateResult(0, None)
    
    async def update_one(self, filter_dict: Dict[str, Any], 
                        update: Dict[str, Any]) -> 'MockUpdateResult':
        await asyncio.sleep(0.001)
        
        for doc_id, doc in self.documents.items():
            if self._matches_filter(doc, filter_dict):
                # Handle $set operations
                if '$set' in update:
                    for key, value in update['$set'].items():
                        self._set_nested_value(doc, key, value)
                
                # Handle $inc operations
                if '$inc' in update:
                    for key, value in update['$inc'].items():
                        current = self._get_nested_value(doc, key) or 0
                        self._set_nested_value(doc, key, current + value)
                
                return MockUpdateResult(1, doc_id)
        
        return MockUpdateResult(0, None)
    
    async def delete_one(self, filter_dict: Dict[str, Any]) -> 'MockDeleteResult':
        await asyncio.sleep(0.001)
        
        for doc_id, doc in list(self.documents.items()):
            if self._matches_filter(doc, filter_dict):
                del self.documents[doc_id]
                return MockDeleteResult(1)
        
        return MockDeleteResult(0)
    
    async def count_documents(self, filter_dict: Dict[str, Any]) -> int:
        await asyncio.sleep(0.001)
        
        count = 0
        for doc in self.documents.values():
            if self._matches_filter(doc, filter_dict):
                count += 1
        return count
    
    def find(self, filter_dict: Optional[Dict[str, Any]] = None) -> 'MockCursor':
        return MockCursor(self.documents, filter_dict or {})
    
    async def create_index(self, keys: Any, unique: bool = False):
        await asyncio.sleep(0.001)
        index_spec = str(keys)
        if index_spec not in self.indexes:
            self.indexes.append(index_spec)
    
    def _matches_filter(self, doc: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Simple filter matching - can be enhanced for complex queries"""
        for key, value in filter_dict.items():
            doc_value = self._get_nested_value(doc, key)
            
            if isinstance(value, dict):
                # Handle operators like $in, $gt, $lt, etc.
                if '$in' in value:
                    if doc_value not in value['$in']:
                        return False
                elif '$gt' in value:
                    if doc_value is None or doc_value <= value['$gt']:
                        return False
                elif '$lt' in value:
                    if doc_value is None or doc_value >= value['$lt']:
                        return False
                elif '$gte' in value:
                    if doc_value is None or doc_value < value['$gte']:
                        return False
                elif '$lte' in value:
                    if doc_value is None or doc_value > value['$lte']:
                        return False
                elif '$ne' in value:
                    if doc_value == value['$ne']:
                        return False
                else:
                    # Exact match for dict values
                    if doc_value != value:
                        return False
            else:
                # Exact match
                if doc_value != value:
                    return False
        return True
    
    def _get_nested_value(self, doc: Dict[str, Any], key: str) -> Any:
        """Get value from nested dict using dot notation"""
        if '.' not in key:
            return doc.get(key)
        
        parts = key.split('.')
        current = doc
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current
    
    def _set_nested_value(self, doc: Dict[str, Any], key: str, value: Any):
        """Set value in nested dict using dot notation"""
        if '.' not in key:
            doc[key] = value
            return
        
        parts = key.split('.')
        current = doc
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


class MockCursor:
    """Mock MongoDB cursor for find operations"""
    
    def __init__(self, documents: Dict[str, Dict], filter_dict: Dict[str, Any]):
        self.documents = documents
        self.filter_dict = filter_dict
        self.sort_key = None
        self.sort_direction = 1
        self.skip_count = 0
        self.limit_count = None
    
    def sort(self, key: str, direction: int = 1) -> 'MockCursor':
        self.sort_key = key
        self.sort_direction = direction
        return self
    
    def skip(self, count: int) -> 'MockCursor':
        self.skip_count = count
        return self
    
    def limit(self, count: int) -> 'MockCursor':
        self.limit_count = count
        return self
    
    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.001)
        
        # Filter documents
        filtered_docs = []
        collection = MockCollection("temp")
        for doc in self.documents.values():
            if collection._matches_filter(doc, self.filter_dict):
                filtered_docs.append(copy.deepcopy(doc))
        
        # Sort if specified
        if self.sort_key is not None:
            sort_key = self.sort_key  # Type narrowing for mypy
            filtered_docs.sort(
                key=lambda x: self._get_nested_value(x, sort_key) or 0,
                reverse=(self.sort_direction == -1)
            )
        
        # Apply skip and limit
        start = self.skip_count
        if self.limit_count:
            end = start + self.limit_count
        else:
            end = len(filtered_docs)
        
        result = filtered_docs[start:end]
        
        # Apply length limit if specified
        if length is not None:
            result = result[:length]
        
        return result
    
    def _get_nested_value(self, doc: Dict[str, Any], key: str) -> Any:
        """Get value from nested dict using dot notation"""
        if '.' not in key:
            return doc.get(key)
        
        parts = key.split('.')
        current = doc
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current
    
    def __aiter__(self):
        """Support async iteration"""
        return MockAsyncIterator(self)


class MockAsyncIterator:
    """Async iterator for mock cursor"""
    
    def __init__(self, cursor: MockCursor):
        self.cursor = cursor
        self.docs = None
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.docs is None:
            self.docs = await self.cursor.to_list()
        
        if self.index >= len(self.docs):
            raise StopAsyncIteration
        
        doc = self.docs[self.index]
        self.index += 1
        return doc


class MockDatabase:
    """Mock MongoDB database"""
    
    def __init__(self, name: str):
        self.name = name
        self.collections: Dict[str, MockCollection] = {}
    
    def __getattr__(self, name: str) -> MockCollection:
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]
    
    def __getitem__(self, name: str) -> MockCollection:
        return self.__getattr__(name)


class MockAsyncIOMotorClient:
    """Mock Motor client for MongoDB"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.databases: Dict[str, MockDatabase] = {}
    
    def __getitem__(self, name: str) -> MockDatabase:
        if name not in self.databases:
            self.databases[name] = MockDatabase(name)
        return self.databases[name]


# Result classes
class MockInsertResult:
    def __init__(self, inserted_id: Any):
        self.inserted_id = inserted_id


class MockUpdateResult:
    def __init__(self, modified_count: int, upserted_id: Any = None):
        self.modified_count = modified_count
        self.upserted_id = upserted_id


class MockDeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count 