from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class NodeType(Enum):
    ENTITY = "entity"
    FACT = "fact"
    EVENT = "event"
    USER_PREFERENCE = "user_preference"
    TASK = "task"
    SUMMARY = "summary"
    SOURCE = "source"
    BELIEF = "belief"

class EdgeType(Enum):
    RELATED_TO = "related_to"
    REFERS_TO = "refers_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    CAUSED_BY = "caused_by"
    TEMPORAL_BEFORE = "temporal_before"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"

class MemoryNode(BaseModel):
    id: str
    content: str
    type: NodeType

class MemoryEdge(BaseModel):
    id: str
    source: str
    target: str
    type: EdgeType