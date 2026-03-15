from pydantic import BaseModel
from typing import List
from memory.schemas import MemoryNode, MemoryEdge

class MemoryGraph(BaseModel):
    nodes: List[MemoryNode]
    edges: List[MemoryEdge]