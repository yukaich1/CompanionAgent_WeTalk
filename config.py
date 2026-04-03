from __future__ import annotations

from pydantic import BaseModel, Field


class SlotBudgetConfig(BaseModel):
    immutable_core: int = 300
    thought_output: int = 400
    web_persona_context: int = 300
    slow_change_state: int = 150
    web_reality_context: int = 300
    semantic_memory: int = 200
    episodic_memory: int = 300
    evidence_chunks: int = 400
    relation_state: int = 100


class MemoryConfig(BaseModel):
    episodic_top_k: int = 5
    semantic_top_k: int = 3
    semantic_upgrade_threshold: int = 3
    semantic_confidence_step_cap: float = Field(default=0.05, ge=0.0, le=1.0)
    relation_delta_cap: float = Field(default=0.05, ge=0.0, le=1.0)


class PersonaConfig(BaseModel):
    parent_chunk_target_tokens: int = 400
    parent_chunk_min_tokens: int = 50
    child_chunk_tokens: int = 100
    child_chunk_stride: int = 50
    vector_top_k: int = 10
    similarity_boundary: float = Field(default=0.5, ge=0.0, le=1.0)
    overlap_deprecate_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class ToolConfig(BaseModel):
    timeout_seconds: int = 15
    search_result_limit: int = 5
    tool_latency_warn_seconds: float = 3.0


class WetalkConfig(BaseModel):
    context_window_tokens: int = 8000
    response_budget_tokens: int = 1800
    slot_budget: SlotBudgetConfig = Field(default_factory=SlotBudgetConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)


DEFAULT_CONFIG = WetalkConfig()
