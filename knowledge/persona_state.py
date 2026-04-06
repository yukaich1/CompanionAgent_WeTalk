from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from knowledge.knowledge_source import KnowledgeSource


class IdentityProfile(BaseModel):
    name: str = ""
    aliases: list[str] = Field(default_factory=list)
    species: str = ""
    archetype: str = ""


class CoreTrait(BaseModel):
    feature: str
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    activation_trigger: list[str] = Field(default_factory=list)
    mandatory_behavior: str = ""
    evidence_tags: list[str] = Field(default_factory=list)


class AbsoluteTaboo(BaseModel):
    description: str
    severity: str = "medium"


class SpeechDNA(BaseModel):
    catchphrases: list[str] = Field(default_factory=list)
    sentence_endings: list[str] = Field(default_factory=list)
    address_rules: dict[str, str] = Field(default_factory=dict)


class InnateBelief(BaseModel):
    content: str
    domain: str = ""
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class ImmutableCore(BaseModel):
    identity: IdentityProfile = Field(default_factory=IdentityProfile)
    core_traits: list[CoreTrait] = Field(default_factory=list)
    absolute_taboos: list[AbsoluteTaboo] = Field(default_factory=list)
    speech_dna: SpeechDNA = Field(default_factory=SpeechDNA)
    innate_beliefs: list[InnateBelief] = Field(default_factory=list)


class AttitudeTowardUser(BaseModel):
    trust: float = Field(default=0.0, ge=0.0, le=1.0)
    affection: float = Field(default=0.0, ge=0.0, le=1.0)
    respect: float = Field(default=0.0, ge=0.0, le=1.0)


class GrowthLogEntry(BaseModel):
    original_trait: str = ""
    evolved_trait: str = ""
    summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SlowChangeLayer(BaseModel):
    attitude_toward_user: AttitudeTowardUser = Field(default_factory=AttitudeTowardUser)
    growth_log: list[GrowthLogEntry] = Field(default_factory=list)


class ParentChunk(BaseModel):
    chunk_id: str
    content: str
    source_level: KnowledgeSource = KnowledgeSource.USER_CANON
    topic_tags: list[str] = Field(default_factory=list)
    trait_tags: list[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    deprecated: bool = False
    version: int = 1
    kind: str = "source_chunk"
    title: str = ""
    metadata: dict = Field(default_factory=dict)
    language: str = "zh"


class ChildChunk(BaseModel):
    chunk_id: str
    parent_id: str
    content: str


class EvidenceVault(BaseModel):
    parent_chunks: list[ParentChunk] = Field(default_factory=list)
    child_chunks: list[ChildChunk] = Field(default_factory=list)


class PersonaState(BaseModel):
    immutable_core: ImmutableCore = Field(default_factory=ImmutableCore)
    slow_change_layer: SlowChangeLayer = Field(default_factory=SlowChangeLayer)
    evidence_vault: EvidenceVault = Field(default_factory=EvidenceVault)
    metadata: dict = Field(default_factory=dict)


class PersonaSystemStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> PersonaState:
        if not self.path.exists():
            return PersonaState()
        raw = self.path.read_text(encoding="utf-8")
        if hasattr(PersonaState, "model_validate_json"):
            return PersonaState.model_validate_json(raw)
        return PersonaState.parse_raw(raw)

    def save(self, state: PersonaState) -> None:
        if hasattr(state, "model_dump_json"):
            payload = state.model_dump_json(indent=2)
        else:
            payload = state.json(indent=2, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")
