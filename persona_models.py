from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PersonaStyleExample(BaseModel):
    text: str = ""
    scene: str = ""
    emotion: str = ""
    rules_applied: List[str] = Field(default_factory=list)
    source: str = ""
    affinity_level: str = "any"


class PersonaSummaryModel(BaseModel):
    character_name: str = ""
    source_label: str = ""
    base_template: Dict[str, Any] = Field(default_factory=dict)
    character_voice_card: str = ""
    display_keywords: List[str] = Field(default_factory=list)
    style_examples: List[PersonaStyleExample] = Field(default_factory=list)


class PersonaSourceSnippet(BaseModel):
    source: str
    title: str = ""
    text: str


class PersonaPreviewModel(BaseModel):
    preview_id: str
    persona_name: str
    work_title: str = ""
    source_label: str
    source_text: str
    base_template_text: str = ""
    snippets: List[PersonaSourceSnippet] = Field(default_factory=list)
    summary: PersonaSummaryModel
    created_at: str
    mode: str = "cold_start"
    committed: bool = False


class PersonaStatusModel(BaseModel):
    persona_name: str
    chunk_count: int
    display_keywords: List[str] = Field(default_factory=list)
