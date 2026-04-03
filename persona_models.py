from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PersonaSectionData(BaseModel):
    voice_style: List[str] = Field(default_factory=list)
    catchphrases: List[str] = Field(default_factory=list)
    addressing_habits: List[str] = Field(default_factory=list)
    sentence_endings: List[str] = Field(default_factory=list)
    personality: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    worldview: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    appearance: List[str] = Field(default_factory=list)
    core_setup: List[str] = Field(default_factory=list)
    life_experiences: List[str] = Field(default_factory=list)


class PersonaSummaryModel(PersonaSectionData):
    style_examples: List[str] = Field(default_factory=list)
    natural_reference_triggers: List[str] = Field(default_factory=list)
    display_keywords: List[str] = Field(default_factory=list)
    section_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    meta_exclusion_words: List[str] = Field(default_factory=list)
    novel_persona_hints: List[str] = Field(default_factory=list)
    voice_hints: List[str] = Field(default_factory=list)
    trait_markers: List[str] = Field(default_factory=list)


class PersonaSourceSnippet(BaseModel):
    source: str
    title: str = ""
    text: str


class PersonaKeywordOption(BaseModel):
    source: str
    title: str = ""
    keywords: List[str] = Field(default_factory=list)


class PersonaPreviewModel(BaseModel):
    preview_id: str
    persona_name: str
    work_title: str = ""
    source_label: str
    source_text: str
    snippets: List[PersonaSourceSnippet] = Field(default_factory=list)
    keyword_options: List[PersonaKeywordOption] = Field(default_factory=list)
    selected_keywords: List[str] = Field(default_factory=list)
    summary: PersonaSummaryModel
    created_at: str
    mode: str = "cold_start"
    committed: bool = False


class PersonaStatusModel(BaseModel):
    persona_name: str
    chunk_count: int
    display_keywords: List[str] = Field(default_factory=list)
