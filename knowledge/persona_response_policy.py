from __future__ import annotations


class PersonaResponsePolicy:
    def __init__(self, persona_system, character_name_getter):
        self.persona_system = persona_system
        self._character_name_getter = character_name_getter

    def has_identity_reference(self) -> bool:
        background = self._base_template_section("00_BACKGROUND")
        profile = background.get("profile", {}) if isinstance(background, dict) else {}
        experiences = list(background.get("key_experiences", []) or []) if isinstance(background, dict) else []
        return bool(profile or experiences)

    def character_name(self) -> str:
        return str(self._character_name_getter() or getattr(self.persona_system, "persona_name", "") or "角色").strip()

    def _base_template_section(self, key: str) -> dict:
        base_template = getattr(self.persona_system, "base_template", {}) or {}
        value = base_template.get(key, {})
        return value if isinstance(value, dict) else {}
