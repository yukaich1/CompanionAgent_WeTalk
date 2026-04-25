from __future__ import annotations

from runtime.runtime_models import TurnInput
from runtime.turn_engine import TurnEngine


class SessionRuntime:
    def __init__(self, system):
        self.system = system
        self.turn_engine = TurnEngine(system)

    def handle_user_turn(self, user_text: str, attached_image=None) -> str:
        response, _trace = self.turn_engine.run_turn(
            TurnInput(user_text=str(user_text or ""), attached_image=attached_image)
        )
        return response
