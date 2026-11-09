"""Keep compact prior-level hypotheses when Duck's explicit transfer notes are empty."""

from .agent import DuckAgent

RETAINED_ON_GAME_OVER = ("world_model", "goal_model", "action_model", "open_questions")


class TransferDuckAgent(DuckAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = "duck-transfer"

    def analyzer(self, loop):
        analyzer = super().analyzer(loop)
        original_update = analyzer._update_summarized_knowledge_from_step_summary
        automatic_note = None

        def update():
            nonlocal automatic_note
            summary = analyzer._last_step_summary or {}
            knowledge = analyzer._summarized_knowledge
            cross_level = knowledge.get("cross_level_notes", "")
            # Intentional model-written transfer notes take precedence. An earlier
            # generated fallback may be replaced by the newest level's hypotheses.
            can_replace = not cross_level or cross_level == automatic_note
            prior = [
                f"{label}: {knowledge[key]}"
                for key, label in (("world_model", "World"), ("action_model", "Actions"))
                if knowledge.get(key)
            ]
            retained = {key: knowledge.get(key, "") for key in RETAINED_ON_GAME_OVER}
            original_update()
            if summary.get("run_complete"):
                if cross_level == automatic_note:
                    knowledge["cross_level_notes"] = ""
                automatic_note = None
            elif summary.get("game_over") and not summary.get("level_transition"):
                # v5: a GAME_OVER restarts the level, not the model's understanding of it.
                knowledge.update(retained)
                knowledge["current_plan"] = ""
                actions = ", ".join(str(a) for a in (summary.get("executed_actions") or [])[-10:])
                mechanics = "The mechanics above still hold; revise the plan and avoid that sequence."
                if self.fix("prompt_text") and not any(retained.values()):
                    # harness_fixes: nothing is listed above this note when no model was retained.
                    mechanics = ("What you learned about this level's mechanics still holds; revise the plan "
                                 "and avoid that sequence.")
                knowledge["recent_findings"] = (
                    f"GAME_OVER at step {summary.get('end_action_num')} after actions "
                    f"[{actions}]; the level restarted from its initial board. {mechanics}"
                )
                self.workspace.append(
                    "knowledge_retained", step=self.steps, keys=sorted(k for k, v in retained.items() if v)
                )
            elif summary.get("level_transition") and can_replace:
                if prior:
                    text = "\n".join(prior)
                    if len(text) > 1400:
                        text = text[:1400] + " [truncated]"
                    automatic_note = (
                        "Prior-level hypotheses, not verified for the new level. "
                        "Re-check mechanics against the new board; old coordinates and "
                        "layouts may no longer apply. Reuse only what the evidence supports.\n"
                        + text
                    )
                    knowledge["cross_level_notes"] = automatic_note
                    self.workspace.append("transfer_notes", step=self.steps, text=automatic_note)
                elif cross_level == automatic_note:
                    knowledge["cross_level_notes"] = ""
                    automatic_note = None

        analyzer._update_summarized_knowledge_from_step_summary = update
        return analyzer
