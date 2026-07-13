"""Frozen-suite metrics for related-memory update selection."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class UpdateSelectionMetrics:
    recall: float
    precision: float
    missed_conflicts: int
    fallback_uses: int
    unrelated_visible_entries: int
    adversarial_pass_rate: float
    passed: bool

    def as_report(self) -> dict[str, float | int | bool]:
        return asdict(self)

def update_selection_metrics(*, expected_ids: set[str], selected_ids: set[str],
    conflict_ids: set[str], fallback_uses: int = 0, unrelated_visible_entries: int = 0,
    adversarial_passes: int = 0, adversarial_total: int = 0) -> UpdateSelectionMetrics:
    true_positive = len(expected_ids & selected_ids)
    recall = true_positive / len(expected_ids) if expected_ids else 1.0
    precision = true_positive / len(selected_ids) if selected_ids else (1.0 if not expected_ids else 0.0)
    missed = len(conflict_ids - selected_ids)
    adversarial = adversarial_passes / adversarial_total if adversarial_total else 1.0
    passed = recall >= .95 and precision >= .80 and missed == 0 and unrelated_visible_entries <= 1 and adversarial == 1.0
    return UpdateSelectionMetrics(recall, precision, missed, fallback_uses,
        unrelated_visible_entries, adversarial, passed)
