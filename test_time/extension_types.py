"""Results shared by the two inference-only extension experiments."""

from dataclasses import dataclass, field

from .types import FloorplanSample, PolicyEvaluation


@dataclass
class ExtensionResult:
    sample: FloorplanSample | None
    evaluation: PolicyEvaluation | None
    status: str
    stats: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    # Named before/after or lookahead artifacts; only `sample` is the returned output.
    extra_samples: dict[str, tuple[FloorplanSample, PolicyEvaluation]] = field(default_factory=dict)
