"""Small, JSON-compatible records shared by generation and evaluation."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class Room:
    label: int
    polygon: list[list[float]]


@dataclass
class Scenario:
    building_use: str = "residential"
    accessible_dwelling: bool = True
    intended_occupants: int | None = 2
    ceiling_height_m: float | None = 2.5
    coordinate_scale_m_per_model_unit: float | None = None
    scale_provenance: str = "missing"
    assumptions: list[str] = field(default_factory=list)
    geometry_grid_m: float = 0.25
    geometry_refine_m: float = 0.05
    habitable_area_mode: str = "zero_thickness_proxy"
    ventilation_mode: str | None = None
    kitchen_extraction_m3_h: float | None = None
    bathroom_extraction_m3_h: float | None = None
    reduced_total_extraction_m3_h: float | None = None
    reduced_kitchen_extraction_m3_h: float | None = None
    bathroom_ventilation_role: str | None = None

    def __post_init__(self):
        if self.habitable_area_mode not in {"zero_thickness_proxy", "unknown"}:
            raise ValueError("habitable_area_mode must be zero_thickness_proxy or unknown")
        if self.scale_provenance not in {"missing", "assumed", "paper_derived", "verified"}:
            raise ValueError("scale_provenance must be missing, assumed, paper_derived, or verified")
        scale = self.coordinate_scale_m_per_model_unit
        if (scale is None) != (self.scale_provenance == "missing"):
            raise ValueError("A supplied scale needs assumed/paper_derived/verified provenance; absent scale needs missing provenance")
        for name in ("coordinate_scale_m_per_model_unit", "ceiling_height_m", "geometry_grid_m", "geometry_refine_m"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a positive finite number")
        if self.intended_occupants is not None and (
            isinstance(self.intended_occupants, bool)
            or not isinstance(self.intended_occupants, int)
            or self.intended_occupants < 1
        ):
            raise ValueError("intended_occupants must be a positive integer or null")
        if self.ventilation_mode not in {None, "occupant_adjustable", "automatic", "fixed"}:
            raise ValueError("ventilation_mode must be occupant_adjustable, automatic, fixed, or null")
        if self.bathroom_ventilation_role not in {None, "bathroom", "other_shower", "mixed", "unknown"}:
            raise ValueError("bathroom_ventilation_role must identify bathroom, other_shower, mixed, unknown, or null")
        for name in ("kitchen_extraction_m3_h", "bathroom_extraction_m3_h",
                     "reduced_total_extraction_m3_h", "reduced_kitchen_extraction_m3_h"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a nonnegative finite number or null")


@dataclass
class FloorplanSample:
    nodes: list[list[float]] = field(default_factory=list)
    edges: list[list[int]] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    raw_nodes: list = field(default_factory=list)
    raw_edges: list = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    scenario: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "rooms": [Room(**room) for room in data.get("rooms", [])]})


@dataclass
class ConstraintSpec:
    id: str
    checker: str
    definition: str
    applicability: str
    thresholds: dict
    required_observations: list[str]
    weight: float
    severity_type: str
    assumptions: list[str]
    source_id: str
    counts_toward_compliance: bool = True
    enabled: bool = True
    group: str = "ungrouped"
    role: str = "check"
    tier: str = "graph_plus_scenario"
    claim_level: str = "assumption_based"
    depends_on: list[str] = field(default_factory=list)
    disabled_reason: str | None = None


@dataclass
class ConstraintResult:
    id: str
    status: str
    severity: float | None
    applicable: bool
    measurable: bool
    source_id: str
    observed_values: dict = field(default_factory=dict)
    threshold_values: dict = field(default_factory=dict)
    assumption_flags: list[str] = field(default_factory=list)
    explanation: str = ""
    counts_toward_compliance: bool = True
    group: str = "ungrouped"
    role: str = "check"
    tier: str = "graph_plus_scenario"
    claim_level: str = "assumption_based"
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class PolicyEvaluation:
    energy: float
    reward: float
    results: list[ConstraintResult]
    structurally_valid: bool
    unknown_count: int
    applicable_count: int
    assumption_based_count: int
    validity_penalty: float
    diagnostics: list[str] = field(default_factory=list)
    measurable_count: int = 0


def overall_validity(evaluation):
    """Overall status under the configured checks, distinct from graph validity.

    Unknown checks cannot establish validity. With no applicable scored checks,
    report not_applicable rather than a vacuous regulatory success.
    """
    data = evaluation if isinstance(evaluation, dict) else asdict(evaluation)
    if not data["structurally_valid"]:
        return "invalid"
    checks = [result for result in data["results"]
              if result.get("counts_toward_compliance", True) and result["applicable"]]
    if any(result["status"] == "fail" for result in checks):
        return "invalid"
    if not checks:
        return "not_applicable"
    if any(result["status"] != "pass" or not result["measurable"] for result in checks):
        return "unknown"
    return "valid"


@dataclass
class Candidate:
    sample: FloorplanSample
    evaluation: PolicyEvaluation
    seed: int
    candidate_id: str = ""
    stats: dict = field(default_factory=dict)


@dataclass
class SamplingResult:
    candidates: list[Candidate]
    selected_index: int
    trace: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
