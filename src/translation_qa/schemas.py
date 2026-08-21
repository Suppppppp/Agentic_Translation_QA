"""Pydantic models shared by the API, pipeline, traces, and benchmark."""

from __future__ import annotations

import hashlib
import unicodedata
from enum import Enum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
TranslationInputText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=5_000),
]
Percentage = Annotated[float, Field(ge=0.0, le=100.0, allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Milliseconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class SchemaModel(BaseModel):
    """Strict base model used at component and API boundaries."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class ExecutionMode(str, Enum):
    BASELINE = "baseline"
    RAG = "rag"
    AGENT = "agent"
    AGENT_RAG = "agent_rag"


class RetrievalMatchType(str, Enum):
    EXACT = "exact"
    VECTOR = "vector"
    HYBRID = "hybrid"


class TranslationErrorType(str, Enum):
    TERM = "term"
    MEANING = "meaning"
    OMISSION_ADDITION = "omission_addition"
    ENTITY_VALUE = "entity_value"
    FLUENCY_GRAMMAR = "fluency_grammar"
    OTHER = "other"


class SourceCoverageKind(str, Enum):
    """Source-side item whose explicit realization may be required."""

    PROPER_NAME = "proper_name"
    STANDALONE_SUBJECT = "standalone_subject"
    TECHNICAL_TERM = "technical_term"


class SourceCoverageStatus(str, Enum):
    """Conservative result of checking one requirement in a candidate."""

    COVERED = "covered"
    MISSING = "missing"
    UNRESOLVED = "unresolved"


class CoverageRecoveryAction(str, Enum):
    """Recovery selected by code for a confirmed source-coverage miss."""

    TARGETED_RAG = "targeted_rag"
    MUST_PRESERVE = "must_preserve"


class SourceCoverageExtractionRule(str, Enum):
    """Source-only evidence that created a coverage requirement."""

    LEADING_TOPIC_SUBJECT = "leading_topic_subject"
    LATIN_IDENTIFIER = "latin_identifier"
    ANALYSIS_KEY_TERM = "analysis_key_term"
    RETRIEVED_GLOSSARY = "retrieved_glossary"


class ManualReviewStatus(str, Enum):
    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"


class ReferenceReviewDecision(str, Enum):
    """A reviewer decision about the public English reference."""

    KEEP_ORIGINAL = "keep_original"
    CORRECTED = "corrected"


class ReferenceReviewStatus(str, Enum):
    """Provenance status for an effective evaluation reference."""

    UNREVIEWED = "unreviewed"
    AI_ASSISTED_DRAFT = "ai_assisted_draft"
    HUMAN_CONFIRMED = "human_confirmed"


class ReferenceOrigin(str, Enum):
    PUBLIC_DATASET = "public_dataset"
    REVIEWER_CORRECTION = "reviewer_correction"


class ManualOutcome(str, Enum):
    """Human comparison of an Agent mode's initial and final candidates."""

    IMPROVED = "improved"
    SAME = "same"
    WORSE = "worse"


class NextAction(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    RETRY_WITH_RAG = "retry_with_rag"
    RETRY_WITH_CONSTRAINTS = "retry_with_constraints"
    STOP = "stop"


class JudgmentDecisionBasis(str, Enum):
    """Authoritative source used to derive the binary quality decision."""

    STRUCTURED_ERROR_TYPES = "structured_error_types"


class JudgmentConsistencyIssue(str, Enum):
    """Auditable contradictions found in a model-produced judgment payload."""

    REPORTED_PASSED_MISMATCH = "reported_passed_mismatch"
    REPORTED_NEXT_ACTION_MISMATCH = "reported_next_action_mismatch"
    SUMMARY_ERROR_WITHOUT_STRUCTURED_ERROR = (
        "summary_error_without_structured_error"
    )
    SUMMARY_PASS_WITH_STRUCTURED_ERROR = "summary_pass_with_structured_error"
    UNSUPPORTED_ERROR_TYPE_NORMALIZED = "unsupported_error_type_normalized"


class StopReason(str, Enum):
    PASSED = "passed"
    MAX_RETRIES = "max_retries"
    UNCHANGED = "unchanged"
    NO_RETRIEVAL_MATCH = "no_retrieval_match"
    COMPONENT_FAILURE = "component_failure"
    CONSTRAINT_FALLBACK = "constraint_fallback"
    AGENT_STOP = "agent_stop"
    BASELINE_COMPLETE = "baseline_complete"
    RAG_COMPLETE = "rag_complete"


class CandidateOrigin(str, Enum):
    NMT = "nmt"
    AGENT_REVISION = "agent_revision"
    FALLBACK = "fallback"


class SelectionReason(str, Enum):
    ONLY_CANDIDATE = "only_candidate"
    PASSED = "passed"
    HIGHEST_QUALITY = "highest_quality"
    HIGHEST_QUALITY_ROLLBACK = "highest_quality_rollback"
    SOURCE_COVERAGE_PRESERVED = "source_coverage_preserved"


class TranslationRequest(SchemaModel):
    text: TranslationInputText


class SourceAnalysis(SchemaModel):
    domain: NonEmptyText | None = None
    key_terms: list[NonEmptyText] = Field(default_factory=list)
    confidence: Confidence


class RetrievalQuery(SchemaModel):
    source_text: NonEmptyText
    domain: NonEmptyText | None = None
    key_terms: list[NonEmptyText] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=20)
    attempt_index: int = Field(default=0, ge=0)


class RetrievalHit(SchemaModel):
    term_id: NonEmptyText
    source_term: NonEmptyText
    target_term: NonEmptyText
    domain: NonEmptyText | None = None
    match_type: RetrievalMatchType
    score: float = Field(allow_inf_nan=False)
    definition: NonEmptyText | None = None
    accepted_target_variants: list[NonEmptyText] = Field(default_factory=list)
    replacement_rules: dict[str, str] = Field(default_factory=dict)


class TermConstraint(SchemaModel):
    source_term: NonEmptyText
    target_term: NonEmptyText
    retrieval_hit_id: NonEmptyText
    target_variants: list[NonEmptyText] = Field(default_factory=list)
    replacement_rules: dict[str, str] = Field(default_factory=dict)


class SourceCoverageRequirement(SchemaModel):
    """A source-anchored item checked without consulting a reference translation."""

    source_term: NonEmptyText
    kinds: list[SourceCoverageKind] = Field(min_length=1)
    accepted_targets: list[NonEmptyText] = Field(default_factory=list)
    extraction_rules: list[SourceCoverageExtractionRule] = Field(min_length=1)
    evidence_hit_ids: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def values_are_unique(self) -> SourceCoverageRequirement:
        if len(self.kinds) != len(set(self.kinds)):
            raise ValueError("coverage requirement kinds must be unique")
        if len(self.extraction_rules) != len(set(self.extraction_rules)):
            raise ValueError("coverage extraction rules must be unique")
        normalized_targets = [
            unicodedata.normalize("NFKC", target).casefold()
            for target in self.accepted_targets
        ]
        if len(normalized_targets) != len(set(normalized_targets)):
            raise ValueError("coverage accepted targets must be unique")
        if len(self.evidence_hit_ids) != len(set(self.evidence_hit_ids)):
            raise ValueError("coverage evidence hit IDs must be unique")
        return self


class MustPreserveConstraint(SchemaModel):
    """Temporary source-preservation instruction for Agent revision only.

    This is deliberately separate from :class:`TermConstraint`, which belongs
    to the NMT/glossary injection path.
    """

    source_term: NonEmptyText
    kinds: list[SourceCoverageKind] = Field(min_length=1)
    accepted_targets: list[NonEmptyText] = Field(default_factory=list)


class SourceCoverageFinding(SchemaModel):
    requirement: SourceCoverageRequirement
    status: SourceCoverageStatus
    matched_target: NonEmptyText | None = None
    recovery_action: CoverageRecoveryAction | None = None
    reason: NonEmptyText

    @model_validator(mode="after")
    def result_is_consistent(self) -> SourceCoverageFinding:
        if self.status is SourceCoverageStatus.COVERED:
            if self.matched_target is None:
                raise ValueError("covered source requirements need a matched target")
            if self.recovery_action is not None:
                raise ValueError("covered source requirements cannot request recovery")
        elif self.status is SourceCoverageStatus.MISSING:
            if self.matched_target is not None:
                raise ValueError("missing source requirements cannot have a target match")
            if self.recovery_action is None:
                raise ValueError("missing source requirements need a recovery action")
        elif self.matched_target is not None or self.recovery_action is not None:
            raise ValueError("unresolved source requirements cannot claim a match or recovery")
        return self


class TranslationCandidate(SchemaModel):
    text: NonEmptyText
    model_id: NonEmptyText


class JudgmentDecisionAudit(SchemaModel):
    """Preserve the LLM's redundant claims beside the code-derived decision."""

    decision_basis: JudgmentDecisionBasis = (
        JudgmentDecisionBasis.STRUCTURED_ERROR_TYPES
    )
    reported_passed: bool
    reported_next_action: NextAction
    reported_error_types: list[str] = Field(default_factory=list)
    code_added_error_types: list[TranslationErrorType] = Field(default_factory=list)
    consistency_issues: list[JudgmentConsistencyIssue] = Field(default_factory=list)


class QualityJudgment(SchemaModel):
    """A concise, inspectable decision rather than hidden chain-of-thought."""

    passed: bool
    quality_score: Confidence = Field(
        description="Agent score for translation quality, independent of confidence."
    )
    error_types: list[TranslationErrorType] = Field(default_factory=list)
    summary: NonEmptyText
    confidence: Confidence
    next_action: NextAction
    suggested_query_terms: list[NonEmptyText] = Field(default_factory=list)
    must_preserve_constraints: list[MustPreserveConstraint] = Field(
        default_factory=list
    )
    decision_audit: JudgmentDecisionAudit | None = None

    @model_validator(mode="after")
    def decision_is_consistent(self) -> QualityJudgment:
        if self.passed != (not self.error_types):
            raise ValueError(
                "passed must be derived from whether structured error types are empty"
            )
        if self.passed:
            if self.next_action is not NextAction.ACCEPT:
                raise ValueError("a passed judgment must use the accept action")
        elif self.next_action is NextAction.ACCEPT:
            raise ValueError("a failed judgment cannot use the accept action")
        return self


class RevisionResult(SchemaModel):
    """Validated JSON payload returned by a structured post-edit call."""

    translation: NonEmptyText


class StageTimings(SchemaModel):
    analysis_ms: Milliseconds = 0.0
    retrieval_ms: Milliseconds = 0.0
    translation_ms: Milliseconds = 0.0
    revision_ms: Milliseconds = 0.0
    judgment_ms: Milliseconds = 0.0
    total_ms: Milliseconds = 0.0


class AttemptTrace(SchemaModel):
    """Trace for one candidate; index zero is the initial translation."""

    attempt_index: int = Field(ge=0)
    candidate_origin: CandidateOrigin = CandidateOrigin.NMT
    parent_attempt_index: int | None = Field(default=None, ge=0)
    requested_action: NextAction | None = None
    applied_action: NextAction | None = None
    retrieval_query: RetrievalQuery | None = None
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    applied_constraints: list[TermConstraint] = Field(default_factory=list)
    applied_must_preserve_constraints: list[MustPreserveConstraint] = Field(
        default_factory=list
    )
    coverage_findings: list[SourceCoverageFinding] = Field(default_factory=list)
    candidate: TranslationCandidate
    judgment: QualityJudgment | None = None
    timings: StageTimings = Field(default_factory=StageTimings)

    @model_validator(mode="after")
    def revision_parent_is_valid(self) -> AttemptTrace:
        if self.candidate_origin is CandidateOrigin.AGENT_REVISION:
            if self.parent_attempt_index is None:
                raise ValueError("an Agent revision must identify its parent attempt")
            if self.parent_attempt_index >= self.attempt_index:
                raise ValueError("parent_attempt_index must precede the revision")
        elif self.parent_attempt_index is not None:
            raise ValueError("only Agent revisions may identify a parent attempt")
        return self


class ComponentCallCounts(SchemaModel):
    analysis: int = Field(default=0, ge=0)
    retrieval: int = Field(default=0, ge=0)
    translation: int = Field(default=0, ge=0)
    revision: int = Field(default=0, ge=0)
    judgment: int = Field(default=0, ge=0)


class TranslationTrace(SchemaModel):
    source_analysis: SourceAnalysis | None = None
    coverage_requirements: list[SourceCoverageRequirement] = Field(
        default_factory=list
    )
    attempts: list[AttemptTrace] = Field(min_length=1)
    final_attempt_index: int = Field(ge=0)
    stop_reason: StopReason
    selection_reason: SelectionReason = SelectionReason.ONLY_CANDIDATE
    total_latency_ms: Milliseconds
    component_call_counts: ComponentCallCounts = Field(
        default_factory=ComponentCallCounts
    )
    warnings: list[NonEmptyText] = Field(default_factory=list)
    model_versions: dict[str, NonEmptyText] = Field(default_factory=dict)

    @model_validator(mode="after")
    def attempts_are_ordered_and_final_exists(self) -> TranslationTrace:
        indices = [attempt.attempt_index for attempt in self.attempts]
        if indices != list(range(len(indices))):
            raise ValueError("attempt indices must be contiguous and start at zero")
        if self.final_attempt_index not in indices:
            raise ValueError("final_attempt_index must identify a recorded attempt")
        return self


class TranslationResponse(SchemaModel):
    request_id: UUID = Field(default_factory=uuid4)
    mode: ExecutionMode
    source_text: NonEmptyText
    translation: NonEmptyText
    retry_count: int = Field(ge=0, le=2)
    final_judgment: QualityJudgment | None = None
    trace: TranslationTrace

    @model_validator(mode="after")
    def response_matches_trace(self) -> TranslationResponse:
        if self.retry_count != len(self.trace.attempts) - 1:
            raise ValueError("retry_count must equal the number of additional attempts")
        final_attempt = self.trace.attempts[self.trace.final_attempt_index]
        if self.translation != final_attempt.candidate.text:
            raise ValueError("translation must match the selected final attempt")
        if self.final_judgment != final_attempt.judgment:
            raise ValueError("final_judgment must match the selected final attempt")
        return self


class ExpectedTerm(SchemaModel):
    source_term: NonEmptyText
    accepted_targets: list[NonEmptyText] = Field(min_length=1)


class ManualJudgmentLabel(SchemaModel):
    """Human gold label for the first Agent judgment in one execution mode.

    An empty error list represents ``NONE`` for a passing candidate. Revision
    labels may omit taxonomy during binary-only annotation, but any supplied
    error values must use ``TranslationErrorType``.
    """

    needs_revision: bool
    review_status: ManualReviewStatus = ManualReviewStatus.CONFIRMED
    primary_error: TranslationErrorType | None = None
    error_types: list[TranslationErrorType] = Field(default_factory=list)

    @model_validator(mode="after")
    def errors_match_binary_label(self) -> ManualJudgmentLabel:
        if not self.needs_revision and (self.primary_error or self.error_types):
            raise ValueError("a PASS manual label cannot contain error types")
        if self.primary_error is not None and self.primary_error not in self.error_types:
            raise ValueError("primary_error must also appear in error_types")
        return self


class ManualOutcomeLabel(SchemaModel):
    """Human gold label comparing attempt zero with the selected final output."""

    outcome: ManualOutcome
    review_status: ManualReviewStatus = ManualReviewStatus.CONFIRMED


class ReferenceProvenance(SchemaModel):
    """Audit trail for a public reference and any pre-evaluation correction.

    ``original_reference_text`` always retains the public dataset value while
    ``EvaluationCase.reference_text`` is the effective offline reference.  None
    of these fields are copied by :meth:`EvaluationCase.to_translation_request`.
    """

    original_reference_text: str = Field(min_length=1)
    original_reference_sha256: Sha256Hex
    effective_origin: ReferenceOrigin
    review_status: ReferenceReviewStatus = ReferenceReviewStatus.UNREVIEWED
    decision: ReferenceReviewDecision | None = None
    reviewer_type: NonEmptyText | None = None
    reviewer: NonEmptyText | None = None
    reviewed_at_utc: NonEmptyText | None = None
    rationale: NonEmptyText | None = None
    reference_review_sha256: Sha256Hex | None = None
    source_feedback_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def decision_matches_origin(self) -> ReferenceProvenance:
        if not self.original_reference_text.strip():
            raise ValueError("original_reference_text must not be blank")
        actual_hash = hashlib.sha256(
            self.original_reference_text.encode("utf-8")
        ).hexdigest()
        if actual_hash != self.original_reference_sha256:
            raise ValueError(
                "original_reference_sha256 must match original_reference_text"
            )
        if self.effective_origin is ReferenceOrigin.REVIEWER_CORRECTION:
            if self.decision is not ReferenceReviewDecision.CORRECTED:
                raise ValueError("a reviewer correction requires decision=corrected")
            if self.rationale is None:
                raise ValueError("a reviewer correction requires a rationale")
        elif self.decision is ReferenceReviewDecision.CORRECTED:
            raise ValueError("decision=corrected requires reviewer_correction origin")
        return self


class EvaluationCase(SchemaModel):
    """Gold fields are for metrics only and must not enter pipeline requests."""

    case_id: NonEmptyText
    source_record_id: NonEmptyText | None = None
    source_text: NonEmptyText
    reference_text: NonEmptyText
    reference_provenance: ReferenceProvenance | None = None
    domain: NonEmptyText | None = None
    scenario_tags: list[NonEmptyText] = Field(default_factory=list)
    selection_note: NonEmptyText | None = None
    expected_terms: list[ExpectedTerm] = Field(default_factory=list)
    manual_judgments: dict[ExecutionMode, ManualJudgmentLabel] = Field(
        default_factory=dict,
        description=(
            "Confirmed or ambiguous human labels for attempt zero, keyed by Agent mode."
        ),
    )
    manual_outcomes: dict[ExecutionMode, ManualOutcomeLabel] = Field(
        default_factory=dict,
        description=(
            "Confirmed or ambiguous human initial-to-final outcome labels, keyed "
            "by Agent mode."
        ),
    )

    @model_validator(mode="after")
    def manual_labels_only_target_agent_modes(self) -> EvaluationCase:
        allowed = {ExecutionMode.AGENT, ExecutionMode.AGENT_RAG}
        for field_name, labels in (
            ("judgment", self.manual_judgments),
            ("outcome", self.manual_outcomes),
        ):
            invalid = set(labels) - allowed
            if invalid:
                values = ", ".join(sorted(mode.value for mode in invalid))
                raise ValueError(
                    f"manual Agent {field_name} labels cannot target: {values}"
                )

        provenance = self.reference_provenance
        if provenance is not None:
            original = provenance.original_reference_text.strip()
            if provenance.effective_origin is ReferenceOrigin.PUBLIC_DATASET:
                if self.reference_text != original:
                    raise ValueError(
                        "a public effective reference must match the original reference"
                    )
            elif self.reference_text == original:
                raise ValueError(
                    "a reviewer correction must differ from the original reference"
                )
        return self

    def to_translation_request(self) -> TranslationRequest:
        """Create a leak-safe request containing only the source sentence."""

        return TranslationRequest(text=self.source_text)


class BenchmarkRequest(SchemaModel):
    dataset_id: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            pattern=r"^[A-Za-z0-9_.-]+$",
        ),
    ] = "evaluation_v1"
    modes: list[ExecutionMode] = Field(
        default_factory=lambda: [
            ExecutionMode.BASELINE,
            ExecutionMode.AGENT_RAG,
        ],
        min_length=1,
    )
    limit: int | None = Field(default=None, ge=1, le=50)
    warmup: bool = True

    @model_validator(mode="after")
    def modes_are_unique(self) -> BenchmarkRequest:
        if len(self.modes) != len(set(self.modes)):
            raise ValueError("benchmark modes must be unique")
        return self


class AgentConfusionCounts(SchemaModel):
    """Counts for first Agent judgments; ``needs revision`` is positive."""

    true_positive: int = Field(default=0, ge=0)
    true_negative: int = Field(default=0, ge=0)
    false_positive: int = Field(default=0, ge=0)
    false_negative: int = Field(default=0, ge=0)


class ModeMetrics(SchemaModel):
    sample_count: int = Field(ge=0)
    terminology_accuracy_pct: Percentage | None = None
    changed_sentence_rate_pct: Percentage | None = None
    successful_correction_rate_pct: Percentage | None = None
    successful_correction_improved_count: int = Field(default=0, ge=0)
    successful_correction_labeled_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Confirmed initial-to-final outcomes whose confirmed initial label "
            "requires revision."
        ),
    )
    mean_latency_ms: Milliseconds
    median_latency_ms: Milliseconds
    p95_latency_ms: Milliseconds
    mean_retry_count: float | None = Field(default=None, ge=0.0, le=2.0)
    retry_distribution: dict[int, int] = Field(default_factory=dict)
    agent_judgment_labeled_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Confirmed manual labels paired with an available first Agent judgment."
        ),
    )
    agent_judgment_accuracy_pct: Percentage | None = None
    agent_revision_recall_pct: Percentage | None = None
    agent_unnecessary_revision_rate_pct: Percentage | None = None
    agent_confusion_counts: AgentConfusionCounts | None = None
    error_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Agent-reported error types on the first attempt.",
    )
    manual_primary_error_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Primary error taxonomy from confirmed manual initial labels.",
    )
    manual_error_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="All error taxonomy values from confirmed manual initial labels.",
    )

    @model_validator(mode="after")
    def counts_are_non_negative(self) -> ModeMetrics:
        if any(retry < 0 or count < 0 for retry, count in self.retry_distribution.items()):
            raise ValueError("retry distribution keys and counts must be non-negative")
        if any(count < 0 for count in self.error_counts.values()):
            raise ValueError("error counts must be non-negative")
        if any(count < 0 for count in self.manual_primary_error_counts.values()):
            raise ValueError("manual primary error counts must be non-negative")
        if any(count < 0 for count in self.manual_error_type_counts.values()):
            raise ValueError("manual error type counts must be non-negative")
        if self.successful_correction_improved_count > (
            self.successful_correction_labeled_count
        ):
            raise ValueError(
                "successful correction count cannot exceed its labeled denominator"
            )
        if self.successful_correction_labeled_count == 0:
            if self.successful_correction_rate_pct is not None:
                raise ValueError(
                    "successful correction rate requires eligible manual outcomes"
                )
        else:
            expected_rate = (
                self.successful_correction_improved_count
                / self.successful_correction_labeled_count
                * 100.0
            )
            if self.successful_correction_rate_pct is None:
                raise ValueError(
                    "eligible manual outcomes require a successful correction rate"
                )
            if abs(self.successful_correction_rate_pct - expected_rate) > 1e-9:
                raise ValueError(
                    "successful correction rate must match its labeled counts"
                )
        if self.agent_confusion_counts is None:
            if self.agent_judgment_labeled_count != 0:
                raise ValueError(
                    "Agent labeled count requires confusion counts"
                )
            if self.agent_judgment_accuracy_pct is not None:
                raise ValueError(
                    "Agent accuracy requires confirmed-label confusion counts"
                )
        else:
            confusion_total = sum(
                (
                    self.agent_confusion_counts.true_positive,
                    self.agent_confusion_counts.true_negative,
                    self.agent_confusion_counts.false_positive,
                    self.agent_confusion_counts.false_negative,
                )
            )
            if confusion_total != self.agent_judgment_labeled_count:
                raise ValueError(
                    "Agent confusion total must equal the labeled count"
                )
            if self.agent_judgment_accuracy_pct is None:
                raise ValueError("Agent confusion counts require an accuracy value")
        return self


class BenchmarkResponse(SchemaModel):
    run_id: UUID = Field(default_factory=uuid4)
    dataset_id: NonEmptyText
    metrics_by_mode: dict[ExecutionMode, ModeMetrics]
    artifact_path: NonEmptyText | None = None
    unavailable_metrics: dict[str, NonEmptyText] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
