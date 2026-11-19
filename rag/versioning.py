"""Document version management and version rules (F3) in ONE self-contained file.

Works with the EXISTING ingestion output: chunks only carry ``source``, ``document_year``,
``file_type``, ``start_index``, ``chunk_index``. Version info lives in each document's header
("Document version / Effective date / Status"), so a ``VersionCatalog`` reads it once and labels
chunks by ``source``. If ingestion later adds ``doc_id``/``version``/``effective_date`` to chunk
metadata, those win automatically - no change needed here.

Flow (see ``VersionService``):  label chunks -> resolve versions -> [rerank] -> version rules.
Hard eligibility comes from version resolution; relevance scores can never override it.
Missing or contradictory metadata is NEVER guessed: the chunk becomes UNRESOLVED with a reason.

Flags: historical_query, superseded_only, version_unresolved, period_not_covered, ambiguous_period.
(``not_found`` = nothing relevant; that belongs to the confidence stage, not here.)
"""
from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- metadata keys -------------------------------------------------------------------------
KEY_DOC_ID, KEY_FAMILY_ID, KEY_VERSION = "doc_id", "family_id", "version"
KEY_EFFECTIVE_DATE, KEY_EFFECTIVE_TO = "effective_date", "effective_to"
KEY_STATUS, KEY_SUPERSEDED_BY, KEY_PAGE, KEY_SOURCE = "status", "superseded_by", "page", "source"
KEY_VERSION_STATUS = "version_status"            # read by the reranker's soft nudge
KEY_VERSION_ELIGIBILITY = "version_eligibility"
KEY_VERSION_ROLE = "version_role"                # answerable chunks only: current | historical
KEY_VERSION_ORIGIN = "version_metadata_origin"   # "catalog" when we filled the fields (audit)
KEY_CROSS_ENCODER_SCORE = "cross_encoder_score"
_VERSION_KEYS = (KEY_DOC_ID, KEY_FAMILY_ID, KEY_VERSION, KEY_EFFECTIVE_DATE)
UNKNOWN_FAMILY = "<unknown>"
_NON_BLOCKING_ISSUES = frozenset({"invalid_page"})

FLAG_HISTORICAL_QUERY = "historical_query"
FLAG_SUPERSEDED_ONLY = "superseded_only"
FLAG_VERSION_UNRESOLVED = "version_unresolved"
FLAG_PERIOD_NOT_COVERED = "period_not_covered"
FLAG_AMBIGUOUS_PERIOD = "ambiguous_period"
FLAG_RERANK_DEGRADED = "rerank_degraded"   # reranker fell back to a weaker scorer or failed


class VersionStatus(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    FUTURE = "future"
    UNRESOLVED = "unresolved"


class Eligibility(str, Enum):
    ELIGIBLE_CURRENT = "eligible_current"
    ELIGIBLE_HISTORICAL = "eligible_historical"
    INELIGIBLE_SUPERSEDED = "ineligible_superseded"
    INELIGIBLE_FUTURE = "ineligible_future"
    UNRESOLVED = "unresolved"


class TemporalIntent(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"


class RejectionReason(str, Enum):
    SUPERSEDED = "superseded_for_current_question"
    FUTURE_EFFECTIVE = "future_effective"
    VERSION_UNRESOLVED = "version_unresolved"


@dataclass(frozen=True)
class TemporalContext:
    """Time frame of the question. ``as_of_start == date.min`` means 'any past period'."""

    intent: TemporalIntent
    as_of_start: date
    as_of_end: date
    today: date
    signals: tuple[str, ...] = ()

    @property
    def open_ended(self) -> bool:
        return self.as_of_start == date.min


# =============================================================================================
# Header parsing
# =============================================================================================
HEADER_SCAN_CHARS = 2000
_HEADER_RE = {
    "version": re.compile(r"^\s*document version\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    "effective_date": re.compile(r"^\s*effective date\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    "status": re.compile(r"^\s*status\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
}
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y")
_TRAILING_YEAR = re.compile(r"[\s_\-]*(?:19|20)\d{2}$")


def parse_date_text(text: str) -> Optional[date]:
    """Parse a header date; None (never a guess) when no known format matches."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def derive_family_id(source: str) -> str:
    """Family id = file stem minus a trailing year, so yearly packs group together.

    Grouping only. Validity always comes from the header's version and effective date.
    """
    stem = Path(source).stem
    return _TRAILING_YEAR.sub("", stem) or stem


def parse_version_header(text: str, source: str) -> dict[str, str]:
    """Return {doc_id, version, effective_date, status} parsed from a header; {} if not a header."""
    head = text[:HEADER_SCAN_CHARS]
    version = _HEADER_RE["version"].search(head)
    eff_raw = _HEADER_RE["effective_date"].search(head)
    eff = parse_date_text(eff_raw.group(1)) if eff_raw else None
    if not version or not eff:
        return {}
    out = {KEY_DOC_ID: derive_family_id(source), KEY_VERSION: version.group(1).strip(),
           KEY_EFFECTIVE_DATE: eff.isoformat()}
    status = _HEADER_RE["status"].search(head)
    if status:
        low = status.group(1).lower()
        if "superseded" in low:
            out[KEY_STATUS] = "superseded"
        elif "current" in low:
            out[KEY_STATUS] = "current"
    return out


# =============================================================================================
# Tolerant metadata parsing
# =============================================================================================
# Text that ingestion may write when it has no real value. It is treated as "missing".
_PLACEHOLDER_VALUES = frozenset({"unknown", "none", "null", "nan", "n/a", "na", "-"})


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_VALUES


def _clean(value: Any) -> Optional[str]:
    if value is None or _is_placeholder(value):
        return None
    return str(value).strip() or None


def _iso(value: Any, name: str, issues: list[str]) -> Optional[date]:
    if value is None or value == "" or _is_placeholder(value):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        issues.append(f"invalid_{name}")
        return None


@dataclass(frozen=True)
class VersionMeta:
    """Version-relevant metadata of one chunk. Problems go to ``issues``; nothing is raised."""

    doc_id: Optional[str] = None
    family_id: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[date] = None
    effective_to: Optional[date] = None
    status: Optional[str] = None
    superseded_by: Optional[str] = None
    page: Optional[int] = None
    issues: tuple[str, ...] = ()

    @property
    def family_key(self) -> Optional[str]:
        return self.family_id or self.doc_id

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        return tuple(i for i in self.issues if i not in _NON_BLOCKING_ISSUES)

    @property
    def is_complete(self) -> bool:
        return bool(self.family_key and self.version and self.effective_date) and not self.blocking_issues

    @classmethod
    def from_metadata(cls, md: Mapping[str, Any]) -> "VersionMeta":
        issues: list[str] = []
        doc_id, family_id, version = _clean(md.get(KEY_DOC_ID)), _clean(md.get(KEY_FAMILY_ID)), _clean(md.get(KEY_VERSION))
        if not (doc_id or family_id):
            issues.append("missing_doc_id")
        if version is None:
            issues.append("missing_version")
        eff = _iso(md.get(KEY_EFFECTIVE_DATE), "effective_date", issues)
        if eff is None and "invalid_effective_date" not in issues:
            issues.append("missing_effective_date")
        eff_to = _iso(md.get(KEY_EFFECTIVE_TO), "effective_to", issues)
        if eff and eff_to and eff_to < eff:
            issues.append("effective_to_before_effective_date")
        status = _clean(md.get(KEY_STATUS))
        if status is not None:
            status = status.lower()
            if status not in {"current", "superseded"}:
                issues.append("invalid_status")
                status = None
        page: Optional[int] = None
        if md.get(KEY_PAGE) is not None:
            try:
                page = int(md[KEY_PAGE])
            except (TypeError, ValueError):
                issues.append("invalid_page")
        return cls(doc_id, family_id, version, eff, eff_to, status, _clean(md.get(KEY_SUPERSEDED_BY)), page, tuple(issues))


# =============================================================================================
# Catalog: knows every version in the corpus, labels chunks by ``source``
# =============================================================================================
class VersionRegistry(Protocol):
    def versions_for(self, family: str) -> Sequence[VersionMeta]: ...


class VersionCatalog:
    """source filename -> version metadata, built from document headers (or later, real records).

    Also acts as the registry that answers "what is the current version?" even when the current
    document was not among the retrieved chunks (needed for superseded_only).
    """

    def __init__(self, entries: Mapping[str, Mapping[str, str]]) -> None:
        self._entries = {name: dict(md) for name, md in entries.items()}

    @classmethod
    def from_markdown_dir(cls, directory: "str | Path") -> "VersionCatalog":
        """Read headers from the processed Markdown files (pass your processed-data directory)."""
        entries: dict[str, dict[str, str]] = {}
        for path in sorted(Path(directory).glob("*.md")):
            parsed = parse_version_header(path.read_text(encoding="utf-8"), path.name)
            if parsed:
                entries[path.name] = parsed
            else:
                logger.warning("No readable version header in %s; its chunks will be UNRESOLVED", path.name)
        return cls(entries)

    @classmethod
    def from_documents(cls, docs: Iterable[Document]) -> "VersionCatalog":
        """Learn from chunks that contain a header (the first chunk of each file) - e.g. at startup."""
        entries: dict[str, dict[str, str]] = {}
        for doc in docs:
            source = _clean(doc.metadata.get(KEY_SOURCE))
            if source and Path(source).name not in entries:
                parsed = parse_version_header(doc.page_content, source)
                if parsed:
                    entries[Path(source).name] = parsed
        return cls(entries)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def versions_for(self, family: str) -> Sequence[VersionMeta]:
        found = [VersionMeta.from_metadata(md) for md in self._entries.values() if md.get(KEY_DOC_ID) == family]
        return tuple(found)

    def label(self, docs: Sequence[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in docs:
            md = doc.metadata
            src = Path(str(md.get(KEY_SOURCE, "")))
            entry = self._entries.get(src.name) or next(
                (e for n, e in self._entries.items() if Path(n).stem == src.stem), None)
            if entry is None:
                out.append(doc)
                continue
            chunk_is_complete = (
                _clean(md.get(KEY_VERSION)) and _clean(md.get(KEY_EFFECTIVE_DATE))
                and (_clean(md.get(KEY_DOC_ID)) or _clean(md.get(KEY_FAMILY_ID)))
            )
            if chunk_is_complete:
                out.append(doc)
                continue
            out.append(doc.model_copy(update={"metadata": {**md, **entry, KEY_VERSION_ORIGIN: "catalog"}}))
        return out


# =============================================================================================
# Resolution
# =============================================================================================
@dataclass(frozen=True)
class VersionDecision:
    family: str
    version: Optional[str]
    effective_date: Optional[date]
    effective_to: Optional[date]
    status: VersionStatus
    eligibility: Eligibility
    superseded_by: Optional[str]
    reason: str


@dataclass(frozen=True)
class FamilyResolution:
    family: str
    requested_as_of: date
    current_version: Optional[str]
    current_effective_date: Optional[date]
    decisions: tuple[VersionDecision, ...]
    conflicts: tuple[str, ...]

    @property
    def current_info_available(self) -> bool:
        return self.current_version is not None


_MERGE_FIELDS = ("effective_date", "effective_to", "status", "superseded_by")


def _decision(family: str, version: str, v: Mapping[str, Any], status: VersionStatus,
              eligibility: Eligibility, successor: Optional[str], reason: str) -> VersionDecision:
    return VersionDecision(family, version, v["effective_date"], v["effective_to"], status,
                           eligibility, successor or v["superseded_by"], reason)


def _eligibility(status: VersionStatus, start: date, end: Optional[date], t: TemporalContext) -> tuple[Eligibility, str]:
    if status is VersionStatus.FUTURE:
        return Eligibility.INELIGIBLE_FUTURE, "not yet effective on the reference date"
    if t.intent is TemporalIntent.CURRENT:
        if status is VersionStatus.CURRENT:
            return Eligibility.ELIGIBLE_CURRENT, "valid on the requested date"
        return Eligibility.INELIGIBLE_SUPERSEDED, "outdated for the requested date"
    if status is VersionStatus.CURRENT:  # historical question: keep the current version for comparison
        return Eligibility.ELIGIBLE_CURRENT, "current version, kept for comparison with historical terms"
    if end is None:
        return Eligibility.UNRESOLVED, "reported superseded but end of validity unknown; cannot prove it applied in the period"
    if start <= t.as_of_end and end >= t.as_of_start:
        return Eligibility.ELIGIBLE_HISTORICAL, "was valid during the requested period"
    return Eligibility.INELIGIBLE_SUPERSEDED, "superseded and not valid during the requested period"


def _resolve_family(family: str, records: Sequence[VersionMeta], t: TemporalContext) -> FamilyResolution:
    status_date = t.today if t.intent is TemporalIntent.HISTORICAL else t.as_of_end
    conflicts: list[str] = []
    family_conflict = False  # True => the ordering of versions cannot be trusted

    usable: dict[str, list[VersionMeta]] = {}
    for rec in records:
        if rec.is_complete:
            usable.setdefault(rec.version, []).append(rec)  # type: ignore[arg-type]
    for rec in records:
        if not rec.is_complete:
            if rec.version in usable:
                conflicts.append(f"incomplete_chunk_metadata:{rec.version}")
            else:  # an unknown, possibly newer version -> cannot order safely
                conflicts.append("incomplete_metadata")
                family_conflict = True

    merged: dict[str, dict[str, Any]] = {}
    taint: dict[str, str] = {}
    for version, recs in usable.items():  # contradictions across chunks are explicit, never averaged
        values: dict[str, Any] = {}
        for fld in _MERGE_FIELDS:
            vals = {getattr(r, fld) for r in recs if getattr(r, fld) is not None}
            if len(vals) > 1:
                conflicts.append(f"inconsistent_{fld}:{version}")
                if fld == "effective_date":
                    family_conflict = True
                else:
                    taint[version] = f"conflicting {fld} across chunks"
                vals = set()
            values[fld] = next(iter(vals)) if vals else None
        if values["effective_date"] is not None:
            merged[version] = values

    timeline = sorted(merged.items(), key=lambda kv: (kv[1]["effective_date"], kv[0]))
    dates = {v: vals["effective_date"] for v, vals in timeline}

    for (va, a), (vb, b) in zip(timeline, timeline[1:]):
        if a["effective_date"] == b["effective_date"]:
            conflicts.append(f"duplicate_effective_date:{va},{vb}")
        elif a["effective_to"] and a["effective_to"] >= b["effective_date"]:
            conflicts.append(f"overlapping_periods:{va},{vb}")
        else:
            continue
        taint.setdefault(va, "versions have duplicate or overlapping validity")
        taint.setdefault(vb, "versions have duplicate or overlapping validity")

    decisions: list[VersionDecision] = []
    for idx, (version, v) in enumerate(timeline):
        start = v["effective_date"]
        nxt_version, nxt = timeline[idx + 1] if idx + 1 < len(timeline) else (None, None)
        has_successor = nxt is not None and nxt["effective_date"] > start
        ends = [e for e in (v["effective_to"], nxt["effective_date"] - timedelta(days=1) if has_successor else None) if e]
        end = min(ends) if ends else None
        successor = nxt_version if has_successor else None

        reported = v["superseded_by"]
        if reported and successor and reported != successor:
            conflicts.append(f"superseded_by_mismatch:{version}")
            taint.setdefault(version, "superseded_by contradicts the version timeline")
        elif reported and reported in dates and dates[reported] <= start:
            conflicts.append(f"superseded_by_not_newer:{version}")
            taint.setdefault(version, "superseded_by points to an older version")

        if start > status_date:
            status = VersionStatus.FUTURE
        elif end is not None and end < status_date:
            status = VersionStatus.SUPERSEDED
        elif v["status"] == "superseded":
            status = VersionStatus.SUPERSEDED  # trust a 'superseded' claim; never promote it to current
        else:
            status = VersionStatus.CURRENT
        if v["status"] == "current" and status is VersionStatus.SUPERSEDED:
            conflicts.append(f"status_contradiction:{version}")
            taint.setdefault(version, "reported current but superseded by dates")

        if version in taint:
            decisions.append(_decision(family, version, v, VersionStatus.UNRESOLVED, Eligibility.UNRESOLVED,
                                       successor, f"unresolved: {taint[version]}"))
            continue
        eligibility, why = _eligibility(status, start, end, t)
        if status is VersionStatus.SUPERSEDED:
            why += f"; replaced by {successor}" if successor else "; no replacement known"
        decisions.append(_decision(family, version, v, status, eligibility, successor, why))

    if family_conflict:  # nothing may be called valid when the ordering itself is in doubt
        decisions = [
            replace(d, status=VersionStatus.UNRESOLVED, eligibility=Eligibility.UNRESOLVED,
                    reason="unresolved: family metadata conflict, version order cannot be trusted")
            if d.eligibility in (Eligibility.ELIGIBLE_CURRENT, Eligibility.ELIGIBLE_HISTORICAL) else d
            for d in decisions
        ]

    current = [d for d in decisions if d.status is VersionStatus.CURRENT]
    if len(current) > 1:  # defensive: should be impossible after the checks above
        conflicts.append("multiple_current_versions")
        decisions = [replace(d, status=VersionStatus.UNRESOLVED, eligibility=Eligibility.UNRESOLVED,
                             reason="unresolved: multiple current versions") if d in current else d for d in decisions]
        current = []

    return FamilyResolution(
        family=family,
        requested_as_of=t.as_of_end,
        current_version=current[0].version if current else None,
        current_effective_date=current[0].effective_date if current else None,
        decisions=tuple(decisions),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )


class VersionManager:
    """Resolves which version of each document family is valid for a question's time frame."""

    def __init__(self, registry: Optional[VersionRegistry] = None) -> None:
        self._registry = registry

    def resolve(self, docs: Sequence[Document], temporal: TemporalContext) -> dict[str, FamilyResolution]:
        by_family: dict[str, list[VersionMeta]] = {}
        for doc in docs:
            meta = VersionMeta.from_metadata(doc.metadata)
            by_family.setdefault(meta.family_key or UNKNOWN_FAMILY, []).append(meta)
        resolutions: dict[str, FamilyResolution] = {}
        for family, records in by_family.items():
            if self._registry is not None and family != UNKNOWN_FAMILY:
                records = [*self._registry.versions_for(family), *records]
            resolutions[family] = _resolve_family(family, records, temporal)
        return resolutions


def find_decision(resolutions: Mapping[str, FamilyResolution], meta: VersionMeta) -> VersionDecision:
    """Decision for ONE chunk. A chunk with its own bad metadata is UNRESOLVED regardless of siblings."""
    family = meta.family_key or UNKNOWN_FAMILY
    if not meta.is_complete:
        problems = ", ".join(meta.blocking_issues) or "missing fields"
        return VersionDecision(family, meta.version, meta.effective_date, meta.effective_to, VersionStatus.UNRESOLVED,
                               Eligibility.UNRESOLVED, meta.superseded_by, f"chunk version metadata incomplete or invalid: {problems}")
    resolution = resolutions.get(family)
    if resolution:
        for decision in resolution.decisions:
            if decision.version == meta.version:
                return decision
    return VersionDecision(family, meta.version, meta.effective_date, meta.effective_to, VersionStatus.UNRESOLVED,
                           Eligibility.UNRESOLVED, meta.superseded_by, "no resolution available for this version")


# =============================================================================================
# Rule 2: temporal intent detection
# =============================================================================================
@dataclass(frozen=True)
class VersionRulesConfig:
    """Question-interpretation settings. No year is hardcoded; ranges are relative to ``today``."""

    historical_markers: tuple[str, ...] = (
        "previous", "previously", "old", "older", "prior", "earlier version", "superseded",
        "no longer", "still apply", "used to", "back then", "was the", "were the", "legacy", "original terms",
    )
    year_pattern: str = r"(?<!\d)(?:19|20)\d{2}(?!\d)"
    max_years_back: int = 15   # older 4-digit numbers are treated as amounts, not years
    currency_prefixes: tuple[str, ...] = ("₹", "$", "£", "€", "rs", "rs.", "inr")

    def __post_init__(self) -> None:
        if any(m != m.lower() for m in self.historical_markers) or self.max_years_back < 1:
            raise ValueError("markers must be lower-case and max_years_back >= 1")


_MONTHS = {n.lower(): i for i, n in enumerate(calendar.month_name) if n}
_MONTHS.update({a.lower(): i for i, a in enumerate(calendar.month_abbr) if a})


def _plausible_year(question: str, match: "re.Match[str]", today: date, cfg: VersionRulesConfig) -> bool:
    """Reject amounts that look like years (e.g. '₹2000') and implausibly old/future numbers."""
    if not (today.year - cfg.max_years_back <= int(match.group()) <= today.year + 1):
        return False
    before = question[max(0, match.start() - 4):match.start()].lower().strip()
    return not any(before.endswith(p) for p in cfg.currency_prefixes)


def _explicit_period(question: str, today: date, cfg: VersionRulesConfig) -> Optional[tuple[date, date, str]]:
    iso = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", question)
    if iso:
        try:
            d = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            return d, d, "date"
        except ValueError:
            pass
    months = "|".join(sorted(_MONTHS, key=len, reverse=True))
    my = re.search(rf"(?<!\w)({months})\.?\s+({cfg.year_pattern})", question, re.IGNORECASE)
    if my:
        y, m = int(my.group(2)), _MONTHS[my.group(1).lower()]
        if today.year - cfg.max_years_back <= y <= today.year + 1:
            return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]), "month"
    years = [int(m.group()) for m in re.finditer(cfg.year_pattern, question) if _plausible_year(question, m, today, cfg)]
    if years:
        return date(min(years), 1, 1), date(max(years), 12, 31), "year"
    return None


def build_temporal_context(question: str, today: date, *, requested_as_of: Optional[date] = None,
                           config: Optional[VersionRulesConfig] = None) -> TemporalContext:
    """As-of priority: explicit API date > date/period in the question > today (injected clock)."""
    cfg = config or VersionRulesConfig()
    lowered = question.lower()
    markers = [m for m in cfg.historical_markers if re.search(rf"(?<!\w){re.escape(m)}(?!\w)", lowered)]
    signals = [f"marker:{m}" for m in markers]
    open_past = TemporalContext(TemporalIntent.HISTORICAL, date.min, today, today, tuple(signals))

    explicit = (requested_as_of, requested_as_of, "request") if requested_as_of else _explicit_period(question, today, cfg)
    if explicit:
        start, end, source = explicit
        signals.append(f"explicit_{source}")
        if end < today:
            return TemporalContext(TemporalIntent.HISTORICAL, start, end, today, tuple(signals))
        if start > today:
            return TemporalContext(TemporalIntent.CURRENT, start, start, today, tuple(signals))
        return open_past if markers else TemporalContext(TemporalIntent.CURRENT, today, today, today, tuple(signals))
    return open_past if markers else TemporalContext(TemporalIntent.CURRENT, today, today, today, ())


# =============================================================================================
# Rules 1, 3-6: eligibility split and flags
# =============================================================================================
@dataclass(frozen=True)
class RejectedDoc:
    """A withheld chunk, for diagnostics only. Never put it in the LLM prompt."""

    doc: Document
    reason: RejectionReason
    detail: str


@dataclass
class RuleOutcome:
    answerable: list[Document] = field(default_factory=list)   # each stamped with version_role
    rejected: list[RejectedDoc] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    version_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    temporal: Optional[TemporalContext] = None
    security_blocked: list[Document] = field(default_factory=list)  # from the reranker, if used


def label_versions(docs: Sequence[Document], resolutions: Mapping[str, FamilyResolution]) -> list[Document]:
    """Stamp version_status / version_eligibility on COPIES (call before reranking)."""
    out = []
    for doc in docs:
        d = find_decision(resolutions, VersionMeta.from_metadata(doc.metadata))
        out.append(doc.model_copy(update={"metadata": {**doc.metadata, KEY_VERSION_STATUS: d.status.value,
                                                      KEY_VERSION_ELIGIBILITY: d.eligibility.value}}))
    return out


def _replacement_text(res: Optional[FamilyResolution]) -> str:
    if res and res.current_info_available:
        return f"current version is {res.current_version} (effective {res.current_effective_date})"
    return "current version information unavailable"


def _supporting(doc: Document, min_relevance: Optional[float]) -> bool:
    score = doc.metadata.get(KEY_CROSS_ENCODER_SCORE)
    return min_relevance is None or score is None or float(score) >= min_relevance


def apply_version_rules(docs: Sequence[Document], resolutions: Mapping[str, FamilyResolution],
                        temporal: TemporalContext, *, min_relevance: Optional[float] = None) -> RuleOutcome:
    """Split ALL scored candidates into answerable vs rejected and compute version flags.

    ``min_relevance`` (raw cross-encoder threshold) only decides which chunks count as supporting
    evidence for superseded_only / version_unresolved. None = every chunk counts.
    """
    out = RuleOutcome(temporal=temporal)
    supporting: list[Eligibility] = []
    covered = False
    families: dict[str, FamilyResolution] = {}
    historical_versions: dict[str, set[str]] = {}
    reasons = {Eligibility.INELIGIBLE_SUPERSEDED: RejectionReason.SUPERSEDED,
               Eligibility.INELIGIBLE_FUTURE: RejectionReason.FUTURE_EFFECTIVE,
               Eligibility.UNRESOLVED: RejectionReason.VERSION_UNRESOLVED}

    for doc in docs:
        decision = find_decision(resolutions, VersionMeta.from_metadata(doc.metadata))
        if decision.family in resolutions:
            families[decision.family] = resolutions[decision.family]
        md = {**doc.metadata, KEY_VERSION_STATUS: decision.status.value, KEY_VERSION_ELIGIBILITY: decision.eligibility.value}

        if decision.eligibility in (Eligibility.ELIGIBLE_CURRENT, Eligibility.ELIGIBLE_HISTORICAL):
            role = "current" if decision.eligibility is Eligibility.ELIGIBLE_CURRENT else "historical"
            out.answerable.append(doc.model_copy(update={"metadata": {**md, KEY_VERSION_ROLE: role}}))
            if role == "historical":
                historical_versions.setdefault(decision.family, set()).add(str(decision.version))
        else:
            detail = f"{decision.reason}; {_replacement_text(resolutions.get(decision.family))}"
            out.rejected.append(RejectedDoc(doc.model_copy(update={"metadata": md}), reasons[decision.eligibility], detail))

        if _supporting(doc, min_relevance):
            supporting.append(decision.eligibility)
            if decision.eligibility is Eligibility.ELIGIBLE_HISTORICAL or (
                decision.eligibility is Eligibility.ELIGIBLE_CURRENT
                and decision.effective_date and decision.effective_date <= temporal.as_of_end
            ):
                covered = True

    if temporal.intent is TemporalIntent.HISTORICAL:
        out.flags.append(FLAG_HISTORICAL_QUERY)
        if not temporal.open_ended:
            if supporting and not covered:
                out.flags.append(FLAG_PERIOD_NOT_COVERED)
            if any(len(v) > 1 for v in historical_versions.values()):
                out.flags.append(FLAG_AMBIGUOUS_PERIOD)

    eligible = (Eligibility.ELIGIBLE_CURRENT, Eligibility.ELIGIBLE_HISTORICAL)
    if supporting and not any(e in eligible for e in supporting):
        if Eligibility.UNRESOLVED in supporting:
            out.flags.append(FLAG_VERSION_UNRESOLVED)
        elif Eligibility.INELIGIBLE_SUPERSEDED in supporting:
            out.flags.append(FLAG_SUPERSEDED_ONLY)
        else:  # only future-effective evidence: validity for today is not established
            out.flags.append(FLAG_VERSION_UNRESOLVED)

    for family, res in families.items():
        out.version_info[family] = {
            "document_id": family,
            "current_version": res.current_version,
            "effective_date": res.current_effective_date.isoformat() if res.current_effective_date else None,
            "current_info_available": res.current_info_available,
            "requested_as_of": temporal.as_of_end.isoformat(),
            "decisions": [{"version": d.version, "effective_date": d.effective_date.isoformat() if d.effective_date else None,
                           "status": d.status.value, "eligibility": d.eligibility.value,
                           "superseded_by": d.superseded_by, "reason": d.reason} for d in res.decisions],
            "conflicts": list(res.conflicts),
        }
    return out


# =============================================================================================
# Orchestration: label -> resolve -> rerank -> rules
# =============================================================================================
def _cap_answerable(docs: list[Document], limit: Optional[int]) -> list[Document]:
    """Keep the first ``limit`` chunks (already best-first). When both current and historical
    chunks exist, keep at least one of each so old-vs-current comparisons still work."""
    if limit is None or len(docs) <= limit:
        return docs
    kept = list(docs[:limit])
    roles = {d.metadata.get(KEY_VERSION_ROLE) for d in docs}
    if len(roles) > 1 and limit >= len(roles):
        for role in roles:
            if all(d.metadata.get(KEY_VERSION_ROLE) != role for d in kept):
                extra = next(d for d in docs if d.metadata.get(KEY_VERSION_ROLE) == role)
                kept = kept[:-1] + [extra]
    return kept


class VersionService:
    """Create once per process. ``run`` wires versioning around the reranker."""

    def __init__(self, catalog: VersionCatalog, *, today_provider: Callable[[], date] = date.today,
                 rules_config: Optional[VersionRulesConfig] = None) -> None:
        self._catalog = catalog
        self._manager = VersionManager(catalog)
        self._today = today_provider
        self._rules_cfg = rules_config

    def run(self, question: str, docs: Sequence[Document], *, reranker: Any = None,
            requested_as_of: Optional[date] = None, min_relevance: Optional[float] = None,
            max_answerable: Optional[int] = None) -> RuleOutcome:
        """``reranker``: object with ``rerank(question, docs, temporal_intent=...)`` -> ``.ranked_docs``
        (the Reranker in rag/reranking.py). None = keep the given order, no scores."""
        temporal = build_temporal_context(question, self._today(), requested_as_of=requested_as_of, config=self._rules_cfg)
        labelled = self._catalog.label(docs)
        resolutions = self._manager.resolve(labelled, temporal)
        candidates = label_versions(labelled, resolutions)

        blocked: list[Document] = []
        degraded = False
        if reranker is not None:  # reranker scores the FULL set; no truncation before the rules run
            result = reranker.rerank(question, candidates, temporal_intent=temporal.intent.value)
            candidates, blocked = result.ranked_docs, result.blocked_docs
            degraded = bool(getattr(result, "degraded", False))
            if degraded:
                min_relevance = None  # the fallback scorer has a different score scale

        outcome = apply_version_rules(candidates, resolutions, temporal, min_relevance=min_relevance)
        outcome.security_blocked = blocked
        if degraded:
            outcome.flags.append(FLAG_RERANK_DEGRADED)
        outcome.answerable = _cap_answerable(outcome.answerable, max_answerable)  # final top-N, optional
        return outcome