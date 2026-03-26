"""
Professor Agent — Expert Profile Scraping Scaffold
====================================================

This scaffold collects and structures public data about domain experts
to build rich professor profiles for tacit knowledge externalization.

Architecture:
  1. Scrapers: One per data source (Scholar, YouTube, University, LinkedIn, etc.)
  2. ProfileBuilder: Merges scraped data into a unified professor profile JSON
  3. ContentIndexer: Indexes the professor's own content for the content database
  4. Validator: Checks profile completeness and quality

Schemas:
  - CONFIG_SCHEMA  : Canonical input schema for build_profile() (see get_config_schema())
  - OUTPUT_SCHEMA  : Canonical output schema (mirrors ProfessorProfile) (see get_output_schema())

Usage:
  python professor_scraper.py --name "Michael Porter" --affiliation "HBS" --skill "Digital Strategy"
  python professor_scraper.py --from-config professors.yaml
  python professor_scraper.py --print-config-schema
  python professor_scraper.py --print-output-schema
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# 1. PROFESSOR PROFILE SCHEMA
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Paper:
    title: str
    year: int
    abstract: str = ""
    citations: int = 0
    journal_or_venue: str = ""
    coauthors: list[str] = field(default_factory=list)
    url: str = ""
    key_concepts: list[str] = field(default_factory=list)  # Extracted by LLM post-scrape

@dataclass
class Course:
    title: str
    institution: str
    level: str = ""            # "mba", "executive_ed", "undergraduate", "phd"
    syllabus_topics: list[str] = field(default_factory=list)
    reading_list: list[str] = field(default_factory=list)
    case_studies: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)
    year: int = 0
    url: str = ""

@dataclass
class Video:
    title: str
    platform: str              # "youtube", "coursera", "edx", "university", "conference"
    url: str = ""
    duration_minutes: int = 0
    transcript_excerpt: str = ""  # First 2000 chars of transcript (fair use)
    key_topics: list[str] = field(default_factory=list)  # Extracted by LLM post-scrape
    year: int = 0
    context: str = ""          # "lecture", "keynote", "panel", "interview", "tutorial"

@dataclass
class Publication:
    """Non-academic publications: articles, blog posts, columns, book chapters"""
    title: str
    outlet: str                # "HBR", "MIT Sloan Review", "LinkedIn", "personal blog", etc.
    url: str = ""
    year: int = 0
    summary: str = ""          # 2-3 sentence summary (NOT full text — copyright)
    key_arguments: list[str] = field(default_factory=list)  # Extracted by LLM post-scrape
    publication_type: str = ""  # "article", "opinion", "book_chapter", "report"

@dataclass
class Book:
    title: str
    year: int
    publisher: str = ""
    summary: str = ""          # Public description / blurb only
    key_themes: list[str] = field(default_factory=list)
    table_of_contents: list[str] = field(default_factory=list)  # Chapter titles only

@dataclass
class ProfessorProfile:
    """
    Complete public profile of a domain expert.
    This is the primary output of the scraping pipeline and the
    primary input to the tacit knowledge externalization pipeline.
    """
    # ── Identity ──
    professor_id: str          # Unique ID: lowercase, hyphenated (e.g., "michael-porter")
    full_name: str
    primary_affiliation: str   # Current institution
    secondary_affiliations: list[str] = field(default_factory=list)
    title_role: str = ""       # "Professor of Strategy", "Chair of Digital Initiative"

    # ── Skill mapping ──
    mapped_skills: list[str] = field(default_factory=list)  # Skills this expert covers
    primary_skill: str = ""    # The skill they're MOST authoritative on
    expertise_domains: list[str] = field(default_factory=list)  # Broader domains

    # ── Research corpus ──
    papers: list[Paper] = field(default_factory=list)
    h_index: int = 0
    total_citations: int = 0
    research_themes: list[str] = field(default_factory=list)  # LLM-extracted themes across papers
    research_evolution: str = ""  # LLM-generated: how their research focus shifted over time

    # ── Teaching corpus ──
    courses: list[Course] = field(default_factory=list)
    teaching_philosophy: str = ""  # LLM-extracted from course descriptions, interviews
    preferred_pedagogy: list[str] = field(default_factory=list)  # "case_method", "simulation", etc.
    signature_cases: list[str] = field(default_factory=list)  # Cases they repeatedly use

    # ── Video/audio corpus ──
    videos: list[Video] = field(default_factory=list)
    explanation_style: str = ""  # LLM-extracted: how they explain complex concepts
    recurring_analogies: list[str] = field(default_factory=list)  # Analogies they reuse

    # ── Written corpus (non-academic) ──
    publications: list[Publication] = field(default_factory=list)
    books: list[Book] = field(default_factory=list)

    # ── Meta ──
    profile_completeness: float = 0.0  # 0-1 score
    last_scraped: str = ""
    scrape_sources: list[str] = field(default_factory=list)  # URLs that were scraped
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════
# 2. FIXED CONFIG SCHEMA & OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════════════

# ── Raw schema dicts (used for docs, validation, and CLI printing) ──

CONFIG_SCHEMA: dict = {
    "$schema": "professor-agent/config/v1",
    "description": (
        "Input config consumed by ProfileBuilder.build_profile(). "
        "Every key is described below with its type, whether it is required, "
        "and which scraper uses it."
    ),
    "fields": {
        "name": {
            "type": "str",
            "required": True,
            "description": "Professor's full name, exactly as it appears on their faculty page.",
            "example": "Sunil Gupta",
            "used_by": ["ProfileBuilder", "YouTubeScraper", "PublicationScraper"],
        },
        "affiliation": {
            "type": "str",
            "required": True,
            "description": "Primary institution (university or school name).",
            "example": "Harvard Business School",
            "used_by": ["ProfileBuilder"],
        },
        "title": {
            "type": "str",
            "required": False,
            "description": "Academic title / role at the institution.",
            "example": "Edward W. Carter Professor of Business Administration",
            "used_by": ["ProfileBuilder"],
        },
        "skills": {
            "type": "list[str]",
            "required": True,
            "description": (
                "All skills this professor has authority on. "
                "These are matched to the platform's skill taxonomy."
            ),
            "example": ["Digital Strategy", "Digital Transformation", "Platform Business Models"],
            "used_by": ["ProfileBuilder", "ContentIndexer"],
        },
        "primary_skill": {
            "type": "str",
            "required": True,
            "description": (
                "The single skill this professor is MOST authoritative on. "
                "Must be one of the values in 'skills'."
            ),
            "example": "Digital Strategy",
            "used_by": ["ProfileBuilder", "ContentIndexer"],
        },
        "domains": {
            "type": "list[str]",
            "required": False,
            "description": "Broader knowledge domains beyond the specific skill list.",
            "example": ["Technology & Strategy", "Digital Business"],
            "used_by": ["ProfileBuilder"],
        },
        "sources": {
            "type": "dict",
            "required": True,
            "description": "All scraping targets for this professor. See nested fields below.",
            "fields": {
                "google_scholar_id": {
                    "type": "str",
                    "required": False,
                    "description": (
                        "Google Scholar author ID. Found in the URL: "
                        "scholar.google.com/citations?user=<ID>"
                    ),
                    "example": "abc123XYZ",
                    "used_by": ["GoogleScholarScraper"],
                },
                "faculty_url": {
                    "type": "str",
                    "required": False,
                    "description": (
                        "Direct URL to the professor's official faculty/bio page. "
                        "Used to scrape courses, syllabi, and CV."
                    ),
                    "example": "https://www.hbs.edu/faculty/Pages/profile.aspx?facId=6547",
                    "used_by": ["UniversityPageScraper"],
                },
                "youtube_search_terms": {
                    "type": "list[str]",
                    "required": False,
                    "description": (
                        "Topic keywords searched alongside the professor's name on YouTube. "
                        "Keep short and specific — these produce the search query "
                        '"{name} {term}" per term.'
                    ),
                    "example": ["digital transformation", "platform strategy"],
                    "used_by": ["YouTubeScraper"],
                },
                "publication_outlets": {
                    "type": "list[str]",
                    "required": False,
                    "description": (
                        "Domain strings used to restrict publication searches. "
                        "If omitted, defaults to hbr.org, sloanreview.mit.edu, "
                        "linkedin.com, wsj.com, ft.com, forbes.com."
                    ),
                    "example": ["hbr.org", "sloanreview.mit.edu"],
                    "used_by": ["PublicationScraper"],
                },
            },
        },
    },
    "example": {
        "name": "Sunil Gupta",
        "affiliation": "Harvard Business School",
        "title": "Edward W. Carter Professor of Business Administration",
        "skills": ["Digital Strategy", "Digital Transformation", "Platform Business Models"],
        "primary_skill": "Digital Strategy",
        "domains": ["Technology & Strategy", "Digital Business"],
        "sources": {
            "google_scholar_id": "abc123XYZ",
            "faculty_url": "https://www.hbs.edu/faculty/Pages/profile.aspx?facId=6547",
            "youtube_search_terms": ["digital transformation", "platform strategy"],
            "publication_outlets": ["hbr.org", "sloanreview.mit.edu"],
        },
    },
}


OUTPUT_SCHEMA: dict = {
    "$schema": "professor-agent/output/v1",
    "description": (
        "Output produced by ProfileBuilder.build_profile() — a ProfessorProfile instance "
        "serialized to JSON. Fields marked LLM are populated by ProfileEnricher, not the scrapers."
    ),
    "root": "ProfessorProfile",
    "types": {
        "ProfessorProfile": {
            "professor_id": {
                "type": "str",
                "source": "derived",
                "description": "Lowercase hyphenated ID derived from the professor's name.",
                "example": "sunil-gupta",
            },
            "full_name": {
                "type": "str",
                "source": "config.name",
                "description": "Professor's full name as provided in the config.",
            },
            "primary_affiliation": {
                "type": "str",
                "source": "config.affiliation",
                "description": "Current institution.",
            },
            "secondary_affiliations": {
                "type": "list[str]",
                "source": "UniversityPageScraper",
                "description": "Any additional affiliations found on the faculty page.",
            },
            "title_role": {
                "type": "str",
                "source": "config.title",
                "description": "Academic title at the primary institution.",
            },
            "mapped_skills": {
                "type": "list[str]",
                "source": "config.skills",
                "description": "All skill taxonomy entries this professor covers.",
            },
            "primary_skill": {
                "type": "str",
                "source": "config.primary_skill",
                "description": "The single most authoritative skill.",
            },
            "expertise_domains": {
                "type": "list[str]",
                "source": "config.domains",
                "description": "Broader knowledge domains.",
            },
            "papers": {
                "type": "list[Paper]",
                "source": "GoogleScholarScraper",
                "description": "Academic papers scraped from Google Scholar.",
            },
            "h_index": {
                "type": "int",
                "source": "GoogleScholarScraper",
                "description": "H-index from Google Scholar.",
            },
            "total_citations": {
                "type": "int",
                "source": "GoogleScholarScraper",
                "description": "Total citation count from Google Scholar.",
            },
            "research_themes": {
                "type": "list[str]",
                "source": "LLM — ProfileEnricher",
                "description": "5-10 recurring intellectual themes extracted across the research corpus.",
            },
            "research_evolution": {
                "type": "str",
                "source": "LLM — ProfileEnricher",
                "description": "Narrative description of how the researcher's focus has shifted over time.",
            },
            "courses": {
                "type": "list[Course]",
                "source": "UniversityPageScraper",
                "description": "Courses scraped from the university faculty/course pages.",
            },
            "teaching_philosophy": {
                "type": "str",
                "source": "LLM — ProfileEnricher",
                "description": "Inferred teaching philosophy from course design choices.",
            },
            "preferred_pedagogy": {
                "type": "list[str]",
                "source": "LLM — ProfileEnricher",
                "description": 'Dominant pedagogical methods, e.g. ["case_method", "simulation"].',
            },
            "signature_cases": {
                "type": "list[str]",
                "source": "LLM — ProfileEnricher",
                "description": "Case studies or examples the professor returns to repeatedly.",
            },
            "videos": {
                "type": "list[Video]",
                "source": "YouTubeScraper",
                "description": "Videos found via YouTube search.",
            },
            "explanation_style": {
                "type": "str",
                "source": "LLM — ProfileEnricher",
                "description": "How the professor explains complex concepts — patterns across videos.",
            },
            "recurring_analogies": {
                "type": "list[str]",
                "source": "LLM — ProfileEnricher",
                "description": "Analogies or metaphors the professor reuses across talks and writings.",
            },
            "publications": {
                "type": "list[Publication]",
                "source": "PublicationScraper",
                "description": "Non-academic articles, columns, and blog posts.",
            },
            "books": {
                "type": "list[Book]",
                "source": "PublicationScraper",
                "description": "Books authored by the professor.",
            },
            "profile_completeness": {
                "type": "float",
                "source": "ProfileBuilder._calculate_completeness()",
                "description": "0.0–1.0 score based on how many corpus sections are populated.",
            },
            "last_scraped": {
                "type": "str (ISO 8601)",
                "source": "ProfileBuilder",
                "description": "Timestamp of the most recent scrape run.",
            },
            "scrape_sources": {
                "type": "list[str]",
                "source": "ProfileBuilder",
                "description": "URLs and source identifiers that were scraped for this profile.",
            },
            "notes": {
                "type": "str",
                "source": "manual / ProfileBuilder",
                "description": "Free-text notes about the profile or scraping issues.",
            },
        },
        "Paper": {
            "title": {"type": "str", "source": "GoogleScholarScraper"},
            "year": {"type": "int", "source": "GoogleScholarScraper"},
            "abstract": {"type": "str", "source": "GoogleScholarScraper", "note": "Public abstract only"},
            "citations": {"type": "int", "source": "GoogleScholarScraper"},
            "journal_or_venue": {"type": "str", "source": "GoogleScholarScraper"},
            "coauthors": {"type": "list[str]", "source": "GoogleScholarScraper"},
            "url": {"type": "str", "source": "GoogleScholarScraper"},
            "key_concepts": {"type": "list[str]", "source": "LLM — ProfileEnricher"},
        },
        "Course": {
            "title": {"type": "str", "source": "UniversityPageScraper"},
            "institution": {"type": "str", "source": "UniversityPageScraper"},
            "level": {
                "type": "str",
                "source": "UniversityPageScraper",
                "allowed_values": ["mba", "executive_ed", "undergraduate", "phd"],
            },
            "syllabus_topics": {"type": "list[str]", "source": "UniversityPageScraper"},
            "reading_list": {"type": "list[str]", "source": "UniversityPageScraper"},
            "case_studies": {"type": "list[str]", "source": "UniversityPageScraper"},
            "learning_objectives": {"type": "list[str]", "source": "UniversityPageScraper"},
            "year": {"type": "int", "source": "UniversityPageScraper"},
            "url": {"type": "str", "source": "UniversityPageScraper"},
        },
        "Video": {
            "title": {"type": "str", "source": "YouTubeScraper"},
            "platform": {
                "type": "str",
                "source": "YouTubeScraper",
                "allowed_values": ["youtube", "coursera", "edx", "university", "conference"],
            },
            "url": {"type": "str", "source": "YouTubeScraper"},
            "duration_minutes": {"type": "int", "source": "YouTubeScraper"},
            "transcript_excerpt": {
                "type": "str",
                "source": "YouTubeScraper",
                "note": "First 2000 characters of transcript only (fair use)",
            },
            "key_topics": {"type": "list[str]", "source": "LLM — ProfileEnricher"},
            "year": {"type": "int", "source": "YouTubeScraper"},
            "context": {
                "type": "str",
                "source": "YouTubeScraper",
                "allowed_values": ["lecture", "keynote", "panel", "interview", "tutorial"],
            },
        },
        "Publication": {
            "title": {"type": "str", "source": "PublicationScraper"},
            "outlet": {"type": "str", "source": "PublicationScraper", "example": "Harvard Business Review"},
            "url": {"type": "str", "source": "PublicationScraper"},
            "year": {"type": "int", "source": "PublicationScraper"},
            "summary": {
                "type": "str",
                "source": "PublicationScraper",
                "note": "2-3 sentence summary — NOT full article text (copyright)",
            },
            "key_arguments": {"type": "list[str]", "source": "LLM — ProfileEnricher"},
            "publication_type": {
                "type": "str",
                "source": "PublicationScraper",
                "allowed_values": ["article", "opinion", "book_chapter", "report"],
            },
        },
        "Book": {
            "title": {"type": "str", "source": "PublicationScraper"},
            "year": {"type": "int", "source": "PublicationScraper"},
            "publisher": {"type": "str", "source": "PublicationScraper"},
            "summary": {
                "type": "str",
                "source": "PublicationScraper",
                "note": "Public blurb / Amazon description only — NOT book content",
            },
            "key_themes": {"type": "list[str]", "source": "LLM — ProfileEnricher"},
            "table_of_contents": {
                "type": "list[str]",
                "source": "PublicationScraper",
                "note": "Chapter titles only",
            },
        },
    },
    "completeness_scoring": {
        "description": (
            "profile_completeness is the fraction of the following checks that pass. "
            "Each check is worth 1/7."
        ),
        "checks": [
            "len(papers) >= 5",
            "len(courses) >= 1",
            "len(videos) >= 3",
            "len(publications) >= 2",
            "len(mapped_skills) >= 1",
            "bool(primary_affiliation)",
            "bool(primary_skill)",
        ],
    },
}


def get_config_schema() -> dict:
    """
    Returns the canonical CONFIG_SCHEMA dict.

    The config schema defines the exact structure that must be passed to
    ProfileBuilder.build_profile(). It specifies:
      - which fields are required vs optional
      - the type of each field
      - which scraper consumes each field
      - example values

    Returns:
        dict: CONFIG_SCHEMA — a nested dict describing all valid config fields.

    Usage:
        schema = get_config_schema()
        print(json.dumps(schema, indent=2))
    """
    return CONFIG_SCHEMA


def get_output_schema() -> dict:
    """
    Returns the canonical OUTPUT_SCHEMA dict.

    The output schema mirrors the ProfessorProfile dataclass and all its
    nested types (Paper, Course, Video, Publication, Book). For each field it
    documents:
      - the Python type
      - which scraper or enricher populates it
      - any allowed values (enums)
      - copyright / data-collection notes

    Returns:
        dict: OUTPUT_SCHEMA — a nested dict describing the full output structure.

    Usage:
        schema = get_output_schema()
        print(json.dumps(schema, indent=2))
    """
    return OUTPUT_SCHEMA


# ═══════════════════════════════════════════════════════════════════════
# 3. CONFIG VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Valid: {self.is_valid}"]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  ✗ {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_config(config: dict) -> ValidationResult:
    """
    Validates a professor config dict against CONFIG_SCHEMA.

    Checks:
      - All required top-level fields are present and non-empty
      - 'sources' dict is present and contains at least one scraping target
      - 'primary_skill' is included in 'skills'
      - Types match (str vs list[str])
      - Warns about optional fields that improve profile completeness

    Args:
        config: The dict to validate, shaped like CONFIG_SCHEMA['example'].

    Returns:
        ValidationResult with is_valid, errors, and warnings.

    Usage:
        result = validate_config(my_config)
        if not result.is_valid:
            raise ValueError(str(result))
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Required top-level fields ──
    required_str_fields = ["name", "affiliation"]
    required_list_fields = ["skills"]
    required_str_singular = ["primary_skill"]

    for f in required_str_fields:
        if not config.get(f):
            errors.append(f"'{f}' is required and must be a non-empty string.")
        elif not isinstance(config[f], str):
            errors.append(f"'{f}' must be a string, got {type(config[f]).__name__}.")

    for f in required_list_fields:
        if not config.get(f):
            errors.append(f"'{f}' is required and must be a non-empty list.")
        elif not isinstance(config[f], list):
            errors.append(f"'{f}' must be a list[str], got {type(config[f]).__name__}.")

    for f in required_str_singular:
        if not config.get(f):
            errors.append(f"'{f}' is required and must be a non-empty string.")

    # ── primary_skill must be in skills ──
    if config.get("primary_skill") and config.get("skills"):
        if config["primary_skill"] not in config["skills"]:
            errors.append(
                f"'primary_skill' (\"{config['primary_skill']}\") must be one of the values in 'skills'."
            )

    # ── sources block ──
    sources = config.get("sources")
    if not sources or not isinstance(sources, dict):
        errors.append("'sources' is required and must be a dict.")
    else:
        known_source_keys = {
            "google_scholar_id", "faculty_url",
            "youtube_search_terms", "publication_outlets",
        }
        scraping_targets = [
            k for k in known_source_keys
            if sources.get(k)
        ]
        if not scraping_targets:
            errors.append(
                "'sources' must contain at least one scraping target: "
                + ", ".join(f"'{k}'" for k in sorted(known_source_keys))
            )

        # Type checks inside sources
        if "google_scholar_id" in sources and not isinstance(sources["google_scholar_id"], str):
            errors.append("'sources.google_scholar_id' must be a string.")
        if "faculty_url" in sources and not isinstance(sources["faculty_url"], str):
            errors.append("'sources.faculty_url' must be a string.")
        if "youtube_search_terms" in sources and not isinstance(sources["youtube_search_terms"], list):
            errors.append("'sources.youtube_search_terms' must be a list[str].")
        if "publication_outlets" in sources and not isinstance(sources["publication_outlets"], list):
            errors.append("'sources.publication_outlets' must be a list[str].")

        # Warn about missing optional sources (each improves completeness)
        if not sources.get("google_scholar_id"):
            warnings.append(
                "'sources.google_scholar_id' not set — papers, h_index, "
                "and total_citations will be empty."
            )
        if not sources.get("faculty_url"):
            warnings.append(
                "'sources.faculty_url' not set — courses and syllabi will be empty."
            )
        if not sources.get("youtube_search_terms"):
            warnings.append(
                "'sources.youtube_search_terms' not set — video corpus will be empty."
            )

    # ── Optional field warnings ──
    if not config.get("title"):
        warnings.append("'title' not set — title_role will be empty in the output profile.")
    if not config.get("domains"):
        warnings.append("'domains' not set — expertise_domains will be empty in the output profile.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. SCRAPERS — One per data source
# ═══════════════════════════════════════════════════════════════════════

class BaseScraper:
    """Base class for all scrapers. Handles rate limiting and caching."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached(self, url: str) -> Optional[str]:
        path = os.path.join(self.cache_dir, self._cache_key(url) + ".json")
        if os.path.exists(path):
            age_hours = (time.time() - os.path.getmtime(path)) / 3600
            if age_hours < 168:  # 7 day cache
                with open(path, "r") as f:
                    return json.load(f)
        return None

    def _set_cached(self, url: str, data):
        path = os.path.join(self.cache_dir, self._cache_key(url) + ".json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _rate_limit(self, seconds: float = 2.0):
        """Respectful rate limiting between requests."""
        time.sleep(seconds)


class GoogleScholarScraper(BaseScraper):
    """
    Scrapes a professor's Google Scholar profile.

    What we collect:
    - Paper titles, years, citation counts, co-authors
    - Abstracts (from Scholar page, not full papers)
    - h-index and total citations
    - Research themes over time

    What we DO NOT collect:
    - Full paper text (copyright)
    - PDF downloads

    In production, use the Scholarly library or SerpAPI for reliable access.
    """

    def scrape(self, scholar_id: str) -> dict:
        """
        Args:
            scholar_id: Google Scholar author ID (from URL: scholar.google.com/citations?user=XXXX)

        Returns:
            Dict with papers, h_index, total_citations
        """
        # ──────────────────────────────────────────────────────
        # PRODUCTION IMPLEMENTATION:
        #
        # Option A: Use `scholarly` library (free, rate-limited)
        #   from scholarly import scholarly
        #   author = scholarly.search_author_id(scholar_id)
        #   author = scholarly.fill(author, sections=['basics', 'publications'])
        #   for pub in author['publications']:
        #       pub = scholarly.fill(pub)  # Gets abstract, citations
        #
        # Option B: Use SerpAPI (paid, reliable, no rate limits)
        #   import serpapi
        #   results = serpapi.search({
        #       "engine": "google_scholar_author",
        #       "author_id": scholar_id,
        #       "api_key": os.environ["SERPAPI_KEY"]
        #   })
        #
        # Option C: Use Semantic Scholar API (free, good for CS/tech)
        #   GET https://api.semanticscholar.org/graph/v1/author/{authorId}
        #       ?fields=papers.title,papers.year,papers.abstract,papers.citationCount
        # ──────────────────────────────────────────────────────

        cached = self._get_cached(f"scholar:{scholar_id}")
        if cached:
            return cached

        # Placeholder — replace with actual API call
        result = {
            "scholar_id": scholar_id,
            "papers": [],
            "h_index": 0,
            "total_citations": 0,
            "status": "NEEDS_IMPLEMENTATION"
        }

        self._set_cached(f"scholar:{scholar_id}", result)
        return result

    def to_papers(self, raw_data: dict) -> list[Paper]:
        """Convert raw Scholar data to Paper objects."""
        papers = []
        for item in raw_data.get("papers", []):
            papers.append(Paper(
                title=item.get("title", ""),
                year=item.get("year", 0),
                abstract=item.get("abstract", ""),
                citations=item.get("citations", 0),
                journal_or_venue=item.get("venue", ""),
                coauthors=item.get("coauthors", []),
                url=item.get("url", ""),
            ))
        return papers


class UniversityPageScraper(BaseScraper):
    """
    Scrapes a professor's university faculty page and course listings.

    What we collect:
    - Bio, research interests, teaching areas
    - Course titles, descriptions, syllabi (if publicly posted)
    - Reading lists and case study references
    - Office hours / teaching approach statements

    Strategy:
    - Faculty bio page: Usually at {university}.edu/faculty/{name}
    - Course catalog: Search for their name in course listings
    - Syllabus PDFs: Often linked from course pages (parse with LLM)
    """

    def scrape(self, faculty_url: str) -> dict:
        """
        Args:
            faculty_url: Direct URL to professor's faculty page

        Returns:
            Dict with bio, courses, research_interests
        """
        # ──────────────────────────────────────────────────────
        # PRODUCTION IMPLEMENTATION:
        #
        # 1. Fetch the faculty page HTML
        #    response = requests.get(faculty_url, headers={"User-Agent": "..."})
        #
        # 2. Parse with BeautifulSoup or use an LLM to extract:
        #    - Bio text
        #    - Research interests
        #    - Course listings with links
        #    - CV link (often a PDF)
        #
        # 3. For each course link, fetch the course page:
        #    - Syllabus PDF link → download and parse with LLM
        #    - Reading list
        #    - Case studies mentioned
        #    - Learning objectives
        #
        # 4. If a CV PDF is available, parse it for:
        #    - Complete publication list
        #    - Teaching history
        #    - Consulting/advisory roles (indicates industry expertise)
        #
        # Key considerations:
        # - Respect robots.txt
        # - Cache aggressively (faculty pages change rarely)
        # - PDF syllabi are the highest-value target
        # ──────────────────────────────────────────────────────

        cached = self._get_cached(faculty_url)
        if cached:
            return cached

        result = {
            "faculty_url": faculty_url,
            "bio": "",
            "research_interests": [],
            "courses": [],
            "cv_url": "",
            "status": "NEEDS_IMPLEMENTATION"
        }

        self._set_cached(faculty_url, result)
        return result

    def to_courses(self, raw_data: dict) -> list[Course]:
        """Convert raw university data to Course objects."""
        courses = []
        for item in raw_data.get("courses", []):
            courses.append(Course(
                title=item.get("title", ""),
                institution=item.get("institution", ""),
                level=item.get("level", ""),
                syllabus_topics=item.get("topics", []),
                reading_list=item.get("readings", []),
                case_studies=item.get("cases", []),
                learning_objectives=item.get("objectives", []),
                url=item.get("url", ""),
            ))
        return courses


class YouTubeScraper(BaseScraper):
    """
    Finds and processes a professor's video content.

    What we collect:
    - Video titles, descriptions, durations
    - Transcript excerpts (first ~2000 chars, for fair use)
    - Key topics per video (LLM-extracted from transcript)

    What we DO NOT collect:
    - Full transcripts (copyright concerns for long lectures)
    - Video downloads

    Strategy:
    - Search YouTube for "{professor_name} {topic}"
    - Filter by channel (university channels, conference channels)
    - Use YouTube Data API for metadata
    - Use youtube-transcript-api for captions
    """

    def scrape(self, professor_name: str, topics: list[str], max_videos: int = 20) -> dict:
        """
        Args:
            professor_name: Full name to search for
            topics: List of topic keywords to search alongside name
            max_videos: Maximum videos to collect

        Returns:
            Dict with video metadata and transcript excerpts
        """
        # ──────────────────────────────────────────────────────
        # PRODUCTION IMPLEMENTATION:
        #
        # 1. YouTube Data API v3 (requires API key):
        #    GET https://www.googleapis.com/youtube/v3/search
        #        ?q={professor_name}+{topic}
        #        &type=video
        #        &maxResults=10
        #        &key={YOUTUBE_API_KEY}
        #
        # 2. For each video, get details:
        #    GET https://www.googleapis.com/youtube/v3/videos
        #        ?id={video_id}
        #        &part=snippet,contentDetails
        #
        # 3. Get transcript (use youtube-transcript-api):
        #    from youtube_transcript_api import YouTubeTranscriptApi
        #    transcript = YouTubeTranscriptApi.get_transcript(video_id)
        #    text = " ".join([t["text"] for t in transcript])
        #    excerpt = text[:2000]  # Fair use: first 2000 chars only
        #
        # 4. Filter for relevance:
        #    - Is the professor actually speaking (not just mentioned)?
        #    - Is it a lecture/talk (not a news clip)?
        #    - Is it from a reputable channel?
        #
        # 5. Classify context:
        #    - "lecture" if from university channel
        #    - "keynote" if from conference channel
        #    - "interview" if from media/podcast channel
        #    - "tutorial" if from educational platform
        # ──────────────────────────────────────────────────────

        cached = self._get_cached(f"youtube:{professor_name}")
        if cached:
            return cached

        result = {
            "professor_name": professor_name,
            "search_topics": topics,
            "videos": [],
            "status": "NEEDS_IMPLEMENTATION"
        }

        self._set_cached(f"youtube:{professor_name}", result)
        return result

    def to_videos(self, raw_data: dict) -> list[Video]:
        """Convert raw YouTube data to Video objects."""
        videos = []
        for item in raw_data.get("videos", []):
            videos.append(Video(
                title=item.get("title", ""),
                platform="youtube",
                url=item.get("url", ""),
                duration_minutes=item.get("duration_minutes", 0),
                transcript_excerpt=item.get("transcript_excerpt", "")[:2000],
                year=item.get("year", 0),
                context=item.get("context", "lecture"),
            ))
        return videos


class PublicationScraper(BaseScraper):
    """
    Scrapes non-academic publications: HBR articles, MIT Sloan Review,
    LinkedIn articles, blog posts, newspaper columns.

    What we collect:
    - Title, outlet, year, URL
    - Summary (2-3 sentences — NOT full text)
    - Key arguments (LLM-extracted from summary/abstract)

    What we DO NOT collect:
    - Full article text (copyright — especially for HBR, WSJ, etc.)

    Strategy:
    - Search Google for "{professor_name} site:hbr.org"
    - Search for "{professor_name} site:sloanreview.mit.edu"
    - Check their LinkedIn profile for articles
    - Check personal website/blog
    """

    def scrape(self, professor_name: str, outlets: list[str] = None) -> dict:
        """
        Args:
            professor_name: Full name to search for
            outlets: List of publication outlets to search

        Returns:
            Dict with publication metadata
        """
        if outlets is None:
            outlets = [
                "hbr.org",
                "sloanreview.mit.edu",
                "linkedin.com",
                "wsj.com",
                "ft.com",
                "forbes.com",
            ]

        # ──────────────────────────────────────────────────────
        # PRODUCTION IMPLEMENTATION:
        #
        # 1. For each outlet, use SerpAPI or Google Custom Search:
        #    GET https://www.googleapis.com/customsearch/v1
        #        ?q={professor_name}
        #        &siteSearch={outlet}
        #        &key={GOOGLE_API_KEY}
        #
        # 2. For each result, fetch the page and extract:
        #    - Title (from <title> or og:title)
        #    - Publication date
        #    - Meta description or og:description (this is the summary)
        #    - DO NOT extract body text
        #
        # 3. For LinkedIn articles:
        #    - Use LinkedIn API (if available) or manual collection
        #    - LinkedIn articles are public if the professor shared them
        #
        # 4. For books:
        #    - Search Google Books API for author name
        #    - Collect: title, publisher, year, description, TOC
        #    - DO NOT collect book content
        # ──────────────────────────────────────────────────────

        cached = self._get_cached(f"publications:{professor_name}")
        if cached:
            return cached

        result = {
            "professor_name": professor_name,
            "outlets_searched": outlets,
            "publications": [],
            "books": [],
            "status": "NEEDS_IMPLEMENTATION"
        }

        self._set_cached(f"publications:{professor_name}", result)
        return result


# ═══════════════════════════════════════════════════════════════════════
# 5. PROFILE BUILDER — Merges all scraped data into unified profile
# ═══════════════════════════════════════════════════════════════════════

class ProfileBuilder:
    """
    Assembles scraped data from multiple sources into a single ProfessorProfile.
    Also runs LLM post-processing to extract themes, patterns, and style.

    Input:  A config dict validated against CONFIG_SCHEMA  (see get_config_schema())
    Output: A ProfessorProfile instance serialized to JSON  (see get_output_schema())
    """

    def __init__(self, output_dir: str = "./data/professors"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.scholar_scraper = GoogleScholarScraper()
        self.university_scraper = UniversityPageScraper()
        self.youtube_scraper = YouTubeScraper()
        self.publication_scraper = PublicationScraper()

    # ── Schema accessors (convenience wrappers) ──

    @staticmethod
    def get_config_schema() -> dict:
        """
        Returns the canonical input config schema for build_profile().
        Identical to the module-level get_config_schema().
        """
        return get_config_schema()

    @staticmethod
    def get_output_schema() -> dict:
        """
        Returns the canonical output schema produced by build_profile().
        Identical to the module-level get_output_schema().
        """
        return get_output_schema()

    @staticmethod
    def validate_config(config: dict) -> ValidationResult:
        """
        Validates a config dict before passing it to build_profile().
        Identical to the module-level validate_config().

        Raises nothing — inspect ValidationResult.is_valid and .errors.
        """
        return validate_config(config)

    def save_templates(self, schema_dir: str = None) -> tuple[str, str]:
        """
        Writes blank-but-valid template files for both the config input and
        the profile output to disk, so users know exactly what to fill in.

        Files written:
          - config_template.json  : A single config object with all fields
                                    present, required fields marked, optional
                                    fields included with empty/null values.
          - output_template.json  : A ProfessorProfile with every field
                                    present but empty, showing the exact shape
                                    of what build_profile() produces.

        Args:
            schema_dir: Directory to write files into.
                        Defaults to self.output_dir.

        Returns:
            Tuple of (config_template_path, output_template_path).
        """
        target_dir = schema_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        # ── Config template ──
        config_template = {
            "name": "",                    # required
            "affiliation": "",             # required
            "title": "",                   # optional
            "skills": [],                  # required — list of skill strings
            "primary_skill": "",           # required — must be one of skills[]
            "domains": [],                 # optional
            "sources": {
                "google_scholar_id": "",   # optional — from scholar.google.com/citations?user=<ID>
                "faculty_url": "",         # optional — direct URL to faculty bio page
                "youtube_search_terms": [], # optional — list of topic keywords
                "publication_outlets": [], # optional — defaults to hbr.org, sloanreview.mit.edu, etc.
            },
        }

        # ── Output template — mirrors ProfessorProfile exactly ──
        output_template = asdict(ProfessorProfile(
            professor_id="",
            full_name="",
            primary_affiliation="",
            secondary_affiliations=[],
            title_role="",
            mapped_skills=[],
            primary_skill="",
            expertise_domains=[],
            papers=[Paper(title="", year=0)],
            h_index=0,
            total_citations=0,
            research_themes=[],
            research_evolution="",
            courses=[Course(title="", institution="")],
            teaching_philosophy="",
            preferred_pedagogy=[],
            signature_cases=[],
            videos=[Video(title="", platform="")],
            explanation_style="",
            recurring_analogies=[],
            publications=[Publication(title="", outlet="")],
            books=[Book(title="", year=0)],
            profile_completeness=0.0,
            last_scraped="",
            scrape_sources=[],
            notes="",
        ))

        cfg_path = os.path.join(target_dir, "config_template.json")
        out_path = os.path.join(target_dir, "output_template.json")

        with open(cfg_path, "w") as f:
            json.dump(config_template, f, indent=2)
        print(f"  Config template saved to {cfg_path}")

        with open(out_path, "w") as f:
            json.dump(output_template, f, indent=2)
        print(f"  Output template saved to {out_path}")

        return cfg_path, out_path

    def save_schemas(self, schema_dir: str = None) -> tuple[str, str]:
        """
        Writes config_schema.json and output_schema.json to disk.

        Args:
            schema_dir: Directory to write the files into.
                        Defaults to self.output_dir (same folder as profiles).

        Returns:
            Tuple of (config_schema_path, output_schema_path).

        Usage:
            builder = ProfileBuilder()
            cfg_path, out_path = builder.save_schemas()
            # → ./data/professors/config_schema.json
            # → ./data/professors/output_schema.json
        """
        target_dir = schema_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        cfg_path = os.path.join(target_dir, "config_schema.json")
        out_path = os.path.join(target_dir, "output_schema.json")

        with open(cfg_path, "w") as f:
            json.dump(get_config_schema(), f, indent=2)
        print(f"  Config schema saved to {cfg_path}")

        with open(out_path, "w") as f:
            json.dump(get_output_schema(), f, indent=2)
        print(f"  Output schema saved to {out_path}")

        return cfg_path, out_path

    # ── Core pipeline ──

    def build_profile(self, config: dict) -> ProfessorProfile:
        """
        Build a complete professor profile from a validated config dict.

        The config must match CONFIG_SCHEMA. Call validate_config() first
        if the config origin is untrusted (e.g., user-provided YAML).

        Args:
            config: Dict shaped like CONFIG_SCHEMA['example']:
            {
                "name": "Sunil Gupta",                          # required str
                "affiliation": "Harvard Business School",       # required str
                "title": "Edward W. Carter Professor...",       # optional str
                "skills": ["Digital Strategy", ...],            # required list[str]
                "primary_skill": "Digital Strategy",           # required str (must be in skills)
                "domains": ["Technology & Strategy", ...],      # optional list[str]
                "sources": {                                    # required dict
                    "google_scholar_id": "abc123",              # optional str  → GoogleScholarScraper
                    "faculty_url": "https://...",               # optional str  → UniversityPageScraper
                    "youtube_search_terms": ["..."],            # optional list → YouTubeScraper
                    "publication_outlets": ["hbr.org", ...],   # optional list → PublicationScraper
                }
            }

        Returns:
            ProfessorProfile — a fully populated profile instance.
            Serialize with: json.dumps(asdict(profile), indent=2)
            Schema: get_output_schema()
        """
        # ── Validate before scraping ──
        result = validate_config(config)
        if not result.is_valid:
            raise ValueError(
                f"Invalid config for '{config.get('name', 'unknown')}':\n{result}"
            )
        if result.warnings:
            for w in result.warnings:
                print(f"  ⚠ {w}")

        prof_id = config["name"].lower().replace(" ", "-").replace(".", "")

        profile = ProfessorProfile(
            professor_id=prof_id,
            full_name=config["name"],
            primary_affiliation=config["affiliation"],
            title_role=config.get("title", ""),
            mapped_skills=config.get("skills", []),
            primary_skill=config.get("primary_skill", ""),
            expertise_domains=config.get("domains", []),
            last_scraped=datetime.now().isoformat(),
        )

        sources = config.get("sources", {})

        # ── Scrape Google Scholar ──
        if sources.get("google_scholar_id"):
            print(f"  Scraping Google Scholar for {config['name']}...")
            scholar_data = self.scholar_scraper.scrape(sources["google_scholar_id"])
            profile.papers = self.scholar_scraper.to_papers(scholar_data)
            profile.h_index = scholar_data.get("h_index", 0)
            profile.total_citations = scholar_data.get("total_citations", 0)
            profile.scrape_sources.append(f"scholar:{sources['google_scholar_id']}")

        # ── Scrape University Page ──
        if sources.get("faculty_url"):
            print(f"  Scraping faculty page for {config['name']}...")
            uni_data = self.university_scraper.scrape(sources["faculty_url"])
            profile.courses = self.university_scraper.to_courses(uni_data)
            profile.scrape_sources.append(sources["faculty_url"])

        # ── Scrape YouTube ──
        if sources.get("youtube_search_terms"):
            print(f"  Scraping YouTube for {config['name']}...")
            yt_data = self.youtube_scraper.scrape(
                config["name"],
                sources["youtube_search_terms"]
            )
            profile.videos = self.youtube_scraper.to_videos(yt_data)
            profile.scrape_sources.append("youtube")

        # ── Scrape Publications ──
        print(f"  Scraping publications for {config['name']}...")
        pub_data = self.publication_scraper.scrape(
            config["name"],
            sources.get("publication_outlets")
        )
        for item in pub_data.get("publications", []):
            profile.publications.append(Publication(
                title=item.get("title", ""),
                outlet=item.get("outlet", ""),
                url=item.get("url", ""),
                year=item.get("year", 0),
                summary=item.get("summary", ""),
                publication_type=item.get("type", "article"),
            ))
        for item in pub_data.get("books", []):
            profile.books.append(Book(
                title=item.get("title", ""),
                year=item.get("year", 0),
                publisher=item.get("publisher", ""),
                summary=item.get("summary", ""),
                table_of_contents=item.get("toc", []),
            ))

        # ── Calculate completeness ──
        profile.profile_completeness = self._calculate_completeness(profile)

        return profile

    def _calculate_completeness(self, profile: ProfessorProfile) -> float:
        """Score profile completeness from 0.0 to 1.0."""
        checks = [
            len(profile.papers) >= 5,
            len(profile.courses) >= 1,
            len(profile.videos) >= 3,
            len(profile.publications) >= 2,
            len(profile.mapped_skills) >= 1,
            bool(profile.primary_affiliation),
            bool(profile.primary_skill),
        ]
        return sum(checks) / len(checks)

    def save_profile(self, profile: ProfessorProfile) -> str:
        """Save profile to JSON file. Returns the file path."""
        path = os.path.join(self.output_dir, f"{profile.professor_id}.json")
        with open(path, "w") as f:
            json.dump(asdict(profile), f, indent=2, default=str)
        print(f"  Saved profile to {path}")
        return path

    def load_profile(self, professor_id: str) -> ProfessorProfile:
        """Load a previously saved profile from disk."""
        path = os.path.join(self.output_dir, f"{professor_id}.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["papers"] = [Paper(**p) for p in data.get("papers", [])]
        data["courses"] = [Course(**c) for c in data.get("courses", [])]
        data["videos"] = [Video(**v) for v in data.get("videos", [])]
        data["publications"] = [Publication(**p) for p in data.get("publications", [])]
        data["books"] = [Book(**b) for b in data.get("books", [])]
        return ProfessorProfile(**data)


# ═══════════════════════════════════════════════════════════════════════
# 6. LLM POST-PROCESSOR — Enriches profile with extracted patterns
# ═══════════════════════════════════════════════════════════════════════

class ProfileEnricher:
    """
    Uses an LLM to analyze the raw scraped data and extract higher-order
    patterns that aren't visible in individual data points.

    This is the bridge between scraping and tacit knowledge externalization.
    It runs AFTER scraping, BEFORE the main externalization pipeline.

    Fields populated by this class are marked "LLM — ProfileEnricher" in OUTPUT_SCHEMA.
    """

    def get_enrichment_prompts(self, profile: ProfessorProfile) -> list[dict]:
        """
        Returns a list of LLM prompts that enrich the profile.
        Each prompt targets one field in OUTPUT_SCHEMA marked as LLM-sourced.
        """
        prompts = []

        # ── 1. Research evolution ──
        if profile.papers:
            paper_timeline = "\n".join([
                f"- ({p.year}) {p.title} [{p.citations} citations]"
                for p in sorted(profile.papers, key=lambda x: x.year)
            ])
            prompts.append({
                "field": "research_evolution",
                "system": "You analyze academic research trajectories. Be specific about shifts in focus, methodology, and framing.",
                "user": f"""Analyze how this researcher's focus has evolved over time:

RESEARCHER: {profile.full_name} ({profile.primary_affiliation})
PUBLICATION TIMELINE:
{paper_timeline}

Describe:
1. What was their early research focus?
2. How did their interests shift over time?
3. What is their current frontier?
4. What does this trajectory reveal about their intellectual priorities?

Be specific — reference actual paper titles to support your analysis."""
            })

        # ── 2. Teaching philosophy ──
        if profile.courses:
            course_info = "\n\n".join([
                f"COURSE: {c.title} ({c.institution}, {c.level})\n"
                f"Topics: {', '.join(c.syllabus_topics)}\n"
                f"Cases: {', '.join(c.case_studies)}\n"
                f"Objectives: {', '.join(c.learning_objectives)}"
                for c in profile.courses
            ])
            prompts.append({
                "field": "teaching_philosophy",
                "system": "You are an expert in pedagogy who can infer teaching philosophy from course design choices.",
                "user": f"""Analyze the teaching approach of {profile.full_name} based on their courses:

{course_info}

Extract their implicit teaching philosophy:
1. Do they favor theory-first or case-first approaches?
2. How do they sequence from foundational to advanced?
3. What pedagogical methods dominate (case discussion, simulation, lecture, workshop)?
4. What patterns in case selection reveal about their worldview?
5. How do they scaffold complexity for different audiences?

Focus on the TACIT choices — the patterns that reveal how this person thinks about teaching, not just what they teach."""
            })

        # ── 3. Explanation style (from videos) ──
        if profile.videos:
            video_excerpts = "\n\n".join([
                f"VIDEO: {v.title} ({v.context}, {v.year})\n"
                f"Excerpt: {v.transcript_excerpt[:500]}..."
                for v in profile.videos[:5]
            ])
            prompts.append({
                "field": "explanation_style",
                "system": "You analyze communication patterns in educational content.",
                "user": f"""Analyze how {profile.full_name} explains complex concepts based on these video excerpts:

{video_excerpts}

Extract their explanation style:
1. Do they start with examples or frameworks?
2. What recurring analogies or metaphors do they use?
3. How do they handle audience questions?
4. What is their pace — do they build slowly or jump to punchlines?
5. How do they make abstract concepts tangible?

Focus on PATTERNS across multiple videos, not individual observations."""
            })

        # ── 4. Key concepts and frameworks ──
        all_content = []
        for p in profile.papers[:10]:
            all_content.append(f"Paper: {p.title} — {p.abstract[:200]}")
        for c in profile.courses:
            all_content.append(f"Course: {c.title} — Topics: {', '.join(c.syllabus_topics)}")
        for pub in profile.publications[:10]:
            all_content.append(f"Article: {pub.title} — {pub.summary}")
        for b in profile.books:
            all_content.append(f"Book: {b.title} — {b.summary}")

        if all_content:
            prompts.append({
                "field": "research_themes",
                "system": "You identify recurring intellectual themes across a researcher's body of work. Output as a JSON array of strings.",
                "user": f"""Identify the 5-10 core intellectual themes across {profile.full_name}'s body of work:

{chr(10).join(all_content)}

Return ONLY a JSON array of theme strings, like:
["Platform economics and two-sided markets", "Digital disruption of incumbents", ...]

Each theme should be specific enough to distinguish this researcher from others in the same field."""
            })

        # ── 5. Signature cases ──
        all_cases = []
        for c in profile.courses:
            all_cases.extend(c.case_studies)
        for v in profile.videos:
            all_cases.extend(v.key_topics)

        if all_cases:
            prompts.append({
                "field": "signature_cases",
                "system": "You identify patterns in an educator's case study selections. Output as a JSON array of strings.",
                "user": f"""These are cases, examples, and companies mentioned across {profile.full_name}'s teaching and talks:

{', '.join(all_cases)}

Identify their SIGNATURE cases — the ones they return to repeatedly, that are central to their teaching.
Also identify the TYPE of cases they prefer (turnaround stories? disruptor cases? incumbent transformation?).

Return a JSON array of the top 5-8 signature cases/examples."""
            })

        return prompts


# ═══════════════════════════════════════════════════════════════════════
# 7. CONTENT DATABASE INDEXER
# ═══════════════════════════════════════════════════════════════════════

class ContentIndexer:
    """
    Indexes the professor's own content and curated external content
    into a structured content database that the professor agent can
    select from during content leveling.

    Three tiers of content:
      Tier 1: Professor's own content (highest authority)
      Tier 2: Content the professor explicitly recommends (reading lists, citations)
      Tier 3: External content on the same topics (curated by the agent)
    """

    @dataclass
    class ContentItem:
        content_id: str
        title: str
        source: str             # "professor_own", "professor_recommended", "external"
        content_type: str       # "paper", "video", "article", "case_study", "book_chapter", "exercise"
        url: str = ""
        professor_id: str = ""
        skill: str = ""
        bloom_levels: list[str] = field(default_factory=list)
        topics: list[str] = field(default_factory=list)
        duration_minutes: int = 0
        difficulty: str = ""    # "introductory", "intermediate", "advanced"
        summary: str = ""
        tier: int = 1           # 1=professor's own, 2=recommended, 3=external

    def index_from_profile(self, profile: ProfessorProfile) -> list:
        """Extract indexable ContentItems from a ProfessorProfile."""
        items = []

        # Tier 1: Papers
        for paper in profile.papers:
            items.append(self.ContentItem(
                content_id=f"{profile.professor_id}-paper-{hashlib.md5(paper.title.encode()).hexdigest()[:8]}",
                title=paper.title,
                source="professor_own",
                content_type="paper",
                url=paper.url,
                professor_id=profile.professor_id,
                skill=profile.primary_skill,
                bloom_levels=["analyze", "evaluate"],
                topics=paper.key_concepts,
                difficulty="advanced",
                summary=paper.abstract[:300],
                tier=1,
            ))

        # Tier 1: Videos
        for video in profile.videos:
            levels = {
                "lecture": ["understand", "apply"],
                "tutorial": ["remember", "understand", "apply"],
                "keynote": ["analyze", "evaluate"],
                "interview": ["understand", "analyze"],
                "panel": ["analyze", "evaluate"],
            }
            items.append(self.ContentItem(
                content_id=f"{profile.professor_id}-video-{hashlib.md5(video.title.encode()).hexdigest()[:8]}",
                title=video.title,
                source="professor_own",
                content_type="video",
                url=video.url,
                professor_id=profile.professor_id,
                skill=profile.primary_skill,
                bloom_levels=levels.get(video.context, ["understand"]),
                topics=video.key_topics,
                duration_minutes=video.duration_minutes,
                difficulty="intermediate",
                summary=video.transcript_excerpt[:200],
                tier=1,
            ))

        # Tier 1: Articles
        for pub in profile.publications:
            items.append(self.ContentItem(
                content_id=f"{profile.professor_id}-pub-{hashlib.md5(pub.title.encode()).hexdigest()[:8]}",
                title=pub.title,
                source="professor_own",
                content_type="article",
                url=pub.url,
                professor_id=profile.professor_id,
                skill=profile.primary_skill,
                bloom_levels=["understand", "analyze"],
                topics=pub.key_arguments,
                difficulty="intermediate",
                summary=pub.summary[:200],
                tier=1,
            ))

        # Tier 2: Reading list items from courses
        for course in profile.courses:
            for reading in course.reading_list:
                items.append(self.ContentItem(
                    content_id=f"{profile.professor_id}-reading-{hashlib.md5(reading.encode()).hexdigest()[:8]}",
                    title=reading,
                    source="professor_recommended",
                    content_type="article",
                    professor_id=profile.professor_id,
                    skill=profile.primary_skill,
                    bloom_levels=["understand", "apply"],
                    difficulty="intermediate",
                    tier=2,
                ))
            for case in course.case_studies:
                items.append(self.ContentItem(
                    content_id=f"{profile.professor_id}-case-{hashlib.md5(case.encode()).hexdigest()[:8]}",
                    title=case,
                    source="professor_recommended",
                    content_type="case_study",
                    professor_id=profile.professor_id,
                    skill=profile.primary_skill,
                    bloom_levels=["apply", "analyze", "evaluate"],
                    difficulty="intermediate",
                    tier=2,
                ))

        return items

    def save_index(self, items: list, output_path: str = "./data/content_index.json"):
        """Save content index to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([asdict(item) for item in items], f, indent=2, default=str)
        print(f"  Indexed {len(items)} content items to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# 8. MANUAL PROFILE TEMPLATE — For the first test
# ═══════════════════════════════════════════════════════════════════════

def create_manual_test_profile() -> ProfessorProfile:
    """
    Creates a MANUALLY populated test profile.

    For your first test, skip automated scraping entirely.
    Spend 30-60 minutes manually collecting this data for one professor.
    This proves the pipeline works before investing in scraping infrastructure.
    """
    profile = ProfessorProfile(
        professor_id="test-professor-dt",
        full_name="[Professor Name]",
        primary_affiliation="[University]",
        title_role="[Title]",
        mapped_skills=["Digital Strategy", "Digital Transformation", "Platform Business Models"],
        primary_skill="Digital Strategy",
        expertise_domains=["Technology & Strategy", "Digital Business"],

        papers=[
            Paper(
                title="[Paper title from Google Scholar]",
                year=2023,
                abstract="[Copy the abstract — this IS public and fair use]",
                citations=150,
                journal_or_venue="[Journal name]",
                coauthors=["Co-author 1", "Co-author 2"],
                url="[Google Scholar link]",
            ),
        ],

        courses=[
            Course(
                title="[Course title]",
                institution="[University]",
                level="executive_ed",
                syllabus_topics=["[Topic 1]", "[Topic 2]", "[Topic 3]"],
                reading_list=["[Reading 1]", "[Reading 2]"],
                case_studies=["[Case 1]", "[Case 2]"],
                learning_objectives=["[Objective 1]", "[Objective 2]"],
                url="[Course page URL]",
            ),
        ],

        videos=[
            Video(
                title="[Video title]",
                platform="youtube",
                url="[YouTube URL]",
                duration_minutes=45,
                transcript_excerpt="[Open video, copy first ~2000 chars of auto-captions]",
                context="lecture",
                year=2023,
            ),
        ],

        publications=[
            Publication(
                title="[Article title]",
                outlet="Harvard Business Review",
                url="[URL]",
                year=2023,
                summary="[Copy the article's meta description or first 2 sentences]",
                publication_type="article",
            ),
        ],

        books=[
            Book(
                title="[Book title]",
                year=2022,
                publisher="[Publisher]",
                summary="[Amazon description / back cover text]",
                table_of_contents=["[Chapter 1 title]", "[Chapter 2 title]"],
            ),
        ],

        last_scraped=datetime.now().isoformat(),
    )

    profile.profile_completeness = 0.85
    return profile


# ═══════════════════════════════════════════════════════════════════════
# 9. CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Professor Agent — Expert Profile Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python professor_scraper.py --save-templates
  python professor_scraper.py --save-templates --schema-dir ./schemas
  python professor_scraper.py --save-schemas
  python professor_scraper.py --validate-config professors.json
  python professor_scraper.py --config professors.json
  python professor_scraper.py --manual-template
        """
    )

    parser.add_argument("--config", type=str, help="Path to professor config JSON (list of config objects)")
    parser.add_argument("--validate-config", type=str, metavar="FILE", help="Validate a config JSON without running scrapers")
    parser.add_argument("--manual-template", action="store_true", help="Generate a blank manual template JSON file")
    parser.add_argument("--save-schemas", action="store_true", help="Write config_schema.json and output_schema.json to --schema-dir and exit")
    parser.add_argument("--save-templates", action="store_true", help="Write config_template.json and output_template.json to --schema-dir and exit")
    parser.add_argument("--schema-dir", type=str, default=None, help="Directory to write schema/template files into (defaults to --output-dir)")
    parser.add_argument("--output-dir", type=str, default="./data/professors")

    args = parser.parse_args()

    # ── Save schema files ──
    if args.save_schemas:
        builder = ProfileBuilder(output_dir=args.output_dir)
        cfg_path, out_path = builder.save_schemas(schema_dir=args.schema_dir)
        print(f"\nSchemas written:")
        print(f"  Config schema : {cfg_path}")
        print(f"  Output schema : {out_path}")
        return

    # ── Save template files ──
    if args.save_templates:
        builder = ProfileBuilder(output_dir=args.output_dir)
        cfg_path, out_path = builder.save_templates(schema_dir=args.schema_dir)
        print(f"\nTemplates written:")
        print(f"  Config template : {cfg_path}")
        print(f"  Output template : {out_path}")
        return

    # ── Validate only ──
    if args.validate_config:
        with open(args.validate_config, "r") as f:
            configs = json.load(f)
        if isinstance(configs, dict):
            configs = [configs]
        all_ok = True
        for cfg in configs:
            result = validate_config(cfg)
            print(f"\n{cfg.get('name', 'unknown')}:")
            print(result)
            if not result.is_valid:
                all_ok = False
        import sys
        sys.exit(0 if all_ok else 1)

    # ── Manual template ──
    if args.manual_template:
        profile = create_manual_test_profile()
        builder = ProfileBuilder(output_dir=args.output_dir)
        path = builder.save_profile(profile)
        print(f"\nManual template saved to {path}")
        print("Edit this file with real professor data, then run the enrichment pipeline.")
        return

    # ── Full pipeline from config ──
    if args.config:
        with open(args.config, "r") as f:
            configs = json.load(f)
        if isinstance(configs, dict):
            configs = [configs]

        builder = ProfileBuilder(output_dir=args.output_dir)
        indexer = ContentIndexer()
        all_content_items = []

        for config in configs:
            print(f"\nBuilding profile for {config['name']}...")
            profile = builder.build_profile(config)
            builder.save_profile(profile)

            items = indexer.index_from_profile(profile)
            all_content_items.extend(items)
            print(f"  Completeness: {profile.profile_completeness:.0%}")

        indexer.save_index(all_content_items)
        print(f"\nDone. Built {len(configs)} profiles, indexed {len(all_content_items)} content items.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()