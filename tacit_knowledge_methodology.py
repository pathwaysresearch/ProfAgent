"""
Professor Agent — Tacit Knowledge Externalization Pipeline
===========================================================

Implements the three-pass tacit knowledge externalization method and the
content selection pipeline. All profile field accesses use the canonical
field names from OUTPUT_SCHEMA in professor_scraper.py.

OUTPUT_SCHEMA field reference (ProfessorProfile):
  Identity      : professor_id, full_name, primary_affiliation,
                  secondary_affiliations, title_role
  Skills        : mapped_skills, primary_skill, expertise_domains
  Research      : papers[{title,year,abstract,citations,journal_or_venue,
                          coauthors,url,key_concepts}],
                  h_index, total_citations, research_themes, research_evolution
  Teaching      : courses[{title,institution,level,syllabus_topics,reading_list,
                            case_studies,learning_objectives,year,url}],
                  teaching_philosophy, preferred_pedagogy, signature_cases
  Video         : videos[{title,platform,url,duration_minutes,transcript_excerpt,
                           key_topics,year,context}],
                  explanation_style, recurring_analogies
  Written       : publications[{title,outlet,url,year,summary,
                                 key_arguments,publication_type}],
                  books[{title,year,publisher,summary,key_themes,
                          table_of_contents}]
  Meta          : profile_completeness, last_scraped, scrape_sources, notes

Pipeline stages:
  1. Pass 1 — Content selection patterns  (WHAT they teach and why)
  2. Pass 2 — Sequencing & scaffolding    (HOW they order learning)
  3. Pass 3 — Synthesis                   (WHY → actionable principles JSON)
  4. Content selection                    (module design from content index)

Usage:
  python tacit_knowledge_pipeline.py \\
      --profile ./data/professors/sunil-gupta.json \\
      --content-index ./data/content_index.json \\
      --skill "Digital Strategy" \\
      --bloom-level analyze \\
      --learner-profile ./learner.json \\
      --output-dir ./data/output/sunil-gupta-digital-strategy-analyze
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS — profile field access using OUTPUT_SCHEMA names
# ═══════════════════════════════════════════════════════════════════════════

def _papers_by_year(profile: dict, min_year: int = 0, max_year: int = 9999) -> list[dict]:
    """Return papers[] filtered by year range, sorted newest first."""
    return sorted(
        [p for p in profile.get("papers", [])
         if min_year <= p.get("year", 0) <= max_year],
        key=lambda p: p.get("year", 0),
        reverse=True,
    )


def _papers_by_citations(profile: dict, top_n: int = 5) -> list[dict]:
    """Return top N papers by citations count."""
    return sorted(
        profile.get("papers", []),
        key=lambda p: p.get("citations", 0),
        reverse=True,
    )[:top_n]


def _format_paper(p: dict) -> str:
    return (
        f"({p.get('year','?')}) \"{p['title']}\" "
        f"[{p.get('citations',0)} citations, {p.get('journal_or_venue','')}]"
    )


def _format_publication(pub: dict) -> str:
    return (
        f"({pub.get('year','?')}) \"{pub['title']}\" "
        f"in {pub.get('outlet','?')} — {pub.get('summary','')}"
    )


def _format_course_block(course: dict) -> str:
    lines = [
        f"COURSE: {course['title']} "
        f"({course.get('institution','')}, level={course.get('level','')})",
        f"  Syllabus topics    : {json.dumps(course.get('syllabus_topics', []))}",
        f"  Case studies       : {json.dumps(course.get('case_studies', []))}",
        f"  Reading list       : {json.dumps(course.get('reading_list', []))}",
        f"  Learning objectives: {json.dumps(course.get('learning_objectives', []))}",
    ]
    return "\n".join(lines)


def _format_video_excerpt(v: dict) -> str:
    return (
        f"VIDEO: \"{v['title']}\" "
        f"(platform={v.get('platform','')}, context={v.get('context','')}, "
        f"year={v.get('year','?')}, duration={v.get('duration_minutes',0)}min)\n"
        f"  Key topics   : {', '.join(v.get('key_topics', []))}\n"
        f"  Transcript   : {v.get('transcript_excerpt','')[:400]}..."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PASS 1 — Content Selection Patterns
# ═══════════════════════════════════════════════════════════════════════════

def build_pass_1_prompt(profile: dict) -> dict:
    """
    PASS 1: What does this professor choose to teach, and why?

    Uses OUTPUT_SCHEMA fields:
      papers            → research trajectory (title, year, citations)
      courses           → syllabus_topics, case_studies, reading_list,
                          learning_objectives
      publications      → title, outlet, year, summary  (non-academic writing)
      books             → title, year, table_of_contents, summary
      research_themes   → LLM-enriched theme list (may be empty pre-enrichment)
      signature_cases   → LLM-enriched case list  (may be empty pre-enrichment)
      primary_skill     → domain label for the prompt
      full_name         → professor identity
      primary_affiliation
    """
    name        = profile["full_name"]
    affiliation = profile.get("primary_affiliation", "")
    skill       = profile.get("primary_skill", "their domain")

    # ── Research trajectory ──
    recent_papers = _papers_by_year(profile, min_year=2020)
    older_papers  = _papers_by_year(profile, max_year=2019)
    top_papers    = _papers_by_citations(profile, top_n=5)

    research_block = (
        "RESEARCH TRAJECTORY:\n"
        f"  Recent (2020+)     : {[_format_paper(p) for p in recent_papers[:10]]}\n"
        f"  Earlier work       : {[_format_paper(p) for p in older_papers[:10]]}\n"
        f"  Most-cited         : {[_format_paper(p) for p in top_papers]}\n"
        f"  H-index            : {profile.get('h_index', 0)}\n"
        f"  Total citations    : {profile.get('total_citations', 0)}\n"
        f"  LLM research themes: {profile.get('research_themes', [])}"
    )

    # ── Course evidence ──
    course_blocks = "\n\n".join(
        _format_course_block(c) for c in profile.get("courses", [])
    ) or "No courses scraped."

    # ── LLM-enriched teaching metadata ──
    teaching_meta = (
        f"Teaching philosophy (LLM-enriched): "
        f"{profile.get('teaching_philosophy', 'not yet enriched')}\n"
        f"Preferred pedagogy : {profile.get('preferred_pedagogy', [])}\n"
        f"Signature cases    : {profile.get('signature_cases', [])}"
    )

    # ── Non-academic publications ──
    recent_pubs = sorted(
        profile.get("publications", []),
        key=lambda p: p.get("year", 0),
        reverse=True,
    )[:8]
    pub_block = "RECENT PUBLIC COMMENTARY:\n" + (
        "\n".join(f"  - {_format_publication(p)}" for p in recent_pubs)
        or "  None scraped."
    )

    # ── Books ──
    book_blocks = "\n\n".join(
        f"BOOK: \"{b['title']}\" ({b.get('year','?')}, {b.get('publisher','')})\n"
        f"  Table of contents : {b.get('table_of_contents', [])}\n"
        f"  Summary           : {b.get('summary','')}\n"
        f"  Key themes (LLM)  : {b.get('key_themes', [])}"
        for b in profile.get("books", [])
    ) or "No books scraped."

    observed = "\n\n---\n\n".join([
        research_block,
        "COURSE DESIGN:\n" + course_blocks,
        teaching_meta,
        pub_block,
        book_blocks,
    ])

    system = f"""You are analyzing the body of work of {name}, a leading expert in \
{skill} at {affiliation}.

Your task: extract the TACIT KNOWLEDGE behind their CONTENT SELECTION decisions — \
the implicit reasoning for what they choose to include, emphasize, sequence, and \
exclude in their teaching.

Following Cho et al. (2024): you are given observed expert behavior. \
Produce FREE-FORM output. Let the patterns in the data drive your structure.

Be SPECIFIC. Reference actual paper titles, course topics, case selections, \
and publication themes. Generic observations are not useful."""

    user = f"""Analyze the following observed expert behavior and extract the tacit \
knowledge behind {name}'s content selection decisions for {skill}.

OBSERVED EXPERT BEHAVIOR:
{observed}

Extract the implicit reasoning behind:

1. CORE CONVICTIONS — Topics that appear across research, teaching, AND public \
commentary. These are the ideas this expert considers foundational. Why do they \
keep returning to them?

2. EMERGING PRIORITIES — Topics added to their teaching or writing in the last 2-3 \
years. What does this reveal about where they think the field is heading?

3. DELIBERATE OMISSIONS — Mainstream topics in {skill} they seem to de-emphasize \
or skip entirely. What does each absence tell us about their worldview?

4. SIGNATURE EXAMPLES — Which companies, cases, or real-world examples do they \
return to repeatedly? Why THESE and not the dozens of alternatives? What makes \
them pedagogically powerful in this professor's view?

5. CROSS-POLLINATION — What ideas from OUTSIDE {skill} do they bring in? \
(economics, psychology, history, etc.) What does their interdisciplinary reach \
reveal about how they think about the domain?

If a pattern emerges that doesn't fit these categories, include it."""

    return {"system": system, "user": user}


# ═══════════════════════════════════════════════════════════════════════════
# PASS 2 — Sequencing & Scaffolding
# ═══════════════════════════════════════════════════════════════════════════

def build_pass_2_prompt(profile: dict, pass_1_output: str) -> dict:
    """
    PASS 2: How does this professor structure and order learning?

    Uses OUTPUT_SCHEMA fields:
      courses  → title, level, syllabus_topics (ordered list = intended sequence),
                 case_studies (ordered = placement in sequence)
      books    → title, table_of_contents (chapter order = intellectual scaffolding)
      videos   → title, context, year  (lecture series sequence)
      explanation_style   → LLM-enriched (may be empty)
      recurring_analogies → LLM-enriched (may be empty)
    """
    name  = profile["full_name"]
    skill = profile.get("primary_skill", "their domain")

    # ── Course sequences ──
    seq_blocks = []
    for course in profile.get("courses", []):
        topics = course.get("syllabus_topics", [])
        cases  = course.get("case_studies", [])
        if topics:
            seq_blocks.append(
                f"COURSE SEQUENCE — \"{course['title']}\" "
                f"(level={course.get('level','?')}, year={course.get('year','?')}):\n"
                + "\n".join(f"  {i+1:02d}. {t}" for i, t in enumerate(topics))
                + (
                    "\n  Cases (in order of appearance):\n"
                    + "\n".join(f"       - {c}" for c in cases)
                    if cases else ""
                )
            )

    # ── Book structures ──
    for book in profile.get("books", []):
        toc = book.get("table_of_contents", [])
        if toc:
            seq_blocks.append(
                f"BOOK STRUCTURE — \"{book['title']}\" ({book.get('year','?')}):\n"
                + "\n".join(f"  Ch.{i+1:02d}: {ch}" for i, ch in enumerate(toc))
            )

    # ── Lecture series ──
    lectures = sorted(
        [v for v in profile.get("videos", []) if v.get("context") == "lecture"],
        key=lambda v: v.get("year", 0),
    )
    if lectures:
        seq_blocks.append(
            "LECTURE SERIES (chronological order):\n"
            + "\n".join(
                f"  ({v.get('year','?')}) \"{v['title']}\" "
                f"[{v.get('duration_minutes',0)}min, {v.get('platform','')}]"
                for v in lectures
            )
        )

    # ── LLM-enriched style metadata ──
    style_meta = (
        f"Explanation style (LLM-enriched) : "
        f"{profile.get('explanation_style', 'not yet enriched')}\n"
        f"Recurring analogies (LLM-enriched): "
        f"{profile.get('recurring_analogies', [])}"
    )

    sequence_text = (
        "\n\n---\n\n".join(seq_blocks) if seq_blocks
        else "No sequencing data available — courses/books not yet scraped."
    )

    system = f"""You are analyzing the pedagogical sequencing choices of {name}.

You have already identified their content selection patterns (provided as prior analysis).
Now focus on HOW they structure and sequence learning — the ordering of concepts, \
scaffolding from simple to complex, and the deliberate placement of different pedagogical \
tools (cases, frameworks, exercises) at specific points.

Expert educators make hundreds of micro-decisions about sequencing that they rarely \
articulate: why this case comes before that framework, why they introduce a concept \
through failure before presenting the model, why the hardest exercise sits in the \
middle rather than the end.

Produce FREE-FORM output. Reference actual sequences from the data."""

    user = f"""You have already extracted this content selection knowledge about {name}:

PRIOR ANALYSIS (Pass 1 — Content Selection):
{pass_1_output}

Now analyze their SEQUENCING AND SCAFFOLDING patterns from these observed structures:

{sequence_text}

STYLE METADATA:
{style_meta}

Extract the tacit knowledge behind:

1. OPENING MOVES — How do they begin a course or book? Provocative case? \
Foundational framework? Historical narrative? Problem statement? \
What does the opening move reveal about their learning theory?

2. SCAFFOLDING LOGIC — How do they build from simple to complex? \
What prerequisite relationships are implied by the topic order? \
Where do they accelerate and where do they slow down?

3. CASE PLACEMENT — When do cases appear in the sequence? Before or after \
frameworks? Do they use cases to motivate theory (case-first) or to apply \
theory (theory-first)? Does this vary by audience level?

4. DIFFICULTY CURVES — Where is the hardest material? Front-loaded, middle, \
or distributed? What does this reveal about their view of productive struggle?

5. SYNTHESIS POINTS — Where do they pause to connect ideas? \
How do they bridge from one major theme to the next?

6. AUDIENCE ADAPTATION — If they teach the same domain at different levels \
(exec ed vs MBA vs public articles), how does the sequencing change? \
What do they add or remove for different audiences?"""

    return {"system": system, "user": user}


# ═══════════════════════════════════════════════════════════════════════════
# PASS 3 — Synthesis into Actionable Principles
# ═══════════════════════════════════════════════════════════════════════════

def build_pass_3_prompt(profile: dict, pass_1_output: str, pass_2_output: str) -> dict:
    """
    PASS 3: Synthesize passes 1 & 2 into a structured JSON artifact.

    The output JSON is the tacit knowledge document consumed by
    build_content_selection_prompt() in the next stage.

    Uses OUTPUT_SCHEMA fields for identity/context only:
      professor_id, full_name, primary_skill,
      preferred_pedagogy (to cross-check LLM synthesis)
    """
    name     = profile["full_name"]
    prof_id  = profile.get("professor_id", "")
    skill    = profile.get("primary_skill", "")
    pedagogy = profile.get("preferred_pedagogy", [])

    system = f"""You are synthesizing the tacit pedagogical knowledge of {name} \
into actionable principles for an AI professor agent.

You have two prior analyses: content selection patterns (Pass 1) and sequencing \
patterns (Pass 2). Synthesize these into the UNDERLYING LEARNING THEORY that \
drives their choices, then translate that into specific, implementable instructions.

The output feeds directly into the content selection and leveling pipeline — \
it must be specific enough for another system to make concrete decisions about \
what content to select, how to sequence it, and how to assess at each Bloom level.

Return ONLY valid JSON. No preamble, no markdown fences."""

    user = f"""Synthesize the tacit knowledge from these two analyses of {name}:

PASS 1 — CONTENT SELECTION PATTERNS:
{pass_1_output}

PASS 2 — SEQUENCING & SCAFFOLDING PATTERNS:
{pass_2_output}

KNOWN PEDAGOGY PREFERENCES (from profile enrichment):
{pedagogy}

Produce a JSON document with this exact structure:

{{
  "professor_id": "{prof_id}",
  "professor_name": "{name}",
  "skill": "{skill}",
  "synthesized_at": "<ISO timestamp>",

  "core_teaching_principles": [
    "5-8 specific principles stated as actionable rules. Each rule must be concrete
     enough to implement. BAD: 'They value practical application'. GOOD: 'Always
     introduce a concept through a failure case before presenting the framework —
     the failure creates a need-to-know that makes the framework sticky. For
     Digital Strategy, use a disrupted incumbent (Kodak, Blockbuster) before
     introducing disruption frameworks.'"
  ],

  "bloom_level_guidance": {{
    "remember": {{
      "content_approach": "How this professor would design Remember-level content",
      "preferred_formats": ["list of formats e.g. concept_card, glossary, timeline"],
      "example": "A specific concrete example of what this would look like"
    }},
    "understand": {{
      "content_approach": "...",
      "preferred_formats": [],
      "example": "..."
    }},
    "apply": {{
      "content_approach": "...",
      "preferred_formats": [],
      "example": "..."
    }},
    "analyze": {{
      "content_approach": "...",
      "preferred_formats": [],
      "example": "..."
    }},
    "evaluate": {{
      "content_approach": "...",
      "preferred_formats": [],
      "example": "..."
    }},
    "create": {{
      "content_approach": "...",
      "preferred_formats": [],
      "example": "..."
    }}
  }},

  "sequencing_rules": [
    "Ordered list of sequencing rules specific enough to implement. Each rule
     should name the exact pedagogical move. Example: 'Open with a disruption
     autopsy case before any frameworks. The case should be from the learner's
     industry if possible.'"
  ],

  "case_selection_criteria": {{
    "must_have_characteristics": ["What makes a case suitable for this professor"],
    "preferred_industries": ["Industries this professor draws examples from"],
    "signature_cases_and_why": {{
      "Case Name": "Why this professor uses this case — what pedagogical work it does"
    }}
  }},

  "assessment_design_principles": [
    "How this professor designs assessments — based on actual assessment approaches
     seen in their courses. Be specific about question types, rubrics, and what
     they consider evidence of mastery at each level."
  ],

  "immersive_learning_design": {{
    "preferred_formats": ["case_discussion", "simulation", "role_play", "project"],
    "design_principles": [
      "Principles for how this professor designs experiential learning"
    ],
    "contextualization_approach": "How to make immersive content relevant to the learner's actual job"
  }},

  "anti_patterns": [
    "Things this professor would NEVER do — as important as positive principles.
     Example: 'Never present a framework without first establishing WHY it exists
     through a concrete problem or failure.'"
  ]
}}"""

    return {"system": system, "user": user}


# ═══════════════════════════════════════════════════════════════════════════
# CONTENT SELECTION — Module design from content index
# ═══════════════════════════════════════════════════════════════════════════

def build_content_selection_prompt(
    tacit_knowledge: dict,
    skill: str,
    bloom_level: str,
    content_index: list[dict],
    learner_profile: dict,
) -> dict:
    """
    Final stage: select content from the index and design the learning module.

    content_index items use the ContentIndexer.ContentItem schema:
      content_id, title, source, content_type, url, professor_id,
      skill, bloom_levels, topics, duration_minutes, difficulty,
      summary, tier

    tacit_knowledge is the Pass 3 JSON output.
    """
    # ── Split by tier ──
    tier_1 = [c for c in content_index if c.get("tier") == 1]
    tier_2 = [c for c in content_index if c.get("tier") == 2]
    tier_3 = [c for c in content_index if c.get("tier") == 3]

    def _format_content_list(items: list[dict], label: str) -> str:
        if not items:
            return f"{label}: No content available."
        lines = []
        for item in items[:30]:  # context window guard
            lines.append(
                f"  [{item['content_id']}] \"{item['title']}\" "
                f"| type={item.get('content_type','?')} "
                f"| bloom={item.get('bloom_levels',[])} "
                f"| difficulty={item.get('difficulty','?')} "
                f"| duration={item.get('duration_minutes',0)}min "
                f"| topics={item.get('topics',[])}"
            )
        return f"{label} ({len(items)} items):\n" + "\n".join(lines)

    catalog = "\n\n".join([
        _format_content_list(tier_1, "TIER 1 — Professor's own content (highest priority)"),
        _format_content_list(tier_2, "TIER 2 — Professor-curated (reading lists, citations)"),
        _format_content_list(tier_3, "TIER 3 — External / agent-curated (needs SME review)"),
    ])

    # ── Extract tacit knowledge sections ──
    tk = tacit_knowledge
    principles   = "\n".join(f"  - {p}" for p in tk.get("core_teaching_principles", []))
    seq_rules    = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(tk.get("sequencing_rules", [])))
    anti_patterns = "\n".join(f"  - {a}" for a in tk.get("anti_patterns", []))
    bloom_guidance = tk.get("bloom_level_guidance", {}).get(bloom_level, {})
    immersive     = tk.get("immersive_learning_design", {})

    # ── Learner context ──
    current_level = learner_profile.get("current_skills", {}).get(skill, "unknown")

    system = f"""You are a Professor Agent embodying the pedagogical approach of \
{tk.get('professor_name', 'the mapped expert')}. You design learning modules by \
selecting content from a curated database, following this professor's externalized \
tacit knowledge.

CRITICAL RULES:
1. ALWAYS prefer Tier 1 content when available and appropriate for the Bloom level.
2. Use Tier 2 to fill gaps Tier 1 cannot cover.
3. Only use Tier 3 when Tiers 1-2 cannot cover a topic — flag all Tier 3 selections
   for SME review in the output.
4. NEVER violate any anti-pattern listed in the professor's tacit knowledge.
5. Follow the sequencing rules in the order given.
6. Match every content item to the target Bloom level using the professor's
   level-specific guidance.

Return ONLY valid JSON. No preamble, no markdown fences."""

    user = f"""Design a complete learning module using the inputs below.

═══ PROFESSOR'S TACIT KNOWLEDGE ═══

CORE TEACHING PRINCIPLES:
{principles}

SEQUENCING RULES (follow in order):
{seq_rules}

GUIDANCE FOR {bloom_level.upper()} LEVEL:
  Content approach  : {bloom_guidance.get('content_approach', 'not specified')}
  Preferred formats : {bloom_guidance.get('preferred_formats', [])}
  Concrete example  : {bloom_guidance.get('example', 'not specified')}

IMMERSIVE LEARNING:
  Preferred formats       : {immersive.get('preferred_formats', [])}
  Design principles       : {immersive.get('design_principles', [])}
  Contextualization       : {immersive.get('contextualization_approach', '')}

ANTI-PATTERNS (NEVER do these):
{anti_patterns}

═══ CONTENT DATABASE ═══

{catalog}

═══ LEARNER PROFILE ═══
  Role              : {learner_profile.get('role', 'not specified')}
  Industry          : {learner_profile.get('industry', 'not specified')}
  Experience        : {learner_profile.get('experience_years', '?')} years
  Current level in {skill}: {current_level}
  Target level      : {bloom_level}

═══ TASK ═══
Design a complete learning module for skill="{skill}", bloom_level="{bloom_level}".

For every content item you select, reference it by content_id and explain which
tacit knowledge principle drove the selection. If a topic is not covered by any
content in the database, emit a CONTENT_GAP entry for the content team to fill.

Output JSON:
{{
  "module_title": "...",
  "professor_id": "{tk.get('professor_id', '')}",
  "skill": "{skill}",
  "target_bloom_level": "{bloom_level}",
  "learner_role": "{learner_profile.get('role', '')}",
  "learner_industry": "{learner_profile.get('industry', '')}",
  "estimated_total_duration_minutes": 0,
  "designed_at": "<ISO timestamp>",

  "pre_assessment": {{
    "purpose": "Establish baseline before the module",
    "questions": [
      {{
        "bloom_level": "remember",
        "question": "...",
        "rationale": "Why this question based on professor's approach"
      }}
    ]
  }},

  "primer": {{
    "sections": [
      {{
        "topic": "...",
        "bloom_level": "...",
        "selected_content": [
          {{
            "content_id": "...",
            "tier": 1,
            "title": "...",
            "usage_rationale": "Which tacit principle drove this selection"
          }}
        ],
        "content_gaps": [
          {{
            "gap_description": "What is missing",
            "suggested_format": "video | article | case_study | exercise",
            "bloom_level": "...",
            "needs_sme_review": true
          }}
        ],
        "duration_minutes": 0
      }}
    ]
  }},

  "immersive_learning": {{
    "format": "...",
    "scenario_description": "...",
    "selected_content": [
      {{
        "content_id": "...",
        "tier": 1,
        "title": "...",
        "usage_rationale": "..."
      }}
    ],
    "design_rationale": "Which professor principles drove this immersive design",
    "learner_deliverable": "What the learner produces at the end",
    "content_gaps": [],
    "duration_minutes": 0
  }},

  "post_assessment": {{
    "questions": [
      {{
        "bloom_level": "{bloom_level}",
        "question": "...",
        "rubric_summary": "How to evaluate the response — what constitutes mastery"
      }}
    ]
  }},

  "sme_review_required": [
    "List of all Tier 3 content_ids selected — these need SME sign-off before deployment"
  ],

  "content_gaps_summary": [
    "Consolidated list of all gaps across primer + immersive — for content team"
  ]
}}"""

    return {"system": system, "user": user}


# ═══════════════════════════════════════════════════════════════════════════
# API CALLER
# ═══════════════════════════════════════════════════════════════════════════
def make_anthropic_caller(model: str = "claude-sonnet-4-20250514", max_tokens: int = 4000):
    """
    Returns a callable(system, user) -> str that hits the Anthropic API.

    Loads API key from .env (ANTHROPIC_API_KEY).
    Uses temperature=0 for deterministic outputs.
    """
    import os
    from dotenv import load_dotenv
    import anthropic

    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment or .env file")

    # Explicitly pass key (more reliable than implicit env usage)
    client = anthropic.Anthropic(api_key=api_key)

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
        )

        # Safer parsing (Anthropic sometimes returns structured content)
        if response.content and len(response.content) > 0:
            return response.content[0].text

        return ""

    return call

def _parse_json_output(raw: str, label: str) -> dict:
    """
    Parse LLM JSON output safely.
    Strips markdown fences if present. Returns {"raw": raw} on failure.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` or ``` ... ```
        cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  WARNING: {label} output was not valid JSON ({e}). Saved as raw.")
        return {"raw": raw}


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    professor_profile_path: str,
    content_index_path: str,
    skill: str,
    bloom_level: str,
    learner_profile: dict,
    output_dir: str = "./data/output",
    api_caller=None,
) -> dict:
    """
    Runs the complete four-stage tacit knowledge externalization pipeline.

    Stages:
      1. Pass 1 — Content selection patterns  → pass_1_content_selection.txt
      2. Pass 2 — Sequencing & scaffolding    → pass_2_sequencing.txt
      3. Pass 3 — Synthesis JSON              → pass_3_tacit_knowledge.json
      4. Content selection                    → content_plan.json

    Args:
        professor_profile_path : Path to ProfessorProfile JSON (output of scraper).
        content_index_path     : Path to content_index.json (output of ContentIndexer).
        skill                  : Target skill string (must match profile.primary_skill
                                 or profile.mapped_skills).
        bloom_level            : One of: remember, understand, apply,
                                 analyze, evaluate, create.
        learner_profile        : Dict — see LEARNER_PROFILE_SCHEMA below.
        output_dir             : Directory for all output files.
        api_caller             : Callable(system: str, user: str) -> str.
                                 If None, uses make_anthropic_caller().

    Returns:
        Dict with keys: pass_1, pass_2, tacit_knowledge, content_plan, output_paths.
    """
    VALID_BLOOM = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
    if bloom_level not in VALID_BLOOM:
        raise ValueError(f"bloom_level must be one of {VALID_BLOOM}, got '{bloom_level}'")

    os.makedirs(output_dir, exist_ok=True)

    if api_caller is None:
        print("  No api_caller provided — using Anthropic API (requires ANTHROPIC_API_KEY).")
        api_caller = make_anthropic_caller()

    # ── Load inputs ──
    print("Loading professor profile and content index...")
    with open(professor_profile_path, "r") as f:
        profile = json.load(f)
    with open(content_index_path, "r") as f:
        content_index = json.load(f)

    # Validate skill against profile
    all_skills = set(profile.get("mapped_skills", [])) | {profile.get("primary_skill", "")}
    if skill not in all_skills:
        print(f"  WARNING: skill '{skill}' not in profile's mapped_skills {all_skills}. Proceeding anyway.")

    output_paths = {}

    # ── Pass 1 ──
    print("\nPass 1 — Content selection patterns...")
    p1 = build_pass_1_prompt(profile)
    pass_1_output = api_caller(p1["system"], p1["user"])
    p1_path = os.path.join(output_dir, "pass_1_content_selection.txt")
    with open(p1_path, "w") as f:
        f.write(pass_1_output)
    output_paths["pass_1"] = p1_path
    print(f"  Done. {len(pass_1_output):,} chars → {p1_path}")

    # ── Pass 2 ──
    print("\nPass 2 — Sequencing & scaffolding...")
    p2 = build_pass_2_prompt(profile, pass_1_output)
    pass_2_output = api_caller(p2["system"], p2["user"])
    p2_path = os.path.join(output_dir, "pass_2_sequencing.txt")
    with open(p2_path, "w") as f:
        f.write(pass_2_output)
    output_paths["pass_2"] = p2_path
    print(f"  Done. {len(pass_2_output):,} chars → {p2_path}")

    # ── Pass 3 ──
    print("\nPass 3 — Synthesis into actionable principles...")
    p3 = build_pass_3_prompt(profile, pass_1_output, pass_2_output)
    pass_3_raw = api_caller(p3["system"], p3["user"])
    tacit_knowledge = _parse_json_output(pass_3_raw, "Pass 3")
    # Stamp synthesis time if the LLM left a placeholder
    if tacit_knowledge.get("synthesized_at", "").startswith("<"):
        tacit_knowledge["synthesized_at"] = datetime.now().isoformat()
    p3_path = os.path.join(output_dir, "pass_3_tacit_knowledge.json")
    with open(p3_path, "w") as f:
        json.dump(tacit_knowledge, f, indent=2)
    output_paths["pass_3"] = p3_path
    print(f"  Done → {p3_path}")

    # ── Content selection ──
    print("\nContent selection — designing learning module...")
    cs = build_content_selection_prompt(
        tacit_knowledge, skill, bloom_level, content_index, learner_profile
    )
    cs_raw = api_caller(cs["system"], cs["user"])
    content_plan = _parse_json_output(cs_raw, "Content selection")
    if content_plan.get("designed_at", "").startswith("<"):
        content_plan["designed_at"] = datetime.now().isoformat()
    cp_path = os.path.join(output_dir, "content_plan.json")
    with open(cp_path, "w") as f:
        json.dump(content_plan, f, indent=2)
    output_paths["content_plan"] = cp_path
    print(f"  Done → {cp_path}")

    # ── Summary ──
    print("\n" + "═" * 60)
    print("PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Professor    : {profile['full_name']}")
    print(f"  Skill        : {skill}")
    print(f"  Bloom level  : {bloom_level}")
    print(f"  Learner role : {learner_profile.get('role', '?')}")
    print(f"  Industry     : {learner_profile.get('industry', '?')}")
    print(f"  Output dir   : {output_dir}")
    print("\n  Files written:")
    for label, path in output_paths.items():
        print(f"    {label:<20} → {path}")

    gaps = (
        content_plan.get("content_gaps_summary", [])
        if isinstance(content_plan, dict) else []
    )
    sme_needed = (
        content_plan.get("sme_review_required", [])
        if isinstance(content_plan, dict) else []
    )
    if gaps:
        print(f"\n  Content gaps ({len(gaps)}) — send to content team:")
        for g in gaps:
            print(f"    - {g}")
    if sme_needed:
        print(f"\n  SME review required ({len(sme_needed)}) — Tier 3 items:")
        for s in sme_needed:
            print(f"    - {s}")

    return {
        "pass_1": pass_1_output,
        "pass_2": pass_2_output,
        "tacit_knowledge": tacit_knowledge,
        "content_plan": content_plan,
        "output_paths": output_paths,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LEARNER PROFILE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

LEARNER_PROFILE_SCHEMA = {
    "$schema": "professor-agent/learner-profile/v1",
    "description": (
        "Learner profile passed to run_pipeline(). "
        "current_skills keys must match skill strings used in the content index."
    ),
    "fields": {
        "role": {
            "type": "str",
            "required": True,
            "description": "Learner's current job title / role.",
            "example": "Business Analyst transitioning to Digital Strategy Manager",
        },
        "industry": {
            "type": "str",
            "required": True,
            "description": "Industry the learner works in.",
            "example": "Financial Services",
        },
        "experience_years": {
            "type": "int",
            "required": True,
            "description": "Total years of professional experience.",
            "example": 7,
        },
        "current_skills": {
            "type": "dict[str, str]",
            "required": True,
            "description": (
                "Map of skill → current Bloom level. "
                "Keys must be skill strings from the content index. "
                "Values must be one of: remember, understand, apply, "
                "analyze, evaluate, create."
            ),
            "example": {"Digital Strategy": "remember", "Data Analysis": "apply"},
        },
        "learning_goals": {
            "type": "list[str]",
            "required": False,
            "description": "Free-text goals the learner has stated.",
            "example": ["Lead digital transformation for a mid-size bank"],
        },
        "time_available_minutes_per_week": {
            "type": "int",
            "required": False,
            "description": "Weekly time budget for learning.",
            "example": 90,
        },
    },
    "example": {
        "role": "Business Analyst transitioning to Digital Strategy Manager",
        "industry": "Financial Services",
        "experience_years": 7,
        "current_skills": {"Digital Strategy": "remember"},
        "learning_goals": ["Lead digital transformation for a mid-size bank"],
        "time_available_minutes_per_week": 90,
    },
}


def get_learner_profile_schema() -> dict:
    """Returns the canonical learner profile schema."""
    return LEARNER_PROFILE_SCHEMA


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Professor Agent — Tacit Knowledge Externalization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with Anthropic API
  python tacit_knowledge_pipeline.py \\
      --profile ./data/professors/sunil-gupta.json \\
      --content-index ./data/content_index.json \\
      --skill "Digital Strategy" \\
      --bloom-level analyze \\
      --learner-profile ./learner.json \\
      --output-dir ./data/output/sunil-gupta-digital-strategy-analyze

  # Dump learner profile schema
  python tacit_knowledge_pipeline.py --print-learner-schema
        """,
    )
    parser.add_argument("--profile",         type=str, help="Path to ProfessorProfile JSON")
    parser.add_argument("--content-index",   type=str, help="Path to content_index.json")
    parser.add_argument("--skill",           type=str, help="Target skill string")
    parser.add_argument("--bloom-level",     type=str,
                        choices=["remember", "understand", "apply", "analyze", "evaluate", "create"],
                        help="Target Bloom level")
    parser.add_argument("--learner-profile", type=str,
                        help="Path to learner profile JSON (see --print-learner-schema)")
    parser.add_argument("--output-dir",      type=str, default="./data/output")
    parser.add_argument("--model",           type=str, default="claude-sonnet-4-20250514",
                        help="Anthropic model to use")
    parser.add_argument("--print-learner-schema", action="store_true",
                        help="Print learner profile schema and exit")

    args = parser.parse_args()

    if args.print_learner_schema:
        print(json.dumps(get_learner_profile_schema(), indent=2))
        return

    # Validate required args for pipeline run
    missing = [f for f in ["profile", "content_index", "skill", "bloom_level", "learner_profile"]
               if not getattr(args, f.replace("-", "_"), None)]
    if missing:
        parser.error(f"Missing required arguments: {[f'--{f}' for f in missing]}")

    with open(args.learner_profile, "r") as f:
        learner = json.load(f)

    run_pipeline(
        professor_profile_path=args.profile,
        content_index_path=args.content_index,
        skill=args.skill,
        bloom_level=args.bloom_level,
        learner_profile=learner,
        output_dir=args.output_dir,
        api_caller=make_anthropic_caller(model=args.model),
    )


if __name__ == "__main__":
    main()