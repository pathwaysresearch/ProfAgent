#!/usr/bin/env python3
"""
PortfolioScraper — Two-stage pipeline
  Stage 1 : Crawl root + internal sublinks → raw markdown
  Stage 1b: Gemini cleans raw markdown → profile.md
  Stage 2 : profile.md → profile_links.md (headings + links only)
  Stage 3 : profile_links.md → links.json (pure Python)
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from pathlib import Path
import aiohttp
import nest_asyncio
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, Response as PlaywrightResponse

nest_asyncio.apply()

BASE_DIR = Path(__file__).parent
# ─────────────────────────── CONFIG ──────────────────────────────────────────

ROOT_URL        = "https://www.isb.edu/faculty-and-research/faculty-directory/deepa-mani"          # ← change this
MAX_DEPTH       = 2                               # how deep to follow internal links
MAX_PAGES       = 25                              # hard cap on pages opened
OUTPUT_DIR      = BASE_DIR / "outputs"


GEMINI_API_KEY  = "[ENCRYPTION_KEY]"   # ← set your key
GEMINI_MODEL    = "gemini-3-flash-preview"

# Retry back-off delays (seconds) — used for API exhaustion
RETRY_DELAYS    = [5, 15, 30, 60]

# URL path fragments that mean "don't crawl this"
IGNORE_KEYWORDS = [
    "logout", "login", "signin", "signup", "register",
    "privacy", "terms", "cookie", "sitemap", "feed",
    "javascript:", "mailto:", "tel:", "#"
]

# Resource types to block in Playwright (we only want text/data)
BLOCKED_RESOURCES = {"image", "media", "font", "stylesheet"}

# ─────────────────────────── DATA CLASSES ────────────────────────────────────

@dataclass
class LinkEntry:
    name: str
    url: str

    def to_md(self) -> str:
        return f"[{self.name}]({self.url})"


@dataclass
class PageData:
    url: str
    heading: str
    content_lines: List[str] = field(default_factory=list)  # DOM in reading order, links inline
    links: List[LinkEntry]   = field(default_factory=list)  # all links (for crawl recursion)
    api_chunks: List[dict]   = field(default_factory=list)  # raw JSON from exhausted APIs


# ─────────────────────────── HELPERS ─────────────────────────────────────────

def profile_slug(url: str) -> str:
    p = urlparse(url)
    slug = (p.netloc + p.path).replace("/", "_").replace(".", "_")
    return slug.strip("_") or "profile"


def is_internal(href: str, root: str) -> bool:
    """True only if href lives under the root URL's path (not just same domain)."""
    parsed_href = urlparse(href)
    parsed_root = urlparse(root)
    if parsed_href.netloc != parsed_root.netloc:
        return False
    # Ensure root path ends with / for clean prefix matching
    root_path = parsed_root.path.rstrip("/") + "/"
    href_path = parsed_href.path
    return href_path.startswith(root_path) or href_path.rstrip("/") == parsed_root.path.rstrip("/")


def should_ignore(url: str) -> bool:
    lower = url.lower()
    return any(kw in lower for kw in IGNORE_KEYWORDS)


def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


# ─────────────────────────── API EXHAUSTION ──────────────────────────────────

class APIExhauster:
    """
    Replays captured XHR/fetch API calls and follows pagination
    (cursor-based or page/offset-based) with retry + back-off.
    Text-only: silently drops binary/non-JSON responses.
    """

    def __init__(self):
        # key = base URL (stripped of pagination params), value = list of raw JSON pages
        self._seen_bases: Set[str] = set()

    def _strip_pagination(self, url: str) -> str:
        """Remove known pagination query params to de-dupe API base."""
        return re.sub(r'([?&])(page|offset|cursor|next_cursor|pageToken)=[^&]*', '', url).rstrip("?&")

    def _extract_cursor(self, data: dict) -> Optional[str]:
        for key in ["next_cursor", "cursor", "nextCursor", "next_page_token", "pageToken", "after"]:
            if val := data.get(key):
                return str(val)
        for sub in ["meta", "pagination", "paging", "links"]:
            node = data.get(sub)
            if isinstance(node, dict):
                for key in ["cursor", "next_cursor", "nextCursor", "next"]:
                    if val := node.get(key):
                        return str(val)
        return None

    def _extract_page_param(self, url: str) -> Optional[tuple]:
        """Returns (param_name, current_value) or None."""
        for param in ["page", "offset"]:
            m = re.search(rf'[?&]{param}=(\d+)', url)
            if m:
                return param, int(m.group(1))
        return None

    def _items_from(self, data: dict) -> list:
        for key in ["data", "items", "results", "entries", "records"]:
            v = data.get(key)
            if isinstance(v, list):
                return v
        return []

    async def _get(self, session: aiohttp.ClientSession, url: str) -> Optional[dict]:
        for delay in RETRY_DELAYS:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=20),
                    headers={"Accept": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        ct = resp.headers.get("Content-Type", "")
                        if "json" not in ct:
                            return None
                        return await resp.json(content_type=None)
                    elif resp.status in (429, 500, 502, 503, 504):
                        print(f"      ! HTTP {resp.status} → retry in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        return None   # 404, auth errors etc — skip silently
            except asyncio.TimeoutError:
                print(f"      ! Timeout → retry in {delay}s")
                await asyncio.sleep(delay)
            except aiohttp.ClientError as e:
                print(f"      ! ClientError ({e}) → retry in {delay}s")
                await asyncio.sleep(delay)
        print(f"      ! Gave up on {url}")
        return None

    async def exhaust(self, first_url: str, first_response: dict, max_extra_pages: int = 8) -> List[dict]:
        base = self._strip_pagination(first_url)
        if base in self._seen_bases:
            return []
        self._seen_bases.add(base)

        chunks = [first_response]
        cursor = self._extract_cursor(first_response)
        page_info = self._extract_page_param(first_url)

        async with aiohttp.ClientSession() as session:
            if cursor:
                # ── Cursor-based pagination ──────────────────────────────
                for _ in range(max_extra_pages):
                    sep = "&" if "?" in first_url else "?"
                    next_url = f"{base}{sep}cursor={cursor}"
                    data = await self._get(session, next_url)
                    if data is None:
                        break
                    chunks.append(data)
                    cursor = self._extract_cursor(data)
                    if not cursor:
                        break

            elif page_info:
                # ── Page/offset-based pagination ─────────────────────────
                param_name, current_val = page_info
                step = current_val if param_name == "offset" and current_val else 1

                for i in range(1, max_extra_pages + 1):
                    next_val = current_val + (i * step if param_name == "offset" else i)
                    next_url = re.sub(
                        rf'([?&]{param_name}=)\d+',
                        lambda m: m.group(1) + str(next_val),
                        first_url
                    )
                    data = await self._get(session, next_url)
                    if data is None:
                        break
                    if not self._items_from(data):  # empty page → done
                        break
                    chunks.append(data)

        return chunks


# ─────────────────────────── CRAWLER ─────────────────────────────────────────

class PortfolioCrawler:
    def __init__(self, root_url: str):
        self.root_url  = root_url
        self.visited:  Set[str]    = set()
        self.pages:    List[PageData] = []
        self.exhausted = APIExhauster()
        self._browser  = None
        self._pw       = None

    async def start(self):
        self._pw      = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)

    async def stop(self):
        if self._browser: await self._browser.close()
        if self._pw:      await self._pw.stop()

    # ── Public entry ─────────────────────────────────────────────────────────

    async def crawl(self, url: str, depth: int = 0):
        url = url.split("#")[0].rstrip("/") or url   # normalise
        if depth > MAX_DEPTH or len(self.visited) >= MAX_PAGES:
            return
        if url in self.visited or should_ignore(url):
            return

        self.visited.add(url)
        indent = "  " * depth
        print(f"{indent}[{len(self.visited)}/{MAX_PAGES}] {url}")

        page_data = await self._scrape(url)
        if page_data:
            self.pages.append(page_data)
            # recurse into internal links only
            for link in page_data.links:
                if is_internal(link.url, self.root_url) and not should_ignore(link.url):
                    await self.crawl(link.url, depth + 1)

    # ── Per-page scrape ───────────────────────────────────────────────────────

    async def _scrape(self, url: str) -> Optional[PageData]:
        ctx = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        # Block heavy resources — we only care about text + JSON
        await ctx.route(
            "**/*",
            lambda route: (
                route.abort()
                if route.request.resource_type in BLOCKED_RESOURCES
                else route.continue_()
            )
        )

        captured: List[tuple] = []   # (url, json_body)

        async def on_response(resp: PlaywrightResponse):
            try:
                if resp.request.resource_type not in ("xhr", "fetch"):
                    return
                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    return
                body = await resp.json()
                captured.append((resp.url, body))
            except Exception:
                pass

        page = await ctx.new_page()
        page.on("response", on_response)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            await page.wait_for_timeout(1_500)

            await self._expand(page)

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Strip noise tags
            for tag in soup.find_all(
                ["script", "style", "noscript", "nav", "footer",
                 "img", "svg", "video", "audio", "iframe", "picture"]
            ):
                tag.decompose()

            heading          = self._heading(soup, url)
            content_lines, links = self._extract_content(soup, url)
            api_data         = await self._exhaust_captured(captured)

            return PageData(url=url, heading=heading, content_lines=content_lines, links=links, api_chunks=api_data)

        except Exception as e:
            print(f"    ! scrape error: {e}")
            return None
        finally:
            await page.close()
            await ctx.close()

    # ── DOM helpers ───────────────────────────────────────────────────────────

    def _heading(self, soup: BeautifulSoup, url: str) -> str:
        for tag in ("h1", "h2", "title"):
            el = soup.find(tag)
            if el:
                return el.get_text(strip=True)
        return urlparse(url).path.strip("/").replace("/", " › ") or "Home"

    # ── PROCESS_TAGS: block-level tags we render as individual lines ──────────
    _PROCESS_TAGS  = {"h1","h2","h3","h4","h5","h6","p","li","td","th","blockquote"}
    _HEADING_MAP   = {"h1":"# ","h2":"## ","h3":"### ","h4":"#### ","h5":"##### ","h6":"###### "}
    _INLINE_TAGS   = {"a","strong","em","b","i","span","code","small","label","abbr"}

    def _extract_content(self, soup: BeautifulSoup, base_url: str):
        """
        Single pass over the DOM in document order.
        Returns (content_lines, links) where content_lines preserves reading order
        with inline [name](url) links exactly where they appear in the HTML.
        """
        from bs4 import NavigableString, Tag

        seen_urls: Set[str] = set()
        all_links: List[LinkEntry] = []

        def _process_inline(el) -> str:
            """Render an element's inline content as a string, turning <a> into [name](url)."""
            parts = []
            for child in el.children:
                if isinstance(child, NavigableString):
                    t = str(child).strip()
                    if t:
                        parts.append(t)
                elif isinstance(child, Tag):
                    if child.name == "a":
                        href = (child.get("href") or "").strip()
                        text = child.get_text(strip=True)
                        if href:
                            full = urljoin(base_url, href).split("#")[0]
                            parsed = urlparse(full)
                            if parsed.scheme in ("http", "https"):
                                name = (
                                    text
                                    or child.get("title", "")
                                    or child.get("aria-label", "")
                                    or "Link"
                                )
                                parts.append(f"[{name}]({full})")
                                if full not in seen_urls:
                                    seen_urls.add(full)
                                    all_links.append(LinkEntry(name=name, url=full))
                                continue
                        # no valid href — fall through to plain text
                        if text:
                            parts.append(text)
                    elif child.name in self._INLINE_TAGS:
                        # recurse inline elements
                        parts.append(_process_inline(child))
                    else:
                        # nested block inside inline context — just grab text
                        t = child.get_text(separator=" ", strip=True)
                        if t:
                            parts.append(t)
            return " ".join(p for p in parts if p)

        def _has_process_ancestor(el) -> bool:
            """True if any ancestor is also a PROCESS_TAG — meaning this el
            will be rendered as part of its parent, not separately."""
            for parent in el.parents:
                if getattr(parent, "name", None) in self._PROCESS_TAGS:
                    return True
            return False

        content_lines: List[str] = []

        for el in soup.find_all(self._PROCESS_TAGS):
            # Skip elements that are nested inside another processable block.
            # The outer block renders everything including this inner one.
            if _has_process_ancestor(el):
                continue

            if el.name in self._HEADING_MAP:
                text = el.get_text(separator=" ", strip=True)
                if text:
                    content_lines.append(f"{self._HEADING_MAP[el.name]}{text}")
            else:
                line = _process_inline(el).strip()
                if len(line) > 15:
                    content_lines.append(line)

        return content_lines, all_links

    async def _expand(self, page: Page):
        """Click collapsed accordions and force-reveal hidden sections."""
        try:
            triggers = await page.locator('button[aria-expanded="false"]').all()
            for t in triggers[:20]:
                try:
                    if await t.is_visible():
                        await t.click(force=True, timeout=800)
                        await page.wait_for_timeout(250)
                except Exception:
                    pass

            await page.evaluate("""
                document.querySelectorAll('[aria-expanded="false"]')
                    .forEach(el => el.setAttribute('aria-expanded','true'));
                document.querySelectorAll('[hidden]')
                    .forEach(el => el.removeAttribute('hidden'));
                document.querySelectorAll('[aria-hidden="true"]')
                    .forEach(el => el.setAttribute('aria-hidden','false'));
                document.querySelectorAll(
                    '[class*="accordion__content"],[class*="collapse"],[class*="dropdown__menu"]'
                ).forEach(el => {
                    el.style.display   = 'block';
                    el.style.maxHeight = 'none';
                    el.style.overflow  = 'visible';
                });
            """)
            await page.wait_for_timeout(500)
        except Exception:
            pass

    async def _exhaust_captured(self, captured: List[tuple]) -> List[dict]:
        all_chunks: List[dict] = []
        for api_url, first_json in captured:
            chunks = await self.exhausted.exhaust(api_url, first_json)
            all_chunks.extend(chunks)
        return all_chunks


# ─────────────────────────── MARKDOWN BUILDER ────────────────────────────────

class MarkdownBuilder:

    def build_full(self, pages: List[PageData]) -> str:
        """
        profile.md — content in exact reading order.
        Headings are above their content. Links appear inline where they are in the DOM.
        API JSON appended at end of each page section.
        """
        parts: List[str] = []
        for page in pages:
            parts.append(f"# {page.heading}")
            parts.append(f"Source: {page.url}\n")

            # Content in DOM order — headings, text, inline links all in place
            parts.extend(page.content_lines)
            parts.append("")

            if page.api_chunks:
                parts.append("## API Data")
                for chunk in page.api_chunks:
                    parts.append("```json")
                    parts.append(json.dumps(chunk, indent=2, ensure_ascii=False))
                    parts.append("```")
                parts.append("")

            parts.append("---\n")

        return "\n".join(parts)


# ─────────────────────────── GEMINI CLEANER ──────────────────────────────────

class GeminiCleaner:
    """
    Sends the raw scraped markdown to Gemini and gets back a clean profile.md.
    Gemini removes nav junk, duplicate headings, index boilerplate, etc.
    Links remain interleaved exactly where they appeared in the content.
    """

    PROMPT = """You are a profile document cleaner.

INPUT: Raw markdown scraped from a person's faculty/portfolio website.
       It may contain navigation menus, site-wide boilerplate, repeated headers,
       index page noise, cookie banners, and other irrelevant content.

TASK: Rewrite it as a single clean, well-structured Markdown profile document.

RULES:
- Keep ALL meaningful content about the person (bio, research, publications,
  awards, courses taught, media, contact info, etc.)
- Keep every [Link name](url) that belongs to actual content — do NOT remove
  or move links; keep them inline exactly where they appear in the text
- Remove: site navigation, breadcrumbs, footer links, "Home / Faculty / ..."
  trails, cookie notices, repeated site-wide headings, index-page card grids
  that just list other people, pagination controls, search bars
- Use clean Markdown headings (## for major sections, ### for sub-sections)
- Do not invent any content that was not in the input
- Return ONLY the cleaned Markdown, no explanation, no fences

INPUT MARKDOWN:
{raw_md}

CLEANED MARKDOWN:"""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def clean(self, raw_md: str) -> str:
        print("  → Sending to Gemini for cleaning...")
        prompt = self.PROMPT.format(raw_md=raw_md)
        for attempt, delay in enumerate(RETRY_DELAYS, 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                cleaned = response.text.strip()
                # Strip accidental code fences if model wraps output
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```[\w]*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                print(f"  ✓ Gemini returned {len(cleaned):,} chars")
                return cleaned
            except Exception as e:
                if attempt < len(RETRY_DELAYS):
                    print(f"  ! Gemini error ({e}) → retry in {delay}s")
                    import time; time.sleep(delay)
                else:
                    print(f"  ! Gemini gave up after {attempt} attempts: {e}")
                    print("  ! Falling back to raw markdown")
                    return raw_md


# ─────────────────────────── LINK JSON BUILDER ───────────────────────────────

class LinkJsonBuilder:
    """
    Parses profile_links.md (headings + link lines only) and produces links.json.
    Pure Python — no LLM needed.

    Output: { "Heading text": ["[Link name](url)", ...], ... }
    Links before the first heading go under "General".
    """

    def build(self, md: str) -> dict:
        result: dict = {}
        current_heading = "General"

        for line in md.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Heading → new key
            if stripped.startswith("#"):
                current_heading = stripped.lstrip("#").strip()
                if current_heading not in result:
                    result[current_heading] = []
                continue

            # Extract all [name](url) entries on this line
            import re
            for name, url in re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', stripped):
                if current_heading not in result:
                    result[current_heading] = []
                result[current_heading].append(f"[{name}]({url})")

        # Drop empty heading keys
        return {k: v for k, v in result.items() if v}


# ─────────────────────────── MAIN ────────────────────────────────────────────

async def main():
    root  = ROOT_URL
    slug  = profile_slug(root)
    outd  = os.path.join(OUTPUT_DIR, slug)
    os.makedirs(outd, exist_ok=True)

    pm_path   = os.path.join(outd, "profile.md")
    pl_path   = os.path.join(outd, "profile_links.md")
    lj_path   = os.path.join(outd, "links.json")

    sep = "=" * 60

    print(f"\n{sep}")
    print("PortfolioScraper")
    print(f"Root   : {root}")
    print(f"Output : {outd}/")
    print(sep + "\n")

    # ── Stage 1: crawl ───────────────────────────────────────────────────────
    print("Stage 1 — Crawling...\n")
    crawler = PortfolioCrawler(root)
    await crawler.start()
    try:
        await crawler.crawl(root)
    finally:
        await crawler.stop()

    print(f"\n✓ Crawled {len(crawler.pages)} pages, "
          f"{sum(len(p.links) for p in crawler.pages)} links found\n")

    # ── Stage 1b: build raw markdown → clean via Gemini → profile.md ─────────
    print("Stage 1b — Building raw markdown...")
    builder = MarkdownBuilder()
    raw_md  = builder.build_full(crawler.pages)
    print(f"  Raw markdown: {len(raw_md):,} chars")

    print("\nStage 1b — Cleaning with Gemini...")
    clean_md = GeminiCleaner().clean(raw_md)

    with open(pm_path, "w", encoding="utf-8") as f:
        f.write(clean_md)
    print(f"✓ profile.md        → {len(clean_md):,} chars")

    # ── Stage 3: parse links → links.json ────────────────────────────────────
    print("\nStage 3 — Building links.json...")
    links_json = LinkJsonBuilder().build(clean_md)

    with open(lj_path, "w", encoding="utf-8") as f:
        json.dump(links_json, f, indent=2, ensure_ascii=False)
    print(f"✓ links.json        → {len(links_json)} top-level groups")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("✓✓✓  COMPLETE")
    print(f"  {pm_path}  — clean profile (Gemini)")
    print(f"  {pl_path}  — headings + links only")
    print(f"  {lj_path}  — link map grouped by heading")
    print(sep + "\n")


if __name__ == "__main__":
    asyncio.run(main())