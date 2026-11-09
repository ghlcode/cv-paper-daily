#!/usr/bin/env python3
"""Fetch the latest arXiv cs.CV (computer vision / deep learning vision) papers.

Outputs, all relative to the repository root:
  data/papers.json          full de-duplicated index
  data/YYYY-MM-DD.md        per-day digest (Beijing time, idempotent rebuild)
  README.md                 auto-refreshed "latest papers" table + archive links

No third-party dependencies: standard library only, so GitHub Actions needs no
pip install step and cannot break because of a package version bump.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
SEARCH_QUERY = "cat:cs.CV"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
USER_AGENT = "cv-paper-daily/1.0 (scheduled GitHub Actions arXiv aggregator)"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_FILE = DATA_DIR / "papers.json"
README_FILE = ROOT / "README.md"

PAPERS_START = "<!-- PAPERS:START -->"
PAPERS_END = "<!-- PAPERS:END -->"
ARCHIVE_START = "<!-- ARCHIVE:START -->"
ARCHIVE_END = "<!-- ARCHIVE:END -->"

CST = timezone(timedelta(hours=8))  # Beijing time
MAX_INDEX_PAPERS = 5000             # keep the repo small
MAX_SEEN_IDS = 8000                 # de-duplication memory, newest first
PAGE_SIZE = 500                     # arXiv API entries per request
API_SLEEP_SECONDS = 3              # arXiv asks for >=3s between API calls


def log(message: str) -> None:
    print(f"[cv-paper-daily] {message}", flush=True)


def clean(text: str | None) -> str:
    """Collapse the multi-line whitespace arXiv puts in titles/abstracts."""
    return " ".join(text.split()) if text else ""


def fetch_page(start: int, page_size: int, retries: int = 4) -> list[dict]:
    """Fetch one page of the Atom feed; retry on network/HTTP/XML errors."""
    params = urllib.parse.urlencode(
        {
            "search_query": SEARCH_QUERY,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"{ARXIV_API}?{params}", headers={"User-Agent": USER_AGENT}
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            papers = parse_feed(payload)
            if not papers and start == 0:
                raise RuntimeError("arXiv returned an empty feed")
            return papers
        except (urllib.error.URLError, TimeoutError, ET.ParseError, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"arXiv fetch failed at start={start} after {retries} attempts: {exc}"
                ) from exc
            wait = 3 * attempt
            log(f"attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    return []  # unreachable


def fetch_feed(max_results: int, cutoff_date: str) -> list[dict]:
    """Fetch the newest entries, paging until we are safely past the cutoff.

    Paging matters after weekends or arXiv's winter break: a single batch of
    200 would otherwise leave the overflow of new submissions undiscovered.
    """
    collected: list[dict] = []
    seen: set[str] = set()
    start = 0
    while start < max_results:
        page_size = min(PAGE_SIZE, max_results - start)
        batch = fetch_page(start, page_size)
        log(f"page start={start}: {len(batch)} entries")
        for paper in batch:
            if paper["id"] not in seen:
                seen.add(paper["id"])
                collected.append(paper)
        if len(batch) < page_size:
            break
        if all(paper["published"][:10] < cutoff_date for paper in batch):
            break
        start += page_size
        if start < max_results:
            time.sleep(API_SLEEP_SECONDS)
    if not collected:
        raise RuntimeError("arXiv returned no usable entries")
    log(f"fetched {len(collected)} unique entries from arXiv")
    return collected


def bootstrap_cutoff(lookback_days: int) -> str:
    """UTC calendar date bounding which unseen papers may be added."""
    return (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).date().isoformat()


def parse_feed(payload: bytes) -> list[dict]:
    root = ET.fromstring(payload)
    papers: list[dict] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = clean(entry.findtext(f"{ATOM}id"))
        if "/abs/" not in raw_id:
            continue
        # "http://arxiv.org/abs/2511.01234v1" -> "2511.01234"
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        if not arxiv_id:
            continue

        published_raw = clean(entry.findtext(f"{ATOM}published"))
        updated_raw = clean(entry.findtext(f"{ATOM}updated"))
        try:
            published = parse_time(published_raw)
        except ValueError:
            continue

        authors = [
            clean(author.findtext(f"{ATOM}name"))
            for author in entry.findall(f"{ATOM}author")
        ]
        authors = [name for name in authors if name]

        categories = [
            category.get("term", "")
            for category in entry.findall(f"{ATOM}category")
            if category.get("term")
        ]
        primary = entry.find(f"{ARXIV}primary_category")
        primary_category = primary.get("term", "") if primary is not None else ""
        if primary_category and primary_category not in categories:
            categories.insert(0, primary_category)

        # Version-less PDF URL always resolves to the newest revision.
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        papers.append(
            {
                "id": arxiv_id,
                "title": clean(entry.findtext(f"{ATOM}title")),
                "authors": authors,
                "abstract": clean(entry.findtext(f"{ATOM}summary")),
                "published": published_raw,
                "updated": updated_raw,
                "published_ts": published.timestamp(),
                "categories": categories,
                "primary_category": primary_category or "cs.CV",
                "comment": clean(entry.findtext(f"{ARXIV}comment")),
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": pdf_url,
            }
        )
    return papers


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def empty_index() -> dict:
    return {"source": SEARCH_QUERY, "updated_at": "", "papers": [], "seen_ids": []}


def load_index() -> dict:
    if not INDEX_FILE.exists():
        return empty_index()
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(data.get("papers"), list):
            data.setdefault("seen_ids", [])  # backwards compatible
            return data
    except (json.JSONDecodeError, OSError) as exc:
        log(f"WARNING: could not read existing index ({exc}); starting a new one")
    return empty_index()


def merge(
    index: dict, fetched: list[dict], lookback_days: int, today: str
) -> tuple[list[dict], int, list[str]]:
    """Merge papers using a seen-ID set plus a stable calendar-day window.

    * seen_ids remembers every ID ever fetched, including ones outside the
      window, so old papers never reappear as "new" on later runs.
    * The window is a UTC calendar date (not a rolling timestamp), so the
      boundary does not drift with the time of day and papers submitted early
      on the boundary day are not permanently skipped.
    * On the first run, papers older than yesterday still enter the index but
      are tagged "bootstrap" and excluded from that day's digest, so the first
      digest is not flooded with back-history.
    """
    bootstrap = not index.get("papers")
    cutoff_date = bootstrap_cutoff(lookback_days)
    # On the first run only the most recent submission day goes into today's
    # digest (weekends have no new papers, so "yesterday" could be empty).
    digest_date = max((p["published"][:10] for p in fetched), default=today)
    existing = {p["id"]: p for p in index.get("papers", []) if p.get("id")}
    old_seen = set(index.get("seen_ids", [])) | set(existing)
    added = 0

    for paper in fetched:
        arxiv_id = paper["id"]
        is_new = arxiv_id not in old_seen
        if paper["published"][:10] < cutoff_date:
            continue
        paper = dict(paper)
        if not is_new:
            if arxiv_id in existing:
                # Refresh metadata (abstract/authors/comments get revised on
                # arXiv) without losing the day it first entered our digest.
                paper["first_seen"] = existing[arxiv_id].get("first_seen", today)
                existing[arxiv_id] = paper
            continue
        paper["first_seen"] = (
            "bootstrap" if bootstrap and paper["published"][:10] < digest_date else today
        )
        existing[arxiv_id] = paper
        added += 1

    papers = sorted(
        existing.values(), key=lambda p: p.get("published_ts", 0), reverse=True
    )[:MAX_INDEX_PAPERS]
    # Keep the newest IDs first so trimming never forgets a recent paper.
    seen_ids = list(
        dict.fromkeys([p["id"] for p in fetched] + list(old_seen))
    )[:MAX_SEEN_IDS]
    return papers, added, seen_ids


def author_label(authors: list[str], limit: int = 3) -> str:
    if not authors:
        return "-"
    label = ", ".join(authors[:limit])
    return f"{label} et al." if len(authors) > limit else label


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_digest(day: str, papers: list[dict], now: datetime) -> str:
    lines = [
        f"# {day} · 计算机视觉新论文",
        "",
        f"> 数据源：arXiv `{SEARCH_QUERY}` ｜ 本次新增 **{len(papers)}** 篇 ｜ "
        f"抓取时间：{now.strftime('%Y-%m-%d %H:%M')} (UTC+8) ｜ 去重键：arXiv ID",
        "",
    ]
    if not papers:
        lines += ["_今日没有新增投稿，可能适逢周末或节假日。_", ""]
    for number, paper in enumerate(papers, start=1):
        lines += [
            f"## {number}. {paper['title']}",
            "",
            f"- **作者**：{escape_cell(author_label(paper['authors']))}",
            f"- **投稿**：{paper['published'][:10]} ｜ **分类**：{', '.join(paper['categories'][:6]) or 'cs.CV'}",
        ]
        if paper.get("comment"):
            lines.append(f"- **备注**：{escape_cell(paper['comment'])}")
        lines += [
            "",
            paper["abstract"] or "_（暂无摘要）_",
            "",
            f"[arXiv 页面]({paper['abs_url']}) · [PDF]({paper['pdf_url']})",
            "",
        ]
    lines += ["---", "", f"由 [cv-paper-daily] 自动生成 · arXiv `{SEARCH_QUERY}`", ""]
    return "\n".join(lines)


def render_papers_table(papers: list[dict], top_n: int, total: int, now: datetime) -> str:
    lines = [
        f"> 共收录 **{total}** 篇 · 最近更新 {now.strftime('%Y-%m-%d %H:%M')} (UTC+8) · "
        "每天 08:23 (UTC+8) 自动更新",
        "",
        "| 投稿日期 | 标题 | 作者 | 分类 | 链接 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for paper in papers[:top_n]:
        lines.append(
            "| {date} | {title} | {authors} | {cat} | [abs]({abs_url}) · [pdf]({pdf_url}) |".format(
                date=paper["published"][:10],
                title=escape_cell(paper["title"]),
                authors=escape_cell(author_label(paper["authors"], 2)),
                cat=escape_cell(paper["primary_category"]),
                abs_url=paper["abs_url"],
                pdf_url=paper["pdf_url"],
            )
        )
    return "\n".join(lines)


def replace_block(content: str, start: str, end: str, body: str) -> str:
    if start not in content or end not in content:
        raise RuntimeError(f"README is missing the {start} / {end} markers")
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n\n{body}\n\n{end}{after}"


def update_readme(papers: list[dict], top_n: int, total: int, now: datetime) -> None:
    content = README_FILE.read_text(encoding="utf-8")
    content = replace_block(
        content, PAPERS_START, PAPERS_END, render_papers_table(papers, top_n, total, now)
    )

    digests = sorted(
        (p.stem for p in DATA_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md")),
        reverse=True,
    )[:14]
    archive_body = "\n".join(f"- [{day}](data/{day}.md)" for day in digests) or "_暂无归档_"
    content = replace_block(content, ARCHIVE_START, ARCHIVE_END, archive_body)
    README_FILE.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-results", type=int, default=800,
                        help="how many recent arXiv entries to request, paged (default: 800)")
    parser.add_argument("--lookback-days", type=int, default=7,
                        help="calendar-day window for unseen papers (default: 7)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="how many papers to show in the README table (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and parse, but write nothing")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    now = datetime.now(CST)
    today = now.strftime("%Y-%m-%d")
    log(f"run date {today} (UTC+8), bootstrap window {args.lookback_days}d, "
        f"max {args.max_results} entries")

    fetched = fetch_feed(args.max_results, bootstrap_cutoff(args.lookback_days))
    if args.dry_run:
        for paper in fetched[:5]:
            log(f"  {paper['published'][:10]}  {paper['id']}  {paper['title'][:70]}")
        log("dry run: no files written")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    papers, added, seen_ids = merge(index, fetched, args.lookback_days, today)
    log(f"new papers: {added}; index total: {len(papers)}; seen ids: {len(seen_ids)}")

    index_out = {
        "source": SEARCH_QUERY,
        "api": ARXIV_API,
        "updated_at": now.isoformat(),
        "total": len(papers),
        "papers": papers,
        "seen_ids": seen_ids,
    }
    INDEX_FILE.write_text(
        json.dumps(index_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    todays = [p for p in papers if p.get("first_seen") == today]
    (DATA_DIR / f"{today}.md").write_text(
        render_digest(today, todays, now), encoding="utf-8"
    )
    update_readme(papers, args.top_n, len(papers), now)
    log(f"wrote data/{today}.md ({len(todays)} papers), data/papers.json, README.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1) from exc
