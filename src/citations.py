import re


CITATION_PATTERN = re.compile(
    r"[\[【]([^\[\]【】\n,]+?),\s*PDF\s+pages?(?:\(s\))?\s*(\d+)(?:\s*[-–]\s*(\d+))?[\]】]",
    re.IGNORECASE,
)


def parse_citations(answer: str) -> list[tuple[str, int, int]]:
    citations = []
    for match in CITATION_PATTERN.finditer(answer):
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if end < start:
            start, end = end, start
        citations.append((match.group(1).strip(), start, end))
    return citations


def _same_file(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).casefold().strip()

    source = normalize(left)
    citation = normalize(right)
    return source == citation or source in citation or citation in source


def _pages_overlap(page_start: int, page_end: int, cite_start: int, cite_end: int) -> bool:
    return page_start <= cite_end and cite_start <= page_end


def matching_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Keep the retrieved source that best matches each citation in the answer."""
    selected = []
    seen = set()
    for source_name, cite_start, cite_end in parse_citations(answer):
        candidates = [
            source
            for source in sources
            if _same_file(str(source.get("source", "")), source_name)
            and _pages_overlap(
                int(source["page_start"]),
                int(source["page_end"]),
                cite_start,
                cite_end,
            )
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda source: float(source.get("score", 0)))
        key = (best["source"], best["page_label"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(best)
    return selected
