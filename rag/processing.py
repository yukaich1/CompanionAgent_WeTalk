from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from rag.models import RAGChunk

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_/-]+")


def estimate_tokens(text: str) -> int:
    tokens = TOKEN_RE.findall(str(text or ""))
    return max(1, len(tokens))


def normalize_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", str(text or "")).strip()


def extract_keywords(text: str, limit: int = 10) -> list[str]:
    stopwords = {
        "角色", "设定", "资料", "内容", "文本", "这个", "那个", "以及", "因为", "所以",
        "可以", "需要", "一个", "一种", "已经", "没有", "相关", "关于", "进行", "用于",
    }
    found: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{2,20}", str(text or "")):
        value = token.strip().lower()
        if value in stopwords or value.isdigit():
            continue
        if value not in found:
            found.append(value)
        if len(found) >= limit:
            break
    return found


def split_markdown_paragraphs(markdown_text: str) -> list[str]:
    text = normalize_text(markdown_text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


class DocumentProcessor:
    def __init__(self) -> None:
        self._markitdown = MarkItDown() if MarkItDown is not None else None

    def convert_path_to_markdown(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if self._markitdown is not None:
            try:
                result = self._markitdown.convert(str(path))
                text = normalize_text(getattr(result, "text_content", "") or getattr(result, "text", ""))
                if text:
                    return text
            except Exception:
                pass
        return self._fallback_path_to_markdown(path)

    def convert_text_to_markdown(self, text: str, title: str = "") -> str:
        body = normalize_text(text)
        if not body:
            return ""
        heading = f"# {title.strip()}\n\n" if title.strip() else ""
        return heading + body

    def extract_sections(self, markdown_text: str) -> list[dict]:
        markdown_text = normalize_text(markdown_text)
        if not markdown_text:
            return []

        sections: list[dict] = []
        heading_stack: list[str] = []
        buffer: list[str] = []
        current_path: list[str] = []

        def flush_section() -> None:
            if not buffer:
                return
            body = "\n".join(buffer).strip()
            buffer.clear()
            if not body:
                return
            paragraphs = split_markdown_paragraphs(body)
            title = current_path[-1] if current_path else ""
            sections.append(
                {
                    "heading_path": list(current_path),
                    "title": title,
                    "body": body,
                    "paragraphs": paragraphs,
                }
            )

        for raw_line in markdown_text.splitlines():
            line = raw_line.rstrip()
            match = HEADING_RE.match(line.strip())
            if match:
                flush_section()
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack[:] = heading_stack[: level - 1]
                if title:
                    heading_stack.append(title)
                current_path = list(heading_stack)
                continue
            buffer.append(line)

        flush_section()
        if not sections and markdown_text.strip():
            sections.append(
                {
                    "heading_path": [],
                    "title": "",
                    "body": markdown_text.strip(),
                    "paragraphs": split_markdown_paragraphs(markdown_text),
                }
            )
        return sections

    def _fallback_path_to_markdown(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".pdf":
            if PdfReader is None:
                raise ValueError("当前环境缺少 pypdf，无法解析 PDF。")
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n\n".join(page for page in pages if page)
            if not text:
                raise ValueError("PDF 没有解析出可用文本。")
            return self.convert_text_to_markdown(text, title=path.stem)
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
            return f"# {path.stem}\n\n```json\n{pretty}\n```"
        if suffix == ".csv":
            rows: list[str] = []
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if any(cell.strip() for cell in row):
                        rows.append(" | ".join(cell.strip() for cell in row))
            return self.convert_text_to_markdown("\n".join(rows), title=path.stem)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.convert_text_to_markdown(text, title=path.stem)


class MarkdownSmartChunker:
    def __init__(self, target_tokens: int = 220, overlap_tokens: int = 40, hard_limit_tokens: int = 320):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.hard_limit_tokens = hard_limit_tokens

    def chunk(
        self,
        markdown_text: str,
        document_id: str,
        source_label: str,
        source_type: str = "document",
        priority: float = 1.0,
        metadata: dict | None = None,
    ) -> list[dict]:
        metadata = dict(metadata or {})
        markdown_text = normalize_text(markdown_text)
        if not markdown_text:
            return []

        units = self._parse_markdown_units(markdown_text)
        chunks: list[dict] = []
        current_units: list[dict] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = unit["token_count"]
            if current_units and current_tokens + unit_tokens > self.target_tokens:
                chunks.append(self._build_chunk(document_id, source_label, source_type, current_units, priority, metadata))
                current_units = list(self._build_overlap_seed(current_units))
                current_tokens = sum(item["token_count"] for item in current_units)

            if unit_tokens > self.hard_limit_tokens:
                for split in self._split_large_unit(unit):
                    if current_units and current_tokens + split["token_count"] > self.target_tokens:
                        chunks.append(self._build_chunk(document_id, source_label, source_type, current_units, priority, metadata))
                        current_units = list(self._build_overlap_seed(current_units))
                        current_tokens = sum(item["token_count"] for item in current_units)
                    current_units.append(split)
                    current_tokens += split["token_count"]
                continue

            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            chunks.append(self._build_chunk(document_id, source_label, source_type, current_units, priority, metadata))

        unique_chunks: list[dict] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = chunk["content"]
            if key and key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        return unique_chunks

    def _parse_markdown_units(self, markdown_text: str) -> list[dict]:
        heading_stack: list[str] = []
        units: list[dict] = []
        paragraph_buffer: list[str] = []

        def flush_paragraph() -> None:
            if not paragraph_buffer:
                return
            text = "\n".join(paragraph_buffer).strip()
            paragraph_buffer.clear()
            if not text:
                return
            units.append({"headings": list(heading_stack), "content": text, "token_count": estimate_tokens(text)})

        for raw_line in markdown_text.splitlines():
            line = raw_line.rstrip()
            match = HEADING_RE.match(line.strip())
            if match:
                flush_paragraph()
                level = len(match.group(1))
                title = match.group(2).strip()
                if title:
                    heading_stack[:] = heading_stack[: level - 1]
                    heading_stack.append(title)
                continue
            if not line.strip():
                flush_paragraph()
                continue
            paragraph_buffer.append(line)

        flush_paragraph()
        return units

    def _split_large_unit(self, unit: dict) -> list[dict]:
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", unit["content"]) if part.strip()]
        if len(sentences) <= 1:
            sentences = [unit["content"][index:index + 280] for index in range(0, len(unit["content"]), 280)]
        splits: list[dict] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            tokens = estimate_tokens(sentence)
            if current and current_tokens + tokens > self.target_tokens:
                text = " ".join(current).strip()
                splits.append({"headings": unit["headings"], "content": text, "token_count": estimate_tokens(text)})
                current = [sentence]
                current_tokens = tokens
            else:
                current.append(sentence)
                current_tokens += tokens
        if current:
            text = " ".join(current).strip()
            splits.append({"headings": unit["headings"], "content": text, "token_count": estimate_tokens(text)})
        return splits

    def _build_overlap_seed(self, units: list[dict]) -> list[dict]:
        if not units:
            return []
        selected: list[dict] = []
        total = 0
        for unit in reversed(units):
            selected.insert(0, unit)
            total += unit["token_count"]
            if total >= self.overlap_tokens:
                break
        return selected

    def _build_chunk(self, document_id: str, source_label: str, source_type: str, units: list[dict], priority: float, metadata: dict) -> dict:
        headings = [heading for unit in units for heading in unit["headings"]]
        deduped_headings: list[str] = []
        for heading in headings:
            if heading not in deduped_headings:
                deduped_headings.append(heading)
        prefix = "\n".join(f"# {'#' * index} {title}" if index else f"# {title}" for index, title in enumerate(deduped_headings[:3]))
        body = "\n\n".join(unit["content"] for unit in units if unit["content"].strip())
        content = "\n\n".join(part for part in (prefix.strip(), body.strip()) if part).strip()
        token_count = estimate_tokens(content)
        keywords = extract_keywords(" ".join(deduped_headings) + "\n" + body)
        chunk_id = f"{document_id}:{abs(hash((tuple(deduped_headings), body[:120]))) % 10_000_000}"
        return RAGChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            source_label=source_label,
            source_type=source_type,
            content=content,
            markdown_path=deduped_headings[:4],
            keywords=keywords,
            token_count=token_count,
            priority=priority,
            metadata=dict(metadata),
        ).dict()
