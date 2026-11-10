"""知识库：文档加载、切片与检索。

检索方案：中文字符二元组 + 英文/数字词 的 BM25，外加同义词扩展。
不引入向量库，是为了让检索结果可解释、可复现，测试里能断言具体命中。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 疑问词、虚词组成的二元组，对相关性没有贡献，去掉以免“公司的…”之类的问题误命中
STOP_TOKENS = {
    "什么", "多少", "怎么", "如何", "可以", "是否", "请问", "一下", "我们", "公司",
    "需要", "哪些", "为什么", "的话", "能不", "不能", "吗", "呢", "的", "了", "是",
    "我", "你", "要", "能", "有", "在", "和", "那", "这", "还", "么", "吧", "啊",
}

_ASCII_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """英文/数字按词，中文按字符二元组；单字片段保留单字。"""
    text = text.lower()
    tokens = _ASCII_RE.findall(text)
    for seg in _CJK_RE.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return [t for t in tokens if t not in STOP_TOKENS]


def normalize(text: str) -> str:
    """用于引用校验：去掉所有空白和常见中英文标点差异。"""
    text = re.sub(r"\s+", "", text)
    return text.replace("，", ",").replace("。", ".").replace("：", ":").replace("；", ";")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    section: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    version: str
    issuer: str
    scope: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(f"{c.section}\n{c.text}" if c.section else c.text for c in self.chunks)

    def meta(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "version": self.version,
                "issuer": self.issuer, "scope": self.scope}


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    header, sep, body = raw.partition("\n---\n")
    if not sep:
        raise ValueError(f"{path.name}: 缺少 '---' 分隔的元数据头")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    for required in ("doc_id", "title", "version"):
        if required not in meta:
            raise ValueError(f"{path.name}: 元数据缺少 {required}")

    doc = Document(meta["doc_id"], meta["title"], meta["version"],
                   meta.get("issuer", "未注明"), meta.get("scope", "未注明"))
    section = ""
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("## "):
            section = lines[0][3:].strip()
            lines = lines[1:]
        if not lines:
            continue
        idx = len(doc.chunks) + 1
        doc.chunks.append(Chunk(doc.doc_id, f"{doc.doc_id}-{idx:02d}", section, "".join(lines)))
    return doc


class KnowledgeBase:
    """BM25 检索 + 同义词扩展。"""

    K1 = 1.5
    B = 0.75
    SYNONYM_WEIGHT = 0.6

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.docs: dict[str, Document] = {}
        for path in sorted(self.data_dir.glob("*.txt")):
            doc = parse_document(path)
            self.docs[doc.doc_id] = doc
        self.synonym_groups = self._load_synonyms()
        self.chunks: list[Chunk] = [c for d in self.docs.values() for c in d.chunks]
        self._chunk_tf = [Counter(tokenize(self._index_text(c))) for c in self.chunks]
        self._chunk_len = [sum(tf.values()) for tf in self._chunk_tf]
        self._avg_len = (sum(self._chunk_len) / len(self._chunk_len)) if self.chunks else 1.0
        df: Counter = Counter()
        for tf in self._chunk_tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _load_synonyms(self) -> list[list[str]]:
        path = self.data_dir / "synonyms.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("groups", [])

    def _index_text(self, chunk: Chunk) -> str:
        title = self.docs[chunk.doc_id].title
        return f"{title} {chunk.section} {chunk.text}"

    def expand_query(self, query: str) -> tuple[dict[str, float], list[str]]:
        """返回 {token: 权重} 和触发的同义词列表。原词权重 1，同义词权重 SYNONYM_WEIGHT。"""
        weights: dict[str, float] = {}
        for t in tokenize(query):
            weights[t] = 1.0
        expanded: list[str] = []
        for group in self.synonym_groups:
            if any(term in query for term in group):
                for term in group:
                    if term in query:
                        continue
                    expanded.append(term)
                    for t in tokenize(term):
                        weights.setdefault(t, self.SYNONYM_WEIGHT)
        return weights, expanded

    def search(self, query: str, top_k: int = 5, min_coverage: float = 0.2,
               exclude_doc_ids: list[str] | None = None) -> dict:
        """coverage = 命中的查询词权重（含同义词）/ 原查询词总权重，上限 1，用来过滤“沾边”命中。"""
        weights, expanded = self.expand_query(query)
        exclude = set(exclude_doc_ids or [])
        if not weights:
            return {"query": query, "expanded_terms": expanded, "hits": []}
        total_weight = sum(w for w in weights.values() if w == 1.0) or 1.0
        scored = []
        for i, chunk in enumerate(self.chunks):
            if chunk.doc_id in exclude:
                continue
            tf = self._chunk_tf[i]
            score, matched = 0.0, 0.0
            for token, w in weights.items():
                f = tf.get(token, 0)
                if not f:
                    continue
                matched += w
                idf = self._idf.get(token, 0.0)
                denom = f + self.K1 * (1 - self.B + self.B * self._chunk_len[i] / self._avg_len)
                score += w * idf * f * (self.K1 + 1) / denom
            coverage = min(1.0, matched / total_weight)
            if score > 0 and coverage >= min_coverage:
                scored.append((score, coverage, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for score, coverage, chunk in scored[:top_k]:
            doc = self.docs[chunk.doc_id]
            hits.append({
                "doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "title": doc.title,
                "version": doc.version, "section": chunk.section,
                "snippet": chunk.text[:80] + ("…" if len(chunk.text) > 80 else ""),
                "score": round(score, 3), "coverage": round(coverage, 3),
            })
        return {"query": query, "expanded_terms": expanded, "hits": hits}

    def read(self, doc_id: str, chunk_id: str | None = None) -> dict:
        doc = self.docs.get(doc_id)
        if doc is None:
            raise KeyError(f"文档 {doc_id} 不存在，可用文档：{', '.join(sorted(self.docs))}")
        if chunk_id:
            for c in doc.chunks:
                if c.chunk_id == chunk_id:
                    return {**doc.meta(), "chunks": [{"chunk_id": c.chunk_id, "section": c.section, "text": c.text}]}
            raise KeyError(f"片段 {chunk_id} 不在文档 {doc_id} 中")
        return {**doc.meta(),
                "chunks": [{"chunk_id": c.chunk_id, "section": c.section, "text": c.text} for c in doc.chunks]}

    def quote_in_doc(self, doc_id: str, quote: str) -> bool:
        doc = self.docs.get(doc_id)
        if doc is None or not quote:
            return False
        return normalize(quote) in normalize(doc.full_text)
