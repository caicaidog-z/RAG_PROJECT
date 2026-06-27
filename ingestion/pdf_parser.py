import hashlib
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from config.settings import AppSettings, get_settings
from ingestion.document_chunker import SemanticDocumentChunker
from utils.logging import log

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing.
    RapidOCR = None


class PDFParser:
    """Parse PDF files into text and OCR-enriched image documents."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.chunker = SemanticDocumentChunker(settings=self.settings)
        self.asset_root = Path(self.settings.pdf_asset_dir)
        self._ocr_engine = None
        self._logged_ocr_unavailable = False

    def parse_pdf_to_documents(self, pdf_file: str) -> List[Document]:
        pdf_path = Path(pdf_file)
        reader = PdfReader(str(pdf_path))
        document_title = self._resolve_document_title(reader, pdf_path)
        page_documents: List[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = self._normalize_text(page.extract_text() or "")
            if page_text:
                page_documents.append(
                    self._build_text_document(
                        document_title=document_title,
                        pdf_path=pdf_path,
                        page_number=page_number,
                        text=page_text,
                    )
                )

            image_documents = self._extract_image_documents(
                document_title=document_title,
                pdf_path=pdf_path,
                page_number=page_number,
                page=page,
            )
            page_documents.extend(image_documents)

        chunked_documents = self.chunker.chunk_documents(page_documents)
        log.info(f"PDF解析后的长度: {len(page_documents)}")
        log.info(f"PDF语义切割后的长度: {len(chunked_documents)}")
        return chunked_documents

    def _resolve_document_title(self, reader: PdfReader, pdf_path: Path) -> str:
        metadata = getattr(reader, "metadata", None)
        raw_title = getattr(metadata, "title", None) if metadata else None
        title = self._normalize_text(raw_title or "")
        return title or pdf_path.stem

    def _build_text_document(
        self,
        document_title: str,
        pdf_path: Path,
        page_number: int,
        text: str,
    ) -> Document:
        return Document(
            page_content=f"{document_title} -> 第{page_number}页\n{text}",
            metadata={
                "title": document_title,
                "category_depth": 1,
                "category": "content",
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "filetype": "application/pdf",
                "page_number": page_number,
                "element_type": "pdf_text",
                "asset_path": "",
            },
        )

    def _extract_image_documents(
        self,
        document_title: str,
        pdf_path: Path,
        page_number: int,
        page,
    ) -> List[Document]:
        images = list(getattr(page, "images", []))
        if not images:
            return []

        page_documents: List[Document] = []
        for image_index, image in enumerate(images, start=1):
            asset_path = self._write_image_asset(pdf_path, page_number, image_index, image)
            ocr_text = self._extract_image_text(image.data)
            if not ocr_text:
                continue

            page_documents.append(
                Document(
                    page_content=(
                        f"{document_title} -> 第{page_number}页图片{image_index}\n"
                        f"{ocr_text}"
                    ),
                    metadata={
                        "title": document_title,
                        "category_depth": 1,
                        "category": "content",
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "filetype": self._guess_image_mime_type(image.name),
                        "page_number": page_number,
                        "element_type": "pdf_image_ocr",
                        "asset_path": str(asset_path),
                    },
                )
            )
        return page_documents

    def _write_image_asset(
        self,
        pdf_path: Path,
        page_number: int,
        image_index: int,
        image,
    ) -> Path:
        asset_dir = self._asset_dir_for_pdf(pdf_path)
        asset_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.name).suffix or ".bin"
        asset_path = asset_dir / f"page_{page_number:04d}_image_{image_index:03d}{suffix}"
        asset_path.write_bytes(image.data)
        return asset_path

    def _asset_dir_for_pdf(self, pdf_path: Path) -> Path:
        digest = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:10]
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_") or "document"
        return self.asset_root / f"{safe_name}_{digest}"

    def _extract_image_text(self, image_bytes: bytes) -> str:
        if not self.settings.pdf_enable_image_ocr:
            return ""

        engine = self._get_ocr_engine()
        if engine is None:
            return ""

        result, _ = engine(image_bytes)
        if not result:
            return ""

        texts = [
            text.strip()
            for _, text, score in result
            if score >= self.settings.pdf_image_ocr_score_threshold and text.strip()
        ]
        merged_text = self._normalize_text("\n".join(texts))
        if len(merged_text) < self.settings.pdf_image_ocr_min_chars:
            return ""
        return merged_text

    def _get_ocr_engine(self):
        if self._ocr_engine is not None:
            return self._ocr_engine

        if RapidOCR is None:
            if not self._logged_ocr_unavailable:
                log.warning("rapidocr_onnxruntime 未安装，PDF 图片OCR将被跳过")
                self._logged_ocr_unavailable = True
            return None

        self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _guess_image_mime_type(self, image_name: str) -> str:
        suffix = Path(image_name).suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "application/octet-stream"

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[ \t]+", " ", text)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()
