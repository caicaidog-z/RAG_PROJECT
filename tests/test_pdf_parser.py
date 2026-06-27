import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.settings import AppSettings
from ingestion.pdf_parser import PDFParser


class _FakeMetadata:
    title = "测试PDF"


class _FakeImage:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data


class _FakePage:
    def __init__(self, text: str, images):
        self._text = text
        self.images = images

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, _path: str):
        self.metadata = _FakeMetadata()
        self.pages = [
            _FakePage("第一页正文", []),
            _FakePage("", [_FakeImage("figure.png", b"fake-image-bytes")]),
        ]


class _FakeOCR:
    def __call__(self, _image_bytes):
        return (
            [
                [[[0, 0], [1, 0], [1, 1], [0, 1]], "图片中的文字", 0.99],
            ],
            None,
        )


class PDFParserTest(unittest.TestCase):
    def test_parse_pdf_to_documents_emits_text_and_image_ocr_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AppSettings(
                pdf_asset_dir=temp_dir,
                pdf_enable_image_ocr=True,
                pdf_image_ocr_score_threshold=0.1,
                pdf_image_ocr_min_chars=1,
            )
            parser = PDFParser(settings=settings)
            parser._ocr_engine = _FakeOCR()

            with patch("ingestion.pdf_parser.PdfReader", _FakeReader):
                documents = parser.parse_pdf_to_documents("/tmp/sample.pdf")

            self.assertEqual(len(documents), 2)

            text_doc = documents[0]
            image_doc = documents[1]

            self.assertEqual(text_doc.metadata["element_type"], "pdf_text")
            self.assertEqual(text_doc.metadata["page_number"], 1)
            self.assertIn("第一页正文", text_doc.page_content)

            self.assertEqual(image_doc.metadata["element_type"], "pdf_image_ocr")
            self.assertEqual(image_doc.metadata["page_number"], 2)
            self.assertTrue(image_doc.metadata["asset_path"].endswith(".png"))
            self.assertIn("图片中的文字", image_doc.page_content)
            self.assertTrue(Path(image_doc.metadata["asset_path"]).exists())


if __name__ == "__main__":
    unittest.main()
