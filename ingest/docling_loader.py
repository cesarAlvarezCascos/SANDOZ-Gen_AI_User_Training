# ingest/docling_loader.py
import os
from modelscope import snapshot_download
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class CustomDocumentLoader:
    """
    Docling loader configured to use RapidOCR with custom ONNX models.
    Converts a PDF into a Docling Document with OCR.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.converter = self._configure_rapidocr_converter()

    def convert(self):
        """Convert the PDF and return a Docling Document object"""
        conversion_result = self.converter.convert(self.file_path)
        return conversion_result.document

    @staticmethod
    def _configure_rapidocr_converter() -> DocumentConverter:
        """
        Configure Docling pipeline to use RapidOCR with ONNX models.
        Downloads RapidOCR models if needed.
        """
        # Download RapidOCR models from Hugging Face via ModelScope
        download_path = snapshot_download(repo_id="RapidAI/RapidOCR")

        det_model_path = os.path.join(
            download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_server_det.onnx"
        )
        rec_model_path = os.path.join(
            download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server_infer.onnx"
        )
        cls_model_path = os.path.join(
            download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        )

        # RapidOCR options
        ocr_options = RapidOcrOptions(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )

        # PDF pipeline options
        pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_options,
            do_ocr=True,
            generate_page_images=True,
            images_scale=1.0,
            do_table_structure=True,
            table_structure_options={"do_cell_matching": True},
        )

        # Document converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return converter
