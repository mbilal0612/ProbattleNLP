
# import pandas as pd
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.chunking import HybridChunker
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions

# pdf_pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
# doc_converter = DocumentConverter(
#     format_options={InputFormat.PDF: PdfFormatOption(
#         pipeline_options=pdf_pipeline_options
#     )}
# )
# chunker = HybridChunker()

# # Load the PDF
# doc = doc_converter.convert("data/pa.pdf").document


# # Extract tables
# tables = doc.tables

# print(tables)

# # Convert the first table to a Pandas DataFrame
# df = pd.DataFrame(tables[0])

# print(df.head())  # View the extracted table


from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import pandas as pd

pdf_pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(
        pipeline_options=pdf_pipeline_options
    )}
)
chunker = HybridChunker(max_tokens=768)

# You can replace this with any list of PDFs you want
# 'file' can be a URL or a local filename for a PDF
PDFs = [
    {'title': "Institute of Business Administration Program Annoucments", 'file': "data/pa.pdf"},

]

data = []
chunk_id = 0
for pdf in PDFs:
    print("Downloading and parsing", pdf['title'])
    doc = doc_converter.convert(pdf['file']).document
    
    # Extract tables
    tables = doc.tables
    if tables:
        # Convert the first table to a Pandas DataFrame
        df = pd.DataFrame(tables[0])
        print(df.head())  # View the extracted table
    
    for chunk in chunker.chunk(dl_doc=doc):
        chunk_dict = chunk.model_dump()
        filename = chunk_dict['meta']['origin']['filename']
        heading = chunk_dict['meta']['headings'][0] if chunk_dict['meta']['headings'] else None
        page_num = chunk_dict['meta']['doc_items'][0]['prov'][0]['page_no']
        data.append(
            {"id": chunk_id, "text": chunk.text, "title": pdf['title'], "filename": filename, "heading": heading, "page_num": page_num}
        )
    print("done parsing document")

