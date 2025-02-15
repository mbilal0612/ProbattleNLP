import fitz
print(fitz.__doc__) 

pdf_path = "data/pa.pdf"
doc = fitz.open(pdf_path)

# Extract text from all pages
text = "\n".join([page.get_text("text") for page in doc])

print(text)