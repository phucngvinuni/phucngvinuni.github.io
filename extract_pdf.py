from pypdf import PdfReader

reader = PdfReader("NguyenHongPhuc_Resume (5).pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(text)