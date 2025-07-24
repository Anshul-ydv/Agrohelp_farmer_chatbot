from fpdf import FPDF
import textwrap

class UTF8PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Add DejaVu Sans font for Unicode support
        self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
        self.add_font('DejaVu', 'I', 'DejaVuSansCondensed-Oblique.ttf', uni=True)
        self.add_font('DejaVu', 'BI', 'DejaVuSansCondensed-BoldOblique.ttf', uni=True)

def create_simple_pdf(text_file, pdf_file):
    """Create a simple PDF without fancy formatting"""
    with open(text_file, 'r') as file:
        content = file.read()

    # Create a simple text file with the content
    with open(pdf_file.replace('.pdf', '.txt'), 'w') as file:
        file.write(content)

    print(f"Text file created: {pdf_file.replace('.pdf', '.txt')}")

    # For demonstration purposes, we'll create a very simple PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Sample Agricultural Guide - Please see the text file for content")
    pdf.output(pdf_file)
    print(f"Simple PDF created: {pdf_file}")

if __name__ == "__main__":
    create_simple_pdf("app/data/sample_agri_guide.txt", "app/data/sample_agri_guide.pdf")
