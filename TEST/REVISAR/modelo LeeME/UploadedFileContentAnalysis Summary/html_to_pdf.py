from weasyprint import HTML, CSS
import os

html_file = '/home/ubuntu/analise_cosmo_robust.html'
pdf_file = '/home/ubuntu/analise_cosmo_robust.pdf'

# Basic CSS for better layout (optional, can be expanded)
css = CSS(string='''
    @page { size: A4; margin: 2cm; }
    body { font-family: sans-serif; line-height: 1.5; }
    h1, h2, h3 { color: #333; margin-top: 1.5em; margin-bottom: 0.5em; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
    th, td { border: 1px solid #ccc; padding: 0.5em; text-align: left; }
    th { background-color: #f2f2f2; }
    img { max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; }
    .abstract { font-style: italic; margin-bottom: 2em; }
    /* Add more styles as needed */
''')

print(f"Converting {html_file} to {pdf_file} using WeasyPrint...")

# Ensure the image path is accessible relative to the HTML or use absolute paths if needed
# The image is referenced as 'mcmc_corner_plot_hz_fullcov_corrected.png' in the HTML
# Assuming it's in the same directory /home/ubuntu/
base_url = '/home/ubuntu/'

HTML(filename=html_file, base_url=base_url).write_pdf(pdf_file, stylesheets=[css])

print(f"PDF successfully generated: {pdf_file}")

