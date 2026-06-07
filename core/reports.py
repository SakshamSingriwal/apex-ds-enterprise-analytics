import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import pandas as pd
def export_report(df, target, problem_type, model=None, format='pdf', include_charts=True):
    if format == 'pdf':
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Ultimate Agentic Data Scientist Pro - Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Dataset info
        story.append(Paragraph(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns", styles['Normal']))
        story.append(Paragraph(f"Target: {target}", styles['Normal']))
        story.append(Paragraph(f"Problem type: {problem_type}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Sample data
        story.append(Paragraph("Sample Data", styles['Heading2']))
        sample_table = Table([df.columns.tolist()] + df.head(5).values.tolist())
        sample_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke)]))
        story.append(sample_table)
        
        doc.build(story)
        return buffer.getvalue()
    else:
        # Simple DOCX
        from docx import Document
        doc = Document()
        doc.add_heading('Ultimate Agentic Data Scientist Pro - Report', 0)
        doc.add_paragraph(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns')
        doc.add_paragraph(f'Target: {target}')
        doc.add_paragraph(f'Problem type: {problem_type}')
        doc.add_heading('Sample Data', level=1)
        table = doc.add_table(rows=min(6, len(df)), cols=len(df.columns))
        for i, col in enumerate(df.columns):
            table.cell(0, i).text = str(col)
        for i in range(min(5, len(df))):
            for j, col in enumerate(df.columns):
                table.cell(i+1, j).text = str(df.iloc[i][col])
        buffer = io.BytesIO()
        doc.save(buffer)
        return buffer.getvalue()
