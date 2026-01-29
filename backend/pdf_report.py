"""
PDF Report Generator for NeuroScan Brain Tumor Detection System
Generates professional medical reports with scan results
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import io
import base64
from datetime import datetime
from PIL import Image as PILImage


def generate_pdf_report(scan_data, patient_data, hospital_data):
    """
    Generate a PDF report for brain tumor scan results

    Args:
        scan_data: Dictionary containing scan information
        patient_data: Dictionary containing patient information
        hospital_data: Dictionary containing hospital information

    Returns:
        BytesIO object containing the PDF
    """
    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#374151'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6,
        leading=14
    )

    # Add header/title
    title = Paragraph("NeuroScan Brain Tumor Detection Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    # Add report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    metadata = Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style)
    elements.append(metadata)
    elements.append(Spacer(1, 0.3 * inch))

    # Hospital Information Section
    hospital_header = Paragraph("Hospital Information", heading_style)
    elements.append(hospital_header)

    hospital_info = [
        ["Hospital Name:", hospital_data.get('hospital_name', 'N/A')],
        ["Hospital Code:", hospital_data.get('hospital_code', 'N/A')],
        ["Doctor/Staff:", hospital_data.get('doctor_name', 'N/A')],
    ]

    hospital_table = Table(hospital_info, colWidths=[2 * inch, 4 * inch])
    hospital_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(hospital_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Patient Information Section
    patient_header = Paragraph("Patient Information", heading_style)
    elements.append(patient_header)

    patient_info = [
        ["Patient Name:", patient_data.get('full_name', 'N/A')],
        ["Patient Code:", patient_data.get('patient_code', 'N/A')],
        ["Access Code:", patient_data.get('access_code', 'N/A')],
        ["Email:", patient_data.get('email', 'N/A')],
        ["Phone:", patient_data.get('phone', 'N/A') or 'N/A'],
        ["Date of Birth:", patient_data.get('date_of_birth', 'N/A') or 'N/A'],
        ["Gender:", patient_data.get('gender', 'N/A') or 'N/A'],
    ]

    patient_table = Table(patient_info, colWidths=[2 * inch, 4 * inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Scan Information Section
    scan_header = Paragraph("Scan Information", heading_style)
    elements.append(scan_header)

    scan_date = scan_data.get('scan_date', 'N/A')
    if scan_date and scan_date != 'N/A':
        try:
            scan_date = datetime.strptime(scan_date, '%Y-%m-%d').strftime('%B %d, %Y')
        except:
            pass

    scan_info = [
        ["Scan Date:", scan_date],
        ["Scan ID:", f"#{scan_data.get('id', 'N/A')}"],
    ]

    scan_table = Table(scan_info, colWidths=[2 * inch, 4 * inch])
    scan_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(scan_table)
    elements.append(Spacer(1, 0.3 * inch))

    # MRI Scan Image
    if scan_data.get('scan_image'):
        try:
            image_header = Paragraph("MRI Scan Image", heading_style)
            elements.append(image_header)

            # Decode base64 image
            image_data = base64.b64decode(scan_data['scan_image'])
            img = PILImage.open(io.BytesIO(image_data))

            # Save to temporary buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Add image to PDF (scaled to fit)
            scan_image = Image(img_buffer, width=4 * inch, height=4 * inch)
            elements.append(scan_image)
            elements.append(Spacer(1, 0.3 * inch))
        except Exception as e:
            print(f"Error adding image: {e}")

    # Analysis Results Section
    results_header = Paragraph("Analysis Results", heading_style)
    elements.append(results_header)

    # Prediction result box
    prediction = scan_data.get('prediction', 'Unknown').upper()
    confidence = scan_data.get('confidence', 0)
    is_tumor = scan_data.get('is_tumor', False)

    # Color coding based on result
    if is_tumor:
        result_color = colors.HexColor('#fee2e2')
        text_color = colors.HexColor('#991b1b')
    else:
        result_color = colors.HexColor('#dcfce7')
        text_color = colors.HexColor('#166534')

    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Normal'],
        fontSize=18,
        textColor=text_color,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceAfter=8,
        spaceBefore=8
    )

    result_data = [[Paragraph(f"{prediction}", result_style),
                    Paragraph(f"Confidence: {confidence:.2f}%", result_style)]]

    result_table = Table(result_data, colWidths=[6 * inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), result_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Probability Distribution
    probabilities = scan_data.get('probabilities', {})
    if probabilities:
        if isinstance(probabilities, str):
            import ast
            try:
                probabilities = ast.literal_eval(probabilities)
            except:
                probabilities = {}

        prob_header = Paragraph("Probability Distribution", heading_style)
        elements.append(prob_header)

        prob_data = [["Tumor Type", "Probability"]]
        for tumor_type, prob in probabilities.items():
            prob_data.append([tumor_type.capitalize(), f"{prob:.2f}%"])

        prob_table = Table(prob_data, colWidths=[3 * inch, 3 * inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 0.2 * inch))

    # Clinical Notes
    notes = scan_data.get('notes', '')
    if notes:
        notes_header = Paragraph("Clinical Notes", heading_style)
        elements.append(notes_header)

        notes_text = Paragraph(notes, normal_style)
        elements.append(notes_text)
        elements.append(Spacer(1, 0.2 * inch))

    # Recommendations
    recommendations_header = Paragraph("Medical Recommendations", heading_style)
    elements.append(recommendations_header)

    if is_tumor:
        recommendations_text = """
        <b>IMPORTANT:</b> This scan indicates the presence of a brain tumor. The following actions are recommended:
        <br/><br/>
        • <b>Immediate consultation</b> with a neurologist or neurosurgeon<br/>
        • <b>Additional imaging</b> may be required for detailed assessment<br/>
        • <b>Biopsy</b> may be necessary to determine tumor type and grade<br/>
        • <b>Treatment planning</b> should begin as soon as possible<br/>
        • <b>Follow-up scans</b> to monitor progression<br/>
        <br/>
        <b>Note:</b> Early detection and treatment significantly improve outcomes.
        """
    else:
        recommendations_text = """
        <b>Result:</b> No tumor detected in this scan. However, the following recommendations apply:
        <br/><br/>
        • <b>Regular monitoring</b> if patient has symptoms or risk factors<br/>
        • <b>Follow-up scan</b> as per physician's recommendation<br/>
        • <b>Maintain healthy lifestyle</b> and report any new symptoms<br/>
        • <b>Consult physician</b> for any concerns or questions<br/>
        <br/>
        <b>Note:</b> This result is based on AI analysis and should be confirmed by a qualified physician.
        """

    recommendations = Paragraph(recommendations_text, normal_style)
    elements.append(recommendations)
    elements.append(Spacer(1, 0.3 * inch))

    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_CENTER,
        leading=12,
        spaceBefore=12
    )

    disclaimer_text = """
    <b>MEDICAL DISCLAIMER:</b> This report is generated by an AI-powered diagnostic system 
    and should be used as a supplementary tool only. All results must be reviewed and confirmed 
    by a qualified medical professional. This report does not constitute medical advice and should 
    not be used as the sole basis for medical decisions.
    """
    disclaimer = Paragraph(disclaimer_text, disclaimer_style)
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(disclaimer)

    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#9ca3af'),
        alignment=TA_CENTER,
        spaceBefore=12
    )

    footer_text = f"""
    NeuroScan Brain Tumor Detection System • Powered by AI & Deep Learning<br/>
    Report ID: {scan_data.get('id', 'N/A')} • Generated: {report_date}
    """
    footer = Paragraph(footer_text, footer_style)
    elements.append(footer)

    # Build PDF
    doc.build(elements)

    # Get the value of the BytesIO buffer and return it
    buffer.seek(0)
    return buffer


def generate_simple_pdf_report(scan_data, patient_data, hospital_data):
    """
    Fallback: Generate a simple text-based PDF if reportlab has issues
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "NeuroScan Brain Tumor Detection Report")

    # Date
    c.setFont("Helvetica", 10)
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    c.drawCentredString(width / 2, height - 70, f"Generated: {report_date}")

    y = height - 120

    # Hospital Info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Hospital Information")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(70, y, f"Hospital: {hospital_data.get('hospital_name', 'N/A')}")
    y -= 15
    c.drawString(70, y, f"Code: {hospital_data.get('hospital_code', 'N/A')}")
    y -= 30

    # Patient Info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Information")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(70, y, f"Name: {patient_data.get('full_name', 'N/A')}")
    y -= 15
    c.drawString(70, y, f"Code: {patient_data.get('patient_code', 'N/A')}")
    y -= 15
    c.drawString(70, y, f"Email: {patient_data.get('email', 'N/A')}")
    y -= 30

    # Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Analysis Results")
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    prediction = scan_data.get('prediction', 'Unknown').upper()
    confidence = scan_data.get('confidence', 0)
    c.drawString(70, y, f"{prediction} - Confidence: {confidence:.2f}%")

    c.save()
    buffer.seek(0)
    return buffer