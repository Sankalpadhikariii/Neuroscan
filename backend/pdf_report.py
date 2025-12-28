
import io
import base64
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbers"""

    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(
            200 * 2.83464567,  # 200mm in points
            15,
            f"Page {self._pageNumber} of {page_count}"
        )


def create_probability_chart(probabilities):
    """Create a bar chart of tumor probabilities"""
    fig, ax = plt.subplots(figsize=(6, 3))

    classes = list(probabilities.keys())
    values = list(probabilities.values())

    colors_map = ['#ef4444' if v == max(values) else '#3b82f6' for v in values]

    ax.barh(classes, values, color=colors_map)
    ax.set_xlabel('Probability (%)', fontsize=10)
    ax.set_title('Tumor Classification Probabilities', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)

    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    # Save to BytesIO
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()

    return img_buffer


def generate_pdf_report(prediction_data, patient_info, image_base64):
    """
    Generate comprehensive PDF report

    Args:
        prediction_data: dict with 'prediction', 'confidence', 'probabilities'
        patient_info: dict with 'name', 'age', 'gender', 'patient_id', 'scan_date', 'notes'
        image_base64: base64 encoded MRI image

    Returns:
        BytesIO buffer containing PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
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

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )

    # ============ TITLE ============
    elements.append(Paragraph("NEUROSCAN", title_style))
    elements.append(Paragraph("Brain Tumor Detection Report", styles['Heading2']))
    elements.append(Spacer(1, 0.3 * inch))

    # ============ REPORT INFO ============
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", body_style))
    elements.append(Spacer(1, 0.2 * inch))

    # ============ PATIENT INFORMATION ============
    elements.append(Paragraph("Patient Information", heading_style))

    patient_data = [
        ['Patient Name:', patient_info.get('name', 'N/A')],
        ['Patient ID:', patient_info.get('patient_id', 'N/A')],
        ['Age:', f"{patient_info.get('age', 'N/A')} years"],
        ['Gender:', patient_info.get('gender', 'N/A')],
        ['Scan Date:', patient_info.get('scan_date', 'N/A')],
    ]

    patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 0.3 * inch))

    # ============ MRI IMAGE ============
    elements.append(Paragraph("MRI Scan Analysis", heading_style))

    try:
        # Decode and add MRI image
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))

        # Save to buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        mri_image = RLImage(img_buffer, width=4 * inch, height=4 * inch)
        elements.append(mri_image)
        elements.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        elements.append(Paragraph(f"Error loading MRI image: {str(e)}", body_style))

    # ============ DETECTION RESULTS ============
    elements.append(Paragraph("Detection Results", heading_style))

    prediction = prediction_data['prediction']
    confidence = prediction_data['confidence']
    is_tumor = prediction.lower() != 'notumor'

    # Result box styling
    result_color = colors.HexColor('#fee2e2') if is_tumor else colors.HexColor('#dcfce7')
    result_text = f"""
    <para alignment="center">
        <b><font size="16" color="{'#dc2626' if is_tumor else '#16a34a'}">
            {prediction.upper()}
        </font></b><br/>
        <font size="12">Confidence: {confidence:.2f}%</font>
    </para>
    """

    result_table = Table([[Paragraph(result_text, body_style)]], colWidths=[6 * inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), result_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#dc2626' if is_tumor else '#16a34a')),
    ]))

    elements.append(result_table)
    elements.append(Spacer(1, 0.3 * inch))

    # ============ PROBABILITY DISTRIBUTION ============
    elements.append(Paragraph("Probability Distribution", heading_style))

    chart_buffer = create_probability_chart(prediction_data['probabilities'])
    chart_image = RLImage(chart_buffer, width=5 * inch, height=2.5 * inch)
    elements.append(chart_image)
    elements.append(Spacer(1, 0.3 * inch))

    # Probability table
    prob_data = [['Tumor Type', 'Probability']]
    for tumor_type, prob in prediction_data['probabilities'].items():
        prob_data.append([tumor_type.capitalize(), f"{prob:.2f}%"])

    prob_table = Table(prob_data, colWidths=[3 * inch, 2 * inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(prob_table)
    elements.append(Spacer(1, 0.3 * inch))

    # ============ MEDICAL RECOMMENDATIONS ============
    elements.append(Paragraph("Medical Recommendations", heading_style))

    if is_tumor:
        recommendations = """
        <b>Action Required:</b> A potential tumor has been detected.<br/><br/>

        <b>Recommended Next Steps:</b><br/>
        1. Consult with a qualified neurologist or oncologist immediately<br/>
        2. Schedule a comprehensive neurological examination<br/>
        3. Consider additional imaging tests (CT scan, PET scan) for confirmation<br/>
        4. Discuss biopsy options with your healthcare provider<br/>
        5. Prepare medical history and current symptoms documentation<br/><br/>

        <b>Important:</b> Early detection significantly improves treatment outcomes. 
        Please schedule an appointment with a specialist as soon as possible.
        """
    else:
        recommendations = """
        <b>Status:</b> No tumor detected in this scan.<br/><br/>

        <b>Recommended Next Steps:</b><br/>
        1. Continue regular health monitoring as advised by your physician<br/>
        2. Maintain a healthy lifestyle and report any new symptoms<br/>
        3. Schedule follow-up scans if recommended by your healthcare provider<br/>
        4. Keep a record of this scan for future medical reference<br/><br/>

        <b>Note:</b> While no tumor was detected, regular medical check-ups remain important 
        for overall health maintenance.
        """

    elements.append(Paragraph(recommendations, body_style))
    elements.append(Spacer(1, 0.3 * inch))

    # ============ ADDITIONAL NOTES ============
    if patient_info.get('notes'):
        elements.append(Paragraph("Additional Notes", heading_style))
        elements.append(Paragraph(patient_info['notes'], body_style))
        elements.append(Spacer(1, 0.3 * inch))

    # ============ DISCLAIMER ============
    elements.append(PageBreak())
    elements.append(Paragraph("Important Disclaimer", heading_style))

    disclaimer_text = """
    This report is generated by an AI-assisted diagnostic system (NeuroScan) and is intended 
    to support medical professionals in their diagnostic process. <b>This is NOT a definitive 
    medical diagnosis.</b><br/><br/>

    <b>Please note:</b><br/>
    • This AI system is a screening tool and should not replace professional medical judgment<br/>
    • All results must be reviewed and confirmed by qualified medical professionals<br/>
    • The accuracy of AI predictions depends on image quality and may vary<br/>
    • False positives and false negatives are possible with any diagnostic tool<br/>
    • This report should be used in conjunction with clinical examination and other diagnostic tests<br/><br/>

    <b>For Medical Professionals:</b><br/>
    This system uses a Convolutional Neural Network (CNN) based on VGG19 architecture, 
    trained on MRI brain scans. Model performance metrics: Training Accuracy: 94.23%, 
    Test Accuracy: 91.61%.<br/><br/>

    Always correlate AI findings with clinical presentation, patient history, and additional 
    diagnostic evidence before making treatment decisions.
    """

    elements.append(Paragraph(disclaimer_text, body_style))
    elements.append(Spacer(1, 0.5 * inch))

    # ============ SIGNATURE SECTION ============
    elements.append(Paragraph("Physician Review", heading_style))

    signature_data = [
        ['Reviewed By:', '_' * 50],
        ['', '(Physician Name & Signature)'],
        ['', ''],
        ['Date:', '_' * 50],
        ['', ''],
        ['Medical License No.:', '_' * 50],
    ]

    signature_table = Table(signature_data, colWidths=[2 * inch, 4 * inch])
    signature_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))

    elements.append(signature_table)

    # ============ BUILD PDF ============
    doc.build(elements, canvasmaker=NumberedCanvas)

    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    # Test the PDF generator
    test_data = {
        'prediction': 'glioma',
        'confidence': 87.5,
        'probabilities': {
            'glioma': 87.5,
            'meningioma': 8.3,
            'notumor': 2.1,
            'pituitary': 2.1
        }
    }

    test_patient = {
        'name': 'John Doe',
        'age': 45,
        'gender': 'Male',
        'patient_id': 'PT-2025-001',
        'scan_date': '2025-01-15',
        'notes': 'Patient reported recurring headaches for 3 weeks.'
    }

    # Create dummy image
    img = Image.new('RGB', (224, 224), color='gray')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    pdf_buffer = generate_pdf_report(test_data, test_patient, img_base64)

    with open('test_report.pdf', 'wb') as f:
        f.write(pdf_buffer.read())

    print("Test PDF generated: test_report.pdf")