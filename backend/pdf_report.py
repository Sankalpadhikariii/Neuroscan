from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import base64
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbers and footer"""

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
        """Add page number and footer"""
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor('#64748b'))

        # Page number
        page_num = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(
            letter[0] - 0.5 * inch,
            0.5 * inch,
            page_num
        )

        # Footer text
        footer_text = "NeuroScan Brain Tumor Detection System • Powered by AI & Deep Learning"
        self.drawCentredString(
            letter[0] / 2.0,
            0.3 * inch,
            footer_text
        )


def generate_pdf_report(scan_data, patient_data, hospital_data):
    """
    Generate a professional PDF report matching the provided template

    Args:
        scan_data: Dict with keys: prediction, confidence, probabilities, scan_id, timestamp, scan_image
        patient_data: Dict with keys: name, id (patient_code), email, phone, date_of_birth, gender
        hospital_data: Dict with keys: name, address, phone

    Returns:
        BytesIO buffer containing the PDF
    """
    try:
        buffer = io.BytesIO()

        # Create PDF with custom canvas for page numbers
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=1 * inch
        )

        # Story will hold all flowable elements
        story = []

        # Styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#475569'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica'
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=12,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            borderPadding=(0, 0, 8, 0),
            borderColor=colors.HexColor('#3b82f6'),
            borderWidth=0,
            leftIndent=0
        )

        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#334155'),
            spaceAfter=8,
            alignment=TA_LEFT
        )

        # ============================================
        # HEADER
        # ============================================
        story.append(Paragraph("NeuroScan Brain Tumor Detection", title_style))
        story.append(Paragraph("Report", subtitle_style))

        # Report metadata
        report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        metadata_text = f"<b>Report Generated:</b> {report_date}"
        story.append(Paragraph(metadata_text, body_style))
        story.append(Spacer(1, 0.3 * inch))

        # ============================================
        # HOSPITAL INFORMATION
        # ============================================
        story.append(Paragraph("Hospital Information", section_heading))

        hospital_info_data = [
            ['Hospital Name:', hospital_data.get('name', 'N/A')],
            ['Hospital Code:', hospital_data.get('code', 'N/A')],
            ['Doctor/Staff:', hospital_data.get('doctor_name', 'N/A')]
        ]

        hospital_table = Table(hospital_info_data, colWidths=[2 * inch, 4 * inch])
        hospital_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#334155')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))

        story.append(hospital_table)
        story.append(Spacer(1, 0.2 * inch))

        # ============================================
        # PATIENT INFORMATION
        # ============================================
        story.append(Paragraph("Patient Information", section_heading))

        patient_info_data = [
            ['Patient Name:', patient_data.get('name', 'N/A')],
            ['Patient Code:', patient_data.get('id', 'N/A')],
            ['Email:', patient_data.get('email', 'N/A')],
            ['Phone:', patient_data.get('phone', 'N/A')],
            ['Date of Birth:', patient_data.get('date_of_birth', 'N/A')],
            ['Gender:', patient_data.get('gender', 'N/A')]
        ]

        patient_table = Table(patient_info_data, colWidths=[2 * inch, 4 * inch])
        patient_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#334155')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))

        story.append(patient_table)
        story.append(Spacer(1, 0.2 * inch))

        # ============================================
        # SCAN INFORMATION
        # ============================================
        story.append(Paragraph("Scan Information", section_heading))

        scan_info_data = [
            ['Scan Date:', scan_data.get('timestamp', datetime.now().strftime('%B %d, %Y'))],
            ['Scan ID:', f"#{scan_data.get('scan_id', 'N/A')}"]
        ]

        scan_table = Table(scan_info_data, colWidths=[2 * inch, 4 * inch])
        scan_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#334155')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))

        story.append(scan_table)
        story.append(Spacer(1, 0.2 * inch))

        # ============================================
        # MRI SCAN IMAGE
        # ============================================
        story.append(Paragraph("MRI Scan Image", section_heading))

        if scan_data.get('scan_image'):
            try:
                # Decode base64 image
                image_data = base64.b64decode(scan_data['scan_image'])
                image = Image.open(io.BytesIO(image_data))

                # Save to temporary buffer
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                # Add to PDF (resize to fit)
                img = RLImage(img_buffer, width=4 * inch, height=4 * inch)
                story.append(img)
            except Exception as e:
                logger.error(f"Error adding scan image: {e}")
                story.append(Paragraph("Scan image unavailable", body_style))
        else:
            story.append(Paragraph("Scan image unavailable", body_style))

        story.append(Spacer(1, 0.3 * inch))

        # ============================================
        # ANALYSIS RESULTS
        # ============================================
        story.append(Paragraph("Analysis Results", section_heading))

        prediction = scan_data.get('prediction', 'Unknown').upper()
        confidence = scan_data.get('confidence', 0)

        # Ensure confidence is a percentage
        if confidence <= 1:
            confidence = confidence * 100

        result_style = ParagraphStyle(
            'ResultStyle',
            parent=styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#16a34a') if prediction == 'NOTUMOR' else colors.HexColor('#dc2626'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        story.append(Paragraph(f"{prediction}", result_style))
        story.append(Paragraph(f"Confidence: {confidence:.2f}%", body_style))
        story.append(Spacer(1, 0.2 * inch))

        # ============================================
        # PROBABILITY DISTRIBUTION TABLE
        # ============================================
        story.append(Paragraph("Probability Distribution", section_heading))

        probabilities = scan_data.get('probabilities', {})

        prob_data = [
            ['Tumor Type', 'Probability'],
            ['Glioma', f"{probabilities.get('glioma', 0):.2f}%"],
            ['Meningioma', f"{probabilities.get('meningioma', 0):.2f}%"],
            ['Notumor', f"{probabilities.get('notumor', 0):.2f}%"],
            ['Pituitary', f"{probabilities.get('pituitary', 0):.2f}%"]
        ]

        prob_table = Table(prob_data, colWidths=[3 * inch, 2 * inch])
        prob_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#334155')),

            # Grid
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),

            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))

        story.append(prob_table)
        story.append(Spacer(1, 0.3 * inch))

        # ============================================
        # MEDICAL RECOMMENDATIONS
        # ============================================
        story.append(Paragraph("Medical Recommendations", section_heading))

        is_tumor = prediction != 'NOTUMOR'

        if is_tumor:
            recommendations = f"""
            <b>Result:</b> {prediction.capitalize()} tumor detected in this scan with {confidence:.2f}% confidence. 
            The following recommendations apply:<br/><br/>
            • Immediate consultation with a neurosurgeon is strongly recommended<br/>
            • Additional imaging studies may be required for detailed assessment<br/>
            • Biopsy may be necessary to determine tumor grade and type<br/>
            • Treatment options should be discussed with oncology specialist<br/>
            • Family history and genetic factors should be evaluated<br/><br/>
            <b>Note:</b> This result is based on AI analysis and must be confirmed by a qualified physician.
            Early detection and proper medical intervention are crucial for better outcomes.
            """
        else:
            recommendations = f"""
            <b>Result:</b> No tumor detected in this scan. However, the following recommendations apply:<br/><br/>
            • Regular monitoring if patient has symptoms or risk factors<br/>
            • Follow-up scan as per physician's recommendation<br/>
            • Maintain healthy lifestyle and report any new symptoms<br/>
            • Consult physician for any concerns or questions<br/><br/>
            <b>Note:</b> This result is based on AI analysis and should be confirmed by a qualified physician.
            """

        story.append(Paragraph(recommendations, body_style))
        story.append(Spacer(1, 0.3 * inch))

        # ============================================
        # DISCLAIMER
        # ============================================
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#64748b'),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            borderWidth=1,
            borderColor=colors.HexColor('#cbd5e1'),
            borderPadding=10,
            backColor=colors.HexColor('#f8fafc')
        )

        disclaimer_text = """
        <b>MEDICAL DISCLAIMER:</b> This report is generated by an AI-powered diagnostic system and should be used as a 
        supplementary tool only. All results must be reviewed and confirmed by a qualified medical professional. This 
        report does not constitute medical advice and should not be used as the sole basis for medical decisions.
        """

        story.append(Paragraph(disclaimer_text, disclaimer_style))

        # ============================================
        # FOOTER INFO
        # ============================================
        footer_info_style = ParagraphStyle(
            'FooterInfo',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=TA_CENTER,
            spaceAfter=0
        )

        story.append(Spacer(1, 0.2 * inch))
        footer_text = f"Report ID: {scan_data.get('scan_id', 'N/A')} • Generated: {report_date}"
        story.append(Paragraph(footer_text, footer_info_style))

        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)

        buffer.seek(0)
        logger.info(f"✅ PDF report generated successfully for scan {scan_data.get('scan_id')}")
        return buffer

    except Exception as e:
        logger.error(f"❌ PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise