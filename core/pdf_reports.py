import io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from fpdf import FPDF
from datetime import datetime

class AuditReport(FPDF):
    def header(self):
        # Logo or Title
        self.set_font("helvetica", "B", 16)
        self.set_text_color(0, 128, 128) # Teal color
        self.cell(0, 10, "Price Verification Audit Report", border=False, ln=True, align="C")
        self.set_font("helvetica", "I", 10)
        self.set_text_color(128, 128, 128)
        
        # Use provided time or fallback to server time
        report_time = getattr(self, "report_time", datetime.now().astimezone())
        current_time = report_time.strftime('%B %d, %Y, %I:%M:%S %p (GMT%z)')
        self.cell(0, 10, f"Generated on: {current_time}", border=False, ln=True, align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Price Verification Tool", align="C")

def _create_importance_chart(feature_importance):
    """
    Creates a horizontal bar chart of feature importance and returns it as a BytesIO object.
    """
    # Sort features by absolute contribution
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    # Take top 10 for clarity in PDF if many
    top_features = sorted_features[:10]
    
    names = [x[0] for x in top_features]
    values = [x[1] for x in top_features]
    colors = ['#EF553B' if v > 0 else '#00CC96' for v in values] # Red for UP, Green for DOWN
    
    plt.figure(figsize=(8, 5))
    plt.barh(names, values, color=colors)
    plt.xlabel('Price Contribution')
    plt.title('Prediction Insights (Top Drivers)')
    plt.gca().invert_yaxis() # Highest impact on top
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    plt.close()
    img_buf.seek(0)
    return img_buf

def generate_prediction_pdf(result, input_data, goods_code, goods_description, report_time=None):
    """
    Generates a PDF bytes object for a single prediction.
    """
    pdf = AuditReport()
    if report_time:
        pdf.report_time = report_time
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Section: Assessment Summary ---
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Assessment Summary", ln=True)
    pdf.ln(2)

    pdf.set_font("helvetica", "", 11)
    # Highlight results
    predicted_price = result.get('predicted_price', 0)
    lower = result.get('lower_bound', 0)
    upper = result.get('upper_bound', 0)
    currency = input_data.get('CURRENCY', 'USD')

    pdf.cell(60, 10, "Predicted Unit Price:", border=0)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 10, f"{currency} {predicted_price:,.2f}", ln=True)
    
    pdf.set_font("helvetica", "", 11)
    pdf.cell(60, 10, "Expected Range:", border=0)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 10, f"{currency} {lower:,.2f} - {upper:,.2f}", ln=True)
    pdf.ln(5)

    # --- Section: Input Details ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Input Details", ln=True)
    pdf.ln(2)

    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 8, "Feature", 1, 0, "C", fill=True)
    pdf.cell(130, 8, "Value", 1, 1, "C", fill=True)
    
    pdf.set_font("helvetica", "", 9)
    # Manual fields for better presentation
    details = {
        "HS Code": f"{goods_code} - {goods_description}",
        "Exporter": input_data.get("EXPORTER"),
        "Exporter's Country": input_data.get("EXPORTER'S COUNTRY"),
        "Importer": input_data.get("IMPORTER"),
        "Country of Origin": input_data.get("COUNTRY_OF_ORIGIN"),
        "Shipment From": input_data.get("SHIPMENT FROM"),
        "Shipment To": input_data.get("SHIPMENT TO"),
        "Trade Year": str(input_data.get("YEAR")),
        "Incoterm": input_data.get("TRADE-TERM"),
        "Quantity": f"{input_data.get('QUANTITY'):,.2f}",
        "Tenor": str(input_data.get("TENOR OF PAYMENT")),
        "Freight": f"{input_data.get('FREIGHT CHARGES'):,.2f}"
    }

    for key, val in details.items():
        pdf.cell(60, 7, key, 1)
        pdf.cell(130, 7, str(val), 1, 1)

    # --- Section: Prediction Insights (Visual) ---
    feature_importance = result.get("feature_importance", {})
    if feature_importance:
        pdf.add_page() # Start insights on a new page as requested
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "3. Prediction Insights", ln=True)
        pdf.ln(2)
        pdf.set_font("helvetica", "I", 9)
        pdf.multi_cell(0, 5, "The following chart visualizes how various factors influenced the final predicted price. "
                            "Red bars indicate factors that increased the price, while green bars indicate those that decreased it.")
        pdf.ln(2)

        # Generate and embed chart
        chart_buf = _create_importance_chart(feature_importance)
        # We use a temporary file-like object approach that fpdf2 supports
        # Position the image (center roughly)
        pdf.image(chart_buf, x=15, w=180)

    # --- Section: Legal Disclaimer ---
    pdf.ln(10)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "Disclaimer: This report is generated by an automated Machine Learning model for assessment support. "
                        "The predicted values are estimates based on historical trade data and should be verified "
                        "against real-market conditions by authorized personnel.")

    return bytes(pdf.output())
