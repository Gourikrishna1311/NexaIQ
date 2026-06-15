import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os


def create_nexaiq_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis('off')
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    def draw_box(x, y, w, h, color, label, sublabel=''):
        box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor=color + '22',
                              edgecolor=color,
                              linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
                label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=color)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.2,
                    sublabel, ha='center', va='center',
                    fontsize=7, color='#aaaaaa')

    def draw_arrow(x1, y1, x2, y2, color='#444444'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->',
                                   color=color, lw=1.5))

    def draw_section_label(x, y, label, color):
        ax.text(x, y, label, fontsize=9, fontweight='bold',
                color=color, alpha=0.7)

    ax.text(9, 12.5, 'NexaIQ — Complete Platform Architecture',
            ha='center', va='center', fontsize=18,
            fontweight='bold', color='#2E86AB')
    ax.text(9, 12.1,
            'Explainable AI-Driven CRM Analytics with Integrated Threat Intelligence',
            ha='center', va='center', fontsize=10, color='#888888')

    draw_section_label(0.3, 11.2, 'DATA SOURCES', '#2E86AB')
    draw_box(0.3, 10.2, 2.8, 0.8, '#2E86AB',
             'IBM Telco Dataset', '7043 customers')
    draw_box(3.3, 10.2, 2.8, 0.8, '#2E86AB',
             'Superstore Sales', 'Revenue data')
    draw_box(6.3, 10.2, 2.8, 0.8, '#2E86AB',
             'Lead Dataset', 'B2B leads')
    draw_box(9.3, 10.2, 2.8, 0.8, '#2E86AB',
             'Access Logs', 'Security data')
    draw_box(12.3, 10.2, 2.8, 0.8, '#2E86AB',
             'Support Tickets', 'Sentiment data')

    draw_section_label(0.3, 9.5, 'ETL LAYER', '#F9C74F')
    draw_box(0.3, 8.5, 17, 0.8, '#F9C74F',
             'Automated ETL Pipeline — clean_data.py + etl_pipeline.py',
             'Extract → Clean → Transform → Load → Schedule')

    for x in [1.7, 4.7, 7.7, 10.7, 13.7]:
        draw_arrow(x, 10.2, x, 9.3, '#2E86AB')

    draw_section_label(0.3, 7.9, 'DATABASE', '#9B59B6')
    draw_box(0.3, 6.9, 17, 0.8, '#9B59B6',
             'PostgreSQL Database — nexaiq_db',
             'customers | sales_orders | leads | access_logs | support_tickets')

    draw_arrow(9, 8.5, 9, 7.7, '#F9C74F')

    draw_section_label(0.3, 6.3, 'INTELLIGENCE LAYER', '#3BB273')
    draw_box(0.3, 5.2, 4, 0.9, '#3BB273',
             'Churn Prediction', 'XGBoost — AUC 0.815')
    draw_box(4.6, 5.2, 4, 0.9, '#3BB273',
             'Sales Forecasting', 'GradientBoosting')
    draw_box(8.9, 5.2, 4, 0.9, '#3BB273',
             'Lead Scoring', 'RandomForest')
    draw_box(13.2, 5.2, 4.1, 0.9, '#E84855',
             'Anomaly Detection', 'Isolation Forest')

    for x in [2.3, 6.6, 10.9, 15.2]:
        draw_arrow(x, 6.9, x, 6.1, '#9B59B6')

    draw_section_label(0.3, 4.6, 'XAI LAYER', '#F39C12')
    draw_box(0.3, 3.6, 8.3, 0.8, '#F39C12',
             'SHAP Explainability — shap_explainer.py',
             'Feature attribution + plain English explanations')
    draw_box(8.9, 3.6, 8.4, 0.8, '#1ABC9C',
             'Sentiment Analysis — sentiment_analysis.py',
             'Ollama LLaMA 3.2 + ticket classification')

    for x in [2.3, 6.6]:
        draw_arrow(x, 5.2, x, 4.4, '#3BB273')
    for x in [10.9, 15.2]:
        draw_arrow(x, 5.2, x, 4.4, '#E84855')

    draw_section_label(0.3, 3.0, 'AI ASSISTANT', '#8E44AD')
    draw_box(0.3, 2.0, 17, 0.8, '#8E44AD',
             'NexaIQ AI Assistant — nexaiq_chat.py',
             'Ollama LLaMA 3.2 + SHAP-to-text + Conversational XAI + KPI Monitor')

    draw_arrow(9, 3.6, 9, 2.8, '#F39C12')

    draw_section_label(0.3, 1.4, 'SECURITY + DEPLOYMENT', '#E74C3C')
    draw_box(0.3, 0.3, 4, 0.9, '#E74C3C',
             'Auth + Security', 'JWT + bcrypt + OWASP')
    draw_box(4.6, 0.3, 4, 0.9, '#E74C3C',
             'AI Governance', 'Activity log + alerts')
    draw_box(8.9, 0.3, 4, 0.9, '#2980B9',
             'HTML Dashboard', 'GitHub Pages — live')
    draw_box(13.2, 0.3, 4.1, 0.9, '#27AE60',
             'Flask REST API', 'localhost:5000')

    for x in [2.3, 6.6, 10.9, 15.2]:
        draw_arrow(x, 2.0, x, 1.2, '#8E44AD')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/architecture_diagram.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0f1117')
    plt.show()
    print("Architecture diagram saved to outputs/architecture_diagram.png")
    print("Open outputs/architecture_diagram.png to see your full platform architecture")


if __name__ == "__main__":
    create_nexaiq_architecture()