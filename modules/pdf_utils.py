from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def df_to_pdf(df: pd.DataFrame, chart_fig: plt.Figure | None = None) -> bytes:
    """Convert dataframe and optional chart to PDF bytes."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        if chart_fig is not None:
            pdf.savefig(chart_fig)
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
