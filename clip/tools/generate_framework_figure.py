#!/usr/bin/env python3
"""
Generate framework diagram for CLIP Multimodal VAD Pipeline.
Outputs: framework_diagram.pdf (vector format suitable for papers)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set figure size and style
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
colors = {
    'data': '#E8E8E8',
    'lexicon': '#FFE5B4',
    'encoder': '#B3D9FF',
    'regressor': '#B3FFB3',
    'loss': '#FFB3B3',
    'output': '#E5B3FF',
    'border_data': '#999999',
    'border_lexicon': '#FF8C00',
    'border_encoder': '#0066CC',
    'border_regressor': '#00CC00',
    'border_loss': '#CC0000',
    'border_output': '#9900CC',
}

def draw_box(ax, x, y, w, h, text, color, border_color, style='normal'):
    """Draw a rounded rectangle box with text."""
    if style == 'normal':
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor=border_color, linewidth=2)
    else:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')

def draw_arrow(ax, x1, y1, x2, y2, style='->', color='black', lw=1.5):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, linewidth=lw,
                           connectionstyle="arc3,rad=0.1" if abs(x2-x1) > 2 else "arc3,rad=0")
    ax.add_patch(arrow)

# Layer 1: Input Data
draw_box(ax, 1.5, 7.5, 1.8, 1.0, 'Image\nI', colors['data'], colors['border_data'])
draw_box(ax, 1.5, 5.5, 1.8, 1.0, 'Text\nT', colors['data'], colors['border_data'])
draw_box(ax, 4.5, 5.5, 1.8, 1.0, 'NRC-VAD\nLexicon', colors['lexicon'], colors['border_lexicon'])

# Layer 2: Pseudo-label Generation
draw_box(ax, 7.5, 5.5, 2.5, 1.0, 'Text Pseudo-label\nGeneration\ny_text in R^3', 
         colors['lexicon'], colors['border_lexicon'])

# Layer 3: Text Branch
draw_box(ax, 10.5, 7.5, 2.2, 1.2, 'CLIP\nText Encoder\nf_text(·)', 
         colors['encoder'], colors['border_encoder'])
draw_box(ax, 13.0, 7.5, 1.8, 1.0, 'Text\nRegressor\ng_text(·)', 
         colors['regressor'], colors['border_regressor'])

# Layer 4: Image Branch
draw_box(ax, 10.5, 4.5, 2.2, 1.2, 'CLIP\nImage Encoder\nf_img(·)', 
         colors['encoder'], colors['border_encoder'])
draw_box(ax, 13.0, 4.5, 1.8, 1.0, 'Image\nRegressor\ng_img(·)', 
         colors['regressor'], colors['border_regressor'])

# Layer 5: Outputs
draw_box(ax, 15.5, 7.5, 1.8, 1.0, 'ŷ_text\n[V, A, D]', 
         colors['output'], colors['border_output'])
draw_box(ax, 15.5, 4.5, 1.8, 1.0, 'ŷ_img\n[V, A, D]', 
         colors['output'], colors['border_output'])

# Layer 6: Loss Functions
draw_box(ax, 18.0, 8.0, 1.8, 0.8, 'L_text\nMSE', 
         colors['loss'], colors['border_loss'])
draw_box(ax, 18.0, 6.5, 1.8, 0.8, 'L_img\nMSE\n(λ·)', 
         colors['loss'], colors['border_loss'])
draw_box(ax, 18.0, 4.8, 2.2, 1.0, 'L_total\n= L_text + λL_img', 
         colors['loss'], colors['border_loss'])

# Arrows: Data flow
draw_arrow(ax, 2.4, 5.5, 6.2, 5.5, color='black')  # Text -> Pseudo
draw_arrow(ax, 5.4, 5.5, 6.2, 5.5, color='orange', lw=2)  # Lexicon -> Pseudo
draw_arrow(ax, 2.4, 7.5, 9.4, 7.5, color='black')  # Image -> Text Encoder (via path)
draw_arrow(ax, 2.4, 5.5, 9.4, 7.5, color='black')  # Text -> Text Encoder
draw_arrow(ax, 2.4, 7.5, 9.4, 4.5, color='black')  # Image -> Image Encoder

# Arrows: Encoder to Regressor
draw_arrow(ax, 11.6, 7.5, 12.1, 7.5, color='blue', lw=2)
draw_arrow(ax, 11.6, 4.5, 12.1, 4.5, color='blue', lw=2)

# Arrows: Regressor to Output
draw_arrow(ax, 13.9, 7.5, 14.6, 7.5, color='green', lw=2)
draw_arrow(ax, 13.9, 4.5, 14.6, 4.5, color='green', lw=2)

# Arrows: Pseudo-label to Loss
draw_arrow(ax, 8.75, 6.0, 17.1, 7.6, color='orange', lw=1.5, style='->')
draw_arrow(ax, 8.75, 5.0, 17.1, 6.9, color='orange', lw=1.5, style='->')

# Arrows: Output to Loss
draw_arrow(ax, 16.4, 7.5, 17.1, 7.6, color='purple', lw=1.5)
draw_arrow(ax, 16.4, 4.5, 17.1, 6.9, color='purple', lw=1.5)

# Arrows: Loss to Total
draw_arrow(ax, 18.9, 8.0, 18.9, 5.3, color='red', lw=2)
draw_arrow(ax, 18.9, 6.9, 18.9, 5.3, color='red', lw=2)

# Annotations
ax.text(10.5, 8.8, 'Pre-trained', ha='center', fontsize=8, style='italic', color='blue')
ax.text(10.5, 3.7, 'Pre-trained', ha='center', fontsize=8, style='italic', color='blue')
ax.text(7.5, 6.8, 'Coverage ≈ 52%', ha='center', fontsize=8, style='italic', color='orange')

# Title
ax.text(10.0, 9.5, 'Joint Multimodal VAD Training Framework', 
        ha='center', fontsize=14, weight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=colors['data'], edgecolor=colors['border_data'], label='Data Input'),
    mpatches.Patch(facecolor=colors['lexicon'], edgecolor=colors['border_lexicon'], label='Lexicon/Pseudo-label'),
    mpatches.Patch(facecolor=colors['encoder'], edgecolor=colors['border_encoder'], label='CLIP Encoder'),
    mpatches.Patch(facecolor=colors['regressor'], edgecolor=colors['border_regressor'], label='Regression Head'),
    mpatches.Patch(facecolor=colors['output'], edgecolor=colors['border_output'], label='VAD Output'),
    mpatches.Patch(facecolor=colors['loss'], edgecolor=colors['border_loss'], label='Loss Function'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('framework_diagram.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('framework_diagram.png', format='png', bbox_inches='tight', dpi=300)
print("Framework diagram saved as framework_diagram.pdf and framework_diagram.png")

