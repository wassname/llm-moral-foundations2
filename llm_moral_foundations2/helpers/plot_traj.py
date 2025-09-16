
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import html
from IPython.display import display, HTML
from cmap import Colormap
import pandas as pd

# choose a map which is dark in the middle
cmap_RdGyGn = Colormap(['red', 'grey', 'green']).to_mpl()


def display_rating_trace(df: pd.DataFrame, title="", key='act_prob', cmap=cmap_RdGyGn, symmetric = False, highlight_tokens=[], s_key="token_strs"):
    """
    Display colored tokens from a chain of thought.

    Args:
    - key: the dataframe columns with the single score to visualise
    """
    # v = df[key].abs().max()
    v = df[key]  
    if symmetric:
        v = v.abs()
        # vmax = v.quantile(0.99)
        vmax = max(v)
        norm = mpl.colors.CenteredNorm(0, vmax)
    else:
        vmin = v.quantile(0.01)
        vmax = v.quantile(0.99)
        vmin = min(v)
        vmax = max(v)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # show colormap
    a = np.array([0, 1])[None]
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap=cmap, norm=norm)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    plt.colorbar(orientation="horizontal", cax=cax, label='rating')
    plt.title(f"Judge rating along chain of thought. {title}")
    plt.show()

    # TODO show \n as <p>
    htmls = f'<h3>{title}</h3>'
    for n,row in df.iterrows():
        token, score = row[s_key], row[key]
        token = token.replace('Ġ', ' ').replace('Ċ', '\n').replace("▁", " ")

        # html escape
        token = html.escape(token)
        # map score → RGBA → hex
        hex_color = mpl.colors.to_hex(cmap(norm(score)))
        if token.startswith("\n"):
            htmls += "<p/>"
        if token in highlight_tokens:
            token = f'!{token}!!'
        h = f'<span title="{score} n={n}" style="color: {hex_color};">{token} </span>'
        if token.endswith("\n"):
            htmls += "<p/>"
        htmls += h
    # render it inline
    display(HTML(htmls))


# display_rating_trace(df_traj.interpolate(method='nearest'))
# df_traj
