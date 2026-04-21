import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import math
import textwrap
from itertools import product
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import pandas as pd


class Thresholds:
    """
    Container for thresholds, supporting numeric (with optional units) 
    and categorical endpoints.
    """

    def __init__(self):
        self._store = {}        # numeric thresholds: column -> value / list / dict of (value, unit)
        self._categorical = {}  # categorical endpoints
        self._cat_colors = {}   # optional colors: column -> dict(category -> color)


    # ----- Numeric thresholds -----
    def add(self, column, threshold=None, name=None, unit=None):
        """
        Add numeric threshold(s) for a column.

        Parameters
        ----------
        column : str
            Column name
        threshold : float, list, tuple, or dict
            Numeric threshold(s)
        name : str, optional
            Name for a single threshold
        unit : str, optional
            Unit of the threshold (e.g., "mg/L")
        """
        def wrap_unit(val):
            return (val, unit) if unit else val

        if name and not isinstance(threshold, (int, float)):
            raise ValueError("Named thresholds must be a single float.")

        if isinstance(threshold, (int, float)):
            if name:
                if column not in self._store or not isinstance(self._store[column], dict):
                    self._store[column] = {}
                self._store[column][name] = wrap_unit(threshold)
            else:
                self._store[column] = wrap_unit(threshold)
        elif isinstance(threshold, (list, tuple)):
            self._store[column] = [wrap_unit(v) for v in threshold]
        elif isinstance(threshold, dict):
            self._store[column] = {k: wrap_unit(v) for k, v in threshold.items()}
        else:
            raise TypeError("Threshold must be float, list, or dict.")

    # ----- Categorical thresholds -----
    def add_categories(self, column, categories, colors=None):
        """
        Add categorical endpoint with optional colors.

        Parameters
        ----------
        column : str
            Column name
        categories : list of str
            List of category names
        colors : list or dict, optional
            Colors for categories. If list, order must match categories.
        """
        if not isinstance(categories, (list, tuple, set)):
            raise TypeError("Categories must be a list, tuple, or set")
        self._categorical[column] = list(categories)

        if colors is not None:
            if isinstance(colors, (list, tuple)):
                if len(colors) != len(categories):
                    raise ValueError("Length of colors must match categories")
                self._cat_colors[column] = {cat: col for cat, col in zip(categories, colors)}
            elif isinstance(colors, dict):
                self._cat_colors[column] = dict(colors)
            else:
                raise TypeError("Colors must be a list, tuple, or dict")

    # ----- Access categorical colors -----
    def colors_for_column(self, column):
        return self._cat_colors.get(column, None)
    
    # ----- Access methods -----
    def numeric_columns(self):
        return list(self._store.keys())

    def categorical_columns(self):
        return list(self._categorical.keys())

    def numeric_for_column(self, column):
        return self._store.get(column)

    def categories_for_column(self, column):
        return self._categorical.get(column)

    # ----- Plotting method for numeric thresholds -----
    def draw(self, ax, column, color="red", linestyle="--", linewidth=1.5):
        thr_values = self._store.get(column)
        if thr_values is None:
            return

        def format_label(val):
            if isinstance(val, tuple):
                v, u = val
                return f"{v} {u}" if u else str(v)
            return str(val)

        if isinstance(thr_values, (int, float, tuple)):
            ax.axhline(thr_values[0] if isinstance(thr_values, tuple) else thr_values,
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       label=f"Threshold = {format_label(thr_values)}")
        elif isinstance(thr_values, (list, tuple)):
            for i, thr in enumerate(thr_values, 1):
                ax.axhline(thr[0] if isinstance(thr, tuple) else thr,
                           color=color, linestyle=linestyle, linewidth=linewidth,
                           label=f"Threshold {i} = {format_label(thr)}")
        elif isinstance(thr_values, dict):
            for name, thr in thr_values.items():
                ax.axhline(thr[0] if isinstance(thr, tuple) else thr,
                           color=color, linestyle=linestyle, linewidth=linewidth,
                           label=f"{name} = {format_label(thr)}")


def wrap_labels(label, width=15):
    if isinstance(label, str):
        return "\n".join(textwrap.wrap(label, width=width))
    return label


def plot_violin_grouped_faceted(df, value_columns, group_column, title,
                                              pngfile=None, thresholds: Thresholds = None,
                                              envelope_alpha=0.15, median_alpha=0.35):
    """
    Classic faceted violin plots with min/max conformal envelope per group.
    """

    ncols = 2
    nrows = (len(value_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows), squeeze=False)

    for idx, value_column in enumerate(value_columns):
        ax = axes[idx // ncols, idx % ncols]

        # Clean data
        df_clean = df.dropna(subset=[value_column, group_column])
        grouped = df_clean.groupby(group_column)[value_column]
        group_labels, data = zip(*[(name.replace("_","\n"), group.values) for name, group in grouped])

        # Violin plot
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#9999ff')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.5)

        # Conformal envelope: min/max per group
        for i, g in enumerate(df_clean[group_column].unique()):
            sub = df_clean[df_clean[group_column] == g]

            # Pick _lower and _upper columns if they exist, else fallback to main column
            lo_col = f"{value_column}_lower" if f"{value_column}_lower" in df_clean.columns else value_column
            up_col = f"{value_column}_upper" if f"{value_column}_upper" in df_clean.columns else value_column

            # Combine all values to get true vertical min/max
            combined_vals = np.concatenate([sub[lo_col].values, sub[up_col].values])
            lo_min = np.nanmin(combined_vals)
            up_max = np.nanmax(combined_vals)

            # Use the violin width for horizontal span
            violin_width = 0.3  # match your violin width
            ax.fill_between([i+1-violin_width, i+1+violin_width], lo_min, up_max,
                            color='orange', alpha=envelope_alpha, zorder=1)


        # Labels and formatting
        ax.set_xticks(range(1, len(group_labels)+1))
        ax.set_xticklabels(group_labels,  ha='right')
        ax.set_xlabel(group_column)
        ax.set_ylabel(value_column)
        ax.set_title(value_column)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Optional thresholds (if available)
        if thresholds is not None:
            thresholds.draw(ax, value_column)

    # Remove unused axes
    for j in range(len(value_columns), nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if pngfile:
        plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_mosaic_faceted(df, cat_columns, group_column, thresholds=None,
                        title=None, pngfile=None):

    ncols = min(2, len(cat_columns))
    nrows = math.ceil(len(cat_columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3*nrows))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif hasattr(axes, "ravel"):
        axes = axes.ravel()
    else:
        axes = list(axes)

    for i, col in enumerate(cat_columns):
        ax = axes[i]

        groups = df[group_column].dropna().unique()
        categories = df[col].dropna().unique()

        raw_counts = df.groupby([group_column, col]).size().to_dict()

        # Fill missing with 0
        for g, c in product(groups, categories):
            raw_counts.setdefault((g, c), 0)

        # Remove zero-count configs
        counts_nonzero = {k: v for k, v in raw_counts.items() if v > 0}
        if len(counts_nonzero) == 0:
            ax.axis("off")
            continue

        # --- TRANSPOSE HERE ---
        # swap keys from (group, category) → (category, group)
        transposed = {(c, g): v for (g, c), v in counts_nonzero.items()}
        # ------------------------

        # Colors
        cat_colors = None
        if thresholds and col in thresholds.categorical_columns():
            cat_colors = thresholds.colors_for_column(col)
            if cat_colors is None:
                categories = thresholds.categories_for_column(col)
                palette = plt.cm.get_cmap("tab10", len(categories))
                cat_colors = {cat: palette(idx) for idx, cat in enumerate(categories)}

        def props(key):
            category, _ = key   # because we transposed
            color = cat_colors.get(category, "lightgrey") if cat_colors else "lightgrey"
            return dict(facecolor=color, edgecolor="black")

        # mosaic with horizontal split first
        _, rects = mosaic(transposed, properties=props, gap=0.01,
                          ax=ax, labelizer=lambda k: "")

        ax.set_title(col)

        # --- X TICKS: categories ---
        xticks, xticklabels = [], []
        for c in categories:
            rects_c = [r for (cat, _), r in rects.items() if cat == c]
            if rects_c:
                xmin = min(r[0] for r in rects_c)
                xmax = max(r[0] + r[2] for r in rects_c)
                xticks.append((xmin + xmax) / 2)
                xticklabels.append(str(c))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=9)

        # --- Y TICKS: materials/groups ---
        yticks, yticklabels = [], []
        for g in groups:
            rects_g = [r for (_, grp), r in rects.items() if grp == g]
            if rects_g:
                ymin = min(r[1] for r in rects_g)
                ymax = max(r[1] + r[3] for r in rects_g)
                yticks.append((ymin + ymax) / 2)
                yticklabels.append(str(g))

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=9)

        # Legend
        if cat_colors:
            handles = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor="black")
                       for color in cat_colors.values()]
            ax.legend(handles, list(cat_colors.keys()),
                      title=col, fontsize=8,
                      loc='center left', bbox_to_anchor=(1, 0.5))

    # Turn off empty axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    if pngfile:
        plt.savefig(pngfile, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


from itertools import product

import pandas as pd
from itertools import product

def get_mosaic_fractions(df, cat_columns, group_column):
    """
    Compute normalized fractions for each (group × column) combination.
    Fractions for each material–endpoint pair sum to 1.

    Returns tidy dataframe:
        column | group | category | count | fraction
    """

    results = []

    for col in cat_columns:
        groups = df[group_column].dropna().unique()
        categories = df[col].dropna().unique()

        # observed counts
        counts = df.groupby([group_column, col]).size().to_dict()

        # ensure all groups × categories exist
        for g, c in product(groups, categories):
            counts.setdefault((g, c), 0)

        # normalize *within each group*
        for g in groups:
            # total count for this material + endpoint
            total_g = sum(counts[(g, c)] for c in categories)

            # if a material has no observations, skip
            if total_g == 0:
                for c in categories:
                    results.append({
                        "column": col,
                        "group": g,
                        "category": c,
                        "count": 0,
                        "fraction": 0.0
                    })
                continue

            # compute normalized fractions
            for c in categories:
                n = counts[(g, c)]
                results.append({
                    "property": col,
                    "tag": g,
                    "category": c,
                    "count": n,
                    "fraction": n / total_g
                })

    return pd.DataFrame(results)


def compute_kde_or_constant(vals, y, width):
    if np.all(vals == vals[0]):  # constant data
        return np.zeros_like(y) + width  # use constant width for plotting
    else:
        kde = gaussian_kde(vals)(y)
        return kde / kde.max() * width


def plot_violin_grouped_halves_with_preds(
        df, value_columns, group_column, title, pngfile=None,
        thresholds=None,
        envelope_alpha=0.15):
    """
    Half-violin plot per group/material with:
    - left half: <base>_lower
    - right half: <base>_upper
    - predicted values as black dots + full predicted violin
    - thin central band showing min/max across lower/upper
    - visual brackets connecting half-violins and predicted violin per material
    - intuitive x-axis labels for lower/upper/predicted
    - color legend
    """
    ncols = 2
    nrows = (len(value_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows), squeeze=False)

    for idx, base in enumerate(value_columns):
        ax = axes[idx // ncols, idx % ncols]

        pred_col  = base
        lower_col = f"{base}_lower"
        upper_col = f"{base}_upper"

        missing = [c for c in [pred_col, lower_col, upper_col] if c not in df.columns]
        if missing:
            ax.text(0.5, 0.5, f"Missing columns:\n{missing}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df_clean = df.dropna(subset=[pred_col, lower_col, upper_col, group_column])
        groups = sorted(df_clean[group_column].unique())

        spacing = 0.3
        pred_spacing = 0.25

        positions_half = np.arange(1, 2*len(groups)+1, 2)  # 1,3,5,... half-violins
        positions_pred = positions_half + 1                  # 2,4,6,... predicted violins

        for i, g in enumerate(groups):
            sub = df_clean[df_clean[group_column] == g]
            lo_vals = sub[lower_col].values
            up_vals = sub[upper_col].values
            pred_vals = sub[pred_col].values

            # KDE
            y_min = min(lo_vals.min(), up_vals.min(), pred_vals.min())
            y_max = max(lo_vals.max(), up_vals.max(), pred_vals.max())
            y = np.linspace(y_min, y_max, 200)

            lo_kde = compute_kde_or_constant(lo_vals, y, spacing)
            up_kde = compute_kde_or_constant(up_vals, y, spacing)
            pred_kde = compute_kde_or_constant(pred_vals, y, pred_spacing)

            lo_kde = lo_kde / lo_kde.max() * spacing
            up_kde = up_kde / up_kde.max() * spacing
            pred_kde = pred_kde / pred_kde.max() * pred_spacing

            # Half-violins
            x_center = positions_half[i]
            ax.fill_betweenx(y, x_center - lo_kde, x_center, facecolor='#FF9999', alpha=0.6, edgecolor='black')
            ax.fill_betweenx(y, x_center, x_center + up_kde, facecolor='#9999FF', alpha=0.6, edgecolor='black')
            ax.plot([x_center - 0.1, x_center], [np.median(lo_vals)]*2, color='darkred', lw=1.5)
            ax.plot([x_center, x_center + 0.1], [np.median(up_vals)]*2, color='darkblue', lw=1.5)

            # Predicted full violin
            x_pred = positions_pred[i]
            ax.fill_betweenx(y, x_pred - pred_kde, x_pred + pred_kde, facecolor='gray', alpha=0.4, edgecolor='black')
            ax.plot([x_pred - 0.05, x_pred + 0.05], [np.median(pred_vals)]*2, color='black', lw=1.5)
            jitter = (np.random.rand(len(pred_vals)) - 0.5) * 0.1
            ax.scatter(np.full_like(pred_vals, x_pred) + jitter, pred_vals, color='black', s=1, alpha=0.5, zorder=3)

            # Conformal envelope
            ax.fill_between([x_center - 0.05, x_center + 0.05], lo_vals.min(), up_vals.max(),
                            color='orange', alpha=envelope_alpha, zorder=1)

            # Bracket for the group
            bracket_height = y_max + 0.05*(y_max - y_min)
            ax.plot([x_center - 0.3, x_pred + 0.3], [bracket_height]*2, color='black', lw=1)
            ax.text((x_center + x_pred)/2, bracket_height + 0.02*(y_max - y_min), 
                    g.replace("_","\n"), ha='center', va='bottom', fontsize=10, clip_on=False)

            # Sub-labels for positions
            label_offset = 0.03*(y_max - y_min)
            ax.text(x_center - 0.25, y_min - label_offset, 'lower', ha='center', va='top', fontsize=8)
            ax.text(x_center + 0.25, y_min - label_offset, 'upper', ha='center', va='top', fontsize=8)
            ax.text(x_pred, y_min - label_offset, 'pred', ha='center', va='top', fontsize=8)

        # Formatting
        ax.set_xlim(0, 2*len(groups)+1)
        ax.set_ylim(y_min - 0.1*(y_max - y_min), bracket_height + 0.1*(y_max - y_min))
        ax.set_xticks([])  # remove old x-ticks
        ax.set_xlabel(group_column)
        ax.set_ylabel(base)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF9999', edgecolor='black', label='lower'),
        Patch(facecolor='#9999FF', edgecolor='black', label='upper'),
        Patch(facecolor='gray', edgecolor='black', label='predicted'),
        Patch(facecolor='orange', edgecolor='orange', alpha=0.3, label='envelope')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Remove unused axes
    for j in range(len(value_columns), nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    if pngfile:
        plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def compute_kde_or_constant(vals, x, width):
    """Return KDE for vals; if constant, return a constant width array"""
    if np.all(vals == vals[0]):
        return np.zeros_like(x) + width
    else:
        kde = gaussian_kde(vals)(x)
        return kde / kde.max() * width


def plot_horizontal_violin_with_preds(
        df, value_columns, group_column, title, pngfile=None,
        thresholds=None, envelope_alpha=0.15):
    """
    Horizontal half-violin plot with compact spacing:
    - Materials/groups on y-axis labeled "Material"
    - Left half = lower, Right half = upper
    - Full predicted violin + predicted points
    - Sub-labels (lower/upper/pred) on right side of each violin
    - Legend above figure
    - thresholds parameter included for future use
    """
    ncols = 2
    nrows = (len(value_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), squeeze=False)

    for idx, base in enumerate(value_columns):
        ax = axes[idx // ncols, idx % ncols]

        pred_col  = base
        lower_col = f"{base}_lower"
        upper_col = f"{base}_upper"

        missing = [c for c in [pred_col, lower_col, upper_col] if c not in df.columns]
        if missing:
            ax.text(0.5, 0.5, f"Missing columns:\n{missing}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df_clean = df.dropna(subset=[pred_col, lower_col, upper_col, group_column])
        groups = sorted(df_clean[group_column].unique())
        n_groups = len(groups)

        spacing = 0.3
        pred_spacing = 0.25

        # Tighter y positions
        y_gap = 1.5
        positions_half = np.arange(n_groups) * y_gap + 0.5
        positions_pred = positions_half + 0.8

        for i, g in enumerate(groups):
            sub = df_clean[df_clean[group_column] == g]
            lo_vals = sub[lower_col].values
            up_vals = sub[upper_col].values
            pred_vals = sub[pred_col].values

            # x-axis range
            x_min = min(lo_vals.min(), up_vals.min(), pred_vals.min())
            x_max = max(lo_vals.max(), up_vals.max(), pred_vals.max())
            x = np.linspace(x_min, x_max, 200)

            kde_lo = compute_kde_or_constant(lo_vals, x, spacing)
            kde_up = compute_kde_or_constant(up_vals, x, spacing)
            kde_pred = compute_kde_or_constant(pred_vals, x, pred_spacing)

            # Half-violins
            y_center = positions_half[i]
            ax.fill_between(x, y_center - kde_lo, y_center, facecolor='#FF9999', alpha=0.6, edgecolor='black')
            ax.fill_between(x, y_center, y_center + kde_up, facecolor='#9999FF', alpha=0.6, edgecolor='black')
            ax.plot([np.median(lo_vals)]*2, [y_center - 0.1, y_center], color='darkred', lw=1.5)
            ax.plot([np.median(up_vals)]*2, [y_center, y_center + 0.1], color='darkblue', lw=1.5)

            # Predicted violin
            y_pred = positions_pred[i]
            ax.fill_between(x, y_pred - kde_pred, y_pred + kde_pred, facecolor='gray', alpha=0.4, edgecolor='black')
            ax.plot([np.median(pred_vals)]*2, [y_pred - 0.05, y_pred + 0.05], color='black', lw=1.5)
            jitter = (np.random.rand(len(pred_vals)) - 0.5) * 0.05
            ax.scatter(pred_vals, y_pred + jitter, color='black', s=1, alpha=0.5, zorder=3)

            # Conformal envelope
            ax.fill_between([lo_vals.min(), up_vals.max()], y_center - 0.05, y_center + 0.05,
                            color='orange', alpha=envelope_alpha, zorder=1)

            # Sub-labels on right
            #label_offset = 0.02*(x_max - x_min)
            #ax.text(x_max + label_offset, y_center - 0.25, 'lower', ha='left', va='center', fontsize=8)
            #ax.text(x_max + label_offset, y_center + 0.25, 'upper', ha='left', va='center', fontsize=8)
            #ax.text(x_max + label_offset, y_pred, 'pred', ha='left', va='center', fontsize=8)

        # Formatting
        ax.set_yticks((positions_half + positions_pred)/2)
        ax.set_yticklabels(groups)
        ax.set_ylim(0, positions_pred[-1] + 1)
        ax.set_xlabel(base)
        ax.set_ylabel("Material")
        ax.grid(True, linestyle='--', alpha=0.3)

    # Legend above figure
    legend_elements = [
        Patch(facecolor='#FF9999', edgecolor='black', label='lower'),
        Patch(facecolor='#9999FF', edgecolor='black', label='upper'),
        Patch(facecolor='gray', edgecolor='black', label='predicted'),
        #Patch(facecolor='orange', edgecolor='orange', alpha=0.3, label='envelope')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    # Remove unused axes
    for j in range(len(value_columns), nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])  # leave space for legend
    if pngfile:
        plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def compute_gaussian_mixture(mus, lowers, uppers, x_points=2000, ci_level=0.9,
                             sigma_floor=None, smooth_sigma=1.0, x_range=None):
    """
    Computes a horizontal Gaussian mixture from conformal intervals.

    Parameters
    ----------
    mus : array-like
        Predicted values (means) of the Gaussians.
    lowers : array-like
        Lower bounds of the conformal intervals.
    uppers : array-like
        Upper bounds of the conformal intervals.
    x_points : int
        Number of points for x-axis.
    ci_level : float
        Confidence level of the intervals (0 < ci_level < 1)
    sigma_floor : float or None
        Minimum sigma to avoid degenerate Gaussians. If None, 0.5% of x-range is used.
    smooth_sigma : float
        Sigma for Gaussian smoothing of the mixture.
    x_range : tuple (xmin, xmax) or None
        Optional fixed x-range. If None, inferred from lowers and uppers.

    Returns
    -------
    x : np.ndarray
        X-axis points.
    mixture_pdf : np.ndarray
        Mixture density values (peak-normalized for plotting).
    """
    mus = np.asarray(mus)
    lowers = np.asarray(lowers)
    uppers = np.asarray(uppers)

    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be between 0 and 1")

    z = norm.ppf((1 + ci_level) / 2)

    # Determine x-range
    if x_range is None:
        xmin = lowers.min()
        xmax = uppers.max()
    else:
        xmin, xmax = x_range

    x = np.linspace(xmin, xmax, x_points)

    # Compute sigmas with optional floor
    sigmas = (uppers - lowers) / (2 * z)
    if sigma_floor is None:
        sigma_floor_local = 0.005 * (xmax - xmin)
    else:
        sigma_floor_local = sigma_floor
    sigmas = np.maximum(sigmas, sigma_floor_local)

    # Gaussian mixture
    pdf_matrix = norm.pdf(x[None, :], loc=mus[:, None], scale=sigmas[:, None])
    mixture_pdf = pdf_matrix.mean(axis=0)

    # Optional smoothing
    if smooth_sigma > 0:
        mixture_pdf = gaussian_filter1d(mixture_pdf, sigma=smooth_sigma)

    # Peak normalization for plotting
    max_pdf = mixture_pdf.max()
    if max_pdf > 0:
        mixture_pdf /= max_pdf

    return x, mixture_pdf


def plot_modeled_mixture_from_conformal(
        df, value_columns, group_column, title, pngfile=None,
        thresholds=None, envelope_alpha=0.15, ci_level=0.90,
        sigma_floor=None, smooth_sigma=1.0, x_points=2000):
    """
    Plots horizontal Gaussian mixture distributions from conformal intervals,
    with numeric thresholds indicated in legend:
        Green: densities below threshold
        Red: densities above threshold
        Blue: densities overlapping threshold
    Uses compute_gaussian_mixture() to calculate the mixtures.
    """

    ncols = 2
    nrows = (len(value_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6*ncols, 3.5*nrows),
        squeeze=False
    )

    for idx, base in enumerate(value_columns):
        ax = axes[idx // ncols, idx % ncols]

        pred_col = base
        lower_col = f"{base}_lower"
        upper_col = f"{base}_upper"

        missing = [c for c in [pred_col, lower_col, upper_col] if c not in df.columns]
        if missing:
            ax.text(0.5, 0.5, f"Missing: {missing}", ha="center", va="center")
            ax.set_axis_off()
            continue

        df_clean = df.dropna(subset=[pred_col, lower_col, upper_col, group_column])
        groups = sorted(df_clean[group_column].unique())
        n_groups = len(groups)

        # Overall x-range
        xmin = df_clean[lower_col].min()
        xmax = df_clean[upper_col].max()
        x_range = (xmin, xmax)

        y_gap = 1.4
        y_positions = np.arange(n_groups)[::-1] * y_gap

        # Preprocess thresholds for this column
        thr_vals = []
        if thresholds is not None:
            thr_val = thresholds.numeric_for_column(pred_col)
            if thr_val is not None:
                if isinstance(thr_val, tuple):
                    thr_vals = [thr_val[0]]
                elif isinstance(thr_val, dict):
                    thr_vals = [v[0] if isinstance(v, tuple) else v for v in thr_val.values()]
                elif isinstance(thr_val, (list, tuple)):
                    thr_vals = [v[0] if isinstance(v, tuple) else v for v in thr_val]
                elif isinstance(thr_val, (int, float)):
                    thr_vals = [thr_val]

        # Flags for legend
        green_flag = red_flag = blue_flag = False

        for yi, g in zip(y_positions, groups):
            sub = df_clean[df_clean[group_column] == g]

            mus = sub[pred_col].values
            lowers = sub[lower_col].values
            uppers = sub[upper_col].values

            # Compute mixture using the extracted function
            x, mixture_pdf = compute_gaussian_mixture(
                mus, lowers, uppers,
                x_points=x_points,
                ci_level=ci_level,
                sigma_floor=sigma_floor,
                smooth_sigma=smooth_sigma,
                x_range=x_range
            )

            # Scale mixture for plotting
            mixture_pdf_plot = mixture_pdf * (y_gap * 0.9)

            # Determine fill color
            fill_color = "steelblue"  # default
            if thr_vals:
                if all(v > xmax for v in thr_vals):
                    fill_color = "green"
                    green_flag = True
                elif all(v < xmin for v in thr_vals):
                    fill_color = "red"
                    red_flag = True
                else:
                    fill_color = "steelblue"
                    blue_flag = True

            # Plot mixture
            ax.fill_between(x, yi, yi + mixture_pdf_plot, alpha=0.6, color=fill_color)
            ax.plot(x, yi + mixture_pdf_plot, color="black", lw=1.0)
            ax.scatter(mus, np.full_like(mus, yi), s=6, color="black", alpha=0.6)

        # Draw vertical lines for thresholds inside range
        for val in thr_vals:
            if xmin <= val <= xmax:
                ax.axvline(val, color="red", linestyle="--", linewidth=1.5)

        # Legend
        legend_handles = []
        if green_flag:
            legend_handles.append(Patch(facecolor='green', alpha=0.6,
                                        label=f"Below threshold ({', '.join(str(v) for v in thr_vals)})"))
        if red_flag:
            legend_handles.append(Patch(facecolor='red', alpha=0.6,
                                        label=f"Above threshold ({', '.join(str(v) for v in thr_vals)})"))
        if blue_flag or not thr_vals:
            legend_handles.append(Patch(facecolor='steelblue', alpha=0.6,
                                        label=f"Spans threshold ({', '.join(str(v) for v in thr_vals)})"))
        ax.legend(handles=legend_handles, fontsize=9)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(groups)
        ax.set_ylim(-0.5, y_positions[0] + 1.5*y_gap)
        ax.set_xlabel(base)
        ax.set_ylabel("Material")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Turn off unused axes
    total_axes = nrows * ncols
    used_axes = len(value_columns)
    if used_axes < total_axes:
        for j in range(used_axes, total_axes):
            r = j // ncols
            c = j % ncols
            axes[r][c].set_axis_off()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pngfile:
        plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def calculate_density_fractions_normalized(
        df, value_columns, group_column, thresholds=None,
        ci_level=0.90, sigma_floor=None, smooth_sigma=1.0, x_points=2000):
    """
    Calculates normalized fraction of the Gaussian mixture density below and above each numeric threshold
    for each group and column in the dataframe.

    Returns a DataFrame with multi-index: (column, group, threshold) and columns:
    fraction_below, fraction_above
    Fractions are normalized so that fraction_below + fraction_above = 1.
    """
    results = []

    for base in value_columns:
        pred_col = base
        lower_col = f"{base}_lower"
        upper_col = f"{base}_upper"

        df_clean = df.dropna(subset=[pred_col, lower_col, upper_col, group_column])
        groups = sorted(df_clean[group_column].unique())

        # Get thresholds for this column
        thr_vals = []
        if thresholds is not None:
            thr_val = thresholds.numeric_for_column(pred_col)
            if thr_val is not None:
                if isinstance(thr_val, tuple):
                    thr_vals = [thr_val[0]]
                elif isinstance(thr_val, dict):
                    thr_vals = [v[0] if isinstance(v, tuple) else v for v in thr_val.values()]
                elif isinstance(thr_val, (list, tuple)):
                    thr_vals = [v[0] if isinstance(v, tuple) else v for v in thr_val]
                elif isinstance(thr_val, (int, float)):
                    thr_vals = [thr_val]

        for g in groups:
            sub = df_clean[df_clean[group_column] == g]
            mus = sub[pred_col].values
            lowers = sub[lower_col].values
            uppers = sub[upper_col].values

            if len(mus) == 0:
                continue

            xmin = lowers.min()
            xmax = uppers.max()

            x, pdf = compute_gaussian_mixture(
                mus, lowers, uppers,
                x_points=x_points,
                ci_level=ci_level,
                sigma_floor=sigma_floor,
                smooth_sigma=smooth_sigma,
                x_range=(xmin, xmax)
            )

            dx = x[1] - x[0]

            # Compute normalized fractions for each threshold
            if thr_vals:
                for t in thr_vals:
                    frac_below = np.sum(pdf[x <= t] * dx)
                    frac_above = np.sum(pdf[x >= t] * dx)
                    total = frac_below + frac_above
                    if total > 0:
                        frac_below /= total
                        frac_above /= total
                    results.append({
                        'property': base,
                        group_column: g,
                        'threshold': t,
                        'fraction_below': frac_below,
                        'fraction_above': frac_above
                    })
            else:
                results.append({
                    'property': base,
                    group_column: g,
                    'threshold': None,
                    'fraction_below': np.nan,
                    'fraction_above': np.nan
                })

    fractions_df = pd.DataFrame(results).set_index(['property', group_column, 'threshold'])
    return fractions_df

