# analysis.py
# -*- coding: utf-8 -*-
"""
Analysis utilities for lithium deposit Monte Carlo simulations.
This module provides functions to filter simulation results, compute depletion
metrics (ERM), derive cost bounds, and generate publication-ready figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def REG_TYP_SCR(x, y, z):
    """
    Filter `simulation_results_list`, keeping only entries with
    "REG" equal to `y` and "Types" equal to `z`.

    Parameters
    ----------
    x : list[list[dict]]
        Simulation results across many runs.
    y : str
        Region to keep.
    z : str
        Deposit type to keep.

    Returns
    -------
    list[list[dict]]
        For each simulation, the filtered sublist.
    """
    filtered_results = []
    for sim in x:
        filtered_sim = [entry for entry in sim if entry.get("REG") == y and entry.get("Types") == z]
        filtered_results.append(filtered_sim)
    return filtered_results


def Production_SCR(x):
    """
    Split `x` into simulations with available `AISC` values and those without.
    For simulations without `AISC`, results are sorted by `RD` in descending order.

    Parameters
    ----------
    x : list[list[dict]]
        The list of simulation results

    Returns
    -------
    filtered_with_AISC : list[list[dict]]
        Simulations containing deposits with `AISC` values
    filtered_without_AISC : list[list[dict]]
        Simulations containing deposits without `AISC`, sorted by RD (descending)
    """
    filtered_with_AISC = []
    filtered_without_AISC = []

    for sim in x:
        with_aisc = [entry for entry in sim if not pd.isna(entry["AISC"])]
        without_aisc = sorted(
            [entry for entry in sim if pd.isna(entry["AISC"])],
            key=lambda entry: entry["RD"],
            reverse=True
        )
        filtered_with_AISC.append(with_aisc)
        filtered_without_AISC.append(without_aisc)

    return filtered_with_AISC, filtered_without_AISC


def sort_by_AISC(x):
    """Sort each simulation by `AISC` in ascending order."""
    sorted_results = [sorted(sim, key=lambda entry: entry["AISC"]) for sim in x]
    return sorted_results


def Depletion_data(x):
    """
    Process simulation results to compute ERM and AISC time series.

    Parameters
    ----------
    x : list[list[dict]]
        Simulations (filtered & sorted).

    Returns
    -------
    erm_df : pd.DataFrame
        DataFrame of ERM time series for each simulation.
    aisc_result : np.ndarray
        AISC values with an initial and a final extrapolated value inserted.
    bound_df : pd.DataFrame
        5th/50th/95th percentile bounds of ERM over time.
    """
    erm_results = []

    # Compute AISC sequence only once (from the first simulation)
    first_sim = x[0]
    AISC = np.array([entry["AISC"] for entry in first_sim])
    initial_aisc = AISC[0] / 2
    AISC = np.insert(AISC, 0, initial_aisc)
    final_aisc = AISC[-1] + (AISC[-1] - AISC[-2]) * 0.5
    aisc_result = np.append(AISC, final_aisc)

    # Compute ERM for each simulation
    for sim in x:
        RD = np.array([entry["RD"] for entry in sim])
        ERM = RD.cumsum() - 0.5 * RD
        ERM = np.insert(ERM, 0, 0)
        final_erm = RD.sum()
        ERM = np.append(ERM, final_erm)
        erm_results.append(ERM)

    # Convert to DataFrame
    erm_df = pd.DataFrame(erm_results).T
    erm_df.columns = [f"Sim_{i+1}" for i in range(len(x))]

    # Compute percentile bounds
    erm_lower = np.percentile(erm_df, 5, axis=1)
    erm_med   = np.percentile(erm_df, 50, axis=1)
    erm_upper = np.percentile(erm_df, 95, axis=1)

    bound_df = pd.DataFrame({
        'Lower Bound': erm_lower,
        'Median':      erm_med,
        'Upper Bound': erm_upper
    })

    return erm_df, aisc_result, bound_df


def ERM_Prec(x):
    """Compute cumulative ERM share as a percentage (0–1 scale)."""
    erm_percent = x["Median"] / x["Median"].iloc[-1]
    return erm_percent


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def draw_depletion(x, y, z):
    """Plot depletion curve: cumulative ERM (%) vs AISC."""
    plt.figure(figsize=(10, 6))
    plt.plot(x * 100, y, 'b-', label='Depletion Curve')  # X-axis as percentage
    plt.xlim(0, 100)
    plt.xlabel('Cumulative ERM (%)')
    plt.ylabel('AISC (2023 USD/LCE)')
    plt.legend(loc='best')
    plt.savefig(f'Depletion Curve {z} (ERM vs AISC)')
    plt.show()


def draw_MCA(x, y, z):
    """Plot ERM vs AISC with 5th/50th/95th bounds."""
    plt.figure(figsize=(10, 6))
    plt.plot(x["Lower Bound"] * 1000, y, 'r--', label='5th Percentile (Lower Bound)')
    plt.plot(x["Median"] * 1000,      y, 'b-',  label='50th Percentile (Median)')
    plt.plot(x["Upper Bound"] * 1000,  y, 'g--', label='95th Percentile (Upper Bound)')

    plt.xlabel(f'ERM for {z} — Li Thousand Metric Tons')
    plt.ylabel('AISC (2023 USD/LCE)')
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.savefig(f'ERM vs AISC for {z} with Upper, Lower, and Median Bounds')
    plt.show()


def draw_barchart(x, y):
    """
    Bar chart where the bar width encodes RD (Median) and the height encodes AIC.
    Bottom annotations show true-value ticks; top shows 25/50/75/100% markers.
    """
    widths = x["Median"].values
    heights = x["AIC"].values
    total_cumulative_rd = x["Cumulative_RD"].iloc[-1]
    total_cumu_value = int(total_cumulative_rd * 1000)

    total_rd = float(np.sum(widths))
    xticks = np.cumsum(widths) - widths / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [(191/255, 29/255, 45/255) if sp == "N" else (41/255, 56/255, 144/255) for sp in x["S&P"]]
    hatches = ["///" if ci == "N" else "" for ci in x["Cost_information"]]

    for i in range(len(xticks)):
        ax.bar(
            xticks[i], heights[i], width=widths[i],
            facecolor="white" if x["Cost_information"].iloc[i] == "N" else colors[i],
            edgecolor="black",
            hatch=hatches[i]
        )

    max_height = float(np.max(heights))
    label_offset = max_height * 0.06

    positions_percent = [total_rd * frac for frac in [0.25, 0.5, 0.75, 1.0]]
    for pos, label in zip(positions_percent, ['25%', '50%', '75%', '100%']):
        ax.text(pos, max_height + label_offset, label, ha='center', va='bottom', fontsize=10)

    raw_step = total_cumu_value / 6 if total_cumu_value > 0 else 1
    magnitude = 10 ** int(np.floor(np.log10(max(raw_step, 1))))

    step = magnitude
    for base in [1, 2, 5, 10]:
        step = base * magnitude
        if total_cumu_value / step <= 10:
            break

    tick_values = list(range(0, total_cumu_value + 1, int(step))) if step > 0 else [0]
    if tick_values and tick_values[-1] > total_cumu_value:
        tick_values = tick_values[:-1]

    tick_positions = [val / total_cumu_value * total_rd if total_cumu_value > 0 else 0 for val in tick_values]

    for position, label in zip(tick_positions, tick_values):
        ax.text(position, -150, f'{label}', color='black', ha='center', va='top', fontsize=10)

    ax.set_xlabel(f"Total {y} Estimated Recoverable Minerals — Li Thousand Metric Tons", labelpad=18)
    ax.set_ylabel("AIC (2023 USD/t LCE)")

    ax.set_xticks([])
    ax.margins(x=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    legend_handles = [
        mpatches.Patch(facecolor=(41/255, 56/255, 144/255), edgecolor='black', label="S&P Operating Deposits"),
        mpatches.Patch(facecolor=(191/255, 29/255, 45/255), edgecolor='black', label="Undeveloped Deposits with Cost Information"),
        mpatches.Patch(facecolor="white", edgecolor='black', hatch="///", label="Undeveloped Deposits without Cost Information")
    ]
    ax.legend(handles=legend_handles, loc="upper left", edgecolor='black')

    plt.savefig(f"Total {y} Estimated Recoverable Minerals.png")
    plt.show()


def draw_bar_country(x, y):
    """
    Region-grouped bar chart with true resource-scale ticks at the bottom
    and percentage markers at the top.
    """
    widths = x["Median"].values
    heights = x["AIC"].values
    total_cumulative_rd = x["Cumulative_RD"].iloc[-1]
    total_cumu_value = int(total_cumulative_rd * 1000)
    total_rd = float(np.sum(widths))
    xticks = np.cumsum(widths) - widths / 2

    region_colors = {
        'LATAM': (250/255, 174/255, 95/255),
        'NAM':   (53/255, 72/255, 152/255),
        'AUS':   (115/255, 171/255, 207/255),
        'CHN':   (151/255, 45/255, 54/255),
        'AFR':   (107/255, 107/255, 107/255),
        'EU':    (133/255, 155/255, 138/255),
        'OTH':   (174/255, 74/255, 138/255),
    }
    used_regions = set(x["REG"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(xticks)):
        region = x["REG"].iloc[i]
        color = region_colors.get(region, (0.5, 0.5, 0.5))
        hatch = "///" if x["Cost_information"].iloc[i] == "N" else ""
        ax.bar(
            xticks[i], heights[i], width=widths[i],
            color=color,
            edgecolor="black",
            hatch=hatch
        )

    max_height = float(np.max(heights))
    label_offset = max_height * 0.06

    positions_percent = [total_rd * frac for frac in [0.25, 0.50, 0.75, 1.00]]
    for pos, label in zip(positions_percent, ['25%', '50%', '75%', '100%']):
        ax.text(pos, max_height + label_offset, label, ha='center', va='bottom', fontsize=10)

    raw_step = total_cumu_value / 6 if total_cumu_value > 0 else 1
    magnitude = 10 ** int(np.floor(np.log10(max(raw_step, 1))))
    step = magnitude
    for base in [1, 2, 5, 10]:
        step = base * magnitude
        if total_cumu_value / step <= 10:
            break

    tick_values = list(range(0, total_cumu_value + 1, int(step))) if step > 0 else [0]
    if tick_values and tick_values[-1] > total_cumu_value:
        tick_values = tick_values[:-1]

    tick_positions = [val / total_cumu_value * total_rd if total_cumu_value > 0 else 0 for val in tick_values]

    for position, label in zip(tick_positions, tick_values):
        ax.text(position, -150, f'{label}', color='black', ha='center', va='top', fontsize=10)

    ax.set_xlabel(f"Total {y} Estimated Recoverable Minerals — Li Thousand Metric Tons", labelpad=18)
    ax.set_ylabel("AIC (2023 USD/t LCE)")
    ax.set_xticks([])
    ax.margins(x=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    legend_handles = [
        mpatches.Patch(facecolor=region_colors[region], edgecolor='black', label=region)
        for region in sorted(used_regions) if region in region_colors
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="white", edgecolor='black', hatch="///",
                       label="Undeveloped Deposits without Cost Information")
    )
    ax.legend(handles=legend_handles, loc="upper left", edgecolor='black')

    plt.savefig(f"Total {y} Estimated Recoverable Minerals by Region.png")
    plt.show()


def draw_final(x, y):
    """
    Final bar chart coloring bars by deposit type ("Ore", "Brine", "Uncon").
    Bottom axis shows true-value ticks; top shows percentage markers.
    """
    widths = x["Median"].values
    heights = x["AIC"].values
    total_cumulative_rd = x["Cumulative_RD"].iloc[-1]
    total_cumu_value = int(total_cumulative_rd * 1000)
    total_rd = float(np.sum(widths))
    xticks = np.cumsum(widths) - widths / 2

    type_colors = {
        'Ore':   (0/255,   82/255, 147/255),
        'Brine': (228/255, 108/255, 10/255),
        'Uncon': (0/255,   128/255, 105/255)
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(xticks)):
        typ = x["Types"].iloc[i]
        color = type_colors.get(typ, (0.5, 0.5, 0.5))
        hatch = "///" if x["Cost_information"].iloc[i] == "N" else ""
        ax.bar(
            xticks[i], heights[i], width=widths[i],
            color=color, edgecolor="black", hatch=hatch
        )

    max_height = float(np.max(heights))
    label_offset = max_height * 0.06

    positions_percent = [total_rd * frac for frac in [0.25, 0.50, 0.75, 1.00]]
    for pos, label in zip(positions_percent, ['25%', '50%', '75%', '100%']):
        ax.text(pos, max_height + label_offset, label, ha='center', va='bottom', fontsize=10)

    raw_step = total_cumu_value / 6 if total_cumu_value > 0 else 1
    magnitude = 10 ** int(np.floor(np.log10(max(raw_step, 1))))
    step = magnitude
    for base in [1, 2, 5, 10]:
        step = base * magnitude
        if total_cumu_value / step <= 10:
            break

    tick_values = list(range(0, total_cumu_value + 1, int(step))) if step > 0 else [0]
    if tick_values and tick_values[-1] > total_cumu_value:
        tick_values = tick_values[:-1]

    tick_positions = [val / total_cumu_value * total_rd if total_cumu_value > 0 else 0 for val in tick_values]

    for position, label in zip(tick_positions, tick_values):
        ax.text(position, -150, f'{label}', color='black', ha='center', va='top', fontsize=10)

    ax.set_xlabel(f"Total {y} Estimated Recoverable Minerals — Li Thousand Metric Tons", labelpad=18)
    ax.set_ylabel("AIC (2023 USD/t LCE)")
    ax.set_xticks([])
    ax.margins(x=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    legend_handles = [
        mpatches.Patch(facecolor=type_colors["Ore"],   edgecolor='black', label="Ore Deposits"),
        mpatches.Patch(facecolor=type_colors["Brine"], edgecolor='black', label="Brine Deposits"),
        mpatches.Patch(facecolor=type_colors["Uncon"], edgecolor='black', label="Unconventional Deposits"),
        mpatches.Patch(facecolor="white", edgecolor='black', hatch="///",
                       label="Undeveloped Deposits without Cost Information")
    ]
    ax.legend(handles=legend_handles, loc="upper left", edgecolor='black')

    plt.savefig(f"Total {y} Estimated Recoverable Minerals by Types.png")
    plt.show()


# ---------------------------------------------------------------------------
# ERM & AISC utilities (no-AISC handling, merging, costs)
# ---------------------------------------------------------------------------

def without_AISC_data(x):
    """
    Compute ERM bounds for simulations without AISC values.
    """
    erm_results = []
    for sim in x:
        RD = np.array([entry["RD"] for entry in sim])
        ERM = RD.cumsum() - 0.5 * RD
        ERM = np.insert(ERM, 0, 0)
        final_erm_value = RD.sum()
        ERM = np.append(ERM, final_erm_value)
        erm_results.append(ERM)

    erm_df = pd.DataFrame(erm_results).T
    erm_df.columns = [f"Sim_{i+1}" for i in range(len(x))]

    erm_lower_bound = np.percentile(erm_df, 5, axis=1)
    erm_median = np.percentile(erm_df, 50, axis=1)
    erm_upper_bound = np.percentile(erm_df, 95, axis=1)

    bound_df = pd.DataFrame({
        'Lower Bound': erm_lower_bound,
        'Median': erm_median,
        'Upper Bound': erm_upper_bound
    })

    return erm_df, bound_df


def AISC_without(x, y, z):
    """
    Interpolate AISC values for a target ERM percentage `x` using
    the known AISC curve defined by (y, z).
    """
    AISC_interp = np.interp(x, y, z)
    return AISC_interp


def rd_aisc_sp(x, y, z, w):
    """
    Compare RD statistics for cases with and without AISC information.

    Parameters
    ----------
    x, z : list[list[dict]]
        Simulation result lists (with AISC / without AISC).
    y, w : np.ndarray
        AISC sequences corresponding to x and z (with boundary padding).
    """
    def compute_rd_bounds(data, aisc_values, cost_label):
        rd_results = [np.array([entry["RD"] for entry in sim]) for sim in data]
        rd_df = pd.DataFrame(rd_results).T
        rd_df.columns = [f"Sim_{i+1}" for i in range(len(data))]

        rd_lower = np.percentile(rd_df, 5, axis=1)
        rd_med   = np.percentile(rd_df, 50, axis=1)
        rd_upper = np.percentile(rd_df, 95, axis=1)

        first = data[0]
        sp_vals    = [entry["S&P"] for entry in first]
        types      = [entry["Types"] for entry in first]
        regs       = [entry["REG"] for entry in first]
        icpex_vals = [entry.get("ICPEX", np.nan) for entry in first]

        return pd.DataFrame({
            "Lower Bound":      rd_lower,
            "Median":           rd_med,
            "Upper Bound":      rd_upper,
            "S&P":              sp_vals,
            "AISC":             aisc_values[1:-1],  # strip the padded ends
            "Cost_information": cost_label,
            "Types":            types,
            "REG":              regs,
            "ICPEX":            icpex_vals
        })

    aisc_with_df    = compute_rd_bounds(x, y, "Y")
    aisc_without_df = compute_rd_bounds(z, w, "N")

    result = pd.concat([aisc_with_df, aisc_without_df], ignore_index=True)
    result["Cumulative_RD"] = result["Median"].cumsum()

    return result


def sort_allcost(df, y):
    """
    Create 'AIC' (All-in-Cost) as:
      - if S&P == 'Y'                  → AIC = AISC
      - if S&P == 'N' and ICPEX notna → AIC = AISC + ICPEX
      - else                           → AIC = AISC * (1 + y)
    Then sort ascending by AIC.
    """
    cond1 = df['S&P'] == 'Y'
    cond2 = (df['S&P'] == 'N') & df['ICPEX'].notna()

    df['AIC'] = np.where(
        cond1,
        df['AISC'],
        np.where(
            cond2,
            df['AISC'] + df['ICPEX'],
            df['AISC'] * (1 + y)
        )
    )
    allcost_bound_df = df.sort_values(by="AIC", ascending=True).reset_index(drop=True)
    return allcost_bound_df


def allcost_allhavecost(x):
    """
    Aggregate RD bounds for the case where all deposits have AISC.
    """
    rd_results = []

    first_sim = x[0]
    AISC = np.array([entry["AISC"] for entry in first_sim])

    for sim in x:
        rd = np.array([entry["RD"] for entry in sim])
        rd_results.append(rd)

    rd_df = pd.DataFrame(rd_results).T
    rd_df.columns = [f"Sim_{i+1}" for i in range(len(x))]

    rd_lower_bound = np.percentile(rd_df, 5, axis=1)
    rd_median      = np.percentile(rd_df, 50, axis=1)
    rd_upper_bound = np.percentile(rd_df, 95, axis=1)

    first = x[0]
    sp_vals    = [entry["S&P"] for entry in first]
    types      = [entry["Types"] for entry in first]
    regs       = [entry["REG"] for entry in first]
    icpex_vals = [entry.get("ICPEX", np.nan) for entry in first]

    allcost_bound_df = pd.DataFrame({
        'Lower Bound': rd_lower_bound,
        'Median':      rd_median,
        'Upper Bound': rd_upper_bound,
        'AISC':        AISC,
        'S&P':         sp_vals,
        'Types':       types,
        'REG':         regs,
        'Cost_information': "Y",
        'ICPEX':       icpex_vals
    })

    allcost_bound_df['Cumulative_RD'] = allcost_bound_df['Median'].cumsum()
    return allcost_bound_df


# ---------------------------------------------------------------------------
# Analysis pipelines
# ---------------------------------------------------------------------------

def run_region_type_analysis(simulation_results_list, region, typ, adjust_factor):
    """
    Execute the complete workflow from filtering to visualization,
    and return the final All-in-Cost DataFrame.
    """
    filtered_simulations = REG_TYP_SCR(simulation_results_list, region, typ)
    filtered_with_AISC, filtered_without_AISC = Production_SCR(filtered_simulations)
    sorted_AISC_deposit = sort_by_AISC(filtered_with_AISC)

    erm_df, aisc_result, bound_df = Depletion_data(sorted_AISC_deposit)
    depletion_prec = ERM_Prec(bound_df)

    draw_MCA(bound_df, aisc_result, f"{region}_{typ}")
    draw_depletion(depletion_prec, aisc_result, f"{region}_{typ}")

    without_erm_df, without_bound_df = without_AISC_data(filtered_without_AISC)
    without_depletion_prec = ERM_Prec(without_bound_df)
    AISC_for_without = AISC_without(without_depletion_prec, depletion_prec, aisc_result)

    AISC_all_bound_df = rd_aisc_sp(sorted_AISC_deposit, aisc_result, filtered_without_AISC, AISC_for_without)
    allcost_df = sort_allcost(AISC_all_bound_df, adjust_factor)
    draw_barchart(allcost_df, f"{region} {typ}")
    return allcost_df


def run_allhavecost_analysis(simulation_results_list, region, typ, adjust_factor):
    """
    Analysis workflow for the case where all deposits have AISC information.
    Returns the final All-in-Cost DataFrame.
    """
    filtered_simulations = REG_TYP_SCR(simulation_results_list, region, typ)
    filtered_with_AISC, _ = Production_SCR(filtered_simulations)
    sorted_AISC_deposit = sort_by_AISC(filtered_with_AISC)

    erm_df, aisc_result, bound_df = Depletion_data(sorted_AISC_deposit)
    depletion_prec = ERM_Prec(bound_df)
    draw_MCA(bound_df, aisc_result, f"{region}_{typ}")
    draw_depletion(depletion_prec, aisc_result, f"{region}_{typ}")

    AISC_all_bound_df = allcost_allhavecost(sorted_AISC_deposit)
    allcost_df = sort_allcost(AISC_all_bound_df, adjust_factor)
    draw_barchart(allcost_df, f"{region} {typ}")
    return allcost_df


def merge_and_plot_allcost(dataframes, label):
    """
    Merge All-in-Cost (AIC) DataFrames from multiple regions, sort them,
    recompute Cumulative_RD, and produce the bar chart.
    """
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)
    merged_df["Cumulative_RD"] = merged_df["Median"].cumsum()
    draw_barchart(merged_df, label)
    return merged_df


def Uncon_SCR(x, z):
    """
    Filter simulations by deposit type (e.g., "Uncon").
    """
    filtered_results = []
    for sim in x:
        filtered_sim = [entry for entry in sim if entry.get("Types") == z]
        filtered_results.append(filtered_sim)
    return filtered_results


def run_unconventional_analysis(simulation_results_list, typ="Uncon", adjust_factor=0.2174):
    """
    Execute the full workflow for Unconventional Lithium deposits and return
    the All-in-Cost (AIC) DataFrame.
    """
    filtered_simulations = Uncon_SCR(simulation_results_list, typ)
    filtered_with_AISC, filtered_without_AISC = Production_SCR(filtered_simulations)
    sorted_AISC_deposit = sort_by_AISC(filtered_with_AISC)

    erm_df, aisc_result, bound_df = Depletion_data(sorted_AISC_deposit)
    depletion_prec = ERM_Prec(bound_df)

    draw_MCA(bound_df, aisc_result, typ)
    draw_depletion(depletion_prec, aisc_result, typ)

    without_erm_df, without_bound_df = without_AISC_data(filtered_without_AISC)
    without_depletion_prec = ERM_Prec(without_bound_df)
    AISC_for_without = AISC_without(without_depletion_prec, depletion_prec, aisc_result)

    AISC_all_bound_df = rd_aisc_sp(
        sorted_AISC_deposit, aisc_result, filtered_without_AISC, AISC_for_without
    )
    allcost_df = sort_allcost(AISC_all_bound_df, adjust_factor)
    draw_barchart(allcost_df, "Unconventional Lithium")
    return allcost_df


def plot_cost_bounds_vs_erm(
    df,
    sps_x: float | None = None,
    aps_x: float | None = None,
    nez_x: float | None = None,
    filename: str = "bounds AIC VS total Li ERM.png",
):
    """
    Plot AIC curves under the 5th/50th/95th ERM bounds and add optional vertical
    reference lines (SPS/APS/NEZ). Returns the working DataFrame with cumulative
    columns.
    """
    bound_lithium_final = df.copy()
    bound_lithium_final['Cumulative_RD_Lower']  = bound_lithium_final['Lower Bound'].cumsum()
    bound_lithium_final['Cumulative_RD']        = bound_lithium_final['Median'].cumsum()
    bound_lithium_final['Cumulative_RD_Higher'] = bound_lithium_final['Upper Bound'].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(bound_lithium_final['Cumulative_RD_Lower'] * 1000,  bound_lithium_final['AIC'], 'r--', label='5% Lower Bound')
    plt.plot(bound_lithium_final['Cumulative_RD'] * 1000,        bound_lithium_final['AIC'], 'b-',  label='50% Bound')
    plt.plot(bound_lithium_final['Cumulative_RD_Higher'] * 1000, bound_lithium_final['AIC'], 'g--', label='95% Higher Bound')

    if sps_x is not None:
        plt.axvline(sps_x * 1000, color=(151/255, 45/255, 54/255), linestyle='-.', label='SPS')
    if aps_x is not None:
        plt.axvline(aps_x * 1000, color=(115/255, 171/255, 207/255), linestyle='-.', label='APS')
    if nez_x is not None:
        plt.axvline(nez_x * 1000, color=(250/255, 174/255, 95/255), linestyle='-.', label='NEZ')

    # Horizontal reference line (e.g., assumed level)
    plt.axhline(7164, color=(174/255, 74/255, 138/255), linestyle=':', linewidth=2, label='7164 assuming level')

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.xlabel('Total Lithium Estimated Recoverable Minerals — Li Thousand Metric Tons')
    plt.ylabel('AIC (2023 USD/t LCE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    return bound_lithium_final
