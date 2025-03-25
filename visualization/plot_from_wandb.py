import wandb
import pandas as pd

from argparse import ArgumentParser

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmms_eval.loggers.wandb_logger import WandbLogger
from lmms_eval.tasks.strokerehab.utils_summarization import SUMMARY_STEPS_METRICS



def _process_name(name):
    """
    'model_name__20250314_094252' -> 'model_name'
    """
    return name.split('__')[0]


def _flatten_dict(d, parent_key="", sep=","):
    """
    Recursively flattens a nested dictionary, naming deeper keys by
    joining them with `sep`. For example, if d = {"a": {"b": 3}},
    you'll get {"a,b": 3}.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Flatten further
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            # Assign the value
            items[new_key] = v
    return items

def _run_info_json_to_pd(dump):
    """
    Given the JSON 'dump' from a .table.json file, return a flattened
    DataFrame. We assume the top-level JSON has 'columns' and 'data'.
    For each row in 'data', if any field is a nested dict, it gets
    flattened with _flatten_dict.
    """
    # Basic table conversion
    df = pd.DataFrame(dump["data"], columns=dump["columns"])

    # Flatten any nested dicts column-by-column
    # We'll build a new list of row dictionaries to handle flattening
    flattened_rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if isinstance(value, dict):
                # Flatten the nested dict with a prefix = column name
                flat_dict = _flatten_dict(value, parent_key=col)
                row_dict.update(flat_dict)
            else:
                # Just keep it as is
                row_dict[col] = value
        flattened_rows.append(row_dict)

    # Build the final flattened DataFrame
    df_flat = pd.DataFrame(flattened_rows)
    return df_flat


def retrieve_runs_info(wandb_runs_location):
    import wandb

    api = wandb.Api()
    runs = api.runs(wandb_runs_location)

    dfs = []
    for run in runs:
        table_meta = run.summary["strokerehab_eval_results"]["path"]
        local_path = run.file(table_meta).download(replace=True)
        with open(local_path.name, "r") as f:
            dump = json.load(f)

        # Convert JSON to a flattened DataFrame
        df = _run_info_json_to_pd(dump)

        # Optionally add a 'model' column (or other run info)
        df["model"] = _process_name(run.name)
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)
    return dfs


def generate_activity_radar_plot(df, metric_name='sr_summary_score,action_f1'):
    """
    Generate a radar plot with Plotly from a DataFrame containing:
      - 'sr_summary_score,action_f1': the metric to aggregate
      - 'sr_summary_score,activity': the dimension (activities)
      - 'model': the entity (model identifier)

    This function aggregates the 'action_f1' scores by taking the mean per model and activity,
    then creates a radar plot where each model is represented as a Scatterpolar trace.
    
    Parameters:
      df (pd.DataFrame): DataFrame with the required columns.
      
    Returns:
      None (displays the Plotly radar plot).
    """
    # SUMMARY_STEPS_METRICS
    import pdb ; pdb.set_trace()
    import plotly.graph_objects as go  # # pip install -U plotly kaleido

    df['sr_summary_score,activity'] = df['sr_summary_score,activity'].str.replace(' left side', '', regex=False)
    df['sr_summary_score,activity'] = df['sr_summary_score,activity'].str.replace(' right side', '', regex=False)

    # Aggregate data: group by model and activity, computing the mean of action_f1
    agg = df.groupby(['model', 'sr_summary_score,activity'])[metric_name].mean().reset_index()
    
    # Pivot the data so that rows are activities and columns are models
    pivot = agg.pivot(index='sr_summary_score,activity', columns='model', values=metric_name)
    pivot = pivot.sort_index()  # ensure consistent ordering for activities
    activities = list(pivot.index)

    fig = go.Figure()
    
    # For each model, add a radar trace to the figure.
    for model in pivot.columns:
        r_values = pivot[model].tolist()
        # Close the loop for the radar plot by appending the first value.
        r_values += [r_values[0]]
        # Also append the first activity to the theta values.
        theta_values = activities + [activities[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill='toself',
            line=dict(width=2),
            name=str(model)
        ))
    
    fig.update_layout(
        paper_bgcolor='white',
        legend_font=dict(size=20),
        
        # Polar subplot (radar) styling
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                visible=True,
                range=[0, 1],      # Adjust to your data range
                gridcolor='lightgray',
                gridwidth=1,
                tickfont=dict(color='black', size=14),
                tickangle=0,
                ticks='outside'
            ),
            angularaxis=dict(
                visible=True,
                gridcolor='lightgray',
                gridwidth=1,
                linecolor='black',
                tickfont=dict(size=20),
                ticks='outside'
            ),
        ),
        
        # Legend styling (e.g., place legend below chart)
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        ),
        
    )

    fig.write_image("visualization/plots/radar_plot.png", scale=3.0)  # pip install -U plotly kaleido


if __name__ == "__main__":
    parser = ArgumentParser(description="Fetch and aggregate data from WandB runs")
    parser.add_argument('--wandb_runs_location', type=str, default="cvfm4rehab/cvfm4rehab_summary", help='WandB runs location in the format username/project_name')
    args = parser.parse_args()

    df = retrieve_runs_info(args.wandb_runs_location)
    generate_activity_radar_plot(df)

    # print(df)
