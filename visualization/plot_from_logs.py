# import pandas as pd

# from argparse import ArgumentParser

# import json

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from lmms_eval.tasks.strokerehab.utils import SUMMARY_STEPS_METRICS
import json
import pandas as pd

# logs are saved in ./logs
# folders are for models (e.g., "llava_vid")


def read_jsonl(file_path):
    """Return flattened dictionary from a jsonl file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.json_normalize(data)
    # logs are saved in ./logs
    # get folder name after logs/
    dirs = file_path.split('/')
    for i, d in enumerate(dirs):
        if d == 'logs' and i + 1 < len(dirs):
            df['model'] = dirs[i + 1]
            break
    return df


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
    import plotly.graph_objects as go  # # pip install -U plotly kaleido

    df['activity'] = df['activity'].str.replace(' left side', '', regex=False)
    df['activity'] = df['activity'].str.replace(' right side', '', regex=False)

    # Aggregate data: group by model and activity, computing the mean of action_f1
    agg = df.groupby(['model', 'activity'])[metric_name].mean().reset_index()
    
    # Pivot the data so that rows are activities and columns are models
    pivot = agg.pivot(index='activity', columns='model', values=metric_name)
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
    df = read_jsonl('./logs/llava_vid/lmms-lab__LLaVA-NeXT-Video-7B-Qwen2/20250315_053415_samples_strokerehab.jsonl')
    generate_activity_radar_plot(df, metric_name="summary_steps_score")

    print(df)
