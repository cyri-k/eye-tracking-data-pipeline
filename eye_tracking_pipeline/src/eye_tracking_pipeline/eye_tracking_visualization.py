import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio
from PIL import Image

from ipywidgets import interact, Dropdown, Checkbox, HBox, VBox 
from IPython.display import display, clear_output

import numpy as np
import pandas as pd

def fixation_proportion_line(aggregated_df,
                             xaxis_title="Time relative to the onset of the stimulus (ms)",
                             yaxis_title="Fixation Proportion (%)"):
    aggregated_df['timeBin'] = pd.cut(aggregated_df["timeFromOnsetMs"], 
                                     bins = np.arange(-4, 3, 0.1).round(decimals=1), 
                                     labels = np.arange(-4, 3, 0.1).round(decimals=1)[1:], 
                                     right=True,ordered=False)
    binned_rois = pd.merge(aggregated_df.loc[:, 'timeBin'], 
             pd.get_dummies(aggregated_df['AOI'], dummy_na=True), 
             left_index=True, right_index=True).groupby('timeBin', observed=False).sum()
    binned_rois = binned_rois.div(binned_rois.sum(axis=1), axis=0)
    binned_rois.index = binned_rois.index.astype(float)*1000
    binned_rois = binned_rois.dropna()
    
    fig = px.line(binned_rois.loc[:, binned_rois.columns.notna()], 
                  color_discrete_sequence=["#EECC66", "#EE99AA","#994455","#004488"])
    
    fig.add_vrect(x0=0, x1=2000, line_width=0, fillcolor="blue", opacity=0.07)
    fig.update_layout(
            xaxis_title=xaxis_title,  # Add x-axis name
            yaxis_title=yaxis_title,  # Add y-axis name
            xaxis=dict(tickmode='linear', tick0=0, dtick=200),  # Set x-axis ticks interval
            yaxis_tickformat=".0%",  # Format y-axis as percentage
            yaxis_range=[-0.02,0.5],
            legend=dict(
                title=dict(
                        text="Variable"
                )
            ),
            font=dict(size=18)
    )
    fig.update_traces(line=dict(width=2))
    
    return fig

def filterable_fixation_proportion_line(aggregated_df: pd.DataFrame, width: int = 1000):
    # --- 1. Prepare unique values for dropdowns ---
    # Get all unique values from the columns, and add an 'All' option
    unique_countries = ['All', 'Not Control Group'] + sorted(aggregated_df['Country'].unique().tolist())
    unique_institutions = ['All'] + sorted(aggregated_df['Institution'].unique().tolist())
    unique_versions = ['All'] + sorted(aggregated_df['Version'].unique().tolist())
    unique_conditions = ['All'] + sorted(aggregated_df['condition'].unique().tolist())
    unique_sessions = ['All'] + sorted(aggregated_df['Session'].dropna().unique().tolist())
    unique_stimuli = ['All'] + sorted(aggregated_df['sound'].dropna().unique().tolist())
    
    # --- 2. Create ipywidgets Dropdown instances ---
    country_dd = Dropdown(options=unique_countries, value='All', description='Country:')
    institution_dd = Dropdown(options=unique_institutions, value='All', description='Institution:')
    version_dd = Dropdown(options=unique_versions, value='All', description='Version:')
    condition_dd = Dropdown(options=unique_conditions, value='All', description='Condition:')
    session_dd = Dropdown(options=unique_sessions, value=1.0, description='Session:')
    stimuli_dd = Dropdown(options=unique_stimuli, value='All', description='Stimuli:')
    
    # --- 3. Create a Plotly FigureWidget for dynamic updates ---
    plot_output = go.FigureWidget()
    # Set the figure to fill available width
    plot_output.layout.width = width  # Let the container control the width
    plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
    
    # --- 4. Define the update function that filters data and redraws the plot ---
    def update_plot_with_filters(country, institution, version, condition, session, stimuli):
        """
        Filters the aggregated DataFrame based on dropdown selections and updates the Plotly graph.
        """
        clear_output(wait=True) # Clear previous plot output in Jupyter
    
        # Create a copy to avoid modifying the original DataFrame
        current_filtered_df = aggregated_df.copy()
    
        # Apply filters based on dropdown selections
        if country != 'All':
            if country == 'Not Control Group':
                current_filtered_df = current_filtered_df[current_filtered_df['Country'] != 'Control Group']
            else:
                current_filtered_df = current_filtered_df[current_filtered_df['Country'] == country]
        if institution != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Institution'] == institution]
        if version != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['Version'] == version]
        if condition != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['condition'] == condition]
        if session != 'All':
            if country != 'Control Group':
                current_filtered_df = current_filtered_df[current_filtered_df['Session'] == session]
        if stimuli != 'All':
            current_filtered_df = current_filtered_df[current_filtered_df['sound'] == stimuli]
    
        # Call your `fixation_proportion_line` function with the *filtered* DataFrame
        fig = fixation_proportion_line(current_filtered_df)
    
        # --- Crucial Fix for ValueError: ---
        plot_output.data = []
    
        for trace in fig.data:
            plot_output.add_trace(trace)
    
        # Update the layout of the FigureWidget (fill width)
        plot_output.layout = fig.layout
        plot_output.layout.autosize = True
        plot_output.layout.width = None
        plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
        plot_output.frames = fig.frames
    
    # --- 5. Initial display of the plot (before any dropdown changes) ---
    update_plot_with_filters(country_dd.value, institution_dd.value, version_dd.value, condition_dd.value, session_dd.value, stimuli_dd.value)
    
    # --- 6. Link dropdowns to the update function using .observe() ---
    country_dd.observe(lambda change: update_plot_with_filters(
        change.new, institution_dd.value, version_dd.value, condition_dd.value, session_dd.value, stimuli_dd.value), names='value')
    institution_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, change.new, version_dd.value, condition_dd.value, session_dd.value, stimuli_dd.value), names='value')
    version_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, change.new, condition_dd.value, session_dd.value, stimuli_dd.value), names='value')
    condition_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, change.new, session_dd.value, stimuli_dd.value), names='value')
    session_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, condition_dd.value, change.new, stimuli_dd.value), names='value')
    stimuli_dd.observe(lambda change: update_plot_with_filters(
        country_dd.value, institution_dd.value, version_dd.value, condition_dd.value, session_dd.value, change.new), names='value')
    
    # --- 7. Arrange and display the widgets and the plot ---
    controls = HBox([session_dd, country_dd, institution_dd, condition_dd, stimuli_dd])
    dashboard = VBox([plot_output, controls], layout={'width': '100%'})
    return dashboard

def filterable_fig(aggregated_df: pd.DataFrame, filtered_cols: list[str], generate_fig, width: int = 1000):
    # --- 1. Prepare unique values for dropdowns ---
    # Get all unique values from the columns, and add an 'All' option
    unique_values = {}
    for col in filtered_cols:
        unique_values[col] = ['All'] + sorted(aggregated_df[col].dropna().unique().tolist())
    
    # --- 2. Create ipywidgets Dropdown instances ---
    dropdowns = {}
    for col, values in unique_values.items():
        dropdowns[col] = Dropdown(options=values, value='All', description=f'{col}:')
    
    # --- 3. Create a Plotly FigureWidget for dynamic updates ---
    plot_output = go.FigureWidget()
    # Set the figure to fill available width
    plot_output.layout.width = width  # Let the container control the width
    plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
    
    # --- 4. Define the update function that filters data and redraws the plot ---
    def update_plot_with_filters(**kwargs):
        """
        Filters the aggregated DataFrame based on dropdown selections and updates the Plotly graph.
        """
        clear_output(wait=True) # Clear previous plot output in Jupyter
    
        # Create a copy to avoid modifying the original DataFrame
        current_filtered_df = aggregated_df.copy()
    
        # Apply filters based on dropdown selections
        for col, value in kwargs.items():
            if value != 'All':
                current_filtered_df = current_filtered_df[current_filtered_df[col] == value]
    
        # Call your `generate_fig` function with the *filtered* DataFrame
        fig = generate_fig(current_filtered_df)
    
        # --- Crucial Fix for ValueError: ---
        plot_output.data = []
    
        for trace in fig.data:
            plot_output.add_trace(trace)
    
        # Update the layout of the FigureWidget (fill width)
        plot_output.layout = fig.layout
        plot_output.layout.autosize = True
        plot_output.layout.width = None
        plot_output.layout.margin = dict(l=20, r=20, t=40, b=20)
        plot_output.frames = fig.frames
    
    # --- 5. Initial display of the plot (before any dropdown changes) ---
    update_plot_with_filters(**{col: dropdowns[col].value for col in filtered_cols})
    
    # --- 6. Link dropdowns to the update function using .observe() ---
    for col, dropdown in dropdowns.items():
        dropdown.observe(lambda change, col=col: update_plot_with_filters(**{col: change.new, **{c: dropdowns[c].value for c in filtered_cols if c != col}}), names='value')
    
    # --- 7. Arrange and display the widgets and the plot ---
    controls = HBox(list(dropdowns.values()))
    dashboard = VBox([plot_output, controls], layout={'width': '100%'})
    return dashboard


filters = [
      {'folder':'0.Session', 'prefix':'0.0.', 'session':'All'},
      {'folder':'0.Session', 'prefix':'0.1.', 'session':1},
      {'folder':'0.Session', 'prefix':'0.2.', 'session':2},
      {'folder':'1.Overall mean', 'prefix':'1.1.', 'condition':'standard'},
      {'folder':'1.Overall mean', 'prefix':'1.2.', 'condition':'alemannic_austria'},
      {'folder':'2.Mean kindergarden', 'prefix':'2.0.', 'institution':'KiGa'},
      {'folder':'2.Mean kindergarden', 'prefix':'2.1.', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'2.Mean kindergarden', 'prefix':'2.2.', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'3.Mean preschool', 'prefix':'3.0.', 'institution':'Grundschule'},
      {'folder':'3.Mean preschool', 'prefix':'3.1.', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'3.Mean preschool', 'prefix':'3.2.', 'condition':'alemannic_austria', 'institution':'Grundschule'},
      {'folder':'4.Mean Germany/4.0.all_institutions', 'prefix':'4.0.0.', 'country':'Deutschland'},
      {'folder':'4.Mean Germany/4.0.all_institutions', 'prefix':'4.0.1.', 'country':'Deutschland', 'condition':'standard'},
      {'folder':'4.Mean Germany/4.0.all_institutions', 'prefix':'4.0.2.', 'country':'Deutschland', 'condition':'alemannic_austria'},
      {'folder':'4.Mean Germany/4.1.kindergarden', 'prefix':'4.1.0.', 'country':'Deutschland', 'institution':'KiGa'},
      {'folder':'4.Mean Germany/4.1.kindergarden', 'prefix':'4.1.1.', 'country':'Deutschland', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'4.Mean Germany/4.1.kindergarden', 'prefix':'4.1.2.', 'country':'Deutschland', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'4.Mean Germany/4.2.grundschule', 'prefix':'4.2.0.', 'country':'Deutschland', 'institution':'Grundschule'},
      {'folder':'4.Mean Germany/4.2.grundschule', 'prefix':'4.2.1.', 'country':'Deutschland', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'4.Mean Germany/4.2.grundschule', 'prefix':'4.2.2.', 'country':'Deutschland', 'condition':'alemannic_austria', 'institution':'Grundschule'},
      {'folder':'5.Mean Austria/5.0.all_institutions', 'prefix':'5.0.0.', 'country':'Österreich'},
      {'folder':'5.Mean Austria/5.0.all_institutions', 'prefix':'5.0.1.', 'country':'Österreich', 'condition':'standard'},
      {'folder':'5.Mean Austria/5.0.all_institutions', 'prefix':'5.0.2.', 'country':'Österreich', 'condition':'alemannic_austria'},
      {'folder':'5.Mean Austria/5.1.kindergarden', 'prefix':'5.1.0.', 'country':'Österreich', 'institution':'KiGa'},
      {'folder':'5.Mean Austria/5.1.kindergarden', 'prefix':'5.1.1.', 'country':'Österreich', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'5.Mean Austria/5.1.kindergarden', 'prefix':'5.1.2.', 'country':'Österreich', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'5.Mean Austria/5.2.grundschule', 'prefix':'5.2.0.', 'country':'Österreich', 'institution':'Grundschule'},
      {'folder':'5.Mean Austria/5.2.grundschule', 'prefix':'5.2.1.', 'country':'Österreich', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'5.Mean Austria/5.2.grundschule', 'prefix':'5.2.2.', 'country':'Österreich', 'condition':'alemannic_austria', 'institution':'Grundschule'},
      {'folder':'6.Mean Switzerland/6.0.all_institutions', 'prefix':'6.0.0.', 'country':'Schweiz'},
      {'folder':'6.Mean Switzerland/6.0.all_institutions', 'prefix':'6.0.1.', 'country':'Schweiz', 'condition':'standard'},
      {'folder':'6.Mean Switzerland/6.0.all_institutions', 'prefix':'6.0.2.', 'country':'Schweiz', 'condition':'alemannic_austria'},
      {'folder':'6.Mean Switzerland/6.1.kindergarden', 'prefix':'6.1.0.', 'country':'Schweiz', 'institution':'KiGa'},
      {'folder':'6.Mean Switzerland/6.1.kindergarden', 'prefix':'6.1.1.', 'country':'Schweiz', 'condition':'standard', 'institution':'KiGa'},
      {'folder':'6.Mean Switzerland/6.1.kindergarden', 'prefix':'6.1.2.', 'country':'Schweiz', 'condition':'alemannic_austria', 'institution':'KiGa'},
      {'folder':'6.Mean Switzerland/6.2.grundschule', 'prefix':'6.2.0.', 'country':'Schweiz', 'institution':'Grundschule'},
      {'folder':'6.Mean Switzerland/6.2.grundschule', 'prefix':'6.2.1.', 'country':'Schweiz', 'condition':'standard', 'institution':'Grundschule'},
      {'folder':'6.Mean Switzerland/6.2.grundschule', 'prefix':'6.2.2.', 'country':'Schweiz', 'condition':'alemannic_austria', 'institution':'Grundschule'},
]
def generate_line_plots_by_filter(aggregated_df, filters=filters):
    # Filtration
    def filter_aggregated_df_copy(df,
                            country = 'All countries', 
                            condition = 'All conditions', 
                            institution = 'All institutions' , 
                            session = 'All'):
        assert country in df.Country.unique() or country == 'All countries', f'"{country}" not in df and not default'
        assert condition in df.condition.unique() or condition =='All conditions', f'"{condition}" not in df and not default'
        assert institution in df.Institution.unique() or institution == 'All institutions', f'"{institution}" not in df and not default'
        assert session in df.Session.unique() or session == 'All', f'"{session}" not in df and not default'
        
        return df[ ~df.ifControlGroup &
                ((df.Country == country) if country in df.Country.unique() else True) &
                ((df.condition == condition) if condition in df.condition.unique() else True) &
                ((df.Institution == institution) if institution in df.Institution.unique() else True) & 
                ((df.Session == session) if session in df.Session.unique() else True) & 
                (df.fixation.notna())].copy()
        
    country_map = {'All countries': 'All-ctry',
        'Deutschland': 'DE',
        'Österreich':'AT',
        'Schweiz':'CH'}
    condition_map = {'All conditions': 'All-cond',
        'standard': 'de-std',
        'alemannic_austria':'de-at'}
    institution_map = {'All institutions': 'All-inst',
        'KiGa': 'KiGa',
        'Grundschule':'GS'}
    aoi_map = {
        'semantic_image' : 'Semantischer Distraktor', 
        'phonological_image' : 'Phonologischer Distraktor', 
        'target_image': 'Target',
        'unrelated_image': 'Bezugsfreier Ablenker',
    }
    aggregated_df = aggregated_df.replace({"AOI": aoi_map})
    # no_session_table = pd.read_csv("../Participants/no_session_participants.csv", delimiter=';')
    for filter_dict in filters:
        filter = {'country':'All countries', 
                'condition':'All conditions',
                'institution':'All institutions',
                'session':1}
        filter.update(filter_dict)
        
    
            
        aggr_fix_df = filter_aggregated_df_copy(
            aggregated_df,
            country = filter['country'],
            condition = filter['condition'],
            institution = filter['institution'],
            session = filter['session'])
        
        filtration_session_count = aggr_fix_df.out.nunique()
        title = f"{country_map[filter['country']]}.{condition_map[filter['condition']]}.{institution_map[filter['institution']]}.Ses{filter['session']}.SesCnt={filtration_session_count}"  
        if 'title' in filter: plot_title = filter['title']
        else: 
                plot_title = filter['prefix'] + title # condition_map[filter['condition']]
        
        fig = fixation_proportion_line(aggr_fix_df)
        fig.add_vrect(x0=0, x1=2000, line_width=0, fillcolor="blue", opacity=0.1)
        fig.update_layout(
                title=title,
                title_font=dict(size=14,
                            color='grey',
                            family='Arial'),
                xaxis_title="Zeit in Abhängigkeit zum Stimulus-Onset (ms)",  # Add x-axis name
                yaxis_title="Fixationsproportion (%)",  # Add y-axis name
                font=dict(size=22),
                title_y=0.05,
                margin=dict(l=20, r=20, t=20, b=170),
        )

        # fig.show(config=fig_config)
        file_dir = "C:/Users/Cyril/Desktop/" + filter_dict['folder'] + '/'
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        file_name = plot_title + '.png'
        print(os.path.join(file_dir, plot_title))
        fig.write_image(os.path.join(file_dir, file_name),
                    format='.png', width=1200, height=470, scale=3,engine='orca')
            
def gaze_heatmap_plot(subject_full_df, input_dir, file_path):
    # Points of interest positions
    pois_x, pois_y = 0.5, 0.25 
    pois_xs = np.array([0, -pois_x, pois_x, pois_x, -pois_x])
    pois_ys = np.array([0, pois_y, pois_y, -pois_y, -pois_y])
    # Regions of interest rectangle size
    roi_x, roi_y = 0.5, 0.5
    
    subject = subject_full_df.Subject.unique()[0]
    version = subject_full_df.Version.unique()[0]
    trialNr = subject_full_df.trial.unique()[0]
    
    subject_full_df = subject_full_df[(subject_full_df.timeFromOnsetMs > -0.5)].copy()
    
    subject_df = subject_full_df[(subject_full_df.fixation.notna())].copy()
    
    averaged_fixations_df = subject_df.loc[:, ['fixation', 'gaze_x', 'gaze_y']].groupby('fixation').mean()
    averaged_fixations_df['firstOccurrence'] = subject_df.loc[:, ['fixation', 'timeFromOnsetMs']].groupby('fixation').min().timeFromOnsetMs

    region_signifier = {0:'target',1:'phon. comp.',2:'sem. comp.',3:'unrelated'}
    stimuli_positions = eval(subject_df.posPerm.iloc[0].replace(' ', ','))
    pos_perm_signifier = [region_signifier[pos] for pos in stimuli_positions]

    conditions_df  = pd.read_csv(os.path.join(
        input_dir,
        file_path,
        subject_df.loc[:,'in'].iloc[0][:-5] + ".csv"
    )).iloc[6:26, :6].reset_index(drop=True)
    conditions_df = conditions_df[conditions_df.sound.apply(lambda x: x not in ['Nüschtern', 'Künschtler', 'Nüstern', 'Künstler'])]
    conditions_df.loc[:,'trial'] = range(1, 1+len(conditions_df))
    conditions_df = conditions_df.set_index('trial')
    conditions_df.rename(columns={'target_image': 'target', 
                                'phonological_image': 'phon. comp.', 
                                'semantic_image': 'sem. comp.', 
                                'unrelated_image': 'unrelated'}, inplace=True)

    fig = go.Figure()

    # Add gaze points scatter plot with time-coded colors
    fig.add_trace(go.Scatter(
        x=subject_df.gaze_x,
        y=subject_df.gaze_y,
        mode='markers',
        marker=dict(
            color=subject_df.fixation,
            colorscale='Viridis',  # Color gradient
            showscale=False,
            opacity=0.7
        ),
        name='Gaze Points'
    ))

    # Add Points of Interest (POIs)
    fig.add_trace(go.Scatter(
        x=pois_xs,
        y=pois_ys,
        mode='markers',
        marker=dict(color='blue', symbol='x', size=10),
        name='Regions of Interest'
    ))

    # Add ROI title
    fig.add_trace(go.Scatter(
        x=pois_xs[1:]-0.2,
        y=pois_ys[1:]+0.2,
        mode='text',
        text=pos_perm_signifier,  # Labels for each POI
        textposition='top right',  # Adjust text position
        name='ROI title'
    ))

    # Add Regions of Interest (ROIs) and Images
    for xp, yp, img_type in zip(pois_xs[1:], pois_ys[1:], pos_perm_signifier):
        image_file = conditions_df.loc[trialNr, img_type]
        fig.add_layout_image(
                dict(
                    source=Image.open(os.path.join(r'C:\Users\Cyril\HESSENBOX\Eye-Tracking_LAVA (Jasmin Devi Nuscheler)\Experiment_PsychoPy\A\Stims', image_file + '.png')),
                    xref="x",
                    yref="y",
                    x=xp-roi_x/2,
                    y=yp+roi_y/2,
                    sizex=roi_x,
                    sizey=roi_y,
                    sizing="stretch",
                    layer="below")
        )
        fig.add_shape(type="rect",
            xref="x", yref="y",
            x0=xp-roi_x/2, y0=yp-roi_y/2,
            x1=xp+roi_x/2, y1=yp+roi_y/2,
            line=dict(
                color="RoyalBlue",
                width=2,
            ),
        )

    # Add Heatmap
    fig.add_trace(go.Histogram2d(
        x=subject_full_df.gaze_x,
        y=subject_full_df.gaze_y,
        colorscale='Blues',
        zmax=10,
        xbins = dict(start=-1, end=1, size=0.075),
        ybins = dict(start=-1, end=1, size=0.075),
        zauto=False,
        opacity=0.5,
        showscale=False,
    ))

    # Add Average Fixation Positions
    fig.add_trace(go.Scatter(
        x=averaged_fixations_df.gaze_x,
        y=averaged_fixations_df.gaze_y,
        mode='markers',
        marker=dict(
            color=averaged_fixations_df.index,  # Use the DataFrame index for coloring
            colorscale='Viridis',  # Choose a color scale
            size=14,               # Marker size
            symbol='cross',         # Use a '+' symbol
            line=dict(
                width=1,           # Border width
                color='ivory'      # Border color
            )
        ),
        name='Average Fixation Positions'
    ))

    # Add a slider to filter data points by time codes
    fig.update_layout(
        sliders=[{
            'steps': [
                {
                    'args': [{'x': [subject_df[subject_df.timeFromOnsetMs < t].gaze_x, 
                                    pois_xs,
                                    pois_xs[1:]-0.2,
                                    subject_full_df[subject_full_df.timeFromOnsetMs < t].gaze_x, 
                                    averaged_fixations_df[averaged_fixations_df.firstOccurrence < t].gaze_x],
                            'y': [subject_df[subject_df.timeFromOnsetMs < t].gaze_y, 
                                    pois_ys, 
                                    pois_ys[1:]+0.2,
                                    subject_full_df[subject_full_df.timeFromOnsetMs < t].gaze_y, 
                                    averaged_fixations_df[averaged_fixations_df.firstOccurrence < t].gaze_y]}],
                    'label': str(t),
                    'method': 'update'
                } for t in averaged_fixations_df.round({'firstOccurrence': 2}).firstOccurrence.iloc[1:]
            ],
            'currentvalue': {'prefix': 'Time: '}
        }],
        title=f'Fixations in time - Audio stimulus: {conditions_df.loc[trialNr, "sound"]}',
        xaxis=dict(title='Gaze X'),
        yaxis=dict(title='Gaze Y')
    )

    fig.update_layout(
        width=1000, 
        height=650,  
        xaxis=dict(
            range= [-0.8,0.8]
        ),
        yaxis=dict(
            range= [-0.55,0.55]
        ),
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )
    fig.show()
    
def dplt_plot(aggregated_df):
    filt_agg_df = aggregated_df[(aggregated_df.Country != "Control Group") &
                                (aggregated_df.Session == 1) &
                                aggregated_df.AOI.notnull()]
    dplt_df = filt_agg_df.groupby(["out", "Country", "condition"]).apply(
        lambda frame: 
            (frame[frame.timeFromOnsetMs >= 0].AOI == "target_image").mean() - \
            (frame[(frame.timeFromOnsetMs < 0) & (frame.TimeFromTrialOnset > 1)].AOI == "target_image").mean(),
            include_groups=False)
    dplt_df = dplt_df.rename("dplt").reset_index()

    dplt_df = dplt_df.loc[:, ["Country", "condition", "dplt"]].groupby(["Country", "condition"]).agg(dpltMean=('dplt', 'mean'), dpltSEM=('dplt', 'sem')).reset_index()
    dplt_mean = pd.pivot(dplt_df, index='Country', columns='condition', values="dpltMean")
    dplt_sem = pd.pivot(dplt_df, index='Country', columns='condition', values="dpltSEM")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='standard',
        x=['Deutschland', 'Schweiz', 'Österreich'], y=dplt_mean.T.loc["standard", :],
        error_y=dict(type='data', array=dplt_sem.T.loc["standard", :])
    ))
    fig.add_trace(go.Bar(
        name='alemannic_austria',
        x=['Deutschland', 'Schweiz', 'Österreich'], y=dplt_mean.T.loc["alemannic_austria", :],
        error_y=dict(type='data', array=dplt_sem.T.loc["alemannic_austria", :])
    ))
    fig.update_layout(barmode='group')
    return fig