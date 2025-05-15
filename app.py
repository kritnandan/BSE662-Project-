import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
import os

# Create a results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Set page config
st.set_page_config(
    page_title="PRL Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Probabilistic Reward Learning (PRL) Analysis Dashboard")
st.markdown("""
This interactive dashboard allows you to analyze data from a Probabilistic Reward Learning experiment.
The experiment includes different blocks with varying reward probabilities (80/20 and 50/50).
""")

@st.cache_data
def load_and_preprocess_data():
    """
    Load data from master.csv and preprocess it:
    1. Filter out practice trials
    2. Handle missing data
    3. Identify optimal choices in 80/20 block
    """
    # Load data
    data_path = 'new_master.csv'
    df = pd.read_csv(data_path)
    
    # Filter out practice trials
    df_filtered = df[(df['is_practice'] == False) & (df['block'] != 'Practice')]
    
    # Handle missing data
    missing_selected = df_filtered['selected_option'].isna().sum()
    missing_rt = df_filtered['response_time_ms'].isna().sum()
    
    # Remove rows with missing selected_option or response_time_ms
    # Create an explicit copy to avoid SettingWithCopyWarning
    df_clean = df_filtered.dropna(subset=['selected_option', 'response_time_ms']).copy()
    
    # Identify optimal choice for 80/20 block
    def get_optimal_choice(row):
        if row['block'] != '80/20':
            return np.nan
        
        # Determine which option has the 0.8 probability
        if row['left_option_prob'] == 0.8:
            return 0  # Left is optimal (0)
        elif row['right_option_prob'] == 0.8:
            return 1  # Right is optimal (1)
        else:
            return np.nan
    
    df_clean['optimal_choice'] = df_clean.apply(get_optimal_choice, axis=1)
    
    # Mark correct choices in 80/20 block - UPDATED LOGIC
    df_clean['chose_optimal'] = False  # Default to False
    
    # Check if selected_option = 0 (left) and left_option_prob = 0.8, OR if selected_option = 1 (right) and right_option_prob = 0.8
    condition1 = (df_clean['selected_option'] == 0) & (df_clean['left_option_prob'] == 0.8)
    condition2 = (df_clean['selected_option'] == 1) & (df_clean['right_option_prob'] == 0.8)
    df_clean.loc[condition1 | condition2, 'chose_optimal'] = True
    
    # For 50/50 block, track if choice matches first choice in the block
    df_5050 = df_clean[df_clean['block'] == '50/50'].copy()
    
    # Group by participant to find first choice in block
    first_choices = df_5050.groupby('participant_id').first()['selected_option']
    
    # Create a dictionary of participant_id -> first_choice
    first_choice_dict = first_choices.to_dict()
    
    # Function to check if the current choice matches first choice
    def matches_first_choice(row):
        if row['block'] != '50/50':
            return np.nan
        return 1 if row['selected_option'] == first_choice_dict.get(row['participant_id']) else 0
    
    df_clean['matches_first_choice'] = df_clean.apply(matches_first_choice, axis=1)
    
    # Add a column for "switched from previous trial" based on shape selection
    df_clean['prev_shape'] = df_clean.groupby(['participant_id', 'block'])['selected_shape'].shift(1)
    df_clean['switched'] = (df_clean['selected_shape'] != df_clean['prev_shape']).astype(int)

    # Calculate total number of switches per participant per block
    df_clean['participant_block'] = df_clean['participant_id'].astype(str) + '_' + df_clean['block']
    switch_counts = df_clean.groupby('participant_block')['switched'].sum().to_dict()
    df_clean['switch_count'] = df_clean['participant_block'].map(switch_counts)
    
    return df_clean, missing_selected, missing_rt

# Load data
with st.spinner("Loading and preprocessing data..."):
    df, missing_selected, missing_rt = load_and_preprocess_data()

# Display data info in sidebar
st.sidebar.header("Dataset Information")
total_participants = df['participant_id'].nunique()
st.sidebar.metric("Number of Participants", total_participants)

# Preprocessing summary
with st.sidebar.expander("Preprocessing Details"):
    st.write(f"Original rows after excluding practice: {len(df) + missing_selected}")
    st.write(f"Missing data in selected_option: {missing_selected} ({missing_selected/(len(df) + missing_selected)*100:.2f}%)")
    st.write(f"Missing data in response_time_ms: {missing_rt} ({missing_rt/(len(df) + missing_rt)*100:.2f}%)")
    st.write(f"Final rows after preprocessing: {len(df)}")

# Global controls in sidebar
st.sidebar.header("Analysis Controls")

# Trial bin size selection
bin_size = st.sidebar.slider("Trial Bin Size", min_value=1, max_value=20, value=10, 
                            help="Number of trials to group in each bin for analysis")

# Add trial bin to DataFrame based on selected bin size
df['trial_bin'] = ((df['trial_number'] - 1) // bin_size) + 1

# Participant selection
all_participants = sorted(df['participant_id'].unique())
selected_participants = st.sidebar.multiselect(
    "Select Participants for Individual Analysis",
    options=["All Participants"] + list(all_participants),
    default=["All Participants"],
    help="Select specific participants to analyze individually"
)

# If 'All Participants' is selected along with other options, keep only 'All Participants'
if "All Participants" in selected_participants and len(selected_participants) > 1:
    selected_participants = ["All Participants"]

# Create filtered dataframe for selected participants
if "All Participants" in selected_participants:
    df_selected = df.copy()
    label_for_all = "Group Average"
else:
    df_selected = df[df['participant_id'].isin(selected_participants)]
    label_for_all = None

# Main tabs for different analyses
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Learning Curves", 
    "Response Times", 
    "Exploration/Exploitation", 
    "Score Dynamics",
    "Individual Differences",
    "Participant Comparison"
])

# ===== 1. Learning Curves Analysis =====
with tab1:
    st.header("Learning Curves Analysis")
    st.markdown("""
    This analysis shows how participants learn to choose the optimal option (in 80/20 block)
    and how consistent their choices are (in 50/50 block) over time.
    """)
    
    # Create figures for learning curves
    fig_learning = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Learning Curve: 80/20 Block", "Choice Stability: 50/50 Block"))
    
    # Function to add data to plots
    def add_learning_data(data, name, color):
        # 80/20 block data
        data_8020 = data[data['block'] == '80/20']
        optimal_by_bin = data_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
        
        # Add to plot
        fig_learning.add_trace(
            go.Scatter(
                x=optimal_by_bin['trial_bin'], 
                y=optimal_by_bin['chose_optimal'],
                mode='lines+markers',
                name=f"{name} (80/20)",
                line=dict(color=color),
                legendgroup=name
            ),
            row=1, col=1
        )
        
        # 50/50 block data
        data_5050 = data[data['block'] == '50/50']
        matches_by_bin = data_5050.groupby('trial_bin')['matches_first_choice'].mean().reset_index()
        
        # Add to plot
        fig_learning.add_trace(
            go.Scatter(
                x=matches_by_bin['trial_bin'], 
                y=matches_by_bin['matches_first_choice'],
                mode='lines+markers',
                name=f"{name} (50/50)",
                line=dict(color=color, dash='dash'),
                legendgroup=name
            ),
            row=1, col=2
        )
    
    # Add group average
    add_learning_data(df, label_for_all or "Group Average", 'royalblue')
    
    # Add individual participants if selected
    colors = px.colors.qualitative.Set2
    if "All Participants" not in selected_participants:
        for i, participant in enumerate(selected_participants):
            participant_data = df[df['participant_id'] == participant]
            add_learning_data(participant_data, f"Participant {participant}", colors[i % len(colors)])
    
    # Update figure layout
    fig_learning.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Add horizontal reference line at 0.5
    fig_learning.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=0, y0=0.5, x1=df['trial_bin'].max(), y1=0.5, row=1, col=1
    )
    fig_learning.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=0, y0=0.5, x1=df['trial_bin'].max(), y1=0.5, row=1, col=2
    )
    
    fig_learning.update_xaxes(title_text="Trial Bin", row=1, col=1)
    fig_learning.update_xaxes(title_text="Trial Bin", row=1, col=2)
    fig_learning.update_yaxes(title_text="Proportion of Optimal Choices", row=1, col=1)
    fig_learning.update_yaxes(title_text="Proportion Matching First Choice", row=1, col=2)
    fig_learning.update_yaxes(range=[0.4, 1.0], row=1, col=1)
    fig_learning.update_yaxes(range=[0.4, 1.0], row=1, col=2)
    
    st.plotly_chart(fig_learning, use_container_width=True)
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("80/20 Block Learning Summary")
        df_8020 = df_selected[df_selected['block'] == '80/20']
        
        # First bin vs last bin
        first_bin = df_8020[df_8020['trial_bin'] == 1]['chose_optimal'].mean()
        last_bin = df_8020[df_8020['trial_bin'] == df_8020['trial_bin'].max()]['chose_optimal'].mean()
        
        # Calculate learning slope (linear regression)
        optimal_by_bin = df_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
        
        if len(optimal_by_bin) > 1:  # Need at least 2 points for regression
            x = optimal_by_bin['trial_bin'].values
            y = optimal_by_bin['chose_optimal'].values
            slope_8020 = np.polyfit(x, y, 1)[0]
        else:
            slope_8020 = 0
        
        metrics_df = pd.DataFrame({
            'Metric': ['First Bin Optimal Rate', 'Last Bin Optimal Rate', 'Learning Slope'],
            'Value': [f"{first_bin:.2f}", f"{last_bin:.2f}", f"{slope_8020:.4f}"]
        })
        
        st.dataframe(metrics_df, hide_index=True)
    
    with col2:
        st.subheader("50/50 Block Choice Stability")
        df_5050 = df_selected[df_selected['block'] == '50/50']
        
        # First bin vs last bin
        first_bin = df_5050[df_5050['trial_bin'] == 1]['matches_first_choice'].mean()
        last_bin = df_5050[df_5050['trial_bin'] == df_5050['trial_bin'].max()]['matches_first_choice'].mean()
        
        # Calculate stability slope
        stability_by_bin = df_5050.groupby('trial_bin')['matches_first_choice'].mean().reset_index()
        
        if len(stability_by_bin) > 1:  # Need at least 2 points for regression
            x = stability_by_bin['trial_bin'].values
            y = stability_by_bin['matches_first_choice'].values
            slope_5050 = np.polyfit(x, y, 1)[0]
        else:
            slope_5050 = 0
        
        metrics_df = pd.DataFrame({
            'Metric': ['First Bin Stability', 'Last Bin Stability', 'Stability Slope'],
            'Value': [f"{first_bin:.2f}", f"{last_bin:.2f}", f"{slope_5050:.4f}"]
        })
        
        st.dataframe(metrics_df, hide_index=True)

# ===== 2. Response Times Analysis =====
with tab2:
    st.header("Response Times Analysis")
    st.markdown("""
    This analysis examines response times (RT) across different conditions and over time.
    """)
    
    # Overall RT metrics by block
    col1, col2 = st.columns(2)
    
    rt_by_block = df_selected.groupby('block')['response_time_ms'].mean()
    rt_std_by_block = df_selected.groupby('block')['response_time_ms'].std() / np.sqrt(df_selected.groupby('block')['response_time_ms'].count())
    
    with col1:
        st.metric("Average RT in 80/20 Block", f"{rt_by_block.get('80/20', 0):.2f} ms")
    
    with col2:
        st.metric("Average RT in 50/50 Block", f"{rt_by_block.get('50/50', 0):.2f} ms")
    
    # RT Dynamics over time
    st.subheader("Response Time Dynamics")
    
    fig_rt = make_subplots(rows=1, cols=2, 
                          subplot_titles=("RT Dynamics: 80/20 Block", "RT Dynamics: 50/50 Block"))
    
    # Function to add RT data to plots
    def add_rt_data(data, name, color):
        # 80/20 block data
        data_8020 = data[data['block'] == '80/20']
        rt_by_bin = data_8020.groupby('trial_bin')['response_time_ms'].mean().reset_index()
        
        fig_rt.add_trace(
            go.Scatter(
                x=rt_by_bin['trial_bin'], 
                y=rt_by_bin['response_time_ms'],
                mode='lines+markers',
                name=f"{name} (80/20)",
                line=dict(color=color),
                legendgroup=name
            ),
            row=1, col=1
        )
        
        # 50/50 block data
        data_5050 = data[data['block'] == '50/50']
        rt_by_bin = data_5050.groupby('trial_bin')['response_time_ms'].mean().reset_index()
        
        fig_rt.add_trace(
            go.Scatter(
                x=rt_by_bin['trial_bin'], 
                y=rt_by_bin['response_time_ms'],
                mode='lines+markers',
                name=f"{name} (50/50)",
                line=dict(color=color, dash='dash'),
                legendgroup=name
            ),
            row=1, col=2
        )
    
    # Add group average
    add_rt_data(df_selected, label_for_all or "Group Average", 'royalblue')
    
    # Add individual participants if selected
    if "All Participants" not in selected_participants:
        for i, participant in enumerate(selected_participants):
            participant_data = df[df['participant_id'] == participant]
            add_rt_data(participant_data, f"Participant {participant}", colors[i % len(colors)])
    
    fig_rt.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig_rt.update_xaxes(title_text="Trial Bin", row=1, col=1)
    fig_rt.update_xaxes(title_text="Trial Bin", row=1, col=2)
    fig_rt.update_yaxes(title_text="Response Time (ms)", row=1, col=1)
    fig_rt.update_yaxes(title_text="Response Time (ms)", row=1, col=2)
    
    st.plotly_chart(fig_rt, use_container_width=True)
    
    # RT analysis by choice type and previous feedback
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RT by Choice Type (80/20 Block)")
        
        df_8020 = df_selected[df_selected['block'] == '80/20']
        
        if not df_8020.empty:
            rt_by_choice = df_8020.groupby('chose_optimal')['response_time_ms'].mean().reset_index()
            rt_by_choice['Choice Type'] = rt_by_choice['chose_optimal'].map({True: 'Optimal', False: 'Suboptimal'})
            
            fig_choice = px.bar(
                rt_by_choice, 
                x='Choice Type', 
                y='response_time_ms',
                color='Choice Type',
                color_discrete_map={'Optimal': 'green', 'Suboptimal': 'red'},
                title="RT by Choice Type",
                labels={'response_time_ms': 'Response Time (ms)'}
            )
            
            st.plotly_chart(fig_choice, use_container_width=True)
            
            # Print values
            optimal_rt = rt_by_choice[rt_by_choice['chose_optimal'] == True]['response_time_ms'].values[0] if True in rt_by_choice['chose_optimal'].values else 0
            suboptimal_rt = rt_by_choice[rt_by_choice['chose_optimal'] == False]['response_time_ms'].values[0] if False in rt_by_choice['chose_optimal'].values else 0
            
            st.markdown(f"""
            - Optimal choice RT: **{optimal_rt:.2f}** ms
            - Suboptimal choice RT: **{suboptimal_rt:.2f}** ms
            - Difference: **{optimal_rt - suboptimal_rt:.2f}** ms
            """)
        else:
            st.info("No data available for the selected participants in the 80/20 block.")
    
    with col2:
        st.subheader("RT by Previous Feedback")
        
        # Add previous trial feedback
        df_selected_copy = df_selected.copy()
        df_selected_copy.loc[:, 'prev_feedback'] = df_selected_copy.groupby(['participant_id', 'block'])['feedback'].shift(1)
        
        # Filter out trials without previous feedback
        df_with_prev = df_selected_copy.dropna(subset=['prev_feedback'])
        
        if not df_with_prev.empty:
            rt_by_feedback = df_with_prev.groupby('prev_feedback')['response_time_ms'].mean().reset_index()
            
            fig_feedback = px.bar(
                rt_by_feedback, 
                x='prev_feedback', 
                y='response_time_ms',
                color='prev_feedback',
                color_discrete_map={'positive': 'green', 'negative': 'red'},
                title="RT by Previous Feedback",
                labels={'response_time_ms': 'Response Time (ms)', 'prev_feedback': 'Previous Feedback'}
            )
            
            st.plotly_chart(fig_feedback, use_container_width=True)
            
            # Print values
            positive_rt = rt_by_feedback[rt_by_feedback['prev_feedback'] == 'positive']['response_time_ms'].values[0] if 'positive' in rt_by_feedback['prev_feedback'].values else 0
            negative_rt = rt_by_feedback[rt_by_feedback['prev_feedback'] == 'negative']['response_time_ms'].values[0] if 'negative' in rt_by_feedback['prev_feedback'].values else 0
            
            st.markdown(f"""
            - After positive feedback: **{positive_rt:.2f}** ms
            - After negative feedback: **{negative_rt:.2f}** ms
            - Difference: **{positive_rt - negative_rt:.2f}** ms
            """)
        else:
            st.info("No data available for RT analysis by previous feedback.")

# ===== 3. Exploration/Exploitation Analysis =====
with tab3:
    st.header("Exploration vs. Exploitation Analysis")
    st.markdown("""
    This analysis examines how participants balance exploration (trying different options) 
    versus exploitation (sticking with a known option).
    """)
    
    # Overall switch rate by block
    switch_by_block = df_selected.groupby(['block', 'participant_id'])['switch_count'].max().groupby('block').mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Number of Switches in 80/20 Block", f"{switch_by_block.get('80/20', 0):.1f}")
    
    with col2:
        st.metric("Average Number of Switches in 50/50 Block", f"{switch_by_block.get('50/50', 0):.1f}")
    
    # Switch Rate Dynamics over time
    st.subheader("Exploration Dynamics (Switch Rate Over Time)")
    
    fig_switch = make_subplots(rows=1, cols=2, 
                              subplot_titles=("Exploration Dynamics: 80/20 Block", 
                                             "Exploration Dynamics: 50/50 Block"))
    
    # Function to add switch rate data to plots
    def add_switch_data(data, name, color):
        # 80/20 block data
        data_8020 = data[data['block'] == '80/20']
        switch_by_bin = data_8020.groupby('trial_bin')['switched'].mean().reset_index()
        
        fig_switch.add_trace(
            go.Scatter(
                x=switch_by_bin['trial_bin'], 
                y=switch_by_bin['switched'],
                mode='lines+markers',
                name=f"{name} (80/20)",
                line=dict(color=color),
                legendgroup=name
            ),
            row=1, col=1
        )
        
        # 50/50 block data
        data_5050 = data[data['block'] == '50/50']
        switch_by_bin = data_5050.groupby('trial_bin')['switched'].mean().reset_index()
        
        fig_switch.add_trace(
            go.Scatter(
                x=switch_by_bin['trial_bin'], 
                y=switch_by_bin['switched'],
                mode='lines+markers',
                name=f"{name} (50/50)",
                line=dict(color=color, dash='dash'),
                legendgroup=name
            ),
            row=1, col=2
        )
    
    # Add group average
    add_switch_data(df_selected, label_for_all or "Group Average", 'royalblue')
    
    # Add individual participants if selected
    if "All Participants" not in selected_participants:
        for i, participant in enumerate(selected_participants):
            participant_data = df[df['participant_id'] == participant]
            add_switch_data(participant_data, f"Participant {participant}", colors[i % len(colors)])
    
    fig_switch.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig_switch.update_xaxes(title_text="Trial Bin", row=1, col=1)
    fig_switch.update_xaxes(title_text="Trial Bin", row=1, col=2)
    fig_switch.update_yaxes(title_text="Switch Rate", row=1, col=1)
    fig_switch.update_yaxes(title_text="Switch Rate", row=1, col=2)
    fig_switch.update_yaxes(range=[0, 1], row=1, col=1)
    fig_switch.update_yaxes(range=[0, 1], row=1, col=2)
    
    st.plotly_chart(fig_switch, use_container_width=True)
    
    # Calculate choice entropy as another measure of exploration
    st.subheader("Choice Entropy Analysis")
    st.markdown("""
    Shannon entropy measures the unpredictability of choices. Higher entropy indicates more exploration.
    """)
    
    def calculate_entropy(choices):
        """Calculate Shannon entropy of choices"""
        unique, counts = np.unique(choices, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))
    
    # Calculate entropy for each participant, block, and trial bin
    if not df_selected.empty:
        entropy_data = []
        
        for (participant, block, bin_num), group in df_selected.groupby(['participant_id', 'block', 'trial_bin']):
            choices = group['selected_option'].values
            if len(choices) > 1:  # Need at least 2 choices to calculate meaningful entropy
                entropy = calculate_entropy(choices)
                entropy_data.append({
                    'participant_id': participant,
                    'block': block,
                    'trial_bin': bin_num,
                    'entropy': entropy
                })
        
        if entropy_data:
            entropy_df = pd.DataFrame(entropy_data)
            
            # Calculate mean entropy by block and trial bin
            if "All Participants" in selected_participants:
                entropy_by_block_bin = entropy_df.groupby(['block', 'trial_bin'])['entropy'].mean().reset_index()
            else:
                entropy_by_block_bin = entropy_df
            
            # Plot entropy over time
            fig_entropy = px.line(
                entropy_by_block_bin, 
                x='trial_bin', 
                y='entropy', 
                color='block',
                color_discrete_map={'80/20': 'blue', '50/50': 'orange'},
                markers=True,
                title="Choice Entropy Over Time",
                labels={'trial_bin': 'Trial Bin', 'entropy': 'Shannon Entropy', 'block': 'Block'}
            )
            
            fig_entropy.update_layout(height=400)
            fig_entropy.update_yaxes(range=[0, 1])
            
            st.plotly_chart(fig_entropy, use_container_width=True)
            
            # Display average entropy by block
            entropy_by_block = entropy_df.groupby('block')['entropy'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Entropy in 80/20 Block", f"{entropy_by_block.get('80/20', 0):.2f}")
            
            with col2:
                st.metric("Average Entropy in 50/50 Block", f"{entropy_by_block.get('50/50', 0):.2f}")
            
            st.info("Higher entropy (closer to 1) indicates more exploration/randomness in choices.")
        else:
            st.info("Not enough data to calculate entropy for the selected participants or trial bins.")
    else:
        st.info("No data available for entropy analysis.")

# ===== 4. Score Dynamics Analysis =====
with tab4:
    st.header("Score Dynamics Analysis")
    st.markdown("""
    This analysis examines how participants accumulate points over time in different blocks.
    """)
    
    # Function to get score data for plotting
    def get_score_data(data, block_type):
        block_data = data[data['block'] == block_type]
        
        if "All Participants" in selected_participants:
            # Group average
            # For 50/50 block, we'll reset the cumulative score to start from 0
            if block_type == '50/50':
                # Get the initial score at the start of 50/50 block for each participant
                initial_scores = {}
                for pid, group in block_data.groupby('participant_id'):
                    # Find the first trial's total_points and subtract from all values
                    initial_score = group['total_points'].iloc[0]
                    initial_scores[pid] = initial_score
                
                # Create adjusted scores
                block_data_copy = block_data.copy()
                block_data_copy['adjusted_total_points'] = block_data_copy.apply(
                    lambda row: row['total_points'] - initial_scores[row['participant_id']], 
                    axis=1
                )
                
                avg_scores = block_data_copy.groupby('trial_number')['adjusted_total_points'].mean().reset_index()
                avg_scores.rename(columns={'adjusted_total_points': 'total_points'}, inplace=True)
                return avg_scores
            else:
                # For 80/20 block, keep as is
                avg_scores = block_data.groupby('trial_number')['total_points'].mean().reset_index()
                return avg_scores
        else:
            # Individual participants
            if block_type == '50/50':
                # For selected participants, also adjust 50/50 scores
                block_data_copy = block_data.copy()
                for pid, group in block_data_copy.groupby('participant_id'):
                    initial_score = group['total_points'].iloc[0]
                    block_data_copy.loc[group.index, 'total_points'] = group['total_points'] - initial_score
                
                return block_data_copy[['participant_id', 'trial_number', 'total_points']]
            else:
                return block_data[['participant_id', 'trial_number', 'total_points']]
    
    # Get data for both blocks
    data_8020 = get_score_data(df_selected, '80/20')
    data_5050 = get_score_data(df_selected, '50/50')
    
    # Create plot
    fig_score = go.Figure()
    
    # Add traces based on selection
    if "All Participants" in selected_participants:
        # Add average traces
        fig_score.add_trace(
            go.Scatter(
                x=data_8020['trial_number'],
                y=data_8020['total_points'],
                mode='lines',
                name='80/20 Block (Avg)',
                line=dict(color='blue')
            )
        )
        
        fig_score.add_trace(
            go.Scatter(
                x=data_5050['trial_number'],
                y=data_5050['total_points'],
                mode='lines',
                name='50/50 Block (Avg)',
                line=dict(color='orange')
            )
        )
    else:
        # Add individual participant traces
        for participant in selected_participants:
            part_data_8020 = data_8020[data_8020['participant_id'] == participant]
            part_data_5050 = data_5050[data_5050['participant_id'] == participant]
            
            fig_score.add_trace(
                go.Scatter(
                    x=part_data_8020['trial_number'],
                    y=part_data_8020['total_points'],
                    mode='lines',
                    name=f'Participant {participant} (80/20)',
                    line=dict(color='blue')
                )
            )
            
            fig_score.add_trace(
                go.Scatter(
                    x=part_data_5050['trial_number'],
                    y=part_data_5050['total_points'],
                    mode='lines',
                    name=f'Participant {participant} (50/50)',
                    line=dict(color='orange')
                )
            )
    
    fig_score.update_layout(
        title="Cumulative Score Over Trials",
        xaxis_title="Trial Number",
        yaxis_title="Cumulative Score (points)",
        height=500
    )
    
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Final scores comparison
    col1, col2, col3 = st.columns(3)
    
    # Calculate final scores for selected participants
    # 80/20 block final score (unchanged)
    final_score_8020 = df_selected[df_selected['block'] == '80/20'].groupby('participant_id')['total_points'].max().mean()
    
    # 50/50 block final score (points gained only in 50/50 block)
    df_5050 = df_selected[df_selected['block'] == '50/50']
    df_8020 = df_selected[df_selected['block'] == '80/20']
    
    # Get starting and ending points for 50/50 block by participant
    score_diff_5050 = {}
    total_final_scores = {}
    
    for pid in df_5050['participant_id'].unique():
        # Get initial score at start of 50/50 block
        initial_score = df_5050[df_5050['participant_id'] == pid]['total_points'].iloc[0]
        
        # Get final score at end of 50/50 block
        final_score = df_5050[df_5050['participant_id'] == pid]['total_points'].max()
        
        # Store the difference (points gained only in 50/50 block)
        score_diff_5050[pid] = final_score - initial_score
        
        # Get final score from 80/20 block
        final_score_80_20 = df_8020[df_8020['participant_id'] == pid]['total_points'].max() if len(df_8020[df_8020['participant_id'] == pid]) > 0 else 0
        
        # Store total final score (sum of 80/20 final + 50/50 points gained)
        total_final_scores[pid] = final_score_80_20 + (final_score - initial_score)
    
    # Calculate average scores
    final_score_5050_only = np.mean(list(score_diff_5050.values())) if score_diff_5050 else 0
    total_final_score = np.mean(list(total_final_scores.values())) if total_final_scores else 0
    
    with col1:
        st.metric("Average Final Score (80/20 Block)", f"{final_score_8020:.2f}")
    
    with col2:
        st.metric("Average Final Score (50/50 Block Only)", f"{final_score_5050_only:.2f}")
        
    with col3:
        st.metric("Average Total Final Score (80/20 + 50/50)", f"{total_final_score:.2f}")
    
    # Score accumulation rate analysis
    st.subheader("Score Accumulation Rate")
    
    # Calculate linear regression for each block's scoring rate
    def calc_score_slope(data):
        if len(data) > 1:
            x = data['trial_number'].values
            y = data['total_points'].values
            slope = np.polyfit(x, y, 1)[0]
            return slope
        else:
            return 0
    
    if "All Participants" in selected_participants:
        slope_8020 = calc_score_slope(data_8020)
        slope_5050 = calc_score_slope(data_5050)
        
        score_slopes = pd.DataFrame({
            'Block': ['80/20 Block', '50/50 Block'],
            'Score Accumulation Rate (points/trial)': [slope_8020, slope_5050]
        })
        
        st.dataframe(score_slopes, hide_index=True)
    else:
        # Calculate for each selected participant
        score_slopes = []
        
        for participant in selected_participants:
            part_data_8020 = data_8020[data_8020['participant_id'] == participant]
            part_data_5050 = data_5050[data_5050['participant_id'] == participant]
            
            slope_8020 = calc_score_slope(part_data_8020)
            slope_5050 = calc_score_slope(part_data_5050)
            
            score_slopes.append({
                'Participant': participant,
                '80/20 Block Rate': slope_8020,
                '50/50 Block Rate': slope_5050
            })
        
        score_slopes_df = pd.DataFrame(score_slopes)
        st.dataframe(score_slopes_df, hide_index=True)

# ===== 5. Individual Differences Analysis =====
with tab5:
    st.header("Individual Differences Analysis")
    st.markdown("""
    This analysis examines how participants differ in their learning, exploration strategies,
    and performance in the 80/20 block.
    """)
    
    # Calculate metrics for each participant in 80/20 block
    if not df.empty:
        df_8020 = df[df['block'] == '80/20']
        
        if not df_8020.empty:
            participant_metrics = []
            
            for participant_id, group in df_8020.groupby('participant_id'):
                # Overall proportion of optimal choices
                optimal_proportion = group['chose_optimal'].mean()
                
                # Average RT
                avg_rt = group['response_time_ms'].mean()
                
                # Average switch rate (focusing on later trials, after learning)
                later_trials = group[group['trial_number'] > 30]  # Second half of trials
                switch_rate = later_trials['switched'].mean() if len(later_trials) > 0 else np.nan
                
                # Final cumulative score
                final_score = group['total_points'].max()
                
                participant_metrics.append({
                    'participant_id': participant_id,
                    'optimal_proportion': optimal_proportion,
                    'avg_rt': avg_rt,
                    'switch_rate': switch_rate,
                    'final_score': final_score
                })
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(participant_metrics)
            
            # Display metrics table
            st.subheader("Participant Metrics in 80/20 Block")
            
            # Format the dataframe for display
            display_df = metrics_df.copy()
            display_df.columns = [
                'Participant ID', 
                'Optimal Choice %', 
                'Avg RT (ms)',
                'Switch Rate (later trials)',
                'Final Score'
            ]
            
            # Format percentages
            display_df['Optimal Choice %'] = display_df['Optimal Choice %'].map(lambda x: f"{x:.1%}")
            display_df['Switch Rate (later trials)'] = display_df['Switch Rate (later trials)'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
            
            # Format as sortable dataframe
            st.dataframe(display_df, hide_index=True)
            
            # Scatter plot matrix
            st.subheader("Relationships Between Metrics")
            st.markdown("Select metrics to compare:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox(
                    "X-axis metric",
                    options=[
                        'optimal_proportion', 
                        'avg_rt', 
                        'switch_rate', 
                        'final_score'
                    ],
                    format_func=lambda x: {
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score'
                    }[x],
                    index=0
                )
            
            with col2:
                y_metric = st.selectbox(
                    "Y-axis metric",
                    options=[
                        'optimal_proportion', 
                        'avg_rt', 
                        'switch_rate', 
                        'final_score'
                    ],
                    format_func=lambda x: {
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score'
                    }[x],
                    index=3
                )
            
            # Create scatter plot
            fig_scatter = px.scatter(
                metrics_df,
                x=x_metric,
                y=y_metric,
                hover_data=['participant_id'],
                labels={
                    'optimal_proportion': 'Optimal Choice %',
                    'avg_rt': 'Average RT (ms)',
                    'switch_rate': 'Switch Rate',
                    'final_score': 'Final Score',
                    'participant_id': 'Participant ID'
                },
                title=f"Relationship between {x_metric.replace('_', ' ').title()} and {y_metric.replace('_', ' ').title()}"
            )
            
            # Highlight selected participants if applicable
            if "All Participants" not in selected_participants:
                selected_metrics = metrics_df[metrics_df['participant_id'].isin(selected_participants)]
                
                if not selected_metrics.empty:
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=selected_metrics[x_metric],
                            y=selected_metrics[y_metric],
                            mode='markers',
                            marker=dict(color='red', size=12, line=dict(width=2, color='DarkSlateGrey')),
                            name='Selected Participants',
                            text=selected_metrics['participant_id']
                        )
                    )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Try to perform clustering if scikit-learn is available
            st.subheader("Participant Clustering")
            
            try:
                from sklearn.cluster import KMeans, AgglomerativeClustering
                from sklearn.preprocessing import StandardScaler
                from scipy.cluster.hierarchy import dendrogram, linkage
                
                # Prepare data for clustering
                cluster_cols = ['optimal_proportion', 'avg_rt', 'switch_rate', 'final_score']
                
                # Handle missing values
                cluster_data = metrics_df[cluster_cols].copy()
                cluster_data = cluster_data.fillna(cluster_data.mean())
                
                # Scale features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Create tabs for different clustering methods
                cluster_tab1, cluster_tab2 = st.tabs(["K-means Clustering", "Hierarchical Clustering"])
                
                with cluster_tab1:
                    # Allow user to select number of clusters for K-means
                    n_clusters_kmeans = st.slider("Number of clusters (K-means)", min_value=2, max_value=5, value=3)
                    
                    # Apply K-means clustering
                    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
                    kmeans_clusters = kmeans.fit_predict(scaled_data)
                    
                    # Add cluster assignment to metrics
                    metrics_df['kmeans_cluster'] = kmeans_clusters
                    
                    # Add to scatter plot
                    fig_kmeans = px.scatter(
                        metrics_df,
                        x=x_metric,
                        y=y_metric,
                        color='kmeans_cluster',
                        hover_data=['participant_id'],
                        labels={
                            'optimal_proportion': 'Optimal Choice %',
                            'avg_rt': 'Average RT (ms)',
                            'switch_rate': 'Switch Rate',
                            'final_score': 'Final Score',
                            'participant_id': 'Participant ID',
                            'kmeans_cluster': 'Cluster'
                        },
                        title=f"Participant Clusters Based on K-means Clustering"
                    )
                    
                    st.plotly_chart(fig_kmeans, use_container_width=True)
                    
                    # Display cluster statistics
                    st.subheader("K-means Cluster Characteristics")
                    
                    kmeans_cluster_stats = metrics_df.groupby('kmeans_cluster')[cluster_cols].mean()
                    
                    # Format for display
                    display_stats_kmeans = kmeans_cluster_stats.copy()
                    display_stats_kmeans.columns = [
                        'Optimal Choice %', 
                        'Avg RT (ms)',
                        'Switch Rate',
                        'Final Score'
                    ]
                    
                    # Format percentages
                    display_stats_kmeans['Optimal Choice %'] = display_stats_kmeans['Optimal Choice %'].map(lambda x: f"{x:.1%}")
                    display_stats_kmeans['Switch Rate'] = display_stats_kmeans['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                    
                    st.dataframe(display_stats_kmeans)
                    
                    # Describe cluster characteristics
                    for cluster in sorted(metrics_df['kmeans_cluster'].unique()):
                        cluster_size = sum(metrics_df['kmeans_cluster'] == cluster)
                        opt_rate = kmeans_cluster_stats.loc[cluster, 'optimal_proportion']
                        switch_rate = kmeans_cluster_stats.loc[cluster, 'switch_rate']
                        final_score = kmeans_cluster_stats.loc[cluster, 'final_score']
                        
                        st.markdown(f"""
                        **Cluster {cluster}** ({cluster_size} participants):
                        - Optimal choice rate: {opt_rate:.1%}
                        - Switch rate (late trials): {switch_rate:.1%}
                        - Final score: {final_score:.1f}
                        """)
                
                with cluster_tab2:
                    # Allow user to select number of clusters for hierarchical clustering
                    n_clusters_hierarchical = st.slider("Number of clusters (Hierarchical)", min_value=2, max_value=5, value=3)
                    
                    # Apply hierarchical clustering
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
                    hierarchical_clusters = hierarchical.fit_predict(scaled_data)
                    
                    # Add cluster assignment to metrics
                    metrics_df['hierarchical_cluster'] = hierarchical_clusters
                    
                    # Create a dendrogram
                    Z = linkage(scaled_data, method='ward')
                    
                    # Plot dendrogram
                    fig_dendrogram, ax = plt.subplots(figsize=(10, 5))
                    dendrogram(Z, ax=ax)
                    ax.set_title('Hierarchical Clustering Dendrogram')
                    ax.set_xlabel('Participant Index')
                    ax.set_ylabel('Distance')
                    ax.axhline(y=ax.get_yticks()[-2], color='r', linestyle='--')
                    
                    st.pyplot(fig_dendrogram)
                    
                    # Add scatter plot with hierarchical clusters
                    fig_hierarchical = px.scatter(
                        metrics_df,
                        x=x_metric,
                        y=y_metric,
                        color='hierarchical_cluster',
                        hover_data=['participant_id'],
                        labels={
                            'optimal_proportion': 'Optimal Choice %',
                            'avg_rt': 'Average RT (ms)',
                            'switch_rate': 'Switch Rate',
                            'final_score': 'Final Score',
                            'participant_id': 'Participant ID',
                            'hierarchical_cluster': 'Cluster'
                        },
                        title=f"Participant Clusters Based on Hierarchical Clustering"
                    )
                    
                    st.plotly_chart(fig_hierarchical, use_container_width=True)
                    
                    # Display cluster statistics
                    st.subheader("Hierarchical Cluster Characteristics")
                    
                    hierarchical_cluster_stats = metrics_df.groupby('hierarchical_cluster')[cluster_cols].mean()
                    
                    # Format for display
                    display_stats_hierarchical = hierarchical_cluster_stats.copy()
                    display_stats_hierarchical.columns = [
                        'Optimal Choice %', 
                        'Avg RT (ms)',
                        'Switch Rate',
                        'Final Score'
                    ]
                    
                    # Format percentages
                    display_stats_hierarchical['Optimal Choice %'] = display_stats_hierarchical['Optimal Choice %'].map(lambda x: f"{x:.1%}")
                    display_stats_hierarchical['Switch Rate'] = display_stats_hierarchical['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                    
                    st.dataframe(display_stats_hierarchical)
                    
                    # Describe cluster characteristics
                    for cluster in sorted(metrics_df['hierarchical_cluster'].unique()):
                        cluster_size = sum(metrics_df['hierarchical_cluster'] == cluster)
                        opt_rate = hierarchical_cluster_stats.loc[cluster, 'optimal_proportion']
                        switch_rate = hierarchical_cluster_stats.loc[cluster, 'switch_rate']
                        final_score = hierarchical_cluster_stats.loc[cluster, 'final_score']
                        
                        st.markdown(f"""
                        **Cluster {cluster}** ({cluster_size} participants):
                        - Optimal choice rate: {opt_rate:.1%}
                        - Switch rate (late trials): {switch_rate:.1%}
                        - Final score: {final_score:.1f}
                        """)
                
                # Add explanation of clustering approaches
                st.markdown("""
                ### Clustering Approach Comparison
                
                **K-means Clustering:**
                - Partitions participants into distinct clusters
                - Each participant belongs to the cluster with the nearest mean
                - Produces spherical clusters
                - Good for identifying groups with clear separation
                
                **Hierarchical Clustering:**
                - Builds a hierarchy of clusters through a bottom-up approach
                - Shows the relationship between participants and subclusters
                - The dendrogram visualizes the hierarchical structure
                - Good for identifying nested patterns and understanding relationships between clusters
                """)
                
            except ImportError:
                st.info("""
                Clustering requires scikit-learn, which is not available in the current environment.
                To enable clustering, install scikit-learn using `pip install scikit-learn`.
                """)
                
                # Simple alternative based on quantiles of optimal choice rate
                st.subheader("Participant Groups")
                st.markdown("""
                Without scikit-learn, we can group participants based on their optimal choice rate.
                """)
                
                # Define the columns for statistics (same as would be used for clustering)
                analysis_cols = ['optimal_proportion', 'avg_rt', 'switch_rate', 'final_score']
                
                # Create groups based on quantiles
                metrics_df['optimal_group'] = pd.qcut(
                    metrics_df['optimal_proportion'], 
                    q=3, 
                    labels=['Low', 'Medium', 'High']
                )
                
                # Display groups
                fig_groups = px.scatter(
                    metrics_df,
                    x=x_metric,
                    y=y_metric,
                    color='optimal_group',
                    hover_data=['participant_id'],
                    labels={
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score',
                        'participant_id': 'Participant ID',
                        'optimal_group': 'Optimal Choice Group'
                    },
                    title=f"Participant Groups Based on Optimal Choice Rate"
                )
                
                st.plotly_chart(fig_groups, use_container_width=True)
                
                # Display group statistics
                st.subheader("Group Characteristics")
                
                group_stats = metrics_df.groupby('optimal_group')[analysis_cols].mean()
                
                # Format for display
                display_group_stats = group_stats.copy()
                display_group_stats.columns = [
                    'Optimal Choice %', 
                    'Avg RT (ms)',
                    'Switch Rate',
                    'Final Score'
                ]
                
                # Format percentages
                display_group_stats['Optimal Choice %'] = display_group_stats['Optimal Choice %'].map(lambda x: f"{x:.1%}")
                display_group_stats['Switch Rate'] = display_group_stats['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                
                st.dataframe(display_group_stats)
                
            # Now add 50/50 block analysis
            st.markdown("---")
            st.header("50/50 Block Individual Differences Analysis")
            st.markdown("""
            This section examines participant differences in the 50/50 block, where both options 
            have equal reward probabilities and thus different strategies may emerge.
            """)
            
            # Calculate metrics for each participant in 50/50 block
            df_5050 = df[df['block'] == '50/50']
            
            if not df_5050.empty:
                participant_metrics_5050 = []
                
                for participant_id, group in df_5050.groupby('participant_id'):
                    # Calculate consistency with first choice
                    first_choice_consistency = group['matches_first_choice'].mean()
                    
                    # Average RT
                    avg_rt = group['response_time_ms'].mean()
                    
                    # Average switch rate 
                    switch_rate = group['switched'].mean()
                    
                    # Choice entropy (randomness/exploration)
                    choices = group['selected_option'].values
                    if len(choices) > 1:  # Need at least 2 choices
                        unique, counts = np.unique(choices, return_counts=True)
                        probs = counts / counts.sum()
                        choice_entropy = -np.sum(probs * np.log2(probs))
                    else:
                        choice_entropy = 0
                    
                    # Get initial score at start of 50/50 block
                    initial_score = group['total_points'].iloc[0]
                    
                    # Get final score at end of 50/50 block
                    final_score = group['total_points'].max()
                    
                    # Store points gained only in 50/50 block
                    final_score_50_only = final_score - initial_score
                    
                    participant_metrics_5050.append({
                        'participant_id': participant_id,
                        'choice_consistency': first_choice_consistency,
                        'avg_rt': avg_rt,
                        'switch_rate': switch_rate,
                        'choice_entropy': choice_entropy,
                        'final_score': final_score_50_only
                    })
                
                # Convert to DataFrame
                metrics_df_5050 = pd.DataFrame(participant_metrics_5050)
                
                # Display metrics table
                st.subheader("Participant Metrics in 50/50 Block")
                
                # Format the dataframe for display
                display_df_5050 = metrics_df_5050.copy()
                display_df_5050.columns = [
                    'Participant ID', 
                    'Choice Consistency %', 
                    'Avg RT (ms)',
                    'Switch Rate',
                    'Choice Entropy',
                    'Points Gained'
                ]
                
                # Format percentages
                display_df_5050['Choice Consistency %'] = display_df_5050['Choice Consistency %'].map(lambda x: f"{x:.1%}")
                display_df_5050['Switch Rate'] = display_df_5050['Switch Rate'].map(lambda x: f"{x:.1%}")
                
                # Format as sortable dataframe
                st.dataframe(display_df_5050, hide_index=True)
                
                # Scatter plot matrix for 50/50 block
                st.subheader("Relationships Between 50/50 Block Metrics")
                st.markdown("Select metrics to compare:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_metric_5050 = st.selectbox(
                        "X-axis metric (50/50)",
                        options=[
                            'choice_consistency', 
                            'avg_rt', 
                            'switch_rate', 
                            'choice_entropy',
                            'final_score'
                        ],
                        format_func=lambda x: {
                            'choice_consistency': 'Choice Consistency %',
                            'avg_rt': 'Average RT (ms)',
                            'switch_rate': 'Switch Rate',
                            'choice_entropy': 'Choice Entropy',
                            'final_score': 'Points Gained'
                        }[x],
                        index=0
                    )
                
                with col2:
                    y_metric_5050 = st.selectbox(
                        "Y-axis metric (50/50)",
                        options=[
                            'choice_consistency', 
                            'avg_rt', 
                            'switch_rate', 
                            'choice_entropy',
                            'final_score'
                        ],
                        format_func=lambda x: {
                            'choice_consistency': 'Choice Consistency %',
                            'avg_rt': 'Average RT (ms)',
                            'switch_rate': 'Switch Rate',
                            'choice_entropy': 'Choice Entropy',
                            'final_score': 'Points Gained'
                        }[x],
                        index=4
                    )
                
                # Create scatter plot
                fig_scatter_5050 = px.scatter(
                    metrics_df_5050,
                    x=x_metric_5050,
                    y=y_metric_5050,
                    hover_data=['participant_id'],
                    labels={
                        'choice_consistency': 'Choice Consistency %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'choice_entropy': 'Choice Entropy',
                        'final_score': 'Points Gained',
                        'participant_id': 'Participant ID'
                    },
                    title=f"Relationship between {x_metric_5050.replace('_', ' ').title()} and {y_metric_5050.replace('_', ' ').title()}"
                )
                
                # Highlight selected participants if applicable
                if "All Participants" not in selected_participants:
                    selected_metrics_5050 = metrics_df_5050[metrics_df_5050['participant_id'].isin(selected_participants)]
                    
                    if not selected_metrics_5050.empty:
                        fig_scatter_5050.add_trace(
                            go.Scatter(
                                x=selected_metrics_5050[x_metric_5050],
                                y=selected_metrics_5050[y_metric_5050],
                                mode='markers',
                                marker=dict(color='red', size=12, line=dict(width=2, color='DarkSlateGrey')),
                                name='Selected Participants',
                                text=selected_metrics_5050['participant_id']
                            )
                        )
                
                st.plotly_chart(fig_scatter_5050, use_container_width=True)
                
                # Try to perform clustering for 50/50 block metrics
                st.subheader("50/50 Block Participant Clustering")
                
                try:
                    from sklearn.cluster import KMeans, AgglomerativeClustering
                    from sklearn.preprocessing import StandardScaler
                    from scipy.cluster.hierarchy import dendrogram, linkage
                    
                    # Prepare data for clustering
                    cluster_cols_5050 = ['choice_consistency', 'avg_rt', 'switch_rate', 'choice_entropy', 'final_score']
                    
                    # Handle missing values
                    cluster_data_5050 = metrics_df_5050[cluster_cols_5050].copy()
                    cluster_data_5050 = cluster_data_5050.fillna(cluster_data_5050.mean())
                    
                    # Scale features
                    scaler_5050 = StandardScaler()
                    scaled_data_5050 = scaler_5050.fit_transform(cluster_data_5050)
                    
                    # Create tabs for different clustering methods
                    cluster_tab1_5050, cluster_tab2_5050 = st.tabs(["K-means Clustering (50/50)", "Hierarchical Clustering (50/50)"])
                    
                    with cluster_tab1_5050:
                        # Allow user to select number of clusters for K-means
                        n_clusters_kmeans_5050 = st.slider("Number of clusters (K-means 50/50)", min_value=2, max_value=5, value=3, 
                                                          key='kmeans_5050')
                        
                        # Apply K-means clustering
                        kmeans_5050 = KMeans(n_clusters=n_clusters_kmeans_5050, random_state=42)
                        kmeans_clusters_5050 = kmeans_5050.fit_predict(scaled_data_5050)
                        
                        # Add cluster assignment to metrics
                        metrics_df_5050['kmeans_cluster'] = kmeans_clusters_5050
                        
                        # Add to scatter plot
                        fig_kmeans_5050 = px.scatter(
                            metrics_df_5050,
                            x=x_metric_5050,
                            y=y_metric_5050,
                            color='kmeans_cluster',
                            hover_data=['participant_id'],
                            labels={
                                'choice_consistency': 'Choice Consistency %',
                                'avg_rt': 'Average RT (ms)',
                                'switch_rate': 'Switch Rate',
                                'choice_entropy': 'Choice Entropy',
                                'final_score': 'Points Gained',
                                'participant_id': 'Participant ID',
                                'kmeans_cluster': 'Cluster'
                            },
                            title=f"Participant Clusters Based on K-means Clustering (50/50 Block)"
                        )
                        
                        st.plotly_chart(fig_kmeans_5050, use_container_width=True)
                        
                        # Display cluster statistics
                        st.subheader("K-means Cluster Characteristics (50/50 Block)")
                        
                        kmeans_cluster_stats_5050 = metrics_df_5050.groupby('kmeans_cluster')[cluster_cols_5050].mean()
                        
                        # Format for display
                        display_stats_kmeans_5050 = kmeans_cluster_stats_5050.copy()
                        display_stats_kmeans_5050.columns = [
                            'Choice Consistency %', 
                            'Avg RT (ms)',
                            'Switch Rate',
                            'Choice Entropy',
                            'Points Gained'
                        ]
                        
                        # Format percentages
                        display_stats_kmeans_5050['Choice Consistency %'] = display_stats_kmeans_5050['Choice Consistency %'].map(lambda x: f"{x:.1%}")
                        display_stats_kmeans_5050['Switch Rate'] = display_stats_kmeans_5050['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                        
                        st.dataframe(display_stats_kmeans_5050)
                        
                        # Describe cluster characteristics
                        for cluster in sorted(metrics_df_5050['kmeans_cluster'].unique()):
                            cluster_size = sum(metrics_df_5050['kmeans_cluster'] == cluster)
                            consistency = kmeans_cluster_stats_5050.loc[cluster, 'choice_consistency']
                            switch_rate = kmeans_cluster_stats_5050.loc[cluster, 'switch_rate']
                            entropy = kmeans_cluster_stats_5050.loc[cluster, 'choice_entropy']
                            points = kmeans_cluster_stats_5050.loc[cluster, 'final_score']
                            
                            st.markdown(f"""
                            **Cluster {cluster}** ({cluster_size} participants):
                            - Choice consistency: {consistency:.1%}
                            - Switch rate: {switch_rate:.1%}
                            - Choice entropy: {entropy:.2f}
                            - Points gained: {points:.1f}
                            """)
                    
                    with cluster_tab2_5050:
                        # Allow user to select number of clusters for hierarchical clustering
                        n_clusters_hierarchical_5050 = st.slider("Number of clusters (Hierarchical 50/50)", min_value=2, max_value=5, value=3,
                                                                key='hierarchical_5050')
                        
                        # Apply hierarchical clustering
                        hierarchical_5050 = AgglomerativeClustering(n_clusters=n_clusters_hierarchical_5050)
                        hierarchical_clusters_5050 = hierarchical_5050.fit_predict(scaled_data_5050)
                        
                        # Add cluster assignment to metrics
                        metrics_df_5050['hierarchical_cluster'] = hierarchical_clusters_5050
                        
                        # Create a dendrogram
                        Z_5050 = linkage(scaled_data_5050, method='ward')
                        
                        # Plot dendrogram
                        fig_dendrogram_5050, ax = plt.subplots(figsize=(10, 5))
                        dendrogram(Z_5050, ax=ax)
                        ax.set_title('Hierarchical Clustering Dendrogram (50/50 Block)')
                        ax.set_xlabel('Participant Index')
                        ax.set_ylabel('Distance')
                        ax.axhline(y=ax.get_yticks()[-2], color='r', linestyle='--')
                        
                        st.pyplot(fig_dendrogram_5050)
                        
                        # Add scatter plot with hierarchical clusters
                        fig_hierarchical_5050 = px.scatter(
                            metrics_df_5050,
                            x=x_metric_5050,
                            y=y_metric_5050,
                            color='hierarchical_cluster',
                            hover_data=['participant_id'],
                            labels={
                                'choice_consistency': 'Choice Consistency %',
                                'avg_rt': 'Average RT (ms)',
                                'switch_rate': 'Switch Rate',
                                'choice_entropy': 'Choice Entropy',
                                'final_score': 'Points Gained',
                                'participant_id': 'Participant ID',
                                'hierarchical_cluster': 'Cluster'
                            },
                            title=f"Participant Clusters Based on Hierarchical Clustering (50/50 Block)"
                        )
                        
                        st.plotly_chart(fig_hierarchical_5050, use_container_width=True)
                        
                        # Display cluster statistics
                        st.subheader("Hierarchical Cluster Characteristics (50/50 Block)")
                        
                        hierarchical_cluster_stats_5050 = metrics_df_5050.groupby('hierarchical_cluster')[cluster_cols_5050].mean()
                        
                        # Format for display
                        display_stats_hierarchical_5050 = hierarchical_cluster_stats_5050.copy()
                        display_stats_hierarchical_5050.columns = [
                            'Choice Consistency %', 
                            'Avg RT (ms)',
                            'Switch Rate',
                            'Choice Entropy',
                            'Points Gained'
                        ]
                        
                        # Format percentages
                        display_stats_hierarchical_5050['Choice Consistency %'] = display_stats_hierarchical_5050['Choice Consistency %'].map(lambda x: f"{x:.1%}")
                        display_stats_hierarchical_5050['Switch Rate'] = display_stats_hierarchical_5050['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                        
                        st.dataframe(display_stats_hierarchical_5050)
                        
                        # Describe cluster characteristics
                        for cluster in sorted(metrics_df_5050['hierarchical_cluster'].unique()):
                            cluster_size = sum(metrics_df_5050['hierarchical_cluster'] == cluster)
                            consistency = hierarchical_cluster_stats_5050.loc[cluster, 'choice_consistency']
                            switch_rate = hierarchical_cluster_stats_5050.loc[cluster, 'switch_rate']
                            entropy = hierarchical_cluster_stats_5050.loc[cluster, 'choice_entropy']
                            points = hierarchical_cluster_stats_5050.loc[cluster, 'final_score']
                            
                            st.markdown(f"""
                            **Cluster {cluster}** ({cluster_size} participants):
                            - Choice consistency: {consistency:.1%}
                            - Switch rate: {switch_rate:.1%}
                            - Choice entropy: {entropy:.2f}
                            - Points gained: {points:.1f}
                            """)
                
                except ImportError:
                    st.info("""
                    Clustering requires scikit-learn, which is not available in the current environment.
                    To enable clustering, install scikit-learn using `pip install scikit-learn`.
                    """)
                    
                    # Simple alternative based on quantiles of choice consistency
                    st.subheader("50/50 Block Participant Groups")
                    st.markdown("""
                    Without scikit-learn, we can group participants based on their choice consistency.
                    """)
                    
                    # Define the columns for statistics
                    analysis_cols_5050 = ['choice_consistency', 'avg_rt', 'switch_rate', 'choice_entropy', 'final_score']
                    
                    # Create groups based on quantiles of choice consistency
                    metrics_df_5050['consistency_group'] = pd.qcut(
                        metrics_df_5050['choice_consistency'], 
                        q=3, 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    # Display groups
                    fig_groups_5050 = px.scatter(
                        metrics_df_5050,
                        x=x_metric_5050,
                        y=y_metric_5050,
                        color='consistency_group',
                        hover_data=['participant_id'],
                        labels={
                            'choice_consistency': 'Choice Consistency %',
                            'avg_rt': 'Average RT (ms)',
                            'switch_rate': 'Switch Rate',
                            'choice_entropy': 'Choice Entropy',
                            'final_score': 'Points Gained',
                            'participant_id': 'Participant ID',
                            'consistency_group': 'Consistency Group'
                        },
                        title=f"Participant Groups Based on Choice Consistency (50/50 Block)"
                    )
                    
                    st.plotly_chart(fig_groups_5050, use_container_width=True)
                    
                    # Display group statistics
                    st.subheader("50/50 Block Group Characteristics")
                    
                    group_stats_5050 = metrics_df_5050.groupby('consistency_group')[analysis_cols_5050].mean()
                    
                    # Format for display
                    display_group_stats_5050 = group_stats_5050.copy()
                    display_group_stats_5050.columns = [
                        'Choice Consistency %', 
                        'Avg RT (ms)',
                        'Switch Rate',
                        'Choice Entropy',
                        'Points Gained'
                    ]
                    
                    # Format percentages
                    display_group_stats_5050['Choice Consistency %'] = display_group_stats_5050['Choice Consistency %'].map(lambda x: f"{x:.1%}")
                    display_group_stats_5050['Switch Rate'] = display_group_stats_5050['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                    
                    st.dataframe(display_group_stats_5050)
                
                # Add comparison between 80/20 and 50/50 strategies
                if not df_8020.empty and len(metrics_df) > 0 and len(metrics_df_5050) > 0:
                    st.markdown("---")
                    st.subheader("Strategy Comparison: 80/20 vs 50/50 Blocks")
                    st.markdown("""
                    This analysis examines how participants' strategies differ between blocks with
                    different reward structures. Does success in one block predict success in another?
                    """)
                    
                    # Merge metrics from both blocks
                    metrics_merged = pd.merge(
                        metrics_df[['participant_id', 'optimal_proportion', 'switch_rate', 'final_score']], 
                        metrics_df_5050[['participant_id', 'choice_consistency', 'switch_rate', 'final_score']],
                        on='participant_id',
                        suffixes=('_8020', '_5050')
                    )
                    
                    # Create correlation analysis
                    corr_matrix = metrics_merged[[
                        'optimal_proportion', 'switch_rate_8020', 'final_score_8020',
                        'choice_consistency', 'switch_rate_5050', 'final_score_5050'
                    ]].corr()
                    
                    # Create heatmap
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(x="Metric", y="Metric", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r',
                        text_auto='.2f'
                    )
                    
                    fig_corr.update_layout(
                        title="Correlation Between 80/20 and 50/50 Block Metrics"
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Highlight key relationships
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot: 80/20 optimal choice vs 50/50 choice consistency
                        fig_strat1 = px.scatter(
                            metrics_merged,
                            x='optimal_proportion',
                            y='choice_consistency',
                            hover_data=['participant_id'],
                            labels={
                                'optimal_proportion': 'Optimal Choice % (80/20)',
                                'choice_consistency': 'Choice Consistency % (50/50)',
                                'participant_id': 'Participant ID'
                            },
                            title="Optimal Choice (80/20) vs Choice Consistency (50/50)"
                        )
                        
                        st.plotly_chart(fig_strat1, use_container_width=True)
                    
                    with col2:
                        # Scatter plot: 80/20 score vs 50/50 score
                        fig_strat2 = px.scatter(
                            metrics_merged,
                            x='final_score_8020',
                            y='final_score_5050',
                            hover_data=['participant_id'],
                            labels={
                                'final_score_8020': 'Final Score (80/20)',
                                'final_score_5050': 'Points Gained (50/50)',
                                'participant_id': 'Participant ID'
                            },
                            title="Performance in 80/20 vs 50/50 Blocks"
                        )
                        
                        st.plotly_chart(fig_strat2, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                    ### Interpretation of Block Comparison
                    
                    The correlation analysis and scatter plots above show how strategies and performance
                    relate between the 80/20 and 50/50 blocks:
                    
                    1. **Optimal Choice vs Consistency**: Does being good at identifying the optimal choice
                       in the 80/20 block relate to being consistent in the 50/50 block?
                       
                    2. **Performance Relationship**: Does high performance in one block predict high 
                       performance in the other?
                       
                    3. **Strategy Transfer**: Do participants maintain similar strategies (e.g., exploration
                       rates) across blocks with different reward structures?
                    """)
            else:
                st.info("No data available for the 50/50 block.")
        else:
            st.info("No data available for the 80/20 block.")
    else:
        st.info("No data available for individual differences analysis.")

# ===== 6. Participant Comparison =====
with tab6:
    st.header("Participant Comparison Analysis")
    st.markdown("""
    This analysis allows you to compare a selected participant's performance against others
    across multiple metrics to identify relative strengths and weaknesses.
    """)
    
    # Select a participant to analyze
    primary_participant = st.selectbox(
        "Select a participant to analyze",
        options=all_participants,
        help="Choose a participant to compare against the rest of the group"
    )
    
    if primary_participant:
        # Get data for the selected participant
        primary_data = df[df['participant_id'] == primary_participant]
        
        # Get data for all other participants for comparison
        other_data = df[df['participant_id'] != primary_participant]
        
        # Calculate key performance metrics for both the primary participant and others
        metrics = {}
        
        # 1. Metric: Optimal choice rate in 80/20 block
        primary_optimal_rate = primary_data[primary_data['block'] == '80/20']['chose_optimal'].mean()
        others_optimal_rate = other_data[other_data['block'] == '80/20']['chose_optimal'].mean()
        
        metrics['Optimal Choice Rate (80/20)'] = {
            primary_participant: primary_optimal_rate,
            'Group Average': others_optimal_rate
        }
        
        # Calculate percentile rank
        all_optimal_rates = df[df['block'] == '80/20'].groupby('participant_id')['chose_optimal'].mean()
        optimal_percentile = sum(all_optimal_rates < primary_optimal_rate) / len(all_optimal_rates) * 100
        
        # 2. Metric: Response time
        primary_rt = primary_data['response_time_ms'].mean()
        others_rt = other_data['response_time_ms'].mean()
        
        metrics['Response Time (ms)'] = {
            primary_participant: primary_rt,
            'Group Average': others_rt
        }
        
        all_rts = df.groupby('participant_id')['response_time_ms'].mean()
        rt_percentile = sum(all_rts > primary_rt) / len(all_rts) * 100  # Lower RT is better
        
        # 3. Metric: Switch rate
        primary_switch_rate = primary_data['switched'].mean()
        others_switch_rate = other_data['switched'].mean()
        
        metrics['Switch Rate'] = {
            primary_participant: primary_switch_rate,
            'Group Average': others_switch_rate
        }
        
        all_switch_rates = df.groupby('participant_id')['switched'].mean()
        switch_percentile = sum(all_switch_rates < primary_switch_rate) / len(all_switch_rates) * 100
        
        # 4. Metric: Final score
        primary_final_score_8020 = primary_data[primary_data['block'] == '80/20']['total_points'].max()
        others_final_score_8020 = other_data[other_data['block'] == '80/20'].groupby('participant_id')['total_points'].max().mean()
        
        metrics['Final Score (80/20)'] = {
            primary_participant: primary_final_score_8020,
            'Group Average': others_final_score_8020
        }
        
        all_final_scores = df[df['block'] == '80/20'].groupby('participant_id')['total_points'].max()
        score_percentile = sum(all_final_scores < primary_final_score_8020) / len(all_final_scores) * 100
        
        # 5. Metric: Learning rate (slope of optimal choices over time)
        primary_8020 = primary_data[primary_data['block'] == '80/20']
        if len(primary_8020) > 0:
            primary_optimal_by_bin = primary_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
            if len(primary_optimal_by_bin) > 1:
                x = primary_optimal_by_bin['trial_bin'].values
                y = primary_optimal_by_bin['chose_optimal'].values
                primary_learning_slope = np.polyfit(x, y, 1)[0]
            else:
                primary_learning_slope = 0
        else:
            primary_learning_slope = 0
            
        # Calculate learning slope for each other participant
        other_learning_slopes = []
        for participant, data in other_data[other_data['block'] == '80/20'].groupby('participant_id'):
            optimal_by_bin = data.groupby('trial_bin')['chose_optimal'].mean().reset_index()
            if len(optimal_by_bin) > 1:
                x = optimal_by_bin['trial_bin'].values
                y = optimal_by_bin['chose_optimal'].values
                slope = np.polyfit(x, y, 1)[0]
                other_learning_slopes.append(slope)
        
        others_learning_slope = np.mean(other_learning_slopes) if other_learning_slopes else 0
        
        metrics['Learning Slope (80/20)'] = {
            primary_participant: primary_learning_slope,
            'Group Average': others_learning_slope
        }
        
        all_learning_slopes = [primary_learning_slope] + other_learning_slopes
        learning_percentile = sum(np.array(other_learning_slopes) < primary_learning_slope) / len(all_learning_slopes) * 100
        
        # Display metrics
        st.subheader("Performance Metrics Comparison")
        
        # Create columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a radar chart for the metrics
            categories = list(metrics.keys())
            
            # Normalize values for radar chart (0 to 1)
            primary_values = []
            group_values = []
            
            for metric in categories:
                # For RT, lower is better, so we invert the normalization
                if metric == 'Response Time (ms)':
                    max_val = max(metrics[metric][primary_participant], metrics[metric]['Group Average'])
                    primary_values.append(1 - (metrics[metric][primary_participant] / max_val))
                    group_values.append(1 - (metrics[metric]['Group Average'] / max_val))
                else:
                    max_val = max(metrics[metric][primary_participant], metrics[metric]['Group Average'])
                    primary_values.append(metrics[metric][primary_participant] / max_val if max_val else 0)
                    group_values.append(metrics[metric]['Group Average'] / max_val if max_val else 0)
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=primary_values,
                theta=categories,
                fill='toself',
                name=f'Participant {primary_participant}',
                line=dict(color='firebrick')
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=group_values,
                theta=categories,
                fill='toself',
                name='Group Average',
                line=dict(color='royalblue')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Performance Profile Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a bar chart showing percentile ranks
            percentiles = {
                'Optimal Choice Rate': optimal_percentile,
                'Response Time': rt_percentile,
                'Switch Rate': switch_percentile,
                'Final Score': score_percentile,
                'Learning Slope': learning_percentile
            }
            
            percentile_df = pd.DataFrame({
                'Metric': list(percentiles.keys()),
                'Percentile': list(percentiles.values())
            })
            
            fig_percentile = px.bar(
                percentile_df,
                x='Metric',
                y='Percentile',
                title=f"Percentile Ranks for Participant {primary_participant}",
                color='Percentile',
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={'Percentile': 'Percentile Rank (%)'}
            )
            
            fig_percentile.update_layout(yaxis_range=[0, 100])
            
            # Add a horizontal line at 50%
            fig_percentile.add_shape(
                type="line",
                x0=-0.5,
                y0=50,
                x1=4.5,
                y1=50,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            st.plotly_chart(fig_percentile, use_container_width=True)
        
        # Display numerical comparison
        st.subheader("Detailed Metric Comparison")
        
        # Create a dataframe for the comparison
        comparison_data = []
        
        for metric, values in metrics.items():
            comparison_data.append({
                'Metric': metric,
                primary_participant: values[primary_participant],
                'Group Average': values['Group Average'],
                'Difference': values[primary_participant] - values['Group Average'],
                'Percentile': percentiles.get(metric.split(' ')[0], None)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format the values
        for col in [primary_participant, 'Group Average', 'Difference']:
            comparison_df[col] = comparison_df.apply(
                lambda row: f"{row[col]:.2f}" if 'Rate' in row['Metric'] or 'Slope' in row['Metric'] 
                else f"{row[col]:.1f}", 
                axis=1
            )
        
        comparison_df['Percentile'] = comparison_df['Percentile'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else 'N/A'
        )
        
        st.dataframe(comparison_df, hide_index=True)
        
        # Interpretation of results
        st.subheader("Performance Interpretation")
        
        # Generate interpretation text based on percentiles
        interpretations = []
        
        if optimal_percentile > 75:
            interpretations.append(f"- **Optimal Choice Rate**: Participant {primary_participant} shows **excellent** performance in choosing the optimal option, better than {optimal_percentile:.1f}% of participants.")
        elif optimal_percentile > 50:
            interpretations.append(f"- **Optimal Choice Rate**: Participant {primary_participant} shows **above average** performance in choosing the optimal option, better than {optimal_percentile:.1f}% of participants.")
        elif optimal_percentile > 25:
            interpretations.append(f"- **Optimal Choice Rate**: Participant {primary_participant} shows **below average** performance in choosing the optimal option, better than only {optimal_percentile:.1f}% of participants.")
        else:
            interpretations.append(f"- **Optimal Choice Rate**: Participant {primary_participant} shows **poor** performance in choosing the optimal option, better than only {optimal_percentile:.1f}% of participants.")
        
        if rt_percentile > 75:
            interpretations.append(f"- **Response Time**: Participant {primary_participant} responds **very quickly**, faster than {rt_percentile:.1f}% of participants.")
        elif rt_percentile > 50:
            interpretations.append(f"- **Response Time**: Participant {primary_participant} responds **somewhat quickly**, faster than {rt_percentile:.1f}% of participants.")
        elif rt_percentile > 25:
            interpretations.append(f"- **Response Time**: Participant {primary_participant} responds **somewhat slowly**, faster than only {rt_percentile:.1f}% of participants.")
        else:
            interpretations.append(f"- **Response Time**: Participant {primary_participant} responds **very slowly**, faster than only {rt_percentile:.1f}% of participants.")
        
        if switch_percentile > 75:
            interpretations.append(f"- **Exploration Strategy**: Participant {primary_participant} shows a **high exploration** strategy, switching options more frequently than {switch_percentile:.1f}% of participants.")
        elif switch_percentile > 50:
            interpretations.append(f"- **Exploration Strategy**: Participant {primary_participant} shows a **moderate exploration** strategy, switching options more frequently than {switch_percentile:.1f}% of participants.")
        elif switch_percentile > 25:
            interpretations.append(f"- **Exploration Strategy**: Participant {primary_participant} shows a **moderate exploitation** strategy, switching options less frequently than {100-switch_percentile:.1f}% of participants.")
        else:
            interpretations.append(f"- **Exploration Strategy**: Participant {primary_participant} shows a **strong exploitation** strategy, switching options less frequently than {100-switch_percentile:.1f}% of participants.")
        
        if score_percentile > 75:
            interpretations.append(f"- **Performance Outcome**: Participant {primary_participant} achieved **excellent** overall performance, scoring better than {score_percentile:.1f}% of participants.")
        elif score_percentile > 50:
            interpretations.append(f"- **Performance Outcome**: Participant {primary_participant} achieved **above average** overall performance, scoring better than {score_percentile:.1f}% of participants.")
        elif score_percentile > 25:
            interpretations.append(f"- **Performance Outcome**: Participant {primary_participant} achieved **below average** overall performance, scoring better than only {score_percentile:.1f}% of participants.")
        else:
            interpretations.append(f"- **Performance Outcome**: Participant {primary_participant} achieved **poor** overall performance, scoring better than only {score_percentile:.1f}% of participants.")
        
        if learning_percentile > 75:
            interpretations.append(f"- **Learning Rate**: Participant {primary_participant} shows **very fast** learning, improving more quickly than {learning_percentile:.1f}% of participants.")
        elif learning_percentile > 50:
            interpretations.append(f"- **Learning Rate**: Participant {primary_participant} shows **above average** learning, improving more quickly than {learning_percentile:.1f}% of participants.")
        elif learning_percentile > 25:
            interpretations.append(f"- **Learning Rate**: Participant {primary_participant} shows **below average** learning, improving more slowly than {100-learning_percentile:.1f}% of participants.")
        else:
            interpretations.append(f"- **Learning Rate**: Participant {primary_participant} shows **very slow** learning, improving more slowly than {100-learning_percentile:.1f}% of participants.")
        
        # Display interpretations
        for interp in interpretations:
            st.markdown(interp)
        
        # Learning curve comparison
        st.subheader("Learning Curve Comparison")
        
        # Create a figure for learning curves
        fig_learning_comp = make_subplots(rows=1, cols=2, 
                                    subplot_titles=("Learning Curve: 80/20 Block", "Switch Rate Over Time"))
        
        # Get data for the primary participant
        primary_8020 = primary_data[primary_data['block'] == '80/20']
        if len(primary_8020) > 0:
            # Optimal choice by bin
            optimal_by_bin = primary_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
            
            fig_learning_comp.add_trace(
                go.Scatter(
                    x=optimal_by_bin['trial_bin'], 
                    y=optimal_by_bin['chose_optimal'],
                    mode='lines+markers',
                    name=f"Participant {primary_participant}",
                    line=dict(color='firebrick', width=3),
                ),
                row=1, col=1
            )
            
            # Switch rate by bin
            switch_by_bin = primary_8020.groupby('trial_bin')['switched'].mean().reset_index()
            
            fig_learning_comp.add_trace(
                go.Scatter(
                    x=switch_by_bin['trial_bin'], 
                    y=switch_by_bin['switched'],
                    mode='lines+markers',
                    name=f"Participant {primary_participant}",
                    line=dict(color='firebrick', width=3),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Calculate average for others
        others_8020 = other_data[other_data['block'] == '80/20']
        if len(others_8020) > 0:
            # Optimal choice by bin
            others_optimal_by_bin = others_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
            
            fig_learning_comp.add_trace(
                go.Scatter(
                    x=others_optimal_by_bin['trial_bin'], 
                    y=others_optimal_by_bin['chose_optimal'],
                    mode='lines+markers',
                    name="Group Average",
                    line=dict(color='royalblue'),
                ),
                row=1, col=1
            )
            
            # Switch rate by bin
            others_switch_by_bin = others_8020.groupby('trial_bin')['switched'].mean().reset_index()
            
            fig_learning_comp.add_trace(
                go.Scatter(
                    x=others_switch_by_bin['trial_bin'], 
                    y=others_switch_by_bin['switched'],
                    mode='lines+markers',
                    name="Group Average",
                    line=dict(color='royalblue'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add horizontal reference line at 0.5 for optimal choice
        fig_learning_comp.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=0, y0=0.5, x1=df['trial_bin'].max(), y1=0.5, row=1, col=1
        )
        
        fig_learning_comp.update_layout(
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        
        fig_learning_comp.update_xaxes(title_text="Trial Bin", row=1, col=1)
        fig_learning_comp.update_xaxes(title_text="Trial Bin", row=1, col=2)
        fig_learning_comp.update_yaxes(title_text="Proportion of Optimal Choices", row=1, col=1)
        fig_learning_comp.update_yaxes(title_text="Switch Rate", row=1, col=2)
        
        st.plotly_chart(fig_learning_comp, use_container_width=True)
        
        # Trial-by-trial choice analysis
        st.subheader("Trial-by-Trial Choice Pattern")
        
        # Create a visualization of choices on each trial for the 80/20 block
        primary_choices = primary_data[primary_data['block'] == '80/20'].sort_values('trial_number')
        
        if not primary_choices.empty:
            # Extract key variables for each trial
            choice_data = primary_choices[['trial_number', 'selected_option', 'chose_optimal', 'feedback']].copy()
            
            # Create plot
            fig_choices = px.scatter(
                choice_data,
                x='trial_number',
                y='selected_option',
                color='chose_optimal',
                symbol='feedback',
                color_discrete_map={True: 'green', False: 'red'},
                symbol_map={'positive': 'circle', 'negative': 'x'},
                labels={
                    'trial_number': 'Trial Number',
                    'selected_option': 'Selected Option (0=Left, 1=Right)',
                    'chose_optimal': 'Chose Optimal Option',
                    'feedback': 'Feedback'
                },
                title=f"Trial-by-Trial Choices for Participant {primary_participant} (80/20 Block)"
            )
            
            fig_choices.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['Left Option', 'Right Option']
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_choices, use_container_width=True)
            
            # Win-stay lose-shift analysis
            st.subheader("Win-Stay Lose-Shift Strategy Analysis")
            
            # Calculate win-stay and lose-shift rates
            choice_data['prev_feedback'] = choice_data['feedback'].shift(1)
            choice_data['prev_choice'] = choice_data['selected_option'].shift(1)
            choice_data['stay'] = choice_data['selected_option'] == choice_data['prev_choice']
            
            # Win-stay rate
            win_trials = choice_data[(choice_data['prev_feedback'] == 'positive')]
            win_stay_rate = win_trials['stay'].mean() if len(win_trials) > 0 else 0
            
            # Lose-shift rate
            lose_trials = choice_data[(choice_data['prev_feedback'] == 'negative')]
            lose_shift_rate = (1 - lose_trials['stay'].mean()) if len(lose_trials) > 0 else 0
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Win-Stay Rate", f"{win_stay_rate:.2f}")
                st.markdown("""
                **Win-Stay Rate**: The proportion of times the participant chose the same option
                after receiving positive feedback on the previous trial.
                """)
            
            with col2:
                st.metric("Lose-Shift Rate", f"{lose_shift_rate:.2f}")
                st.markdown("""
                **Lose-Shift Rate**: The proportion of times the participant chose a different option
                after receiving negative feedback on the previous trial.
                """)
            
            # Compare to group averages
            others_choices = other_data[other_data['block'] == '80/20'].sort_values(['participant_id', 'trial_number'])
            others_choices['prev_feedback'] = others_choices.groupby('participant_id')['feedback'].shift(1)
            others_choices['prev_choice'] = others_choices.groupby('participant_id')['selected_option'].shift(1)
            others_choices['stay'] = others_choices['selected_option'] == others_choices['prev_choice']
            
            others_win_trials = others_choices[others_choices['prev_feedback'] == 'positive']
            others_win_stay_rate = others_win_trials['stay'].mean() if len(others_win_trials) > 0 else 0
            
            others_lose_trials = others_choices[others_choices['prev_feedback'] == 'negative']
            others_lose_shift_rate = (1 - others_lose_trials['stay'].mean()) if len(others_lose_trials) > 0 else 0
            
            # Create bar chart comparing WSLS rates
            wsls_data = pd.DataFrame({
                'Strategy': ['Win-Stay Rate', 'Lose-Shift Rate'],
                f'Participant {primary_participant}': [win_stay_rate, lose_shift_rate],
                'Group Average': [others_win_stay_rate, others_lose_shift_rate]
            })
            
            wsls_data_melted = pd.melt(
                wsls_data, 
                id_vars=['Strategy'],
                var_name='Participant',
                value_name='Rate'
            )
            
            fig_wsls = px.bar(
                wsls_data_melted,
                x='Strategy',
                y='Rate',
                color='Participant',
                barmode='group',
                title='Win-Stay Lose-Shift Strategy Comparison',
                labels={'Rate': 'Proportion of Trials'}
            )
            
            fig_wsls.update_layout(yaxis_range=[0, 1])
            
            st.plotly_chart(fig_wsls, use_container_width=True)
    else:
        st.info("Please select a participant to analyze.")

# Add some resources and explanations at the bottom
st.markdown("---")
st.markdown("""
### About Probabilistic Reward Learning (PRL)

In PRL tasks, participants learn to choose between two options whose chance of delivering a reward varies by block:
- **80/20 block**: one shape yields a reward 80% of the time, the other only 20%.
- **50/50 block**: both shapes yield a reward 50% of the time.

Under asymmetric contingencies (80/20), participants typically learn to favor the high-reward shape over successive trials. When both options are equal (50/50), choices tend to remain around chance and switching/exploration remains higher.

---

### Metrics Explained

- **Cumulative Reward** (`cum_reward`): Total points a participant earned in a block, summing all trial outcomes.
- **Optimal Choice Rate** (`POC`): Proportion of trials in which the participant selected the higher-probability (80%) shape in the 80/20 block.
- **Switch Rate** (`switch_rate`): Fraction of trials where the participant chose a different shape than on the immediately preceding trial.
- **Switch Count** (`switch_count`): Raw number of such switches across all trials in a block.
- **Average Response Time** (`Avg_RT`): Mean decision latency (in ms) across all trials in a block.
- **Difference Scores**:
  - **POC_diff**: `POC(80/20) - POC(50/50)`  
  - **switch_diff**: `switch_rate(80/20) - switch_rate(50/50)`  
  - **avg_RT_diff**: `Avg_RT(80/20) - Avg_RT(50/50)`  

These difference scores capture how behavior shifts when moving from an informative (80/20) to an uninformative (50/50) environment.

---

### Analysis Tips

- Use the sidebar participant selector to focus on individual performance.
- Adjust the trial bin size to zoom into early vs. late learning phases.
- Compare metrics between the two blocks to see how learning and exploration adapt under different reward probabilities.
""")
