import pandas as pd
import numpy as np
import os

# Path to the master CSV file
file_path = os.path.join(os.path.dirname(__file__), "master.csv")

# Read the CSV file
data = pd.read_csv(file_path)

# Filter out practice trials
data = data[data['is_practice'] == False]

# Initialize empty lists to store results
results = []

# Get unique participants
participants = data['participant_id'].unique()

for pid in participants:
    participant_data = data[data['participant_id'] == pid]
    
    # Get unique conditions (blocks) for this participant, excluding practice
    conditions = participant_data['block'].unique()
    
    for condition in conditions:
        condition_data = participant_data[participant_data['block'] == condition]
        
        # Skip if fewer than 20 trials
        if len(condition_data) < 20:
            continue
        
        # Total number of trials
        total_trials = len(condition_data)
        
        # Calculate cumulative reward
        cum_reward = condition_data['points_earned'].sum()
        
        # Calculate proportion of optimal choices (POC)
        # For 80/20 condition
        if condition == "80/20":
            # When left_option_prob is 0.8, optimal choice is left (0)
            # When right_option_prob is 0.8, optimal choice is right (1)
            condition_data['optimal_choice'] = np.where(
                condition_data['left_option_prob'] == 0.8, 0,
                np.where(condition_data['right_option_prob'] == 0.8, 1, np.nan)
            )
        # For 50/50 condition (no optimal choice)
        else:
            # For 50/50, there's no optimal choice, but we'll count selection of the higher reward option
            condition_data['optimal_choice'] = np.nan
        
        # Calculate POC (excluding trials with no selected option)
        valid_trials = condition_data.dropna(subset=['selected_option'])
        if condition == "80/20" and len(valid_trials) > 0:
            optimal_choices = 0
            for _, trial in valid_trials.iterrows():
                # If selected right (1) and right has 0.8 probability
                if trial['selected_option'] == 1 and trial['right_option_prob'] == 0.8:
                    optimal_choices += 1
                # If selected left (0) and left has 0.8 probability
                elif trial['selected_option'] == 0 and trial['left_option_prob'] == 0.8:
                    optimal_choices += 1
            
            poc = optimal_choices / len(valid_trials)
        else:
            poc = np.nan  # No optimal choice for 50/50
        
        # Calculate switch rate and count
        # First, get all the selected shapes in sequence
        selected_shapes = valid_trials['selected_shape'].values
        
        # Count switches (when current shape != previous shape)
        switch_count = 0
        for i in range(1, len(selected_shapes)):
            if selected_shapes[i] != selected_shapes[i-1]:
                switch_count += 1
        
        # Calculate switch rate
        switch_rate = switch_count / (len(selected_shapes) - 1) if len(selected_shapes) > 1 else 0
        
        # Calculate average response time
        avg_rt = valid_trials['response_time_ms'].mean()
        
        # Calculate metrics for first 10 trials
        first_10 = valid_trials.head(10)
        if len(first_10) > 0:
            # POC for first 10 trials
            if condition == "80/20":
                optimal_choices_first_10 = 0
                for _, trial in first_10.iterrows():
                    # If selected right (1) and right has 0.8 probability
                    if trial['selected_option'] == 1 and trial['right_option_prob'] == 0.8:
                        optimal_choices_first_10 += 1
                    # If selected left (0) and left has 0.8 probability
                    elif trial['selected_option'] == 0 and trial['left_option_prob'] == 0.8:
                        optimal_choices_first_10 += 1
                
                poc_first_10 = optimal_choices_first_10 / len(first_10)
            else:
                poc_first_10 = np.nan
            
            # Switch rate for first 10 trials
            selected_first_10 = first_10['selected_shape'].values
            switch_count_first_10 = 0
            for i in range(1, len(selected_first_10)):
                if selected_first_10[i] != selected_first_10[i-1]:
                    switch_count_first_10 += 1
            
            switch_rate_first_10 = switch_count_first_10 / (len(selected_first_10) - 1) if len(selected_first_10) > 1 else 0
            
            # Average RT for first 10 trials
            avg_rt_first_10 = first_10['response_time_ms'].mean()
        else:
            poc_first_10 = np.nan
            switch_rate_first_10 = np.nan
            avg_rt_first_10 = np.nan
        
        # Calculate metrics for last 10 trials
        last_10 = valid_trials.tail(10)
        if len(last_10) > 0:
            # POC for last 10 trials
            if condition == "80/20":
                optimal_choices_last_10 = 0
                for _, trial in last_10.iterrows():
                    # If selected right (1) and right has 0.8 probability
                    if trial['selected_option'] == 1 and trial['right_option_prob'] == 0.8:
                        optimal_choices_last_10 += 1
                    # If selected left (0) and left has 0.8 probability
                    elif trial['selected_option'] == 0 and trial['left_option_prob'] == 0.8:
                        optimal_choices_last_10 += 1
                
                poc_last_10 = optimal_choices_last_10 / len(last_10)
            else:
                poc_last_10 = np.nan
            
            # Switch rate for last 10 trials
            selected_last_10 = last_10['selected_shape'].values
            switch_count_last_10 = 0
            for i in range(1, len(selected_last_10)):
                if selected_last_10[i] != selected_last_10[i-1]:
                    switch_count_last_10 += 1
            
            switch_rate_last_10 = switch_count_last_10 / (len(selected_last_10) - 1) if len(selected_last_10) > 1 else 0
            
            # Average RT for last 10 trials
            avg_rt_last_10 = last_10['response_time_ms'].mean()
        else:
            poc_last_10 = np.nan
            switch_rate_last_10 = np.nan
            avg_rt_last_10 = np.nan
        
        # Calculate differences
        poc_diff = poc_last_10 - poc_first_10
        switch_diff = switch_rate_last_10 - switch_rate_first_10
        avg_rt_diff = avg_rt_last_10 - avg_rt_first_10
        
        # Add results to list
        results.append({
            'PID': pid,
            'condition': condition,
            'cum_reward': cum_reward,
            'POC': poc,
            'switch_rate': switch_rate,
            'switch_count': switch_count,
            'Avg_RT': avg_rt,
            'POC_diff': poc_diff,
            'switch_diff': switch_diff,
            'avg_RT_diff': avg_rt_diff
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Split the results by condition
results_5050 = results_df[results_df['condition'] == '50/50']
results_8020 = results_df[results_df['condition'] == '80/20']

# Save to CSV files
results_df.to_csv(os.path.join(os.path.dirname(__file__), "prl_metrics_combined.csv"), index=False)
results_5050.to_csv(os.path.join(os.path.dirname(__file__), "prl_metrics_5050.csv"), index=False)
results_8020.to_csv(os.path.join(os.path.dirname(__file__), "prl_metrics_8020.csv"), index=False)

print("Analysis complete. Results saved to CSV files.")