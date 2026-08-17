"""Preprocessing helpers: raw trial records into per subject evidence.

Extracted verbatim from notebooks/data_preprocessing_with_reward.ipynb so that
the same code backs both the notebook and scripts/preprocess_03_build_evidence.py.

analyze_all_subjects_behavior turns the reward augmented trial table into the
per subject records everything downstream reads. The two format functions
produce the evidence dictionaries the fitting code consumes: the plain one stops
at the subject's decision, the full sequence one runs to the deadline.
"""
import os

import numpy as np
import pandas as pd


number_of_3_choieTrial={}
def analyze_all_subjects_behavior(human_meg_data: pd.DataFrame) -> tuple[pd.DataFrame, int, list]:
    """
    Processes raw behavioral data for all subjects and combines the results.

    This function iterates through each subject found in the input DataFrame,
    extracts their nested trial data, and computes 'action_taken' and 'reward'
    for each trial. It then concatenates the processed data for all subjects
    into a single output DataFrame.

    Args:
        human_meg_data (pd.DataFrame): The main DataFrame from the pickle file,
                                      containing 'userID' and 'data' columns.

    Returns:
        tuple[pd.DataFrame, int, list]: A tuple containing:
        - pd.DataFrame: A new, single DataFrame containing the processed trial data
                        for all subjects, with added 'action_taken' and 'reward' columns.
        - int: A counter for the total number of trials across all subjects where
               total left evidence equaled total right evidence.
        - list: A list of tuples recording details for every trial with equal evidence.
                Each tuple is (total_left, total_right, assigned_action).
    """
    processed_subjects_list = {}
    total_equal_evidence_counter = 0
    all_even_cards_choices = []

    user_id_list = human_meg_data['userID'].unique()

    for user_id in user_id_list:
        # Use your helper function to get the data for one subject
        subj_data = get_subject_data(human_meg_data, user_id)
        if subj_data is None:
            continue

        # Work on a copy to avoid SettingWithCopyWarning
        subj_data_processed = subj_data.copy()
        # replace choiceTrial values of 3 with NaN. 
        # subj_data_processed.loc[subj_data_processed['choiceTrial'] == 3, 'choiceTrial'] = np.nan
        # count how many 3 in the choiceTrial overall 
        number_of_3_choieTrial[user_id] = (subj_data_processed['choiceTrial'] == 3).sum()


        

        # Loop through each game for the subject to process it
        for block in subj_data_processed['block'].unique():
            for game in subj_data_processed['game'].unique():
                filter_mask = (subj_data_processed['block'] == block) & (subj_data_processed['game'] == game)
                game_indices = subj_data_processed[filter_mask].index
                
                if game_indices.empty:
                    continue
                user_data_slice = subj_data_processed.loc[game_indices]
                if (user_data_slice['choiceTrial'] == 3).any():
                    subj_data_processed.loc[subj_data_processed['choiceTrial'] == 3, 'choiceTrial'] = np.nan
                    # Refresh the slice so the rest of the loop sees the NaNs
                    user_data_slice = subj_data_processed.loc[game_indices]
                    num_draws_game=len(user_data_slice['currEvLeft'])
                    # calculate the number of draws to decision, so number of rows in the game until the first non Nan value in the action_taken column
                    decision_index = (user_data_slice['choiceTrial'].first_valid_index())
                    try:
                        decision_pos = user_data_slice.index.get_loc(decision_index)

                        dist2ch = np.full_like(user_data_slice['choiceTrial'], np.nan, dtype=np.float64)

                        just_before_actions = decision_pos 

                        dist2ch[:just_before_actions] = np.arange(just_before_actions, 0, -1)
                        # write in the distance2choice column 
                        subj_data_processed.loc[game_indices, 'distance2choice'] = dist2ch
                        user_data_slice = subj_data_processed.loc[game_indices]
                    except:
                        print(" None Decision trials, do nothing ")
                        


                else:
                    num_draws_game=len(user_data_slice['currEvLeft'])
                    # calculate the number of draws to decision, so number of rows in the game until the first non Nan value in the action_taken column
                    decision_index = (user_data_slice['choiceTrial'].first_valid_index())



                if decision_index is not None:
                    decision_pos = user_data_slice.index.get_loc(decision_index)
                    num_draws = decision_pos + 1   

                else:
                    num_draws=num_draws_game
                # --- Action Taken Calculation ---
                final_action_series = user_data_slice['choiceTrial'].dropna()

                if not final_action_series.empty:
                    final_action = int(final_action_series.iloc[0])
                    # if matches_yellow.all():
                    assigned_action = 0 if final_action == 1 else 1 if final_action == 2 else np.nan
                    # else:
                        # assigned_action = 1 if final_action == 1 else 0 if final_action == 2 else np.nan
                    
                    subj_data_processed.loc[game_indices, 'action_taken'] = assigned_action
                    nan_chosen_idx = user_data_slice[user_data_slice['choiceTrial'].isna()].index
                    subj_data_processed.loc[nan_chosen_idx, 'action_taken'] = np.nan
                else:
                    assigned_action = 2  # Missing
                    subj_data_processed.loc[game_indices, 'action_taken'] = assigned_action
                # horizon_condition
                horizon_condition=user_data_slice['termination'].iloc[0]

                # --- Reward Calculation ---
                total_left = user_data_slice['totEvLeft'].iloc[-1]
                total_right = user_data_slice['totEvRight'].iloc[-1]
                # calculate the number of draws before the subject decided, this happed when the action taken is not Nan. 
                # add a column for the number of draws to decision for each game
                subj_data_processed.loc[game_indices, 'num_draws'] = num_draws

                if assigned_action != 2:
                    # If totals tie -> reward = 0 regardless of action; still record the event
                    if total_left == total_right:
                        total_equal_evidence_counter += 1
                        all_even_cards_choices.append({
                            "userID": user_id,
                            "block": int(block),
                            "game": int(game),
                            "total_left": int(total_left),
                            "total_right": int(total_right),
                            "assigned_action": int(assigned_action) if not pd.isna(assigned_action) else np.nan,
                            "num_draws_game": int(num_draws_game),
                            "num_draws": int(num_draws),
                            "horizon_condition": int(horizon_condition)
                        })
                        # subj_data_processed.loc[game_indices, 'reward'] = 0
                    else:
                        # if total_left > total_right:
                        #     correct_bool = user_data_slice['chosen'] == 1
                        # elif total_left < total_right:
                        #     correct_bool = user_data_slice['chosen'] == 2
                        # reward = correct_bool.map({True: 2, False: -2})
                        # subj_data_processed.loc[game_indices, 'reward'] = reward

                        nan_chosen_idx = user_data_slice[user_data_slice['chosen'].isna()].index[:-1]
                        # subj_data_processed.loc[nan_chosen_idx, 'reward'] = np.nan
                        # if decision_index is not None:
                            # subj_data_processed.loc[decision_index, 'reward'] = reward.tolist()[-1]
                            
                elif assigned_action == 2:
                    # missing action: normally -1, but if totals tie set reward = 0 and record
                    if total_left == total_right:
                        total_equal_evidence_counter += 1
                        all_even_cards_choices.append({
                            "userID": user_id,
                            "block": int(block),
                            "game": int(game),
                            "total_left": int(total_left),
                            "total_right": int(total_right),
                            "assigned_action": int(assigned_action) if not pd.isna(assigned_action) else np.nan,
                            "num_draws_game": int(num_draws_game),
                            "num_draws": int(num_draws),
                            "horizon_condition": int(horizon_condition)
                        })
#
        
        # Add the processed data for this subject to our list
        processed_subjects_list[user_id] = subj_data_processed

    # Convert the dict to a DataFrame with 'userID' and nested DataFrame in 'data'
    final_processed_df = pd.DataFrame({
        "userID": list(processed_subjects_list.keys()),
        "data": list(processed_subjects_list.values())
    })
    

    return final_processed_df, total_equal_evidence_counter, all_even_cards_choices

if __name__ == '__main__':
    # Assume 'project_root' is defined. For this example, we'll set it to the current directory.
    # --- Step 1: Load the main data file ---
    # This path needs to be correct for your system.
    file_path = os.path.join(project_root, "data/TrHu_NHB_light/data_MEG/behdat_reward.pkl")
    
    try:
        human_meg_data = pd.read_pickle(file_path)
        print(f"Successfully loaded data for {len(human_meg_data)} subjects.")

        # --- Step 2: Call the main processing function ---
        processed_df, equal_evidence_count, equal_evidence_details = analyze_all_subjects_behavior(human_meg_data)

        # --- Step 3: Display the results ---
        print("\n" + "="*50)
        print("           Behavioral Data Processing Complete")
        print("="*50 + "\n")
        
        print("--- Sample of the Final Processed DataFrame ---")
        # Display a sample of the final combined and processed data
        print(processed_df.head(10))
        print("\nDataFrame Info:")
        processed_df.info()
        
        print("\n" + "="*50 + "\n")

        print("--- Summary Statistics ---")
        print(f"Total frequency of trials with equal evidence: {equal_evidence_count}")
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {file_path}")
        print("Please ensure the 'project_root' variable and file path are correct.")
    # Save the dataset now with the same format as before but modified at the end 
    processed_data_path= os.path.join(project_root, "data/TrHu_NHB_light/data_MEG/behdat_preprocessed.pkl")
    processed_df.to_pickle(processed_data_path)
    print(f"Preprocessed dataset saved to {processed_data_path}")

# ==============================================================================
# 1. HELPER FUNCTION
# ==============================================================================
def get_subject_data(human_meg_data: pd.DataFrame, user_id: str) -> pd.DataFrame | None:
    subject_row = human_meg_data.loc[human_meg_data['userID'] == user_id]
    if not subject_row.empty:
        return subject_row['data'].values[0]
    else:
        print(f"Warning: No data found for userID: {user_id}")
        return None

# ==============================================================================
# 3. FORMATTING FUNCTION
# ==============================================================================
def format_human_data_for_modeling(processed_human_data: pd.DataFrame,subject_id:int):
    """
    Formats processed data from a single subject for modeling.

    This function groups the data by game and block, calculates key metrics for each game
    (like number of draws until a decision), and structures the data into two main formats:
    1. A summary DataFrame with one row per game.
    2. A dictionary of DataFrames containing detailed draw-by-draw evidence, split by horizon condition.
    """

    evidence_lists, reward_list, num_draws_list, final_action_list = [], [], [], []
    termination_conditions_list = []
    all_games_data_long, all_games_data_short = [], []

    # Create the groupby object. This is an efficient way to prepare for group-wise operations.
    game_groups = processed_human_data.groupby(['block', 'game'])

    # FIX 1: Correctly iterate over the groupby object.
    # The iterator yields a tuple containing the group name (e.g., (block_val, game_val))
    # and the corresponding DataFrame slice (game_data).
    for name, game_data in game_groups:
        # The check for empty is generally not needed with groupby, as it only
        # creates groups for combinations that actually exist in the data.
        # However, keeping it does no harm.
        if game_data.empty:
            continue

        # FIX 2: Robustly calculate the number of draws to decision.
        # This new logic correctly finds the position of the decision within the game,
        # regardless of the main DataFrame's index values.
        decision_index = game_data['choiceTrial'].first_valid_index()
        
        if decision_index:
            # Get the integer position (e.g., 0, 1, 2...) of the decision within the game_data slice.
            decision_pos = game_data.index.get_loc(decision_index)
            num_draws = decision_pos + 1
        else:
            # If no decision was made (e.g., all 'chosen' values are NaN),
            # then the number of draws is the total number of rows for that game.
            num_draws = len(game_data)
        
        final_action_series = game_data['action_taken'].dropna()
        final_action = int(final_action_series.iloc[0]) if not final_action_series.empty else 2
        try:
            final_reward=game_data['reward'].dropna().iloc[-1]
        except:
            print(game_data['reward'],game_data['reward'].iloc[-1])
        horizon_condition = game_data['termination'].iloc[0]
        
        # Calculate cumulative evidence up to the point of decision
        cum_yellow_ev = game_data['currEvLeft'].cumsum()
        cum_blue_ev = game_data['currEvRight'].cumsum()

        game_draw_details = []
        # The loop now correctly uses the number of draws until a decision was made.
        for i in range(num_draws):
            game_draw_details.append([i + 1, int(cum_yellow_ev.iloc[i]), int(cum_blue_ev.iloc[i]), 2,0])
        
        # Set the final action in the last recorded draw
        if game_draw_details:
            game_draw_details[-1][3] = int(final_action)
           
            game_draw_details[-1][4] = int(final_reward)

        # Append the detailed game data to the correct list based on termination condition
        if horizon_condition == 1:
            all_games_data_long.append(game_draw_details)
        elif horizon_condition == 2:
            all_games_data_short.append(game_draw_details)

        # Append summary statistics for this game to the main lists
        evidence_lists.append(cum_yellow_ev.tolist())
        reward_list.append(final_reward)
        num_draws_list.append(num_draws)
        final_action_list.append(final_action)
        termination_conditions_list.append(horizon_condition)

    # Create the summary DataFrame from the collected lists
    data_summary = pd.DataFrame({
        "subject_id": [subject_id] * len(reward_list),
        "ev": evidence_lists,
        "outcome": reward_list,
        "num_draws": num_draws_list,
        "action": final_action_list,
        "termination": termination_conditions_list
    })

    # Create the dictionary of DataFrames for detailed evidence fitting
    df_long = pd.DataFrame({'draw_yellow_blue_action_outcome': all_games_data_long})
    df_short = pd.DataFrame({'draw_yellow_blue_action_outcome': all_games_data_short})

    evidence_dict = {'long': df_long, 'short': df_short}
    evidence_dict_short={'short':df_short}
    evidence_dict_long={'long':df_long}

    return data_summary, evidence_dict_short,evidence_dict_long,evidence_dict



# ==============================================================================
# 3. FORMATTING FUNCTION
# ==============================================================================
def format_human_data_for_modeling_full_sequence(processed_human_data: pd.DataFrame,subject_id:int):
    """
    Formats processed data from a single subject for modeling.

    This function groups the data by game and block, calculates key metrics for each game
    (like number of draws until a decision), and structures the data into two main formats:
    1. A summary DataFrame with one row per game.
    2. A dictionary of DataFrames containing detailed draw-by-draw evidence, split by horizon condition.
    """

    evidence_lists, reward_list, num_draws_list, final_action_list = [], [], [], []
    termination_conditions_list = []
    all_games_data_long, all_games_data_short = [], []

    # Create the groupby object. This is an efficient way to prepare for group-wise operations.
    game_groups = processed_human_data.groupby(['block', 'game'])

    # and the corresponding DataFrame slice (game_data).
    for _, game_data in game_groups:

        decision_index = game_data['choiceTrial'].first_valid_index()
        if decision_index is not None:
            decision_pos = game_data.index.get_loc(decision_index)

        num_draws = len(game_data)

        
        final_action=int(game_data['action_taken'].dropna().iloc[-1])
        final_reward=int(game_data['reward'].dropna().iloc[-1])
        horizon_condition = game_data['termination'].iloc[0]
        
        # Calculate cumulative evidence up to the point of decision
        cum_yellow_ev = game_data['currEvLeft'].cumsum()
        cum_blue_ev = game_data['currEvRight'].cumsum()

        game_draw_details = []
        
        for i in range(num_draws):
            if decision_index is not None:# none missing games
                
                if i==decision_pos or i>decision_pos:
                    game_draw_details.append([i + 1, int(cum_yellow_ev.iloc[i]), int(cum_blue_ev.iloc[i]), int(final_action), (final_reward)]) # post decision, repeat action and rewards
                else:
                    game_draw_details.append([i + 1, int(cum_yellow_ev.iloc[i]), int(cum_blue_ev.iloc[i]), 2,0]) # prior to decision
            else: # for missing games
                if i<num_draws-1:
                    game_draw_details.append([i + 1, int(cum_yellow_ev.iloc[i]), int(cum_blue_ev.iloc[i]), 2,0])
                else:
                    game_draw_details.append([i + 1, int(cum_yellow_ev.iloc[i]), int(cum_blue_ev.iloc[i]), int(final_action),final_reward])
        
        # Append the detailed game data to the correct list based on termination condition
        if horizon_condition == 1:
            all_games_data_long.append(game_draw_details)
        elif horizon_condition == 2:
            all_games_data_short.append(game_draw_details)
        print(game_draw_details)

        # Append summary statistics for this game to the main lists
        evidence_lists.append(cum_yellow_ev.tolist())
        reward_list.append(final_reward)
        num_draws_list.append(num_draws)
        final_action_list.append(final_action)
        termination_conditions_list.append(horizon_condition)

    # Create the summary DataFrame from the collected lists
    data_summary = pd.DataFrame({
        "subject_id": [subject_id] * len(reward_list),
        "ev": evidence_lists,
        "outcome": reward_list,
        "num_draws": num_draws_list,
        "action": final_action_list,
        "termination": termination_conditions_list
    })

    # Create the dictionary of DataFrames for detailed evidence fitting
    df_long = pd.DataFrame({'draw_yellow_blue_action_outcome': all_games_data_long})
    df_short = pd.DataFrame({'draw_yellow_blue_action_outcome': all_games_data_short})
    evidence_dict = {'long': df_long, 'short': df_short}
    evidence_dict_short={'short':df_short}
    evidence_dict_long={'long':df_long}

    return data_summary, evidence_dict_short,evidence_dict_long,evidence_dict


