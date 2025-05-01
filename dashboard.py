import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from math import radians, sin, cos, sqrt, atan2

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go



def validate_team_occurrences(df):
    """
    Validates if TeamIDs appear exactly 3 times in their corresponding course columns based on 'Gang' value.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns: 'Gang', 'TeamID', 'Vorspeise_TeamID', 
        'Hauptgang_TeamID', 'Nachspeise_TeamID'
    
    Returns
    -------
    dict
        Dictionary containing validation results for each course type:
        {
            'appetizer': bool,
            'main': bool,
            'dessert': bool
        }
    """
    # Initialize result dictionary
    result = {
        'appetizer': True,
        'main': True,
        'dessert': True
    }
    
    # Check appetizer rows
    appetizer_teams = df[df['Gang'] == 'appetizer']['TeamID'].unique()
    for team_id in appetizer_teams:
        if df['Vorspeise_TeamID'].value_counts()[team_id] != 3:
            result['appetizer'] = False
            break
    
    # Check main course rows
    main_teams = df[df['Gang'] == 'main']['TeamID'].unique()
    for team_id in main_teams:
        if df['Hauptgang_TeamID'].value_counts()[team_id] != 3:
            result['main'] = False
            break
    
    # Check dessert rows
    dessert_teams = df[df['Gang'] == 'dessert']['TeamID'].unique()
    for team_id in dessert_teams:
        if df['Nachspeise_TeamID'].value_counts()[team_id] != 3:
            result['dessert'] = False
            break
    
    return result



def process_registrations(df_original, df):
    """
    Process registrations from original dataframe and update food preferences in the target dataframe.
    For couple registrations, both persons get the same food preference.
    Also adds a column 'Paaranmeldung' to indicate if the row represents a couple registration.
    
    Parameters
    ----------
    df_original : pandas.DataFrame
        Original dataframe containing registration information with columns:
        'ID', 'ID_2', 'FoodPreference'
    df : pandas.DataFrame
        Target dataframe to be updated with columns:
        'Person1_ID', 'Person2_ID'
        
    Returns
    -------
    tuple
        (updated_df, registration_counts)
        - updated_df: DataFrame with new columns 'Person1_FoodPreference', 'Person2_FoodPreference',
          and 'Paaranmeldung'
        - registration_counts: dict with keys 'couple' and 'single' containing counts
    """
    # Initialize new columns for food preferences and couple registration indicator
    df['Person1_FoodPreference'] = None
    df['Person2_FoodPreference'] = None
    df['Paaranmeldung'] = 0  # Default value is 0
    
    # Count registrations
    couple_rows = df_original[df_original['ID_2'].notna()]
    single_rows = df_original[df_original['ID_2'].isna()]
    
    registration_counts = {
        'couple': len(couple_rows),
        'single': len(single_rows)
    }
    
    # Create sets of couple IDs for faster lookup
    couple_id_pairs = set()
    for _, row in couple_rows.iterrows():
        # Add both combinations of the pair
        couple_id_pairs.add((row['ID'], row['ID_2']))
        couple_id_pairs.add((row['ID_2'], row['ID']))
    
    # Process couple registrations
    for _, row in couple_rows.iterrows():
        matches = df[
            ((df['Person1_ID'] == row['ID']) & (df['Person2_ID'] == row['ID_2'])) |
            ((df['Person1_ID'] == row['ID_2']) & (df['Person2_ID'] == row['ID']))
        ]
        
        for _, match in matches.iterrows():
            # Set food preferences
            df.loc[match.name, 'Person1_FoodPreference'] = row['FoodPreference']
            df.loc[match.name, 'Person2_FoodPreference'] = row['FoodPreference']
            # Mark as couple registration
            df.loc[match.name, 'Paaranmeldung'] = 1
    
    # Process single registrations
    for _, row in single_rows.iterrows():
        # Find rows where Person1_ID matches
        person1_matches = df[df['Person1_ID'] == row['ID']]
        if not person1_matches.empty:
            df.loc[person1_matches.index, 'Person1_FoodPreference'] = row['FoodPreference']
            
        # Find rows where Person2_ID matches
        person2_matches = df[df['Person2_ID'] == row['ID']]
        if not person2_matches.empty:
            df.loc[person2_matches.index, 'Person2_FoodPreference'] = row['FoodPreference']
    
    return df, registration_counts


def create_master_food_preference(updated_df):
    """
    Creates a new column 'MasterFoodPreference' based on the food preferences in 
    'Person1_FoodPreference' and 'Person2_FoodPreference' columns.
    
    Parameters
    ----------
    updated_df : pandas.DataFrame
        DataFrame containing 'Person1_FoodPreference' and 'Person2_FoodPreference' columns
        with values: 'meat' (0), 'none' (1), 'vegan' (2), 'veggie' (3)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new 'MasterFoodPreference' column added based on the mapping:
        - If either preference is meat (0), result is meat (0)
        - If either preference is veggie (3) and other is none (1), result is veggie (3)
        - If either preference is vegan (2), result is vegan (2) unless other is meat (0)
        - If both preferences are none (1), result is none (1)
    """
    # Create food preference mapping dictionary
    food_pref_map = {
        'meat': 0,
        'none': 1,
        'vegan': 2,
        'veggie': 3
    }
    
    # Create mapping for final preferences
    preference_mapping = {
        (0, 0): 0, (0, 1): 0, (1, 0): 0,
        (0, 2): 2, (2, 0): 2, (0, 3): 3,
        (3, 0): 3, (1, 1): 1, (1, 2): 2,
        (2, 1): 2, (1, 3): 3, (3, 1): 3,
        (2, 2): 2, (2, 3): 2, (3, 2): 2,
        (3, 3): 3
    }
    
    # Convert food preferences to numerical values
    food1_numeric = updated_df['Person1_FoodPreference'].map(food_pref_map)
    food2_numeric = updated_df['Person2_FoodPreference'].map(food_pref_map)
    
    # Create tuples of preferences for mapping
    preference_pairs = list(zip(food1_numeric, food2_numeric))
    
    # Map the pairs to final preference
    updated_df['MasterFoodPreference'] = [preference_mapping[pair] for pair in preference_pairs]

    # Assuming your dataframe is called 'df'
    food_pref_map_reversed = {v: k for k, v in food_pref_map.items()}
    updated_df['MasterFoodPreference'] = updated_df['MasterFoodPreference'].map(food_pref_map_reversed)
    
    return updated_df


def find_shared_addresses(df):
    # Group by all address components and Gang
    grouped = df.groupby([
        'Longitude', 
        'Latitude', 
        'Wohnungsnummer', 
        'Stockwerk',
        'Gang'
    ])['TeamID'].agg(list).reset_index()
    
    # Create a copy of the DataFrame where multiple teams share locations
    shared_locations = grouped[grouped['TeamID'].str.len() > 1].copy()
    
    # Safely add the num_teams column using loc
    shared_locations.loc[:, 'num_teams'] = shared_locations['TeamID'].str.len()
    
    # Sort by number of teams (descending) and Gang
    shared_locations = shared_locations.sort_values(['num_teams', 'Gang'], ascending=[False, True])
    
    return shared_locations


def analyze_couple_characteristics(df):
    """
    Analyze frequency counts of age, gender, and food preference combinations for couples
    (where Paaranmeldung == 0)
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing couple information
    
    Returns:
    dict: Dictionary containing different frequency analyses
    """
    # Filter for Paaranmeldung == 0
    couples_df = df[df['Paaranmeldung'] == 0].copy()
    
    # Age analysis
    couples_df['age_combination'] = couples_df.apply(
        lambda x: f"{min(x['Person1_Alter'], x['Person2_Alter'])}-{max(x['Person1_Alter'], x['Person2_Alter'])}",
        axis=1
    )
    age_counts = couples_df['age_combination'].value_counts()
    
    # Gender combination analysis
    couples_df['gender_combination'] = couples_df.apply(
        lambda x: '-'.join(sorted([x['Person1_Geschlecht'], x['Person2_Geschlecht']])),
        axis=1
    )
    gender_counts = couples_df['gender_combination'].value_counts()
    
    # Food preference combination analysis
    couples_df['food_combination'] = couples_df.apply(
        lambda x: '-'.join(sorted([x['Person1_FoodPreference'], x['Person2_FoodPreference']])),
        axis=1
    )
    food_counts = couples_df['food_combination'].value_counts()
    
    # Age difference analysis
    couples_df['age_difference'] = couples_df.apply(
        lambda x: abs(x['Person1_Alter'] - x['Person2_Alter']),
        axis=1
    )
    age_diff_stats = {
        'mean_difference': couples_df['age_difference'].mean(),
        'max_difference': couples_df['age_difference'].max(),
        'min_difference': couples_df['age_difference'].min()
    }
    
    return {
        'age_combinations': age_counts,
        'gender_combinations': gender_counts,
        'food_preference_combinations': food_counts,
        'age_difference_stats': age_diff_stats
    }


def analyze_food_preferences(df):
    """
    Analyze food preferences for each course/meetup by identifying which teams meet and their preferences.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with columns TeamID, Gang, VorspeiseTeamID, HauptgangTeamID, 
                      NachspeiseTeamID, and MasterFoodPreference
    
    Returns:
    pd.DataFrame: DataFrame with columns Course, Host_TeamID, Team1_ID, Team2_ID, Team3_ID,
                 Team1_Preference, Team2_Preference, Team3_Preference, Preference_Combination
    """
    # Create empty lists to store the results
    results = []
    
    # Map German course names to English
    course_mapping = {
        'appetizer': 'Vorspeise',
        'main': 'Hauptgang',
        'dessert': 'Nachspeise'
    }
    
    # For each course type
    for course_eng, course_ger in course_mapping.items():
        # Find all teams hosting this course
        host_teams = df[df['Gang'] == course_eng]
        
        for _, host_row in host_teams.iterrows():
            host_team_id = host_row['TeamID']
            
            # Find guest teams based on the course
            if course_eng == 'appetizer':
                guest_teams = df[df['Vorspeise_TeamID'] == host_team_id]
            elif course_eng == 'main':
                guest_teams = df[df['Hauptgang_TeamID'] == host_team_id]
            else:  # dessert
                guest_teams = df[df['Nachspeise_TeamID'] == host_team_id]
            
            # Get the team IDs and their preferences
            teams = [guest_teams.iloc[0]['TeamID'],
                    guest_teams.iloc[1]['TeamID'],
                    host_team_id]
            
            # Get preferences for each team
            preferences = []
            for team_id in teams:
                pref = df[df['TeamID'] == team_id]['MasterFoodPreference'].iloc[0]
                preferences.append(pref)
            
            # Create preference combination string
            pref_combination = f"{preferences[0]}, {preferences[1]}, {preferences[2]}"
            
            # Add to results
            results.append({
                'Course': course_ger,
                'Host_TeamID': host_team_id,
                'Team1_ID': teams[0],
                'Team2_ID': teams[1],
                'Team3_ID': teams[2],
                'Team1_Preference': preferences[0],
                'Team2_Preference': preferences[1],
                'Team3_Preference': preferences[2],
                'Preference_Combination': pref_combination
            })
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort by Course
    result_df = result_df.sort_values('Course')
    
    return result_df


def analyze_course_preferences(df):
    """
    Analyzes food preference compatibility for each course meetup by gathering
    all participants (hosts and guests) and their food preferences.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing course assignments and food preferences with columns:
        'TeamID', 'Gang', 'Vorspeise_TeamID', 'Hauptgang_TeamID', 'Nachspeise_TeamID',
        'Person1_FoodPreference', 'Person2_FoodPreference'
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing for each meetup:
        - Host TeamID
        - Course type
        - List of all participants' food preferences
        - Number of different food preferences in the meetup
    """
    # Initialize empty lists to store meetup data
    meetups = []
    
    # Process each team's hosting event
    for _, row in df.iterrows():
        host_team_id = row['TeamID']
        course_type = row['Gang']
        
        # Initialize set of participating teams
        participating_teams = {host_team_id}  # Host team
        
        # Add guest teams based on course type
        if course_type == 'appetizer':
            # Find teams who have this team's ID in their Vorspeise_TeamID
            guest_teams = df[df['Vorspeise_TeamID'] == host_team_id]['TeamID'].tolist()
        elif course_type == 'main':
            # Find teams who have this team's ID in their Hauptgang_TeamID
            guest_teams = df[df['Hauptgang_TeamID'] == host_team_id]['TeamID'].tolist()
        else:  # dessert
            # Find teams who have this team's ID in their Nachspeise_TeamID
            guest_teams = df[df['Nachspeise_TeamID'] == host_team_id]['TeamID'].tolist()
            
        participating_teams.update(guest_teams)
        
        # Gather food preferences of all participants
        food_preferences = []
        for team_id in participating_teams:
            team_data = df[df['TeamID'] == team_id].iloc[0]
            if pd.notna(team_data['Person1_FoodPreference']):
                food_preferences.append(team_data['Person1_FoodPreference'])
            if pd.notna(team_data['Person2_FoodPreference']):
                food_preferences.append(team_data['Person2_FoodPreference'])
        
        # Create meetup record
        meetups.append({
            'Host_TeamID': host_team_id,
            'Course': course_type,
            'Participant_Preferences': food_preferences,
            'Unique_Preferences_Count': len(set(food_preferences)),
            'Preferences_Distribution': pd.Series(food_preferences).value_counts().to_dict()
        })
    
    # Create DataFrame from meetups data
    result_df = pd.DataFrame(meetups)
    
    # Sort by course type and host team ID
    result_df = result_df.sort_values(['Course', 'Host_TeamID'])
    
    return result_df


def count_preference_combinations(meetup_analysis):
    """
    Count frequency of unique preference distributions, treating distributions
    with the same values but different orders as identical.
    
    Parameters
    ----------
    meetup_analysis : pandas.DataFrame
        DataFrame containing meetup analysis with column 'Preferences_Distribution'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - Distribution: str representation of the preference distribution
        - Count: number of occurrences
        - Course: list of courses where this distribution appears
    """
    # Convert dictionary strings to frozenset of tuples for hashable comparison
    def normalize_distribution(dist_dict):
        # Convert string representation of dict to actual dict if needed
        if isinstance(dist_dict, str):
            dist_dict = eval(dist_dict)
        # Convert to frozenset of (preference, count) tuples for order-invariant comparison
        return frozenset(dist_dict.items())
    
    # Create a list to store normalized distributions and their metadata
    distributions = []
    
    for _, row in meetup_analysis.iterrows():
        dist = normalize_distribution(row['Preferences_Distribution'])
        course = row['Course']
        distributions.append((dist, course))
    
    # Count unique combinations
    from collections import defaultdict
    combination_counts = defaultdict(lambda: {'count': 0, 'courses': set()})
    
    for dist, course in distributions:
        combination_counts[dist]['count'] += 1
        combination_counts[dist]['courses'].add(course)
    
    # Convert to DataFrame
    result = []
    for dist, data in combination_counts.items():
        # Convert frozenset back to dictionary for readable output
        dist_dict = dict(dist)
        result.append({
            'Distribution': str(dist_dict),
            'Count': data['count'],
            'Courses': sorted(list(data['courses']))
        })
    
    # Create DataFrame and sort by count in descending order
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values('Count', ascending=False)
    
    return result_df


def count_pair_frequencies(df, col1, col2):
    """
    Count frequencies of pairs where order doesn't matter
    """
    # Create pairs and sort them to make (a,b) equal to (b,a)
    pairs = [tuple(sorted([row[col1], row[col2]])) for _, row in df.iterrows()]
    return Counter(pairs)


# Split the combinations and sort them
def normalize_combination(combo):
    prefs = combo.split(', ')
    return ', '.join(sorted(prefs))


def plot_coordinates_plotly(df):
    """
    Create an interactive scatter plot of geographical coordinates using Plotly.
    """
    fig = px.scatter(
        df,
        x='Kitchen_Longitude',
        y='Kitchen_Latitude',
        title='Ortsangaben der Anmeldungen',
        opacity=0.6,
        width=600,
        height=600
    )
    
    fig.update_traces(
        marker=dict(size=10),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title='Kitchen_Longitude',
        yaxis_title='Kitchen_Latitude',
        xaxis_gridcolor='rgba(0,0,0,0.1)',
        yaxis_gridcolor='rgba(0,0,0,0.1)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


def find_inconsistent_kitchens(df):
    """
    Find kitchens with inconsistent data based on their location coordinates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing kitchen information with columns:
        Kitchen_Longitude, Kitchen_Latitude, Kitchen_Story, 
        Kitchen_Appartment, Kitchen_Allow_Multi_Allocations
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only inconsistent kitchen entries,
        sorted by location coordinates
    """
    cols = ["Kitchen_Longitude", "Kitchen_Latitude", "Kitchen_Story", 
            "Kitchen_Appartment", "Kitchen_Allow_Multi_Allocations"]
    
    # Group by location and check for inconsistencies
    inconsistent_groups = (
        df.sort_values(cols)
        .groupby(['Kitchen_Longitude', 'Kitchen_Latitude'])
        .filter(lambda g: (g['Kitchen_Story'].nunique() > 1) or 
                         (g['Kitchen_Appartment'].nunique() > 1) or 
                         (g['Kitchen_Allow_Multi_Allocations'].nunique() > 1))
    )

    return inconsistent_groups[cols]

def get_frequent_rows(df, column_name, min_frequency=1):
    """
    Get rows where values in the specified column appear more than min_frequency times.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe
    column_name : str
        Name of the column to check for duplicates
    min_frequency : int
        Minimum frequency threshold
        
    Returns
    -------
    pandas.DataFrame
        Rows where the column value appears more than min_frequency times
    """
    # Count frequencies
    value_counts = df[column_name].value_counts()
    
    # Get values appearing more than once
    duplicate_values = value_counts[value_counts > min_frequency].index.tolist()
    
    # Filter rows
    frequent_rows = df[df[column_name].isin(duplicate_values)]
    
    return frequent_rows

def find_names(df, name_list):
    """
    Find rows containing any of the specified names.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe
    name_list : list
        List of names to search for
        
    Returns
    -------
    pandas.DataFrame
        Rows containing any of the specified names
    """
    # Create mask for name matches
    mask = df['Name'].str.contains('|'.join(name_list), case=False, na=False)
    
    # Return matching rows
    return df[mask]

def get_team_routes(df):
    """
    Calculate the walking routes for each team during the Running Dinner event.
    
    This function processes the input DataFrame to create a comprehensive route
    overview for each team, showing where they need to be for each course.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing Running Dinner information with columns:
        - TeamID : unique identifier for each team
        - Gang : course that the team hosts (appetizer, main, dessert)
        - Longitude : team's home longitude
        - Latitude : team's home latitude
        - Vorspeise_TeamID : team hosting appetizer
        - Hauptgang_TeamID : team hosting main course
        - Nachspeise_TeamID : team hosting dessert
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the complete route for each team with columns:
        - TeamID : team identifier
        - Course : course name (Vorspeise, Hauptgang, Nachspeise)
        - Host_TeamID : ID of team hosting this course
        - Host_Longitude : longitude of host location
        - Host_Latitude : latitude of host location
        - Team_Home_Longitude : team's home longitude
        - Team_Home_Latitude : team's home latitude
        - Team_Host_Course : course that this team hosts
    """
    routes = []
    
    # Process each team's route information
    for _, team in df.iterrows():
        team_id = team['TeamID']
        team_host_course = team['Gang']
        
        # Create route entries for each course
        courses = [
            ('Vorspeise', team['Vorspeise_TeamID']),
            ('Hauptgang', team['Hauptgang_TeamID']),
            ('Nachspeise', team['Nachspeise_TeamID'])
        ]
        
        # Add route information for each course
        for course_name, host_team_id in courses:
            host_info = df[df['TeamID'] == host_team_id].iloc[0]
            routes.append({
                'TeamID': team_id,
                'Course': course_name,
                'Host_TeamID': host_team_id,
                'Host_Longitude': host_info['Longitude'],
                'Host_Latitude': host_info['Latitude'],
                'Team_Home_Longitude': team['Longitude'],
                'Team_Home_Latitude': team['Latitude'],
                'Team_Host_Course': team_host_course
            })
    
    # Create DataFrame and sort by team and course order
    routes_df = pd.DataFrame(routes)
    course_order = {'Vorspeise': 1, 'Hauptgang': 2, 'Nachspeise': 3}
    routes_df['CourseOrder'] = routes_df['Course'].map(course_order)
    routes_df = routes_df.sort_values(['TeamID', 'CourseOrder']).drop('CourseOrder', axis=1)
    
    return routes_df

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Parameters
    ----------
    lat1 : float
        Latitude of the first point in decimal degrees
    lon1 : float
        Longitude of the first point in decimal degrees
    lat2 : float
        Latitude of the second point in decimal degrees
    lon2 : float
        Longitude of the second point in decimal degrees
        
    Returns
    -------
    float
        Distance between the points in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def calculate_distances(routes_df):
    """
    Calculate the walking distances between consecutive locations for each team.
    
    This function handles three different cases:
    1. Teams hosting appetizer:
       - Distance 1: 0 (starts at home)
       - Distance 2: Home -> Main course location
       - Distance 3: Main course location -> Dessert location
    
    2. Teams hosting main course:
       - Distance 1: Home -> Appetizer location
       - Distance 2: Appetizer location -> Home (same as Distance 1)
       - Distance 3: Home -> Dessert location
    
    3. Teams hosting dessert:
       - Distance 1: Home -> Appetizer location
       - Distance 2: Appetizer location -> Main course location
       - Distance 3: Main course location -> Home
    
    Parameters
    ----------
    routes_df : pandas.DataFrame
        DataFrame containing the routes from get_team_routes()
    
    Returns
    -------
    pandas.DataFrame
        Routes DataFrame with additional column:
        - Distance_to_next_km : distance to next location in kilometers
    """
    result_df = routes_df.copy()
    result_df['Distance_to_next_km'] = 0.0
    
    # Process each team's distances
    for team_id in result_df['TeamID'].unique():
        team_routes = result_df[result_df['TeamID'] == team_id]
        host_course = team_routes.iloc[0]['Team_Host_Course']
        
        if host_course == 'appetizer':
            # Case 1: Team hosts appetizer
            # First distance is 0 (they're at home)
            result_df.loc[team_routes.index[0], 'Distance_to_next_km'] = 0.0
            
            # Second distance: Home to main course
            result_df.loc[team_routes.index[1], 'Distance_to_next_km'] = haversine_distance(
                team_routes.iloc[0]['Team_Home_Latitude'],
                team_routes.iloc[0]['Team_Home_Longitude'],
                team_routes.iloc[1]['Host_Latitude'],
                team_routes.iloc[1]['Host_Longitude']
            )
            
            # Third distance: Main course to dessert
            result_df.loc[team_routes.index[2], 'Distance_to_next_km'] = haversine_distance(
                team_routes.iloc[1]['Host_Latitude'],
                team_routes.iloc[1]['Host_Longitude'],
                team_routes.iloc[2]['Host_Latitude'],
                team_routes.iloc[2]['Host_Longitude']
            )
            
        elif host_course == 'main':
            # Case 2: Team hosts main course
            # Calculate distance between home and appetizer (used for both ways)
            dist_to_appetizer = haversine_distance(
                team_routes.iloc[0]['Team_Home_Latitude'],
                team_routes.iloc[0]['Team_Home_Longitude'],
                team_routes.iloc[0]['Host_Latitude'],
                team_routes.iloc[0]['Host_Longitude']
            )
            
            # First and second distances are the same (to appetizer and back home)
            result_df.loc[team_routes.index[0], 'Distance_to_next_km'] = dist_to_appetizer
            result_df.loc[team_routes.index[1], 'Distance_to_next_km'] = dist_to_appetizer
            
            # Third distance: Home to dessert
            result_df.loc[team_routes.index[2], 'Distance_to_next_km'] = haversine_distance(
                team_routes.iloc[0]['Team_Home_Latitude'],
                team_routes.iloc[0]['Team_Home_Longitude'],
                team_routes.iloc[2]['Host_Latitude'],
                team_routes.iloc[2]['Host_Longitude']
            )
            
        else:  # host_course == 'dessert'
            # Case 3: Team hosts dessert
            # First distance: Home to appetizer
            result_df.loc[team_routes.index[0], 'Distance_to_next_km'] = haversine_distance(
                team_routes.iloc[0]['Team_Home_Latitude'],
                team_routes.iloc[0]['Team_Home_Longitude'],
                team_routes.iloc[0]['Host_Latitude'],
                team_routes.iloc[0]['Host_Longitude']
            )
            
            # Second distance: Appetizer to main
            result_df.loc[team_routes.index[1], 'Distance_to_next_km'] = haversine_distance(
                team_routes.iloc[0]['Host_Latitude'],
                team_routes.iloc[0]['Host_Longitude'],
                team_routes.iloc[1]['Host_Latitude'],
                team_routes.iloc[1]['Host_Longitude']
            )
            
            # Third distance: Main course to home
            result_df.loc[team_routes.index[2], 'Distance_to_next_km'] = haversine_distance(
                team_routes.iloc[1]['Host_Latitude'],
                team_routes.iloc[1]['Host_Longitude'],
                team_routes.iloc[0]['Team_Home_Latitude'],
                team_routes.iloc[0]['Team_Home_Longitude']
            )
    
    # Remove helper columns before returning
    result_df = result_df.drop(['Team_Home_Latitude', 'Team_Home_Longitude', 'Team_Host_Course'], axis=1)
    
    return result_df

def calculate_team_distance_stats(routes_with_distances):
    """
    Calculate mean distance and standard deviation for each team.
    
    Parameters
    ----------
    routes_with_distances : pandas.DataFrame
        DataFrame containing the routes and distances for each team
        Must contain columns: 'TeamID', 'Distance_to_next_km'
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing distance statistics for each team with columns:
        - TeamID: team identifier
        - mean_distance_km: mean distance in kilometers
        - std_distance_km: standard deviation of distances in kilometers
    """
    # Group by TeamID and calculate statistics
    distance_stats = routes_with_distances.groupby('TeamID').agg({
        'Distance_to_next_km': ['mean', 'std']
    }).round(3)  # Round to 3 decimal places
    
    # Flatten column names and rename for clarity
    distance_stats.columns = ['mean_distance_km', 'std_distance_km']
    distance_stats = distance_stats.reset_index()
    
    return distance_stats

def simple_paginated_dataframe(df, title="DataFrame", page_size=10):
    """A simpler pagination implementation that should be more robust"""
    
    st.write(f"**{title}** (Total rows: {len(df)})")
    
    # Simple page number input
    page = st.number_input(
        "Page number", 
        min_value=1, 
        max_value=max(1, len(df) // page_size + 1),
        value=1,
        key=f"{title}_simple_page"
    )
    
    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    # Show row range
    st.write(f"Showing rows {start_idx+1} to {end_idx}")
    
    # Display dataframe
    st.dataframe(df.iloc[start_idx:end_idx])







# Set page config
st.set_page_config(layout="wide", page_title="Running-Dinner Analyse")

# File upload section
st.sidebar.header("Upload csv-Files")
input_file = st.sidebar.file_uploader("Upload Input Data File (giessen_anmeldungen.csv)", type="csv")
output_file = st.sidebar.file_uploader("Upload Output Data File (giessen_verteilung.csv)", type="csv")

if not input_file or not output_file:
    st.warning("Please upload both input and output CSV files to see the dashboard.")
    st.stop()

# Load data
@st.cache_data
def load_data(input_file, output_file):
    df_original = pd.read_csv(input_file)
    df = pd.read_csv(output_file)
    return df_original, df

df_original, df = load_data(input_file, output_file)

# Main title
st.title("Running-Dinner Dashboard")

# Geographical Distribution Section
st.header("Geographische Verteilung der Anmeldungen")

# Create and display the plot
try:
    geo_fig = plot_coordinates_plotly(df_original)
    st.plotly_chart(geo_fig, use_container_width=True)
    
    # Add some statistics about the geographical distribution
    col_geo1, col_geo2 = st.columns(2)
    
except Exception as e:
    st.error(f"Could not create geographical plot. Please ensure the data contains 'Kitchen_Latitude' and 'Kitchen_Longitude' columns. Error: {str(e)}")

# Basic Statistics Section
st.header("Basis Statistiken")
col1, col2 = st.columns(2)

with col1:
    st.metric("Gesamtanzahl Teams nach der Verteilung:", len(df))
    
    # Team validation
    validation_results = validate_team_occurrences(df)
    st.text("Jedes Team durchläuft drei unterschiedliche Gänge:")
    for course, valid in validation_results.items():
        st.write(f"{course.capitalize()}: {'✅' if valid else '❌'}")

with col2:
    # Teams with accessibility needs
    st.write("Teams mit Bedarf an Barrierefreiheit:")
    if "Barrierefreiheitsbedarf" in df_original.columns:
        accessibility_df = df_original[df_original["Barrierefreiheitsbedarf"] == 1]
        if not accessibility_df.empty:
            st.dataframe(accessibility_df)
        else:
            st.write("Keine Teams benötigen Barrierefreiheit")
    else:
        st.write("Der Datensatz enthält keine Informationen über den Barrierefreiheitsbedarf!")

# Registration Analysis Section
st.header("Analyse der Anmeldedaten")
col3, col4 = st.columns(2)

with col3:
    # Process registrations and create master food preferences
    updated_df, counts = process_registrations(df_original, df)
    updated_df = create_master_food_preference(updated_df)
    
    st.subheader("Anzahl der Anmeldungen", divider=True)
    st.write(f"Pärchenanmeldungen: {counts['couple']}")
    st.write(f"Einzelanmeldungen: {counts['single']} (Max. Potenzial für Pärchen: {counts['single'] // 2})")
    
    st.subheader("Nach Durchführung der Verteilung", divider=True)
    couple_count = updated_df['Paaranmeldung'].value_counts().get(1, 0)
    single_count = updated_df['Paaranmeldung'].value_counts().get(0, 0)
    
    # Create a pie chart for registration distribution
    fig = px.pie(
        values=[couple_count, single_count],
        names=['Pärchen', 'Singles'],
        title='Verteilung der Pärchen- und Einzelanmeldungen'
    )
    st.plotly_chart(fig)

with col4:
    # Shared kitchens analysis
    st.subheader("Übersicht Küchendoppelbelegung", divider=True)
    shared_kitchens = find_shared_addresses(updated_df)
    if not shared_kitchens.empty:
        st.dataframe(shared_kitchens)

# Single Registration Analysis Section
st.header("Analyse der verteilten Einzelanmeldungen")

# Filter for single registrations
singles_df = updated_df[updated_df['Paaranmeldung'] == 0]
all_genders = pd.concat([singles_df['Person1_Geschlecht'], singles_df['Person2_Geschlecht']])
gender_counts = all_genders.value_counts()

col5, col6 = st.columns(2)

with col5:
    st.subheader("Geschlechter-Verteilung")
    fig = px.bar(
        x=gender_counts.index,
        y=gender_counts.values,
        title='Geschlechterverteilung der Einzelanmeldungen'
    )
    st.plotly_chart(fig)

with col6:
    # Gender combinations
    st.subheader("Geschlechter-Kombinationen")
    gender_pairs = count_pair_frequencies(singles_df, 'Person1_Geschlecht', 'Person2_Geschlecht')
    gender_df = pd.DataFrame(gender_pairs.items(), columns=['Kombination', 'Anzahl'])
    st.dataframe(gender_df)

# Food Preferences Section
st.header("Analyse Essensvorlieben der verteilten Einzelanmeldungen")
col7, col8 = st.columns(2)

with col7:
    # Food preference combinations for single registrations only
    couples_df = updated_df[updated_df['Paaranmeldung'] == 0]
    food_pairs = count_pair_frequencies(couples_df, 'Person1_FoodPreference', 'Person2_FoodPreference')
    food_df = pd.DataFrame([{'Combination': f"{pair[0]}, {pair[1]}", 'Count': count} 
                           for pair, count in food_pairs.items()])
    
    fig = px.bar(
        food_df,
        x='Combination',
        y='Count',
        title='Kombination Essensvorlieben (Einzelanmeldungen)'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

with col8:
    # Analyze couple characteristics
    results = analyze_couple_characteristics(updated_df)
    
    st.dataframe(food_df)
    st.write("Altersdifferenzen:")
    age_stats = pd.DataFrame({
        'Metric': results['age_difference_stats'].keys(),
        'Value': results['age_difference_stats'].values()
    })
    st.dataframe(age_stats)


# Preference Combinations Section
st.header("Team Master-Essensvorlieben Kombinationen")

# Analyze food preferences and normalize combinations
preferences_df = analyze_food_preferences(updated_df)
preferences_df['Preference_Combination'] = preferences_df['Preference_Combination'].apply(normalize_combination)
preference_counts = preferences_df['Preference_Combination'].value_counts()

fig = px.bar(
    x=preference_counts.index,
    y=preference_counts.values,
    title='Verteilung der Team-Präferenzkombinationen'
)
fig.update_layout(xaxis_tickangle=-45)
st.write("Übersicht Master-Essensvorlieben der jeweiligen Teams die sich im Laufe des Abends treffen")
st.plotly_chart(fig)

# Master Food Preferences Overview Section
st.header("Übersicht Master-Essensvorlieben")
master_food_counts = updated_df["MasterFoodPreference"].value_counts()

# Create bar chart using plotly
fig = px.bar(
    x=master_food_counts.index,
    y=master_food_counts.values,
    title='Verteilung der Master-Essensvorlieben der verteilten Teams',
    labels={'x': 'Food Preference', 'y': 'Count'}
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)


# Add a separator
st.markdown("---")

# Create a new section for the additional analyses
st.header("Weitere Analysen")

# Topic 1: Show all rows with inconsistencies
# st.subheader("1. Einzelanmeldungen mit Inkonsistenzen")
# inconsistent_kitchens = find_inconsistent_kitchens(df_original)
# st.write(f"Found {len(inconsistent_kitchens)} inconsistent registrations")
# st.dataframe(inconsistent_kitchens)

# For Topic 1: Show all rows with inconsistencies
st.subheader("1. Einzelanmeldungen mit Inkonsistenzen")
inconsistent_kitchens = find_inconsistent_kitchens(df_original)
st.write(f"Es wurden {len(inconsistent_kitchens)} inkonsistenzen in Einzelanmeldungen gefunden")
simple_paginated_dataframe(inconsistent_kitchens, title="Einzelanmeldungen mit Inkonsistenzen", page_size=10)

# Topic 2: Show all duplicate registrations
st.subheader("2. Anzeige aller Doppelanmeldungen")
frequent_names = get_frequent_rows(df_original, 'Name', min_frequency=1)
st.write(f"Es wurden {len(frequent_names)} Doppelanmeldungen gefunden")
st.dataframe(frequent_names)
simple_paginated_dataframe(frequent_names, title="Anzeige aller Doppelanmeldungen", page_size=10)

# Topic 3: Check if specific names are in the dataset
st.subheader("3. Befinden sich unsere Namen noch im Datensatz?")
names = ['Schirin', 'Ramin', 'Eichhorn']
matching_rows = find_names(df_original, names)
st.write(f"Es wurden {len(matching_rows)} matches für folgende Namen gefunden: {', '.join(names)}")
#st.dataframe(matching_rows)
simple_paginated_dataframe(matching_rows, title="Namen im Datensatz", page_size=10)

# Topic 4: Overview of distances for each team
st.subheader("4. Übersicht der Distanzen für jedes Team")
routes_df = get_team_routes(df)
routes_with_distances = calculate_distances(routes_df)
#st.dataframe(routes_with_distances)
simple_paginated_dataframe(routes_with_distances, title="Übersicht der Distanzen für jedes Team", page_size=10)


# Topic 5: Maximum distance for any team from location A to B
st.subheader("5. Maximale Distanz die ein Team von Gang A zu Gang B nehmen muss")
max_distance = routes_with_distances["Distance_to_next_km"].max()
st.write(f"Maximum Distanz: {max_distance:.3f} km")

# Topic 6: Overview of average total distances for each team
st.subheader("6. Übersicht der durchschnittlichen Distanzen für jedes Team")
st.write("Durschschnitt aus Dinstanzen (Vorspeise, Hauptspeise, Nachspeise) für jedes Team")
team_distance_stats = calculate_team_distance_stats(routes_with_distances)
st.dataframe(team_distance_stats)

# Topic 7: Box plot of average total distances for each team
st.subheader("7. Verteilung der durchschnittlichen Distanzen für jedes Team")

# Create a matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
team_distance_stats["mean_distance_km"].plot.box(ax=ax)
ax.set_title("Verteilung der durchschnittlichen Distanzen")
ax.set_ylabel("Distance (km)")
ax.grid(True, linestyle='--', alpha=0.7)

# Display the plot in Streamlit
st.pyplot(fig)

# Create an interactive version with Plotly
fig_plotly = px.box(team_distance_stats, y="mean_distance_km", 
                    title="Verteilung der durchschnittlichen Distanzen (Interactive)")
fig_plotly.update_layout(
    yaxis_title="Distance (km)",
    height=500,
    width=700
)
st.plotly_chart(fig_plotly)

# Add a footer
st.markdown("---")
