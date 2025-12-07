import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Sbopen, Pitch
from scipy import stats
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3949AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">⚽ Soccer Player Performance Analytics</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard analyzes soccer player performance using StatsBomb data.
Visualize player actions, compare statistics, and understand performance through z-score analysis.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Competition selection
    competition_options = {
        "Women's World Cup 2019": (72, 30),
        "UEFA Euro 2024": (55, 43),
        "FIFA World Cup 2022": (43, 106),
        "Premier League 2022/23": (2, 42)
    }
    
    selected_comp = st.selectbox(
        "Select Competition:",
        list(competition_options.keys())
    )
    
    competition_id, season_id = competition_options[selected_comp]
    
    # Player name input
    player_name = st.text_input(
        "Enter Player Name:",
        value="Ellen White",
        help="Enter the full name of the player you want to analyze"
    )
    
    # Match selection
    st.subheader("Match Selection")
    use_specific_match = st.checkbox("Use specific match ID", value=True)
    
    if use_specific_match:
        match_id = st.number_input(
            "Match ID:",
            min_value=1,
            value=69301,
            step=1
        )
    else:
        match_id = None
    
    # Analysis options
    st.subheader("Analysis Options")
    show_shot_map = st.checkbox("Show Shot Map", value=True)
    show_pass_map = st.checkbox("Show Pass Map", value=True)
    show_carry_map = st.checkbox("Show Carry/Dribble Map", value=True)
    show_zscore_analysis = st.checkbox("Show Z-Score Analysis", value=True)
    
    # Visualization settings
    st.subheader("Visualization Settings")
    arrow_color = st.color_picker("Arrow Color", value="#FFFFFF")
    arrow_thickness = st.slider("Base Arrow Thickness", 1.0, 5.0, 2.0, 0.5)
    
    # Action buttons
    st.markdown("---")
    analyze_button = st.button("🚀 Analyze Player", type="primary", use_container_width=True)
    reset_button = st.button("🔄 Reset Analysis", use_container_width=True)

# Initialize parser
@st.cache_resource
def get_parser():
    return Sbopen()

parser = get_parser()

# Function to draw pitch
def draw_pitch():
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = plt.subplots(figsize=(10, 7))
    pitch.draw(ax=ax)
    return fig, ax, pitch

# Main analysis function
def analyze_player(player_name, competition_id, season_id, match_id, arrow_color, arrow_thickness):
    """Main function to analyze player performance"""
    
    results = {}
    
    try:
        # Load match data
        with st.spinner(f"Loading match data for {player_name}..."):
            if match_id:
                try:
                    df_event, df_related, df_freeze, df_tactics = parser.event(match_id)
                    results['match_loaded'] = True
                    
                    if player_name in df_event['player_name'].values:
                        player_events = df_event[df_event["player_name"] == player_name]
                        results['player_found'] = True
                        results['player_events'] = player_events
                        results['total_events'] = len(player_events)
                        
                        # Get player info
                        position = player_events['position_name'].iloc[0] if pd.notna(player_events['position_name'].iloc[0]) else "Unknown"
                        team = player_events['team_name'].iloc[0] if pd.notna(player_events['team_name'].iloc[0]) else "Unknown"
                        results['position'] = position
                        results['team'] = team
                        
                        # Event type breakdown
                        event_types = player_events['type_name'].value_counts()
                        results['event_types'] = event_types
                        
                    else:
                        results['player_found'] = False
                        results['available_players'] = df_event['player_name'].dropna().unique()[:10]
                        
                except Exception as e:
                    st.error(f"Error loading match: {e}")
                    results['match_loaded'] = False
            else:
                st.warning("Please enter a match ID")
                return results
        
        if results.get('player_found', False):
            player_events = results['player_events']
            
            # Display player info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", results['total_events'])
            with col2:
                st.metric("Position", results['position'])
            with col3:
                st.metric("Team", results['team'])
            
            # Event breakdown
            st.subheader("📊 Event Breakdown")
            event_df = pd.DataFrame(results['event_types'].head(10)).reset_index()
            event_df.columns = ['Event Type', 'Count']
            st.dataframe(event_df, use_container_width=True)
            
            # Shot Map
            if show_shot_map:
                st.subheader("🎯 Shot Map")
                shots = player_events[player_events["type_name"] == "Shot"]
                
                if not shots.empty:
                    fig, ax, pitch = draw_pitch()
                    
                    for idx, row in shots.iterrows():
                        xg = row.get('shot_statsbomb_xg', 0)
                        # Color based on xG value
                        if xg > 0.3:
                            color = 'red'
                        elif xg > 0.1:
                            color = 'orange'
                        else:
                            color = 'yellow'
                        
                        pitch.scatter(
                            row['x'], row['y'],
                            s=200 + (xg * 500),
                            color=color,
                            edgecolor='black',
                            linewidth=1.5,
                            ax=ax,
                            zorder=3,
                            alpha=0.8
                        )
                        
                        if pd.notna(xg) and xg > 0:
                            ax.text(row['x'], row['y'] + 1.5,
                                    f"{xg:.2f}",
                                    ha='center', fontsize=9, color='black',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                    
                    ax.set_title(f"{player_name} – Shot Map\nMatch {match_id}", fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    # Shot statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Shots", len(shots))
                    with col2:
                        if 'shot_statsbomb_xg' in shots.columns:
                            total_xg = shots['shot_statsbomb_xg'].sum()
                            st.metric("Total xG", f"{total_xg:.2f}")
                    with col3:
                        if 'shot_statsbomb_xg' in shots.columns and len(shots) > 0:
                            avg_xg = shots['shot_statsbomb_xg'].mean()
                            st.metric("Avg xG per Shot", f"{avg_xg:.2f}")
                else:
                    st.info(f"No shots recorded for {player_name} in this match")
            
            # Pass Map
            if show_pass_map:
                st.subheader("🎯 Pass Map")
                passes = player_events[player_events["type_name"] == "Pass"]
                
                if not passes.empty:
                    fig, ax, pitch = draw_pitch()
                    
                    # Separate successful and unsuccessful passes
                    successful_passes = passes[pd.isna(passes['outcome_name']) | (passes['outcome_name'] != 'Incomplete')]
                    unsuccessful_passes = passes[passes['outcome_name'] == 'Incomplete'] if 'outcome_name' in passes.columns else pd.DataFrame()
                    
                    # Plot successful passes
                    if not successful_passes.empty:
                        pitch.arrows(
                            successful_passes['x'], successful_passes['y'],
                            successful_passes['end_x'], successful_passes['end_y'],
                            width=arrow_thickness,
                            headwidth=8,
                            headlength=8,
                            color=arrow_color,
                            edgecolor='green',
                            linewidth=0.8,
                            alpha=0.85,
                            ax=ax,
                            zorder=2,
                            label='Successful'
                        )
                    
                    # Plot unsuccessful passes
                    if not unsuccessful_passes.empty:
                        pitch.arrows(
                            unsuccessful_passes['x'], unsuccessful_passes['y'],
                            unsuccessful_passes['end_x'], unsuccessful_passes['end_y'],
                            width=arrow_thickness,
                            headwidth=8,
                            headlength=8,
                            color=arrow_color,
                            edgecolor='red',
                            linewidth=0.8,
                            alpha=0.7,
                            ax=ax,
                            zorder=2,
                            label='Unsuccessful',
                            linestyle='--'
                        )
                    
                    ax.set_title(f"{player_name} – Pass Map\nMatch {match_id}", fontsize=14, fontweight='bold')
                    
                    # Add legend
                    if not successful_passes.empty or not unsuccessful_passes.empty:
                        ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Pass statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Passes", len(passes))
                    with col2:
                        if not passes.empty and 'outcome_name' in passes.columns:
                            success_rate = (passes['outcome_name'] != 'Incomplete').mean() * 100
                            st.metric("Pass Success Rate", f"{success_rate:.1f}%")
                    with col3:
                        if 'pass_length' in passes.columns:
                            avg_pass_length = passes['pass_length'].mean()
                            st.metric("Avg Pass Length", f"{avg_pass_length:.1f} yards")
                else:
                    st.info(f"No passes recorded for {player_name} in this match")
            
            # Carry/Dribble Map
            if show_carry_map:
                st.subheader("🎯 Carry/Dribble Map")
                carries = player_events[player_events["type_name"] == "Carry"]
                
                if not carries.empty:
                    fig, ax, pitch = draw_pitch()
                    
                    # Calculate distances for thickness variation
                    if all(col in carries.columns for col in ['x', 'y', 'end_x', 'end_y']):
                        carries['distance'] = np.sqrt(
                            (carries['end_x'] - carries['x'])**2 + 
                            (carries['end_y'] - carries['y'])**2
                        )
                        carries['thickness'] = arrow_thickness + (carries['distance'] / 40)
                    
                    # Plot carries
                    pitch.arrows(
                        carries['x'], carries['y'],
                        carries['end_x'], carries['end_y'],
                        width=carries.get('thickness', arrow_thickness) if 'thickness' in carries.columns else arrow_thickness,
                        headwidth=10,
                        headlength=10,
                        color=arrow_color,
                        edgecolor='orange',
                        linewidth=0.8,
                        alpha=0.9,
                        ax=ax,
                        zorder=2
                    )
                    
                    ax.set_title(f"{player_name} – Carry/Dribble Map\nMatch {match_id}", fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    # Carry statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Carries", len(carries))
                    with col2:
                        if 'duration' in carries.columns:
                            avg_duration = carries['duration'].mean()
                            st.metric("Avg Duration", f"{avg_duration:.1f}s")
                    with col3:
                        if all(col in carries.columns for col in ['x', 'y', 'end_x', 'end_y']):
                            distances = np.sqrt((carries['end_x'] - carries['x'])**2 + (carries['end_y'] - carries['y'])**2)
                            avg_distance = distances.mean()
                            st.metric("Avg Distance", f"{avg_distance:.1f} yards")
                else:
                    st.info(f"No carries recorded for {player_name} in this match")
            
            # Z-Score Analysis
            if show_zscore_analysis:
                st.subheader("📈 Z-Score Performance Analysis")
                
                with st.spinner("Loading tournament data for comparison..."):
                    try:
                        # Load multiple matches for comparison
                        df_matches = parser.match(competition_id=competition_id, season_id=season_id)
                        match_ids_all = df_matches['match_id'].unique()
                        
                        # Load a subset of matches
                        match_ids_subset = match_ids_all[:10]  # First 10 matches
                        all_tournament_events = []
                        
                        progress_bar = st.progress(0)
                        for i, m_id in enumerate(match_ids_subset):
                            try:
                                df_e, _, _, _ = parser.event(m_id)
                                all_tournament_events.append(df_e)
                                progress_bar.progress((i + 1) / len(match_ids_subset))
                            except:
                                continue
                        
                        progress_bar.empty()
                        
                        if all_tournament_events:
                            all_events_df = pd.concat(all_tournament_events, ignore_index=True)
                            
                            # Filter for forwards
                            forwards_df = all_events_df[all_events_df['position_name'].str.contains('Forward', na=False)]
                            
                            if not forwards_df.empty and player_name in forwards_df['player_name'].values:
                                # Calculate statistics
                                player_stats = forwards_df.groupby('player_name').agg({
                                    'shot_statsbomb_xg': 'sum',
                                    'id': 'count',
                                    'type_name': lambda x: (x == 'Shot').sum(),
                                    'pass_length': lambda x: x.mean() if not x.isnull().all() else 0,
                                }).rename(columns={
                                    'id': 'total_actions',
                                    'type_name': 'total_shots',
                                    'shot_statsbomb_xg': 'total_xg',
                                    'pass_length': 'avg_pass_length'
                                })
                                
                                # Fill NaN values
                                player_stats['total_xg'] = player_stats['total_xg'].fillna(0)
                                player_stats['avg_pass_length'] = player_stats['avg_pass_length'].fillna(0)
                                
                                # Calculate derived metrics
                                player_stats['xg_per_shot'] = player_stats['total_xg'] / player_stats['total_shots'].replace(0, 1)
                                
                                # Calculate z-scores
                                metrics_to_analyze = ['total_xg', 'total_shots', 'xg_per_shot', 'total_actions']
                                
                                for metric in metrics_to_analyze:
                                    if metric in player_stats.columns:
                                        player_stats[f'z_{metric}'] = stats.zscore(player_stats[metric].fillna(0))
                                
                                if player_name in player_stats.index:
                                    target_stats = player_stats.loc[player_name]
                                    
                                    # Display z-scores
                                    st.markdown("### 📊 Z-Score Analysis")
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    z_metrics = [
                                        ('total_xg', 'Total xG'),
                                        ('total_shots', 'Total Shots'),
                                        ('xg_per_shot', 'xG per Shot'),
                                        ('total_actions', 'Total Actions')
                                    ]
                                    
                                    for idx, (metric, label) in enumerate(z_metrics):
                                        z_col = f'z_{metric}'
                                        if z_col in target_stats:
                                            with [col1, col2, col3, col4][idx]:
                                                z_score = target_stats[z_col]
                                                color = "green" if z_score > 0 else "red" if z_score < 0 else "gray"
                                                st.metric(label, f"{z_score:.2f}")
                                                if z_score > 0:
                                                    st.caption("Above Average 📈")
                                                elif z_score < 0:
                                                    st.caption("Below Average 📉")
                                                else:
                                                    st.caption("Average ⏸️")
                                    
                                    # Create z-score comparison chart
                                    st.markdown("### 📈 Z-Score Comparison Chart")
                                    comparison_metrics = ['total_xg', 'total_shots', 'xg_per_shot']
                                    comparison_labels = ['Total xG', 'Total Shots', 'xG per Shot']
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    x_pos = np.arange(len(comparison_metrics))
                                    
                                    # Get target player z-scores
                                    target_z_scores = []
                                    for metric in comparison_metrics:
                                        z_col = f'z_{metric}'
                                        if z_col in target_stats:
                                            target_z_scores.append(target_stats[z_col])
                                        else:
                                            target_z_scores.append(0)
                                    
                                    # Plot target player
                                    bars = ax.bar(x_pos, target_z_scores, alpha=0.7)
                                    
                                    # Color bars based on value
                                    for bar, score in zip(bars, target_z_scores):
                                        if score > 0:
                                            bar.set_color('green')
                                        elif score < 0:
                                            bar.set_color('red')
                                        else:
                                            bar.set_color('gray')
                                    
                                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                                    ax.set_xticks(x_pos)
                                    ax.set_xticklabels(comparison_labels, rotation=45, ha='right')
                                    ax.set_ylabel('Z-Score')
                                    ax.set_title(f'{player_name} Z-Score Performance\n(Compared to Tournament Forwards)')
                                    ax.grid(True, alpha=0.3, axis='y')
                                    
                                    # Add value labels on bars
                                    for i, (bar, score) in enumerate(zip(bars, target_z_scores)):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                                f'{score:.2f}',
                                                ha='center', va='bottom' if height >= 0 else 'top')
                                    
                                    st.pyplot(fig)
                                    
                                    # Statistical test
                                    st.markdown("### 📊 Statistical Significance")
                                    if len(player_stats) > 1:
                                        target_xg = target_stats['total_xg']
                                        other_xg = player_stats[player_stats.index != player_name]['total_xg']
                                        
                                        if len(other_xg) > 1 and np.std(other_xg) > 0:
                                            t_stat, p_value = stats.ttest_1samp(other_xg, target_xg)
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("t-statistic", f"{t_stat:.3f}")
                                            with col2:
                                                st.metric("p-value", f"{p_value:.3f}")
                                            
                                            if p_value < 0.05:
                                                st.success("✅ Statistically significant difference from average (p < 0.05)")
                                                if target_xg > other_xg.mean():
                                                    st.info(f"{player_name} has significantly HIGHER xG than tournament average")
                                                else:
                                                    st.info(f"{player_name} has significantly LOWER xG than tournament average")
                                            else:
                                                st.warning("⚠️ No statistically significant difference from average (p ≥ 0.05)")
                                                st.info(f"{player_name}'s xG is not significantly different from tournament average")
                                    
                                    # Percentile rankings
                                    st.markdown("### 📊 Percentile Rankings")
                                    percentiles = {}
                                    for metric in ['total_xg', 'total_shots', 'xg_per_shot']:
                                        if metric in player_stats.columns:
                                            target_value = target_stats[metric]
                                            all_values = player_stats[metric]
                                            percentile = stats.percentileofscore(all_values, target_value)
                                            percentiles[metric] = percentile
                                    
                                    # Display percentiles
                                    col1, col2, col3 = st.columns(3)
                                    percentile_labels = {
                                        'total_xg': 'Total xG',
                                        'total_shots': 'Total Shots',
                                        'xg_per_shot': 'xG per Shot'
                                    }
                                    
                                    for idx, (metric, label) in enumerate(percentile_labels.items()):
                                        if metric in percentiles:
                                            with [col1, col2, col3][idx]:
                                                percentile = percentiles[metric]
                                                st.metric(f"{label} Percentile", f"{percentile:.1f}%")
                                                if percentile >= 75:
                                                    st.caption("🏆 Elite")
                                                elif percentile >= 50:
                                                    st.caption("👍 Above Average")
                                                elif percentile >= 25:
                                                    st.caption("👎 Below Average")
                                                else:
                                                    st.caption("⚠️ Bottom Quartile")
                                    
                                    # Show top comparable players
                                    st.markdown("### 👥 Most Similar Players")
                                    
                                    # Calculate similarity (Euclidean distance in z-space)
                                    similarity_scores = []
                                    target_z_vector = []
                                    for metric in ['total_xg', 'total_shots', 'xg_per_shot']:
                                        z_col = f'z_{metric}'
                                        if z_col in target_stats:
                                            target_z_vector.append(target_stats[z_col])
                                    
                                    for other_player in player_stats.index:
                                        if other_player != player_name:
                                            other_z_vector = []
                                            for metric in ['total_xg', 'total_shots', 'xg_per_shot']:
                                                z_col = f'z_{metric}'
                                                if z_col in player_stats.columns:
                                                    other_z_vector.append(player_stats.loc[other_player, z_col])
                                            
                                            if len(other_z_vector) == len(target_z_vector):
                                                distance = np.sqrt(np.sum((np.array(target_z_vector) - np.array(other_z_vector))**2))
                                                similarity_scores.append((other_player, distance))
                                    
                                    if similarity_scores:
                                        similarity_scores.sort(key=lambda x: x[1])
                                        similar_df = pd.DataFrame(similarity_scores[:5], columns=['Player', 'Distance'])
                                        st.dataframe(similar_df, use_container_width=True)
                                    
                            else:
                                st.warning(f"{player_name} not found in tournament data. Try analyzing a different match.")
                        else:
                            st.error("Could not load tournament data for comparison.")
                            
                    except Exception as e:
                        st.error(f"Error in z-score analysis: {e}")
        
        elif not results.get('player_found', False) and results.get('match_loaded', False):
            st.error(f"Player '{player_name}' not found in match {match_id}")
            st.info("Available players in this match:")
            for player in results.get('available_players', []):
                st.write(f"- {player}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    return results

# Main app logic
if analyze_button:
    with st.container():
        results = analyze_player(player_name, competition_id, season_id, match_id, arrow_color, arrow_thickness)
        
        # Display summary
        if results.get('player_found', False):
            st.success(f"✅ Analysis complete for {player_name}!")
            
            # Summary statistics
            st.subheader("📋 Performance Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Match Performance")
                st.write(f"**Match ID:** {match_id}")
                st.write(f"**Total Events:** {results.get('total_events', 0)}")
                st.write(f"**Position:** {results.get('position', 'N/A')}")
                st.write(f"**Team:** {results.get('team', 'N/A')}")
            
            with col2:
                st.markdown("### Event Distribution")
                if 'event_types' in results:
                    event_df = pd.DataFrame(results['event_types'].head(5)).reset_index()
                    event_df.columns = ['Event Type', 'Count']
                    st.dataframe(event_df, hide_index=True, use_container_width=True)
        
        # Add explanations
        with st.expander("📖 Understanding the Analysis"):
            st.markdown("""
            ### How to interpret the visualizations:
            
            **Shot Map:**
            - 🔴 Red circles: High xG shots (> 0.3)
            - 🟡 Yellow circles: Medium xG shots (0.1-0.3)
            - 🟢 Green circles: Low xG shots (< 0.1)
            - Size indicates xG value (larger = higher xG)
            
            **Pass Map:**
            - Solid white arrows: Successful passes
            - Dashed white arrows: Unsuccessful passes
            - Green outline: Successful completion
            - Red outline: Unsuccessful attempt
            
            **Carry/Dribble Map:**
            - White arrows with orange outline
            - Thickness indicates carry distance
            - Shows player movement with the ball
            
            **Z-Score Analysis:**
            - **Z-score = 0**: Average performance
            - **Z-score > 0**: Above average (positive is better)
            - **Z-score < 0**: Below average (negative is worse)
            - **|Z-score| > 1.96**: Statistically significant difference
            """)
            
            st.markdown("""
            ### Key Metrics Explained:
            
            **xG (Expected Goals):**
            - Measures quality of scoring chances
            - Higher xG = better chance to score
            - Total xG: Sum of all chance qualities
            - xG per shot: Average chance quality
            
            **Z-Scores:**
            - Shows how player compares to peers
            - Based on tournament forward average
            - Positive = better than average
            - Negative = worse than average
            
            **Percentile Rankings:**
            - Shows relative standing
            - 90th percentile = top 10% of players
            - 50th percentile = exactly average
            - 25th percentile = bottom quarter
            """)

elif reset_button:
    st.rerun()
else:
    # Initial state - show instructions
    st.info("👈 Configure your analysis in the sidebar and click 'Analyze Player' to begin!")
    
    # Quick examples
    st.markdown("### 🚀 Quick Start Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Women's World Cup 2019**
        - Player: Ellen White
        - Match ID: 69301
        """)
    
    with col2:
        st.markdown("""
        **Other Players to Try:**
        - Stina Blackstenius
        - Sam Kerr
        - Alex Morgan
        - Marta
        """)
    
    with col3:
        st.markdown("""
        **Features:**
        - Shot visualization
        - Pass analysis
        - Dribble mapping
        - Statistical comparison
        - Z-score analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, StatsBomb, and mplsoccer</p>
    <p>Data courtesy of StatsBomb | Analytics Dashboard v1.0</p>
</div>
""", unsafe_allow_html=True)
