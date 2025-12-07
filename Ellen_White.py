# app.py
import streamlit as st
from mplsoccer import Sbopen, Pitch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

st.set_page_config(layout="wide", page_title="Player Match Analysis (WWC 2019)")

st.title("WWC 2019 — Player Match Analysis")
st.caption("Stina Blackstenius example adapted to a Streamlit app. Uses StatsBomb event data via mplsoccer.Sbopen().")

# ---------------------------------------------------------
# CACHING: Sbopen parser & loaded match/event data
# ---------------------------------------------------------
@st.cache_resource
def get_parser():
    return Sbopen()

@st.cache_data(show_spinner=False)
def load_matches(parser, competition_id=72, season_id=30):
    try:
        df_match = parser.match(competition_id=competition_id, season_id=season_id)
        return df_match
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_event(parser, match_id):
    try:
        # returns tuple (df_event, df_related, df_freeze, df_tactics)
        df_tuple = parser.event(match_id)
        return df_tuple
    except Exception as e:
        return None

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    competition_id = st.number_input("Competition ID", value=72, step=1)
    season_id = st.number_input("Season ID", value=30, step=1)
    default_player = "Stina Blackstenius"
    player_name = st.text_input("Player name", value=default_player)
    run_tournament_analysis = st.checkbox("Run tournament analysis (slower)", value=False)
    find_matches_btn = st.button("Find matches for player")

# ---------------------------------------------------------
# Initialize parser & load match list
# ---------------------------------------------------------
parser = get_parser()
df_match = load_matches(parser, competition_id=int(competition_id), season_id=int(season_id))

if df_match.empty:
    st.error("Could not load matches for the specified competition/season. Check competition_id & season_id.")
    st.stop()

st.sidebar.success(f"Loaded {len(df_match)} matches")

# ---------------------------------------------------------
# Find matches where the player appears
# ---------------------------------------------------------
def find_player_matches_in_tournament(parser, df_match, player_name, limit=50):
    """Search match ids where provided player_name appears. Returns list of match_ids."""
    found = []
    match_ids = df_match['match_id'].unique()
    # iterate but limit to avoid super long loop in UI
    for idx, m in enumerate(match_ids):
        try:
            df_evt_tuple = load_event(parser, int(m))
            if df_evt_tuple is None:
                continue
            df_event = df_evt_tuple[0]
            if player_name in df_event['player_name'].values:
                found.append(int(m))
            if len(found) >= 10:
                break
        except Exception:
            continue
        # small progress display handled by UI (no prints)
    return found

# If user pressed button, find matches
if find_matches_btn:
    with st.spinner("Searching matches for player..."):
        player_matches = find_player_matches_in_tournament(parser, df_match, player_name)
    if player_matches:
        st.success(f"Found {len(player_matches)} matches with {player_name}.")
    else:
        st.info(f"No matches found for {player_name} in this competition (search limited).")
        # Suggest forwards from the first match
        try:
            df_event_tuple = load_event(parser, int(df_match['match_id'].iloc[0]))
            df_event_samp = df_event_tuple[0]
            forwards = df_event_samp[df_event_samp['position_name'].str.contains('Forward', na=False)]
            alt = forwards['player_name'].dropna().unique()[:8]
            st.write("Suggested players from a sample match:")
            st.write(list(alt))
        except Exception:
            st.write("No sample match available to suggest players.")
else:
    # default: try to find small sample automatically (non-blocking)
    player_matches = find_player_matches_in_tournament(parser, df_match, player_name)

# Provide match selection
if player_matches:
    selected_match = st.selectbox("Select match id (where player found)", options=player_matches, index=0)
else:
    # fallback: show first 20 matches to choose from
    fallback_matches = df_match['match_id'].unique()[:20].tolist()
    selected_match = st.selectbox("Select match id (fallback)", options=fallback_matches)

st.write(f"Selected match: **{selected_match}**")

# ---------------------------------------------------------
# Load selected match events
# ---------------------------------------------------------
evt_tuple = load_event(parser, int(selected_match))
if evt_tuple is None:
    st.error(f"Could not load events for match {selected_match}.")
    st.stop()

df_event, df_related, df_freeze, df_tactics = evt_tuple
st.write(f"Events loaded: {len(df_event)}")

# If player not in match, suggest available forwards
if player_name not in df_event['player_name'].values:
    st.warning(f"{player_name} not found in match {selected_match}.")
    available = df_event['player_name'].dropna().unique()[:40]
    pick = st.selectbox("Select an available player from this match", options=available)
    if pick:
        player_name = pick
        st.info(f"Now analyzing: {player_name}")

# Filter player events
player_events = df_event[df_event["player_name"] == player_name]
if player_events.empty:
    st.warning(f"No events for {player_name} in match {selected_match}. App will still show tournament-level analysis if requested.")
else:
    st.subheader(f"Player: {player_name}   —   Team: {player_events['team_name'].iloc[0] if 'team_name' in player_events.columns and len(player_events)>0 else 'Unknown'}")

# ---------------------------------------------------------
# Helper: draw pitch (returns fig, ax, pitch)
# ---------------------------------------------------------
def draw_pitch_fig(figsize=(10,7)):
    pitch_obj = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = plt.subplots(figsize=figsize)
    pitch_obj.draw(ax=ax)
    return fig, ax, pitch_obj

# ---------------------------------------------------------
# Shot Map
# ---------------------------------------------------------
st.markdown("### Shot Map")
shots = player_events[player_events["type_name"] == "Shot"] if not player_events.empty else pd.DataFrame()
if not shots.empty:
    fig, ax, pitch = draw_pitch_fig()
    # annotate each shot and size by xG if available
    for _, row in shots.iterrows():
        x = row.get('x', None)
        y = row.get('y', None)
        if pd.isna(x) or pd.isna(y):
            continue
        xg = row.get('shot_statsbomb_xg', 0) if 'shot_statsbomb_xg' in row.index else 0
        size = 150 + (float(xg) * 500) if not pd.isna(xg) else 150
        color = 'red' if xg > 0.3 else 'yellow' if xg > 0.1 else 'green'
        pitch.scatter(x, y, s=size, color=color, edgecolor='black', linewidth=1.0, ax=ax, zorder=3, alpha=0.9)
        if not pd.isna(xg) and xg > 0:
            ax.text(x, y + 1.5, f"{xg:.2f}", ha='center', fontsize=8, color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    ax.set_title(f"{player_name} – Shot Map (Match {selected_match})", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    # Show shot stats
    st.write("Shot statistics:")
    st.write({
        "Total shots": len(shots),
        "Total xG (if available)": float(shots['shot_statsbomb_xg'].sum()) if 'shot_statsbomb_xg' in shots.columns else "N/A",
        "Avg xG per shot": float(shots['shot_statsbomb_xg'].mean()) if 'shot_statsbomb_xg' in shots.columns and len(shots)>0 else "N/A"
    })
else:
    st.info(f"No shots for {player_name} in this match.")

# ---------------------------------------------------------
# Pass Map
# ---------------------------------------------------------
st.markdown("### Pass Map")
passes = player_events[player_events["type_name"] == "Pass"] if not player_events.empty else pd.DataFrame()
if not passes.empty and all(col in passes.columns for col in ['x','y','end_x','end_y']):
    fig, ax, pitch = draw_pitch_fig()
    # classify successful vs unsuccessful
    successful = passes[pd.isna(passes['outcome_name']) | (passes['outcome_name'] != 'Incomplete')]
    unsuccessful = passes[passes['outcome_name'] == 'Incomplete'] if 'outcome_name' in passes.columns else pd.DataFrame()
    # compute distances -> thickness
    if not successful.empty:
        successful = successful.copy()
        successful['distance'] = np.sqrt((successful['end_x'] - successful['x'])**2 + (successful['end_y'] - successful['y'])**2)
        successful['thickness'] = 1.5 + (successful['distance'] / 50)
    if not unsuccessful.empty:
        unsuccessful = unsuccessful.copy()
        unsuccessful['distance'] = np.sqrt((unsuccessful['end_x'] - unsuccessful['x'])**2 + (unsuccessful['end_y'] - unsuccessful['y'])**2)
        unsuccessful['thickness'] = 1.5 + (unsuccessful['distance'] / 50)
    # plot
    for _, row in successful.iterrows():
        pitch.arrows(row['x'], row['y'], row['end_x'], row['end_y'],
                     width=row.get('thickness',1.5), headwidth=(8+row.get('thickness',1.5)*2),
                     headlength=(8+row.get('thickness',1.5)*2), color='white', ax=ax, zorder=2, alpha=0.85)
    for _, row in unsuccessful.iterrows():
        pitch.arrows(row['x'], row['y'], row['end_x'], row['end_y'],
                     width=row.get('thickness',1.5), headwidth=(8+row.get('thickness',1.5)*2),
                     headlength=(8+row.get('thickness',1.5)*2), color='white', ax=ax, zorder=2, alpha=0.6, linestyle='--')
    # legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color='white', lw=3, label='Successful pass'),
        Line2D([0],[0], color='white', lw=3, label='Unsuccessful pass', linestyle='--')
    ]
    ax.legend(handles=legend_elems, loc='upper right')
    ax.set_title(f"{player_name} – Pass Map (Match {selected_match})", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    # pass stats
    st.write("Pass statistics:")
    pass_stats = {
        "Total passes": len(passes),
    }
    if 'outcome_name' in passes.columns:
        pass_stats["Success rate (%)"] = float((passes['outcome_name'] != 'Incomplete').mean() * 100)
    if 'pass_length' in passes.columns:
        pass_stats["Avg pass length"] = float(passes['pass_length'].mean())
    st.write(pass_stats)
else:
    st.info("No complete pass geometry (x,end_x) available for this player in this match to draw pass map.")

# ---------------------------------------------------------
# Carry Map
# ---------------------------------------------------------
st.markdown("### Carry / Dribble Map")
carries = player_events[player_events["type_name"] == "Carry"] if not player_events.empty else pd.DataFrame()
if not carries.empty and all(col in carries.columns for col in ['x','y','end_x','end_y']):
    fig, ax, pitch = draw_pitch_fig()
    carries = carries.copy()
    carries['distance'] = np.sqrt((carries['end_x'] - carries['x'])**2 + (carries['end_y'] - carries['y'])**2)
    carries['thickness'] = 2.0 + (carries['distance'] / 40)
    for _, row in carries.iterrows():
        pitch.arrows(row['x'], row['y'], row['end_x'], row['end_y'],
                     width=row.get('thickness',2.0), headwidth=(10+row.get('thickness',2.0)*2),
                     headlength=(10+row.get('thickness',2.0)*2), color='white', edgecolor='orange', ax=ax, alpha=0.9, zorder=2)
    ax.set_title(f"{player_name} – Carry/Dribble Map (Match {selected_match})", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    st.write({
        "Total carries": len(carries),
        "Average carry distance (yards)": float(carries['distance'].mean())
    })
else:
    st.info("No carries with start/end coordinates for this player in this match.")

# ---------------------------------------------------------
# Comprehensive summary (short)
# ---------------------------------------------------------
st.markdown("### Quick Match Summary")
if not player_events.empty:
    st.write({
        "Match": int(selected_match),
        "Total events (player)": int(len(player_events)),
        "Top event types": player_events['type_name'].value_counts().head(8).to_dict()
    })
else:
    st.write("No player events available for this match.")

# ---------------------------------------------------------
# Tournament-level analysis (optional)
# ---------------------------------------------------------
if run_tournament_analysis:
    st.markdown("---")
    st.header("Tournament-level analysis (sample of matches for speed)")
    st.info("For speed, this analysis loads up to 5 matches where the player was found.")
    with st.spinner("Loading tournament samples..."):
        # find up to 5 matches containing the player across the tournament
        match_ids = df_match['match_id'].unique()
        found_matches = []
        for m in match_ids:
            try:
                df_evt_tuple = load_event(parser, int(m))
                if df_evt_tuple is None:
                    continue
                df_ev = df_evt_tuple[0]
                if player_name in df_ev['player_name'].values:
                    found_matches.append(int(m))
                if len(found_matches) >= 5:
                    break
            except Exception:
                continue
        if not found_matches:
            st.error("No tournament matches found for player (or search limited).")
        else:
            st.write(f"Using {len(found_matches)} match(es) for sample tournament analysis: {found_matches}")
            all_events = []
            for m in found_matches:
                df_evt_tuple = load_event(parser, int(m))
                if df_evt_tuple is None:
                    continue
                all_events.append(df_evt_tuple[0])
            if all_events:
                all_df = pd.concat(all_events, ignore_index=True)
                forwards_df = all_df[all_df['position_name'].str.contains('Forward', na=False)]
                if forwards_df.empty:
                    st.error("No forwards found in loaded sample matches.")
                else:
                    # compute player-level aggregates
                    player_stats = forwards_df.groupby('player_name').agg({
                        'shot_statsbomb_xg': 'sum',
                        'id': 'count',
                        'type_name': lambda x: (x == 'Shot').sum(),
                        'pass_length': lambda x: x.mean() if not x.isnull().all() else 0,
                        'duration': 'mean'
                    }).rename(columns={'id':'total_actions', 'type_name':'total_shots', 'shot_statsbomb_xg':'total_xg', 'pass_length':'avg_pass_length', 'duration':'avg_action_duration'})
                    player_stats['total_xg'] = player_stats['total_xg'].fillna(0)
                    player_stats['xg_per_shot'] = player_stats['total_xg'] / player_stats['total_shots'].replace(0,1)
                    # actions per match (approx)
                    player_stats['actions_per_match'] = player_stats['total_actions'] / max(1, len(found_matches))
                    st.write("Sample player stats (top 10):")
                    st.dataframe(player_stats.sort_values('total_xg', ascending=False).head(10))
                    # if our player exists show z-scores
                    if player_name in player_stats.index:
                        target = player_stats.loc[player_name]
                        st.subheader(f"Tournament sample summary for {player_name}")
                        st.write({
                            "Total xG (sample)": float(target['total_xg']),
                            "Total shots (sample)": float(target['total_shots']),
                            "xG per shot (sample)": float(target['xg_per_shot']),
                            "Actions per match (approx)": float(target['actions_per_match'])
                        })
                        # z-scores
                        metrics = ['total_xg','total_shots','xg_per_shot','total_actions','avg_pass_length']
                        for m in metrics:
                            if m in player_stats.columns:
                                player_stats[f'z_{m}'] = scipy_stats.zscore(player_stats[m].fillna(0))
                        zcols = [c for c in player_stats.columns if c.startswith('z_')]
                        st.write("Z-scores (sample):")
                        st.dataframe(player_stats[[c for c in player_stats.columns if c in zcols]].loc[[player_name]])
                        # simple bar chart for z_total_xg
                        if f'z_total_xg' in player_stats.columns:
                            fig, ax = plt.subplots(figsize=(8,4))
                            top = player_stats.sort_values(f'z_total_xg', ascending=False).head(12)
                            ax.barh(top.index, top[f'z_total_xg'])
                            ax.axvline(x=0, color='black', linewidth=1)
                            ax.set_title("Top players by z_total_xg (sample)")
                            st.pyplot(fig)
                    else:
                        st.warning(f"{player_name} not in tournament sample stats.")
            else:
                st.error("No events loaded for sample matches.")

st.markdown("---")
st.caption("App created from provided analysis script. Modify player name or match to explore other players.")
