"""
app.py — AI Dungeon Master · Ultimate Edition
"""
import streamlit as st

st.set_page_config(
    page_title="AI Dungeon Master",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend import (
    init_memory, gen_story_stream, sum_turn, find_npc,
    sum_npc_turn, GENRE_OPENINGS, post_turn_updates,
    retrieve_intent_only,
)
from character_memory import add_char_mem
from quest import add_quest, get_quests
from dice import parse_and_roll, format_rolls
from location import detect_location
from character_sheet import update_sheet
from combat import update_combat, enemy_hp_bar_html, default_combat_state
from mood import detect_mood, get_mood_style
from lore import extract_and_store_lore, get_all_lore, LORE_ICONS
from relationships import update_relationship, get_tier
from stats import increment, is_dead, respawn, format_stats

# ── ML memory modules ─────────────────────────────────────────────────────
from emotion_tracker import get_emotion_history
from intent_classifier import INTENTS
from memory_consolidator import get_memory_count
from narrative_graph import graph_stats

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap');

:root {
    --gold:    #c9a84c;
    --gold2:   #e8c87a;
    --dark:    #0a0806;
    --panel:   #100d0a;
    --panel2:  #161210;
    --border:  #2a1e10;
    --text:    #ece0c8;
    --muted:   #7a6a50;
    --red:     #8b1a1a;
    --green:   #2d6a3f;
    --accent:  #c9a84c;
}

html, body, [data-testid="stApp"] {
    background: var(--dark);
    color: var(--text);
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 18px;
}

/* Grain texture overlay */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

[data-testid="stHeader"] { display: none; }

/* ── Title ── */
.dm-title {
    font-family: 'Cinzel Decorative', serif;
    font-size: 2rem;
    color: var(--gold);
    text-shadow: 0 0 40px rgba(201,168,76,0.5), 0 2px 4px rgba(0,0,0,0.8);
    letter-spacing: 0.06em;
    margin: 0;
    line-height: 1.2;
}
.dm-subtitle {
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 0.95rem;
    color: var(--muted);
    letter-spacing: 0.02em;
    margin-top: 0.2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.sidebar-section-title {
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.72rem;
    color: var(--gold) !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.35rem;
    margin: 1.1rem 0 0.6rem;
}

/* ── HP / enemy bars ── */
.bar-wrap {
    background: #0f0a06;
    border: 1px solid var(--border);
    border-radius: 3px;
    height: 8px;
    width: 100%;
    margin: 3px 0 8px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* ── DM message — parchment card ── */
.dm-message-card {
    background: linear-gradient(135deg, #110e09 0%, #0d0b07 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    font-family: 'Crimson Pro', serif;
    font-size: 1.08rem;
    line-height: 1.75;
    color: var(--text);
    transition: border-left-color 0.6s ease;
    margin: 0.2rem 0;
}

/* ── Mood banner ── */
.mood-banner {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    margin-bottom: 0.5rem;
    padding: 0.2rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 20px;
    background: #0d0a06;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #130f0a !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-size: 1rem !important;
    border-radius: 5px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 16px rgba(201,168,76,0.12) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    background: #0f0c08 !important;
    border-left: 3px solid var(--gold) !important;
    color: var(--text) !important;
    font-family: 'Crimson Pro', serif;
}

hr { border-color: var(--border) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e160a, #2e2010) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.1em;
    border-radius: 3px !important;
    transition: all 0.25s;
    padding: 0.4rem 0.8rem !important;
}
.stButton > button:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
    box-shadow: 0 0 12px rgba(201,168,76,0.2) !important;
}

/* ── Dice box ── */
.dice-box {
    background: #0d0a06;
    border: 1px solid var(--gold);
    border-radius: 5px;
    padding: 0.5rem 0.85rem;
    margin: 0.35rem 0;
    font-family: 'Crimson Pro', serif;
    font-size: 1rem;
    color: var(--gold2);
}
.dice-crit {
    color: #ffd700;
    font-weight: 600;
    text-shadow: 0 0 8px rgba(255,215,0,0.5);
}
.dice-fumble {
    color: #c83a2a;
    font-weight: 600;
}

/* ── Combat panel ── */
.combat-panel {
    background: linear-gradient(135deg, #1a0505, #120808);
    border: 1px solid #5c1a1a;
    border-radius: 5px;
    padding: 0.7rem 0.9rem;
    margin: 0.5rem 0;
    animation: pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 rgba(139,26,26,0); }
    50%       { box-shadow: 0 0 12px rgba(139,26,26,0.4); }
}
.combat-turn-badge {
    display: inline-block;
    background: #8b1a1a;
    color: #ffd0d0;
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    margin-bottom: 0.4rem;
}

/* ── Relationship chip ── */
.rel-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
}
.rel-score-bar {
    display: inline-block;
    width: 60px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    position: relative;
    vertical-align: middle;
    margin-left: 6px;
}
.rel-score-fill {
    position: absolute;
    top: 0; bottom: 0;
    border-radius: 2px;
    transition: width 0.4s, left 0.4s;
}

/* ── Lore entry ── */
.lore-entry {
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
    line-height: 1.5;
}
.lore-name {
    color: var(--gold2);
    font-weight: 600;
}
.lore-desc { color: var(--muted); }

/* ── Quest item ── */
.quest-item {
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
    color: var(--gold2);
}

/* ── Inventory pill ── */
.inv-pill {
    display: inline-block;
    background: #1a1208;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.12rem 0.5rem;
    margin: 0.12rem 0.15rem 0.12rem 0;
    font-size: 0.82rem;
    color: var(--text);
}

/* ── Location breadcrumb ── */
.loc-trail { font-size: 0.8rem; color: var(--muted); line-height: 1.7; }
.loc-current { color: var(--gold2); font-weight: 600; }

/* ── Change events ── */
.event-line {
    font-size: 0.88rem;
    color: #a0d0b0;
    line-height: 1.9;
    font-family: 'Crimson Pro', serif;
}

/* ── Death screen ── */
.death-screen {
    text-align: center;
    padding: 2rem;
    border: 2px solid #8b1a1a;
    border-radius: 8px;
    background: linear-gradient(135deg, #1a0505, #0d0303);
    margin: 1rem 0;
}
.death-title {
    font-family: 'Cinzel Decorative', serif;
    font-size: 2.5rem;
    color: #8b1a1a;
    text-shadow: 0 0 30px rgba(139,26,26,0.6);
    letter-spacing: 0.1em;
}

/* ── Stats grid ── */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    font-size: 0.82rem;
}
.stat-cell {
    background: #0f0c08;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.3rem 0.5rem;
    color: var(--muted);
}
.stat-cell b { color: var(--gold2); }

/* ── Tab styling ── */
[data-testid="stTab"] button {
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── Setup screen ── */
.setup-hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.setup-tagline {
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 1.2rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* ── Scrollable ── */
section.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
    max-width: 860px;
}

/* ── ML: Intent badge ── */
.intent-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: #0f0c08; border: 1px solid var(--border);
    border-radius: 20px; padding: 0.15rem 0.55rem;
    font-size: 0.72rem; color: var(--muted);
    font-family: "Crimson Pro", serif; letter-spacing: 0.04em;
    margin-right: 0.3rem; margin-bottom: 0.2rem;
}
.intent-badge.primary { border-color: var(--gold); color: var(--gold2); }

/* ── ML: Emotion sparkline ── */
.emo-row {
    display: flex; align-items: flex-end; gap: 2px;
    height: 28px; margin: 4px 0 6px;
}
.emo-bar { flex: 1; border-radius: 2px 2px 0 0; min-height: 2px; transition: height 0.3s; }

/* ── ML: Memory stats ── */
.ml-stat {
    display: inline-block; background: #0c0a07;
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.2rem 0.5rem; font-size: 0.75rem; color: var(--muted);
    margin: 0.15rem 0.2rem 0.15rem 0;
}
.ml-stat b { color: var(--gold2); }
.consolidation-notice {
    font-size: 0.78rem; color: #4a8c5c;
    border-left: 2px solid #2d6a3f; padding-left: 0.5rem;
    margin: 0.3rem 0; font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def hp_bar(hp, max_hp, height=8):
    pct = max(0, min(100, int(hp / max_hp * 100))) if max_hp else 0
    color = "#2d6a3f" if pct > 60 else ("#b5770d" if pct > 30 else "#8b1a1a")
    return (
        f"<div class='bar-wrap' style='height:{height}px'>"
        f"<div class='bar-fill' style='width:{pct}%;background:{color};height:{height}px'></div>"
        f"</div>"
    )


def rel_bar_html(score: int) -> str:
    """Centred relationship bar: left=hostile, right=ally."""
    pct = (score + 100) / 200 * 100   # map -100..100 → 0..100%
    color = "#2d6a3f" if score >= 0 else "#8b1a1a"
    # bar fills from centre (50%) outward
    if score >= 0:
        left = 50
        width = pct - 50
    else:
        left = pct
        width = 50 - pct
    return (
        f"<div class='rel-score-bar'>"
        f"<div class='rel-score-fill' style='left:{left}%;width:{width}%;background:{color}'></div>"
        f"</div>"
    )


# ── Setup screen ────────────────────────────────────────────────────────────

def show_setup():
    st.markdown("""
    <div class='setup-hero'>
        <div class='dm-title'>⚔️ AI Dungeon Master</div>
        <div class='setup-tagline'>A world that remembers. NPCs that evolve. Your legend, forged in darkness.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        player_name = st.text_input("🧙 Your Name", value="Adventurer", max_chars=30)
        campaign = st.text_input(
            "📖 Campaign Name", value="my_adventure", max_chars=30,
            help="Each campaign saves separately. Resume any time by using the same name."
        )
        st.markdown("<div style='font-size:0.8rem;color:#5a4a30;margin-top:-0.5rem'>"
                    "Saves to: ./campaigns/{name}/</div>", unsafe_allow_html=True)
    with col2:
        genre = st.selectbox("🌍 Setting", list(GENRE_OPENINGS.keys()), index=3)
        preview = GENRE_OPENINGS[genre][:160] + "…"
        st.markdown(
            f"<div style='background:#0f0c08;border:1px solid #2a1e10;border-radius:5px;"
            f"padding:0.8rem;font-size:0.92rem;color:#7a6a50;min-height:80px'>"
            f"<em>{preview}</em></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("⚔️  Begin Your Adventure", use_container_width=True):
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in campaign).lower()
            st.session_state._setup_done = True
            st.session_state._campaign = safe
            st.session_state._genre = genre
            st.session_state._player_name = player_name
            st.rerun()


# ── Gate ───────────────────────────────────────────────────────────────────
if "_setup_done" not in st.session_state:
    show_setup()
    st.stop()

init_memory(
    st,
    campaign=st.session_state._campaign,
    genre=st.session_state._genre,
    player_name=st.session_state._player_name,
)

char  = st.session_state.character
mood  = st.session_state.current_mood
m_emoji, m_accent, m_border = get_mood_style(mood)
combat = st.session_state.combat


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown(
        f"<div style='padding:0.6rem 0 0.2rem'>"
        f"<div style='font-family:Cinzel Decorative,serif;font-size:0.9rem;color:#c9a84c'>"
        f"⚔️ Dungeon Master</div>"
        f"<div style='font-size:0.75rem;color:#5a4a30'>{st.session_state._campaign} · {st.session_state._genre}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Mood
    st.markdown(
        f"<div class='mood-banner'>{m_emoji} Atmosphere: <b style='color:{m_accent}'>{mood.title()}</b></div>",
        unsafe_allow_html=True,
    )

    # ── Combat ──
    if combat["active"]:
        st.markdown(
            f"<div class='combat-panel'>"
            f"<div class='combat-turn-badge'>⚔ COMBAT · TURN {combat['turn']}</div>"
            + enemy_hp_bar_html(combat["enemy_hp"], combat["enemy_max_hp"], combat["enemy_name"])
            + f"</div>",
            unsafe_allow_html=True,
        )

    # ── Character ──
    st.markdown("<div class='sidebar-section-title'>🧙 Character</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1rem;color:#e8c87a;font-weight:600'>{char['name']}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.82rem;color:#7a6a50'>❤ HP: {char['hp']}/{char['max_hp']}</div>"
        + hp_bar(char["hp"], char["max_hp"]),
        unsafe_allow_html=True,
    )
    cg, cs = st.columns(2)
    cg.markdown(f"<div style='font-size:0.85rem'>🪙 <b style='color:#c9a84c'>{char['gold']}</b> gold</div>", unsafe_allow_html=True)
    cs.markdown(f"<div style='font-size:0.85rem'>🔄 <b style='color:#c9a84c'>{st.session_state.stats['turns']}</b> turns</div>", unsafe_allow_html=True)

    # ── Inventory ──
    st.markdown("<div class='sidebar-section-title'>🎒 Inventory</div>", unsafe_allow_html=True)
    if char["inventory"]:
        pills = "".join(f"<span class='inv-pill'>{i}</span>" for i in char["inventory"])
        st.markdown(f"<div style='line-height:2'>{pills}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#5a4a30;font-size:0.85rem'>Empty</span>", unsafe_allow_html=True)

    # ── Location ──
    st.markdown("<div class='sidebar-section-title'>🗺️ Trail</div>", unsafe_allow_html=True)
    locs = st.session_state.location_history
    crumbs = []
    for i, loc in enumerate(locs[-6:][::-1]):
        if i == 0:
            crumbs.append(f"<span class='loc-current'>📍 {loc}</span>")
        else:
            crumbs.append(f"<span>↑ {loc}</span>")
    st.markdown(f"<div class='loc-trail'>{'<br>'.join(crumbs)}</div>", unsafe_allow_html=True)

    # ── Relationships ──
    rels = st.session_state.relationships
    if rels:
        st.markdown("<div class='sidebar-section-title'>🤝 Relations</div>", unsafe_allow_html=True)
        for name, score in sorted(rels.items(), key=lambda x: -abs(x[1]))[:6]:
            emoji, label, color = get_tier(score)
            st.markdown(
                f"<div class='rel-chip'>"
                f"<span style='color:{color}'>{emoji} {name}</span>"
                f"<span style='font-size:0.75rem;color:#5a4a30'>{label}"
                + rel_bar_html(score)
                + f"</span></div>",
                unsafe_allow_html=True,
            )

    # ── Quests ──
    st.markdown("<div class='sidebar-section-title'>📜 Quests</div>", unsafe_allow_html=True)
    quests = get_quests(st.session_state.quest_retriever)
    if quests:
        for q in quests[-6:]:
            st.markdown(f"<div class='quest-item'>⚔ {q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#5a4a30;font-size:0.85rem'>No quests yet…</span>", unsafe_allow_html=True)

    # ── Dice ──
    st.markdown("<div class='sidebar-section-title'>🎲 Quick Dice</div>", unsafe_allow_html=True)
    dice_input = st.text_input("", key="sidebar_dice", placeholder="e.g. 1d20, 2d6+3", label_visibility="collapsed")
    if dice_input:
        results = parse_and_roll(dice_input)
        if results:
            for r in results:
                is_crit = (r.rolls[0] == int(dice_input.split("d")[1][:2]) if "d20" in dice_input else False)
                is_fumble = (r.rolls == [1] and "d20" in dice_input)
                cls = "dice-crit" if is_crit else ("dice-fumble" if is_fumble else "")
                label = " 🌟 CRITICAL!" if is_crit else (" 💀 FUMBLE!" if is_fumble else "")
                st.markdown(
                    f"<div class='dice-box'><span class='{cls}'>{r.display}{label}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<span style='color:#5a4a30;font-size:0.8rem'>No dice notation found.</span>", unsafe_allow_html=True)

    # ── Export ──
    st.markdown("<div class='sidebar-section-title'>📄 Export</div>", unsafe_allow_html=True)
    if st.button("Download Log", use_container_width=True):
        log = "\n\n---\n\n".join(
            f"**{'DM' if m['role']=='assistant' else char['name']}:**\n{m['content']}"
            for m in st.session_state.messages
        )
        header = (
            f"# {st.session_state._campaign} — Adventure Log\n\n"
            f"**Genre:** {st.session_state._genre}  \n"
            f"**Player:** {char['name']}  \n"
            f"**Turns:** {st.session_state.stats['turns']}  \n\n---\n\n"
        )
        st.download_button(
            "⬇ Save Markdown", header + log,
            f"{st.session_state._campaign}.md", "text/markdown",
            use_container_width=True,
        )

    # ── ML: Emotional Arc Sparkline ──
    st.markdown("<div class='sidebar-section-title'>🧠 Memory Engine</div>", unsafe_allow_html=True)
    emo_hist = get_emotion_history(st.session_state.emotion_state)
    if emo_hist:
        bars_html = "<div class='emo-row'>"
        for pt in emo_hist[-20:]:
            c = pt["compound"]
            h = int(abs(c) * 26) + 2
            col = "#2d6a3f" if c > 0.1 else ("#8b1a1a" if c < -0.1 else "#5a4a30")
            bars_html += f"<div class='emo-bar' style='height:{h}px;background:{col}' title='{pt["tone"]} (turn {pt["turn"]})' ></div>"
        bars_html += "</div>"
        last = emo_hist[-1]
        st.markdown(
            f"<div style='font-size:0.75rem;color:#7a6a50;margin-bottom:2px'>Emotional Arc — {last['tone']}</div>"
            + bars_html,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span style='color:#5a4a30;font-size:0.78rem'>Arc builds as you play…</span>", unsafe_allow_html=True)

    # ── ML: Intent display ──
    last_intent = st.session_state.get("last_intent")
    if last_intent:
        badges = ""
        for i, intent in enumerate(last_intent.top_k[:3]):
            cls = "intent-badge primary" if i == 0 else "intent-badge"
            badges += f"<span class='{cls}'>{intent}</span>"
        st.markdown(
            f"<div style='font-size:0.75rem;color:#5a4a30;margin-bottom:3px'>Last Intent</div>{badges}",
            unsafe_allow_html=True,
        )

    # ── ML: Memory / Graph stats ──
    mem_count = get_memory_count(st.session_state.retriever.vectorstore)
    g_stats = graph_stats(st.session_state.narrative_graph)
    st.markdown(
        f"<div style='margin-top:6px'>"
        f"<span class='ml-stat'>📚 <b>{mem_count}</b> memories</span>"
        f"<span class='ml-stat'>🕸 <b>{g_stats['nodes']}</b> entities</span>"
        f"<span class='ml-stat'>🔗 <b>{g_stats['edges']}</b> links</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Consolidation notice
    if st.session_state.get("_consolidation_ran"):
        info = st.session_state._consolidation_ran
        st.markdown(
            f"<div class='consolidation-notice'>✨ Memory consolidated: {info.get('reduced', 0)} redundant memories merged</div>",
            unsafe_allow_html=True,
        )
        st.session_state._consolidation_ran = None

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#5a4a30;line-height:1.8'>"
        "<b style='color:#7a6a50'>Commands</b><br>"
        "/q — quests &nbsp; /c — character<br>"
        "/lore — world journal &nbsp; /stats — stats<br>"
        "/help — all commands<br>"
        "<b style='color:#7a6a50'>Dice in action</b><br>"
        "Type 1d20 anywhere in your action"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────
# Header
st.markdown(
    f"<div class='dm-title'>⚔️ AI Dungeon Master</div>"
    f"<div class='dm-subtitle'>{st.session_state._genre} · {st.session_state._campaign} · "
    f"<span style='color:{m_accent}'>{m_emoji} {mood.title()}</span></div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Render history ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🎲"):
            st.markdown(
                f"<div class='dm-message-card' style='border-left-color:{m_accent}'>"
                f"{msg['content']}</div>",
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("user", avatar="🧙"):
            st.markdown(msg["content"])

# ── Toasts ────────────────────────────────────────────────────────────────
for q in st.session_state.get("new_quest_flash", []):
    st.toast(f"📜 Quest: {q}", icon="⚔️")
st.session_state.new_quest_flash = []

if st.session_state.get("new_lore_flash"):
    e = st.session_state.new_lore_flash
    icon = LORE_ICONS.get(e["type"], "📖")
    st.toast(f"{icon} Lore discovered: {e['name']}", icon="📖")
    st.session_state.new_lore_flash = None

for ev in st.session_state.get("relationship_events", []):
    st.toast(ev.replace("**", ""), icon="🤝")
st.session_state.relationship_events = []


# ── Commands ───────────────────────────────────────────────────────────────
COMMANDS = {
    "/q": "quests", "/c": "character", "/lore": "lore",
    "/stats": "stats", "/help": "help",
    "q": "quests",  # backwards compat
}

if user_text := st.chat_input(
    f"What does {char['name']} do? (/help for commands)"
):
    stripped = user_text.lower().strip()

    if stripped in COMMANDS:
        cmd = COMMANDS[stripped]

        if cmd == "quests":
            qs = get_quests(st.session_state.quest_retriever)
            body = "\n".join(f"- ⚔ {q}" for q in qs) if qs else "No quests yet. Venture forth!"
            st.info(f"**📜 Quests & Achievements**\n\n{body}")

        elif cmd == "character":
            inv = ", ".join(char["inventory"]) or "nothing"
            rels = st.session_state.relationships
            rel_block = "\n".join(
                f"  {get_tier(s)[0]} {n}: {get_tier(s)[1]}"
                for n, s in sorted(rels.items(), key=lambda x: -x[1])
            ) if rels else "  No established relationships."
            st.info(
                f"**🧙 {char['name']}**\n\n"
                f"❤️ HP: {char['hp']}/{char['max_hp']}\n\n"
                f"🪙 Gold: {char['gold']}\n\n"
                f"🎒 Inventory: {inv}\n\n"
                f"📍 Location: {st.session_state.location_history[-1]}\n\n"
                f"**Relationships:**\n{rel_block}"
            )

        elif cmd == "lore":
            entries = get_all_lore(st.session_state.lore_db)
            if entries:
                grouped: dict[str, list] = {}
                for e in entries:
                    grouped.setdefault(e["type"], []).append(e)
                parts = []
                for t, items in grouped.items():
                    icon = LORE_ICONS.get(t, "📖")
                    parts.append(f"**{icon} {t.title()}s**")
                    for item in items:
                        parts.append(f"- **{item['name']}**: {item['description']}")
                st.info("**📖 World Lore Journal**\n\n" + "\n".join(parts))
            else:
                st.info("📖 Your lore journal is empty. Explore to discover creatures, places, and factions.")

        elif cmd == "stats":
            st.info(f"**📊 Session Statistics**\n\n{format_stats(st.session_state.stats)}")

        elif cmd == "help":
            st.info(
                "**📖 Commands**\n\n"
                "- **/q** — Quests & achievements\n"
                "- **/c** — Character sheet + relationships\n"
                "- **/lore** — World lore journal\n"
                "- **/stats** — Session statistics\n"
                "- **/help** — This message\n\n"
                "**🎲 Dice** — Include in your action text:\n"
                "*I roll 1d20 to persuade the guard, 2d6+3 for damage*\n"
                "Supported: `1d20`, `2d6`, `d100`, `3d8+2`\n\n"
                "**Natural 20** → Critical! **Natural 1** → Fumble!"
            )
        st.stop()

    # ── Normal turn ────────────────────────────────────────────────────────

    # Dice
    dice_results = parse_and_roll(user_text)
    if dice_results:
        st.session_state.stats = increment(st.session_state.stats, "dice_rolled", len(dice_results))

    # User message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="🧙"):
        st.markdown(user_text)
        if dice_results:
            for r in dice_results:
                is_crit   = len(r.rolls) == 1 and r.rolls[0] == 20 and "d20" in user_text.lower()
                is_fumble = len(r.rolls) == 1 and r.rolls[0] == 1  and "d20" in user_text.lower()
                label = " 🌟 CRITICAL!" if is_crit else (" 💀 FUMBLE!" if is_fumble else "")
                cls   = "dice-crit" if is_crit else ("dice-fumble" if is_fumble else "")
                st.markdown(
                    f"<div class='dice-box'>🎲 <span class='{cls}'>{r.display}{label}</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Stream DM response ─────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🎲"):
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.short_term_memory)

        dice_ctx = ""
        if dice_results:
            dice_ctx = "\n[Dice results: " + "; ".join(
                f"{r.notation}={'CRITICAL HIT' if (len(r.rolls)==1 and r.rolls[0]==20) else 'FUMBLE' if (len(r.rolls)==1 and r.rolls[0]==1) else str(r.total+r.modifier)}"
                for r in dice_results
            ) + ". Let the result naturally drive success or failure.]"

        full_prompt = user_text + dice_ctx

        placeholder = st.empty()
        full_resp = ""
        # Pre-classify intent for sidebar display (fast, no API call)
        st.session_state.last_intent = retrieve_intent_only(user_text)

        known_npcs = list(st.session_state.relationships.keys())
        for chunk in gen_story_stream(
            full_prompt, hist,
            st.session_state.retriever,
            st.session_state.char_retriever,
            genre=st.session_state._genre,
            relationships=st.session_state.relationships,
            location_history=st.session_state.location_history,
            emotion_state=st.session_state.emotion_state,
            narrative_graph=st.session_state.narrative_graph,
            npc_names=known_npcs,
        ):
            full_resp += chunk
            placeholder.markdown(
                f"<div class='dm-message-card' style='border-left-color:{m_accent}'>"
                f"{full_resp}▌</div>",
                unsafe_allow_html=True,
            )
        placeholder.markdown(
            f"<div class='dm-message-card' style='border-left-color:{m_accent}'>"
            f"{full_resp}</div>",
            unsafe_allow_html=True,
        )

    # ── Post-turn updates ──────────────────────────────────────────────────

    st.session_state.messages.append({"role": "assistant", "content": full_resp})
    st.session_state.stats = increment(st.session_state.stats, "turns")
    turn_count = st.session_state.stats["turns"]

    # Short-term memory
    st.session_state.short_term_memory.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": full_resp},
    ])
    if len(st.session_state.short_term_memory) > 8:
        st.session_state.short_term_memory = st.session_state.short_term_memory[-8:]

    # ── ML post-turn: emotion arc + graph extraction + consolidation ──────
    updated_emo, updated_graph, consolidation_info = post_turn_updates(
        user_text=user_text,
        full_resp=full_resp,
        retriever=st.session_state.retriever,
        narrative_graph=st.session_state.narrative_graph,
        emotion_state=st.session_state.emotion_state,
        turn_count=turn_count,
        campaign=st.session_state._campaign,
    )
    st.session_state.emotion_state   = updated_emo
    st.session_state.narrative_graph = updated_graph
    if consolidation_info.get("ran"):
        st.session_state._consolidation_ran = consolidation_info

    # NPC memory + relationships
    turn_text = f"Player: {user_text}\nDM: {full_resp}"
    npc_name = find_npc(turn_text)
    if npc_name:
        npc_sum = sum_npc_turn(npc_name, user_text, full_resp)
        if npc_sum:
            add_char_mem(st.session_state.char_retriever, npc_name, npc_sum)

        # Relationship update
        updated_rels, rel_event = update_relationship(
            st.session_state.relationships, npc_name, user_text, full_resp
        )
        st.session_state.relationships = updated_rels
        if rel_event:
            st.session_state.relationship_events.append(rel_event)

    # Quest tracking
    new_quest = add_quest(st.session_state.quest_retriever, user_text, full_resp)
    if new_quest:
        st.session_state.new_quest_flash.append(new_quest)
        st.session_state.stats = increment(st.session_state.stats, "quests_completed")

    # Location tracking
    old_loc = st.session_state.location_history[-1]
    new_loc = detect_location(full_resp, old_loc)
    if new_loc != old_loc:
        st.session_state.location_history.append(new_loc)
        st.session_state.stats = increment(st.session_state.stats, "locations_visited")

    # Mood detection
    st.session_state.current_mood = detect_mood(full_resp)

    # Lore extraction
    lore_entry = extract_and_store_lore(st.session_state.lore_db, full_resp)
    if lore_entry:
        st.session_state.new_lore_flash = lore_entry

    # Combat tracking
    updated_combat, combat_events = update_combat(
        st.session_state.combat, user_text, full_resp
    )
    st.session_state.combat = updated_combat
    if combat_events:
        # Check for enemy defeated
        for ev in combat_events:
            if "Victory" in ev:
                st.session_state.stats = increment(st.session_state.stats, "enemies_defeated")
        # Show combat events inline
        events_html = "<br>".join(combat_events)
        with st.chat_message("assistant", avatar="⚔️"):
            st.markdown(f"<div class='event-line'>{events_html}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": "\n".join(combat_events)})

    # Character sheet update
    updated_sheet, sheet_changes = update_sheet(st.session_state.character, full_resp)
    st.session_state.character = updated_sheet
    if sheet_changes:
        for ch in sheet_changes:
            if "Gained" in ch:
                st.session_state.stats = increment(st.session_state.stats, "items_found")
        changes_html = "<br>".join(sheet_changes)
        with st.chat_message("assistant", avatar="📋"):
            st.markdown(f"<div class='event-line'>{changes_html}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": "\n".join(sheet_changes)})

    # ── Death check ────────────────────────────────────────────────────────
    if is_dead(st.session_state.character):
        updated_char, updated_stats, death_narrative = respawn(
            st.session_state.character, st.session_state.stats
        )
        st.session_state.character = updated_char
        st.session_state.stats = updated_stats
        with st.chat_message("assistant", avatar="💀"):
            st.markdown(
                f"<div class='death-screen'>"
                f"<div class='death-title'>💀 YOU HAVE FALLEN 💀</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(death_narrative)
        st.session_state.messages.append({"role": "assistant", "content": death_narrative})

    st.rerun()
