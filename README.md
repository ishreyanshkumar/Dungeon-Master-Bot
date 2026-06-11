# ⚔️ AI Dungeon Master — Ultimate Edition

An interactive text-based RPG powered by Groq LLaMA with a deep, living world engine and advanced cognitive architecture.

![AI Dungeon Master](https://img.shields.io/badge/Powered_by-Groq_LLaMA-c9a84c?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-2d6a3f?style=for-the-badge)

---

## ✨ Features

### 🧠 Advanced Cognitive Architecture (New in Ultimate Edition)
The core of the Dungeon Master is its state-of-the-art cognitive architecture that ensures a dynamic and evolving campaign.
| Feature | Details |
|---|---|
| **Narrative Knowledge Graph** | Extracts and maintains an intricate graph of entities (characters, items, locations) and their relationships. |
| **Emotion Tracker** | Performs sentiment analysis on player and DM interactions, generating a living emotional arc. |
| **Intent Classifier** | Real-time Zero-Shot classification of player intent (Combat, Exploration, Social, etc.) to tailor DM responses. |
| **Memory Consolidator** | Periodically merges and compresses redundant memories, optimizing context windows and retaining crucial lore. |
| **Memory Ranker** | Ranks and retrieves the most contextually relevant memories using vector similarity heuristics. |

### 🌍 World Systems
| Feature | Details |
|---|---|
| **Streaming Narrative** | The Dungeon Master's text streams word-by-word in real time for immersion. |
| **Mood / Atmosphere** | LLM classifies scene tone (tense/wondrous/dreadful…) — UI accent colors adapt dynamically. |
| **Location Tracking** | Generates a persistent breadcrumb trail of every discovered location. |
| **World Lore Journal** | Auto-extracts creatures, places, factions, and items into an evolving `/lore` encyclopedia. |
| **Dynamic Combat Tracker** | Automatically detects combat states, manages turn order, and tracks enemy HP bars live. |
| **NPC Relationships** | Tracks trust/hostility scores per NPC, announcing tier changes as your actions impact them. |

### 🦸 Player Systems
| Feature | Details |
|---|---|
| **Dynamic Character Sheet** | HP, gold, and inventory are automatically updated by parsing the DM's narrative narrative. |
| **Integrated Dice Engine** | Natural language parsing of dice rolls (e.g., `1d20`, `2d6+3`) within action text, featuring critical hit and fumble detection. |
| **Death & Respawn** | Dropping to 0 HP triggers a unique death screen, enforces a gold penalty, and handles resurrection mechanics. |
| **Comprehensive Statistics** | Tracks campaign turns, enemies defeated, items looted, locations discovered, quests completed, and deaths. |
| **Adventure Log Export** | Download your entire epic saga as a beautifully formatted Markdown log. |

---

## 🚀 Setup & Installation

Ensure you have Python 3.10+ installed on your system.

```bash
# Clone the repository
git clone https://github.com/miniCoder6/dungeon-master-bot.git
cd dungeon-master-bot

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your Groq API Key
echo 'GROQ_API_KEY="your-key-here"' > .env

# Launch the Ultimate Edition
streamlit run app.py
```

*Get a free Groq key at [console.groq.com](https://console.groq.com).*

---

## 🎮 In-Game Commands

You can seamlessly integrate these commands in the chat bar:

| Command | Action |
|---|---|
| `/q` | View active quests & unlocked achievements |
| `/c` | Display your full character sheet and NPC relationship standings |
| `/lore` | Open the world lore journal and encyclopedia |
| `/stats` | View lifetime session statistics |
| `/help` | List all available commands |
| `1d20`, `2d6+3` | Include dice notations directly in your action prompt (e.g., "I attack the goblin `1d20`"). |

---

## 📁 Repository Structure

```text
├── app.py                   ← Main Streamlit UI frontend
├── backend.py               ← Story generation and memory orchestration logic
├── config.py                ← Shared LLM and Embeddings singletons
├── Cognitive Architecture:
│   ├── emotion_tracker.py   ← VADER-based emotional arc tracking
│   ├── intent_classifier.py ← Zero-Shot NLP player intent classification
│   ├── memory_consolidator.py ← Redundant memory merging and cleanup
│   ├── memory_ranker.py     ← Contextual retrieval ranking
│   └── narrative_graph.py   ← Entity relationship graph construction
├── Game Systems:
│   ├── character_memory.py  ← Per-NPC metadata-filtered ChromaDB stores
│   ├── character_sheet.py   ← Narrative-driven HP/Gold/Inventory tracker
│   ├── combat.py            ← Turn-based combat state manager
│   ├── dice.py              ← Regex-based NdM+Modifier parser
│   ├── location.py          ← Real-time location change detection
│   ├── lore.py              ← World encyclopedia auto-builder
│   ├── mood.py              ← Scene atmosphere and aesthetic classifier
│   ├── quest.py             ← Quest and achievement detection
│   ├── relationships.py     ← NPC trust and hostility tracker
│   └── stats.py             ← Comprehensive session and mortality stats
└── requirements.txt         ← Project dependencies
```

## 🛠 Tech Stack

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **Large Language Model:** Groq LLaMA 3.1 8B (via [Groq](https://groq.com/))
- **Orchestration:** [LangChain LCEL](https://python.langchain.com/) (LangChain Expression Language)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) for long-term semantic memory
- **Embeddings:** [HuggingFace Embeddings](https://huggingface.co/) (Sentence Transformers)
- **Natural Language Processing:** `scikit-learn`, `vaderSentiment`
- **Language:** Python 3.10+
