"""
narrative_graph.py — In-memory world knowledge graph.

The problem with flat memory:
  All memories are stored as independent text strings. The system knows
  "Player met Zara" and "Zara works for the Thieves' Guild" but doesn't
  know to connect these — so "Tell me about the Thieves' Guild" won't
  surface Zara as relevant even though she's the connection.

Solution — a lightweight entity-relationship graph:
  We maintain a directed graph where nodes are named entities (NPCs,
  locations, factions, items) and edges are relationships ("works_for",
  "located_in", "owns", "enemy_of", "ally_of", etc.).

  When building retrieval context, we:
    1. Detect entities in the player's query
    2. Walk 1-2 hops in the graph from those entities
    3. Pull the memories associated with neighbouring nodes too

  This gives the DM awareness of indirect connections — the world feels
  like it has STRUCTURE, not just a flat list of events.

Implementation:
  Pure Python dict-based adjacency list (no external graph library).
  Persisted to JSON alongside the campaign. LLM extracts edges each turn.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    source:     str    # Entity name
    relation:   str    # Relationship type
    target:     str    # Entity name
    confidence: float = 1.0


@dataclass
class NarrativeGraph:
    # adjacency: source → list of (relation, target, confidence)
    adjacency:  Dict[str, List[Tuple[str, str, float]]] = field(default_factory=dict)
    # entity types: name → type label
    entity_types: Dict[str, str] = field(default_factory=dict)


# ── LLM extraction prompt ────────────────────────────────────────────────────

_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "Extract named entity relationships from the DM's response.\n"
    "Return ONLY a JSON array (possibly empty []) of objects:\n"
    "  {{\"source\": \"EntityA\", \"relation\": \"relation_type\", \"target\": \"EntityB\", "
    "\"source_type\": \"npc|place|faction|item\", \"target_type\": \"npc|place|faction|item\"}}\n\n"
    "Valid relation types: works_for, ally_of, enemy_of, located_in, owns, guards, "
    "member_of, leads, fears, seeks, created_by, destroyed_by, related_to\n\n"
    "Only extract clear, named relationships. No pronouns. Max 3 per turn.\n"
    "DM Response: {dm_response}\n\nJSON array only:"
)
_extract_chain = _EXTRACT_PROMPT | llm | StrOutputParser()


# ── Persistence ─────────────────────────────────────────────────────────────

def _graph_path(campaign: str) -> str:
    os.makedirs(f"./campaigns/{campaign}", exist_ok=True)
    return f"./campaigns/{campaign}/narrative_graph.json"


def load_graph(campaign: str) -> NarrativeGraph:
    """Load the graph from disk, or create a fresh one."""
    path = _graph_path(campaign)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            g = NarrativeGraph()
            g.adjacency     = data.get("adjacency", {})
            g.entity_types  = data.get("entity_types", {})
            return g
        except Exception:
            pass
    return NarrativeGraph()


def save_graph(graph: NarrativeGraph, campaign: str):
    """Persist the graph to disk."""
    path = _graph_path(campaign)
    try:
        with open(path, "w") as f:
            json.dump({
                "adjacency":    graph.adjacency,
                "entity_types": graph.entity_types,
            }, f, indent=2)
    except Exception:
        pass


# ── Graph operations ─────────────────────────────────────────────────────────

def add_edge(graph: NarrativeGraph, edge: GraphEdge,
             src_type: str = "unknown", tgt_type: str = "unknown"):
    """Add a directed edge to the graph (idempotent on same src/rel/tgt)."""
    src, rel, tgt = edge.source.strip(), edge.relation.strip(), edge.target.strip()
    if not src or not tgt or src == tgt:
        return

    graph.entity_types.setdefault(src, src_type)
    graph.entity_types.setdefault(tgt, tgt_type)

    neighbours = graph.adjacency.setdefault(src, [])
    # Avoid duplicates
    for existing_rel, existing_tgt, _ in neighbours:
        if existing_rel == rel and existing_tgt == tgt:
            return
    neighbours.append((rel, tgt, edge.confidence))

    # Also add reverse (undirected convenience for retrieval)
    rev = graph.adjacency.setdefault(tgt, [])
    for er, et, _ in rev:
        if er == f"inv_{rel}" and et == src:
            return
    rev.append((f"inv_{rel}", src, edge.confidence))


def neighbours_of(graph: NarrativeGraph, entity: str, hops: int = 2) -> Set[str]:
    """
    BFS walk from entity up to `hops` hops.
    Returns the set of all reachable entity names (not including the start).
    """
    visited: Set[str] = set()
    frontier = {entity}
    for _ in range(hops):
        next_frontier: Set[str] = set()
        for node in frontier:
            for rel, tgt, _ in graph.adjacency.get(node, []):
                if tgt not in visited and tgt != entity:
                    next_frontier.add(tgt)
        visited.update(next_frontier)
        frontier = next_frontier
    return visited


def graph_context_for_query(
    graph: NarrativeGraph,
    query: str,
    max_entities: int = 5,
) -> str:
    """
    Detect entities mentioned in the query, walk the graph,
    and return a brief context string of relevant relationships.
    """
    if not graph.adjacency:
        return ""

    # Simple entity detection: capitalised words / known entities
    query_tokens = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", query))
    known = set(graph.adjacency.keys())

    # Match query tokens to known entities (case-insensitive)
    matched = {e for e in known if any(e.lower() in qt.lower() or qt.lower() in e.lower()
                                       for qt in query_tokens)}
    if not matched:
        return ""

    # Gather all neighbours within 2 hops
    all_relevant: Set[str] = set(matched)
    for entity in matched:
        all_relevant.update(neighbours_of(graph, entity, hops=2))

    all_relevant = set(list(all_relevant)[:max_entities])

    # Build human-readable relationship lines
    lines = []
    for src in all_relevant:
        for rel, tgt, _ in graph.adjacency.get(src, []):
            if rel.startswith("inv_"):
                continue   # Skip reverse edges for readability
            if tgt in all_relevant or src in matched:
                lines.append(f"  {src} —[{rel}]→ {tgt}")
        if len(lines) >= 10:
            break

    if not lines:
        return ""
    return "World Connections:\n" + "\n".join(lines[:8])


# ── Turn extraction ──────────────────────────────────────────────────────────

def extract_and_update_graph(graph: NarrativeGraph, dm_response: str, campaign: str):
    """
    Extract entity relationships from the latest DM response and add them to the graph.
    Persists the updated graph to disk.
    """
    try:
        raw = _extract_chain.invoke({"dm_response": dm_response}).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        items = json.loads(raw)
        if not isinstance(items, list):
            return

        for item in items[:3]:   # Max 3 edges per turn
            src  = str(item.get("source", "")).strip()
            rel  = str(item.get("relation", "related_to")).strip()
            tgt  = str(item.get("target", "")).strip()
            stype = str(item.get("source_type", "unknown"))
            ttype = str(item.get("target_type", "unknown"))

            if src and tgt and len(src) < 40 and len(tgt) < 40:
                add_edge(graph, GraphEdge(source=src, relation=rel, target=tgt),
                         src_type=stype, tgt_type=ttype)

        save_graph(graph, campaign)

    except Exception:
        pass


def graph_stats(graph: NarrativeGraph) -> dict:
    """Return basic graph statistics."""
    nodes = len(graph.adjacency)
    edges = sum(len(v) for v in graph.adjacency.values()) // 2  # Undirected
    return {"nodes": nodes, "edges": edges}
