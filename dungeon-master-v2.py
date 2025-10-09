import os
import groq
from dotenv import load_dotenv
from collections import deque
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- CONFIGURATION ---
SHORT_TERM_MEMORY_TURNS = 5
LONG_TERM_MEMORY_TURNS = 30
SUMMARY_TRIGGER_COUNT = 3 # Summarize every 3 turns

class MemoryManager:
    """Manages the AI's short-term and long-term memory."""

    def __init__(self, llm_client):
        print("Initializing Memory Manager...")
        # Short-term memory for recent events (~5 turns) 
        self.short_term_memory = deque(maxlen=SHORT_TERM_MEMORY_TURNS)
        
        # Long-term memory for crucial past events (~30 turns) [cite: 21]
        self.long_term_memory = []
        self.turn_history_for_summary = []
        
        # RAG component: embedding model for semantic search 
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_client = llm_client
        print("Memory Manager initialized.")

    def add_turn(self, player_input, ai_response):
        """Adds a turn to both short-term and long-term memory trackers."""
        turn_text = f"Player: {player_input}\nDM: {ai_response}"
        self.short_term_memory.append(turn_text)
        self.turn_history_for_summary.append(turn_text)

        # After a few turns, create a summary for long-term memory
        if len(self.turn_history_for_summary) >= SUMMARY_TRIGGER_COUNT:
            self.summarize_and_store()

    def summarize_and_store(self):
        """Uses the LLM to summarize recent events and stores it in long-term memory."""
        print("\n--- Creating a long-term memory summary... ---")
        text_to_summarize = "\n".join(self.turn_history_for_summary)
        
        summary_prompt = (
            "Summarize the following key events from this part of the adventure "
            "into 1-2 concise sentences. Focus on choices, consequences, new characters, "
            "and significant discoveries. Example: 'The heroes defeated the goblin scout "
            "and found a mysterious map in its pouch.'\n\n"
            f"Adventure Log:\n{text_to_summarize}\n\nSummary:"
        )
        
        summary = self.llm_client.generate(summary_prompt, max_tokens=100)
        
        if summary:
            print(f"New Memory: {summary}")
            embedding = self.embedding_model.encode(summary, convert_to_tensor=False)
            self.long_term_memory.append({"summary": summary, "embedding": embedding})
            self.turn_history_for_summary.clear() # Reset for the next batch
        print("--- Summary creation complete. ---\n")

    def get_short_term_context(self):
        """Returns a string of recent events."""
        return "\n".join(self.short_term_memory)

    def get_long_term_context(self, current_input, top_k=3):
        """Retrieves the most relevant long-term memories using semantic search."""
        if not self.long_term_memory:
            return ""

        input_embedding = self.embedding_model.encode(current_input, convert_to_tensor=False)
        
        # Get all stored embeddings and calculate cosine similarity
        stored_embeddings = np.array([item["embedding"] for item in self.long_term_memory])
        similarities = util.cos_sim(input_embedding, stored_embeddings)[0]
        
        # Get the indices of the top_k most similar summaries
        top_indices = np.argsort(similarities)[-top_k:]
        
        relevant_memories = [self.long_term_memory[i]["summary"] for i in top_indices if similarities[i] > 0.4]
        
        return "\n".join(reversed(relevant_memories)) # Most relevant first

class LLMClient:
    """A wrapper for the Groq API client."""
    def __init__(self):
        load_dotenv()
        self.client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        print("LLM Client initialized with model:", self.model)

    def generate(self, prompt, max_tokens=250):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error communicating with LLM: {e}")
            return "The Dungeon Master seems to be pondering deeply and is silent for a moment."

class PromptEngine:
    """Constructs the final prompt for the LLM."""
    def __init__(self):
        # This system prompt guides the AI to act as a Dungeon Master [cite: 16]
        self.system_prompt = (
            "You are an expert Dungeon Master for a text-based tabletop role-playing game. "
            "Your role is to weave a compelling and coherent story. "
            "Describe the world, narrate the outcomes of the player's actions, and play the role of non-player characters (NPCs). "
            "Keep your responses descriptive but concise (2-4 sentences). "
            "You must use the provided memories to ensure the world is consistent and player choices have lasting consequences. "
            "Never break character. Address the player directly as 'you'."
        )

    def create_prompt(self, short_term_context, long_term_context, player_input):
        return (
            f"{self.system_prompt}\n\n"
            f"--- RECENT EVENTS (Short-Term Memory) ---\n{short_term_context}\n\n"
            f"--- KEY PAST EVENTS (Long-Term Memory) ---\n{long_term_context}\n\n"
            f"--- CURRENT SITUATION ---\n"
            f"Player: {player_input}\n"
            f"DM:"
        )

def main():
    """The main game loop."""
    print("Starting the AI Dungeon Master...")
    llm_client = LLMClient()
    memory = MemoryManager(llm_client)
    prompt_engine = PromptEngine()
    
    # Starting scenario
    print("\n--- Welcome to the Adventure! ---")
    initial_prompt = "You stand at the entrance of the Whispering Caves, a place rumored to hold an ancient artifact. A cold breeze carrying the scent of damp earth and something... reptilian... blows from the dark opening. What do you do?"
    print("DM:", initial_prompt)
    
    turn_count = 0
    while turn_count < LONG_TERM_MEMORY_TURNS:
        player_input = input("\n> ")
        if player_input.lower() in ["quit", "exit"]:
            print("Ending the adventure. Farewell!")
            break
            
        short_term_context = memory.get_short_term_context()
        long_term_context = memory.get_long_term_context(player_input)
        
        final_prompt = prompt_engine.create_prompt(short_term_context, long_term_context, player_input)
        
        ai_response = llm_client.generate(final_prompt)
        print("\nDM:", ai_response)
        
        memory.add_turn(player_input, ai_response)
        turn_count += 1
        print(f"--- Turn {turn_count}/{LONG_TERM_MEMORY_TURNS} ---")

    print(f"\nGame finished after {turn_count} turns. The application remained stable.") # 

if __name__ == "__main__":
    main()