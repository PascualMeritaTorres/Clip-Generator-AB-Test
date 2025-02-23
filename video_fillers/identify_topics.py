from typing import Dict, List
import json
import os
import re
import fal_client
from dotenv import load_dotenv

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, removing punctuation,
    and standardizing whitespace.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace (convert multiple spaces to single space)
    text = ' '.join(text.split())
    return text

def identify_topics(transcription: str) -> Dict[str, List[str]]:
    """
    Use Fal AI's Claude 3.5 Sonnet to identify topics and their associated words in chronological order.
    The topics should split the transcription into sequential parts that can be reconstructed
    to form the original text when concatenated in order.
    
    Args:
        transcription (str): Full text transcription
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping topic names to word lists
        
    Raises:
        AssertionError: If LLM response is invalid
        RuntimeError: If multiple LLM attempts fail
    """
    base_prompt = f"""
    Split this text into sequential topics, where each topic represents an entity or object that is being talked about.
    The words in each topic must appear in the exact same order as in the original text.
    When concatenating all the words from all topics in order, it must reconstruct the original text exactly.

    Rules:
    1. Separate different words in the topic name by whitespaces
    2. The words in each topic must be taken sequentially from the text
    3. Every word from the original text must be included exactly once
    4. The order of topics must match the text flow
    5. No adding, removing, or modifying words

    Return ONLY a JSON where:
    - Each key is a topic name (e.g., "spies", "world war 2", "the bible, etc.)
    - Each value is a list of the exact words from that section of text

    Example:
    Text: "What is the one spy trick you would teach everyone"
    Output: {{
        "spies": ["What", "is", "the", "one", "spy", "trick", "you", "would", "teach", "everyone"],
        "surroundings": ["I", "would", "teach", "everyone", "to", "always", "look", "at", "their", "surroundings"]
    }}

    Text: {transcription}
    """
    
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"[DEBUG] identify_topics: Attempt {attempt + 1}")
        try:
            result = fal_client.subscribe(
                "fal-ai/any-llm",
                arguments={
                    "model": "anthropic/claude-3.5-sonnet",
                    "prompt": base_prompt + "\n\nRemember: Output only valid JSON." if attempt > 0 else base_prompt
                }
            )
            
            # Parse the output from the result
            result_json = json.loads(result["output"])
            assert isinstance(result_json, dict), "LLM response must be a dictionary"
            
            # Verify that concatenating all words reconstructs the original text
            all_words = []
            for words in result_json.values():
                all_words.extend(words)
            reconstructed_text = " ".join(all_words)
            
            # Normalize both texts for comparison
            normalized_original = normalize_text(transcription)
            normalized_reconstructed = normalize_text(reconstructed_text)
            
            # Compare normalized versions
            assert normalized_reconstructed == normalized_original, \
                f"Topics don't reconstruct original text exactly.\nOriginal (normalized): {normalized_original}\nReconstructed (normalized): {normalized_reconstructed}"
            
            print(f"[DEBUG] identify_topics: Topics identified: {list(result_json.keys())}")
            return result_json
            
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"[DEBUG] identify_topics: Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed to get valid JSON response after {max_attempts} attempts") from e
            continue

def read_transcription(jsonl_path: str) -> str:
    """
    Read transcription from a JSONL file.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        
    Returns:
        str: Transcription text
        
    Raises:
        AssertionError: If file doesn't exist or isn't a JSONL file
    """
    print(f"[DEBUG] read_transcription: Reading transcription from file: {jsonl_path}")
    assert os.path.exists(jsonl_path), f"Transcription file not found: {jsonl_path}"
    assert jsonl_path.endswith('.jsonl'), "Transcription file must be JSONL format"
    
    with open(jsonl_path, 'r') as f:
        data = json.load(f)
    assert isinstance(data, dict) and 'text' in data, "Invalid JSONL format"
    print("[DEBUG] read_transcription: Finished reading transcription.")
    return data['text']

def main():
    # Load environment variables
    load_dotenv()
    
    # Read transcription
    transcription = read_transcription("audio_transcription.jsonl")
    print("[DEBUG] Main: Transcription loaded successfully.")
    
    # Identify topics
    topics = identify_topics(transcription)
    
    # Save topics to file
    with open('identified_topics.json', 'w') as f:
        json.dump(topics, f, indent=2)
    print("[DEBUG] Main: Topics saved to identified_topics.json")
    
    # Print topics for verification
    print("\nIdentified Topics:")
    for topic, words in topics.items():
        print(f"\n{topic}:")
        print(" ".join(words[:10]) + "..." if len(words) > 10 else " ".join(words))

if __name__ == "__main__":
    main() 