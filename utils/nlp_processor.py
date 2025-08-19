import spacy
from typing import Dict, Any

# Load the small English model
# This block handles the download if the model isn't already present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def process_text_with_nlp(text: str) -> Dict[str, Any]:
    """
    Processes a raw text string with spaCy to get linguistic annotations.
    
    Returns a dictionary with entities and POS tags.
    """
    try:
        doc = nlp(text)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return {
            "entities": entities,
            "pos_tags": pos_tags
        }
    except Exception as e:
        print(f"Error processing text with spaCy: {str(e)}")
        return {
            "entities": [],
            "pos_tags": []
        }

if __name__ == '__main__':
    # Example usage for testing the function
    sample_text = "Apple Inc. is headquartered in Cupertino, California. It was founded by Steve Jobs in 1976."
    result = process_text_with_nlp(sample_text)
    print("Entities:")
    print(result["entities"])
    print("\nPOS Tags:")
    print(result["pos_tags"])
