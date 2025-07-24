import os
import logging
from dotenv import load_dotenv
from app.models.language_model import LanguageModel

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test the OpenAI integration."""
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("OpenAI API key not set. Please update the .env file with your API key.")
        return
    
    try:
        # Initialize language model
        print("Initializing OpenAI language model...")
        model = LanguageModel(model_type="gpt", model_size="gpt-3.5-turbo")
        
        # Generate a response
        prompt = "What are sustainable farming practices for rice cultivation?"
        print(f"\nPrompt: {prompt}")
        
        print("\nGenerating response...")
        response = model.generate_agricultural_advice(prompt)
        
        print("\nResponse:")
        print(response)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check your OpenAI API key and internet connection.")

if __name__ == "__main__":
    test_openai_integration()
