import os
import logging
from typing import Dict, Any, Optional

# Import for OpenAI
from langchain_openai import ChatOpenAI

# Imports for Hugging Face models (only loaded when needed)
try:
    from langchain_community.llms import HuggingFacePipeline
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class LanguageModel:
    """
    Wrapper for language models used in AgriSaarthi.
    """

    def __init__(self, model_type: str = "gpt", model_size: str = "gpt-3.5-turbo"):
        """
        Initialize the language model.

        Args:
            model_type: Type of model to use ("falcon", "flan-t5", or "gpt")
            model_size: Size of the model ("small", "base", or "large")
        """
        # Force model_type to "gpt" if Hugging Face is not available
        if model_type in ["falcon", "flan-t5"] and not HF_AVAILABLE:
            logging.warning(f"{model_type} model requested but Hugging Face libraries not available. Defaulting to 'gpt'.")
            model_type = "gpt"
            model_size = "gpt-3.5-turbo"

        self.model_type = model_type
        self.model_size = model_size
        self.model = None
        self.tokenizer = None
        self.llm = None

        # Load the appropriate model
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the specified language model.
        """
        try:
            # Check for GPU availability if torch is available
            if HF_AVAILABLE:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logging.info(f"Using device: {device}")
            else:
                device = "cpu"
                logging.info("Using CPU (torch not available)")

            if self.model_type == "falcon":
                # Check if Hugging Face libraries are available
                if not HF_AVAILABLE:
                    logging.error("Hugging Face libraries not available. Please install transformers, torch, and langchain-community.")
                    raise ImportError("Required libraries not installed. Run: pip install transformers torch langchain-community")

                # Use Falcon model (smaller and faster than GPT)
                if self.model_size == "small":
                    model_name = "tiiuae/falcon-7b-instruct"
                else:
                    model_name = "tiiuae/falcon-40b-instruct"

                logging.info(f"Loading Falcon model: {model_name}")

                try:
                    # Load in 8-bit to reduce memory usage
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        load_in_8bit=True,
                        trust_remote_code=True
                    )

                    # Create a text generation pipeline
                    text_generation_pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=40,
                        repetition_penalty=1.1
                    )

                    # Create LangChain wrapper
                    self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
                except Exception as e:
                    logging.error(f"Error loading Falcon model: {str(e)}")
                    raise ValueError(f"Failed to load Falcon model: {str(e)}. Consider using 'gpt' model type instead.")

            elif self.model_type == "flan-t5":
                # Check if Hugging Face libraries are available
                if not HF_AVAILABLE:
                    logging.error("Hugging Face libraries not available. Please install transformers, torch, and langchain-community.")
                    raise ImportError("Required libraries not installed. Run: pip install transformers torch langchain-community")

                # Use Flan-T5 model (good for instruction following)
                if self.model_size == "small":
                    model_name = "google/flan-t5-base"
                elif self.model_size == "base":
                    model_name = "google/flan-t5-large"
                else:
                    model_name = "google/flan-t5-xl"

                logging.info(f"Loading Flan-T5 model: {model_name}")

                try:
                    from transformers import T5Tokenizer, T5ForConditionalGeneration

                    self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                    )

                    # Create a text generation pipeline
                    text_generation_pipeline = pipeline(
                        "text2text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_tokens=512
                    )

                    # Create LangChain wrapper
                    self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
                except Exception as e:
                    logging.error(f"Error loading Flan-T5 model: {str(e)}")
                    raise ValueError(f"Failed to load Flan-T5 model: {str(e)}. Consider using 'gpt' model type instead.")

            elif self.model_type == "gpt":
                # Use OpenAI GPT model (requires API key)
                from langchain_openai import ChatOpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key or api_key == "your_openai_api_key_here":
                    logging.error("OpenAI API key not found in environment variables")
                    raise ValueError("OpenAI API key not found or not set. Please set the OPENAI_API_KEY environment variable in the .env file.")

                if self.model_size == "gpt-3.5-turbo":
                    model_name = "gpt-3.5-turbo"
                else:
                    model_name = "gpt-4"

                logging.info(f"Using OpenAI model: {model_name}")

                # Create ChatOpenAI instance
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=0.7,
                    api_key=api_key
                )

            else:
                logging.error(f"Unsupported model type: {self.model_type}")
                raise ValueError(f"Unsupported model type: {self.model_type}")

            logging.info(f"Successfully loaded {self.model_type} model")

        except Exception as e:
            logging.error(f"Error loading language model: {str(e)}")
            raise

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for a given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        if not self.llm:
            logging.error("Language model not initialized")
            return "I'm having trouble generating a response. Please try again later."

        try:
            # Handle different types of LLM responses
            if self.model_type == "gpt":
                # For ChatOpenAI, we need to use the invoke method and extract content
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            else:
                # For other LLMs that return strings directly
                response = self.llm(prompt)
                return response

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating a response. Please try again."

    def generate_agricultural_advice(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate agricultural advice for a specific query.

        Args:
            query: User query about agriculture
            context: Optional context from retrieved documents

        Returns:
            Generated agricultural advice
        """
        # Create a prompt for agricultural advice
        prompt = f"""
        You are AgriSaarthi, an AI assistant for farmers focused on sustainable agriculture.
        Provide helpful, practical advice for the following query:

        Query: {query}
        """

        if context:
            prompt += f"""

            Use the following information to inform your response:
            {context}
            """

        prompt += """

        Your response should:
        1. Be clear and easy to understand for farmers
        2. Provide practical, actionable steps
        3. Focus on sustainable and eco-friendly practices
        4. Consider the constraints of small-scale farming
        5. Include both traditional knowledge and modern techniques when appropriate

        Response:
        """

        return self.generate_response(prompt)

    def generate_disease_treatment(self, crop: str, disease: str, severity: str = "moderate") -> str:
        """
        Generate treatment recommendations for a crop disease.

        Args:
            crop: Type of crop
            disease: Identified disease
            severity: Severity of the disease ("mild", "moderate", or "severe")

        Returns:
            Treatment recommendations
        """
        # Create a prompt for disease treatment
        prompt = f"""
        You are AgriSaarthi, an AI assistant for farmers focused on sustainable agriculture.
        Provide a detailed treatment plan for the following crop disease:

        Crop: {crop}
        Disease: {disease}
        Severity: {severity}

        Your treatment plan should include:
        1. Immediate actions to contain the disease
        2. Step-by-step treatment methods
        3. Organic/natural treatment options
        4. Chemical treatment options (if necessary)
        5. Preventive measures for the future
        6. Signs of recovery to look for

        Format your response as a clear, actionable treatment plan that farmers can follow.

        Treatment Plan:
        """

        return self.generate_response(prompt)
