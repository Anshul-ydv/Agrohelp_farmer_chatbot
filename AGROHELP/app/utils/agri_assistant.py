import logging
from typing import Dict, Any, List, Optional

from app.models.language_model import LanguageModel
from app.models.vision_model import VisionModel
from app.models.translation_model import TranslationModel
from app.services.retrieval_service import RetrievalService
from app.utils.helpers import parse_disease_severity, format_recommendations

class AgriAssistant:
    """
    Main assistant class that orchestrates all components of AgriSaarthi.
    """
    
    def __init__(
        self,
        llm_type: str = "falcon",
        llm_size: str = "small",
        vision_model_type: str = "resnet",
        translation_model: str = "nllb",
        vector_store: Any = None
    ):
        """
        Initialize the AgriSaarthi assistant.
        
        Args:
            llm_type: Type of language model to use
            llm_size: Size of the language model
            vision_model_type: Type of vision model to use
            translation_model: Type of translation model to use
            vector_store: Vector store for document retrieval
        """
        logging.info("Initializing AgriSaarthi assistant")
        
        # Initialize language model
        self.language_model = LanguageModel(model_type=llm_type, model_size=llm_size)
        logging.info(f"Initialized language model: {llm_type} ({llm_size})")
        
        # Initialize vision model
        self.vision_model = VisionModel(model_type=vision_model_type)
        logging.info(f"Initialized vision model: {vision_model_type}")
        
        # Initialize translation model
        self.translation_model = TranslationModel(model_type=translation_model)
        logging.info(f"Initialized translation model: {translation_model}")
        
        # Initialize retrieval service
        self.retrieval_service = RetrievalService(llm=self.language_model.llm, vector_store=vector_store)
        logging.info("Initialized retrieval service")
        
        # Set vector store if provided
        self.vector_store = vector_store
        if vector_store:
            self.retrieval_service.set_vector_store(vector_store)
            logging.info("Vector store set in retrieval service")
    
    def set_vector_store(self, vector_store: Any) -> None:
        """
        Set or update the vector store.
        
        Args:
            vector_store: Vector store to use for retrieval
        """
        self.vector_store = vector_store
        self.retrieval_service.set_vector_store(vector_store)
        logging.info("Updated vector store in assistant and retrieval service")
    
    def process_query(self, query: str, target_lang: str = "English") -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query
            target_lang: Target language for the response
            
        Returns:
            Response in the target language
        """
        logging.info(f"Processing query: {query}")
        
        try:
            # Translate query to English if needed
            if target_lang != "English":
                source_lang = self.translation_model.detect_language(query)
                if source_lang != "English":
                    english_query = self.translation_model.translate(query, source_lang, "English")
                    logging.info(f"Translated query from {source_lang} to English")
                else:
                    english_query = query
            else:
                english_query = query
            
            # Check if we have a vector store for retrieval
            if self.vector_store:
                # Use retrieval-based QA
                retrieval_result = self.retrieval_service.retrieve_answer(english_query)
                response = retrieval_result["answer"]
                logging.info("Generated response using retrieval-based QA")
            else:
                # Use direct LLM generation
                response = self.language_model.generate_agricultural_advice(english_query)
                logging.info("Generated response using direct LLM generation")
            
            # Translate response to target language if needed
            if target_lang != "English":
                response = self.translation_model.translate(response, "English", target_lang)
                logging.info(f"Translated response to {target_lang}")
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            error_msg = f"I'm sorry, I encountered an error while processing your query. Please try again."
            
            # Translate error message if needed
            if target_lang != "English":
                error_msg = self.translation_model.translate(error_msg, "English", target_lang)
            
            return error_msg
    
    def process_image(self, image_path: str, target_lang: str = "English") -> Dict[str, Any]:
        """
        Process an image for crop disease detection and generate recommendations.
        
        Args:
            image_path: Path to the image file
            target_lang: Target language for the response
            
        Returns:
            Dictionary containing detection results and recommendations
        """
        logging.info(f"Processing image: {image_path}")
        
        try:
            # Detect crop disease
            detection_result = self.vision_model.predict(image_path)
            
            # Check for errors
            if "error" in detection_result:
                return detection_result
            
            # Extract information
            crop = detection_result["crop"]
            condition = detection_result["condition"]
            is_healthy = detection_result["is_healthy"]
            confidence = detection_result["confidence"]
            
            # Generate recommendations if the crop is diseased
            if not is_healthy:
                # Parse severity based on confidence
                severity = parse_disease_severity(confidence)
                
                # Generate treatment recommendations
                recommendations = self.language_model.generate_disease_treatment(
                    crop=crop,
                    disease=condition,
                    severity=severity
                )
                
                # Translate recommendations if needed
                if target_lang != "English":
                    recommendations = self.translation_model.translate(
                        recommendations, "English", target_lang
                    )
            else:
                # Generate healthy crop recommendations
                recommendations = self.language_model.generate_agricultural_advice(
                    f"What are the best practices for maintaining a healthy {crop} crop?"
                )
                
                # Translate recommendations if needed
                if target_lang != "English":
                    recommendations = self.translation_model.translate(
                        recommendations, "English", target_lang
                    )
            
            # Create result dictionary
            result = {
                "crop": crop,
                "condition": condition,
                "is_healthy": is_healthy,
                "confidence": confidence,
                "recommendations": recommendations
            }
            
            logging.info(f"Successfully processed image: {crop} with {condition}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            error_msg = f"I'm sorry, I encountered an error while processing your image. Please try again."
            
            # Translate error message if needed
            if target_lang != "English":
                error_msg = self.translation_model.translate(error_msg, "English", target_lang)
            
            return {
                "error": str(e),
                "crop": "Unknown",
                "condition": "Unknown",
                "is_healthy": False,
                "confidence": 0.0,
                "recommendations": error_msg
            }
    
    def generate_crop_care_plan(self, crop: str, region: str = None, season: str = None, target_lang: str = "English") -> str:
        """
        Generate a customized crop care plan.
        
        Args:
            crop: Type of crop
            region: Growing region (optional)
            season: Growing season (optional)
            target_lang: Target language for the response
            
        Returns:
            Crop care plan in the target language
        """
        logging.info(f"Generating crop care plan for {crop}")
        
        try:
            # Create prompt for crop care plan
            prompt = f"Generate a comprehensive care plan for growing {crop}"
            
            if region:
                prompt += f" in {region}"
            
            if season:
                prompt += f" during {season} season"
            
            prompt += ". Include soil preparation, planting, irrigation, fertilization, pest management, and harvesting guidelines. Focus on sustainable and organic practices."
            
            # Generate crop care plan
            care_plan = self.language_model.generate_response(prompt)
            
            # Translate care plan if needed
            if target_lang != "English":
                care_plan = self.translation_model.translate(care_plan, "English", target_lang)
                logging.info(f"Translated crop care plan to {target_lang}")
            
            return care_plan
            
        except Exception as e:
            logging.error(f"Error generating crop care plan: {str(e)}")
            error_msg = f"I'm sorry, I encountered an error while generating the crop care plan. Please try again."
            
            # Translate error message if needed
            if target_lang != "English":
                error_msg = self.translation_model.translate(error_msg, "English", target_lang)
            
            return error_msg
