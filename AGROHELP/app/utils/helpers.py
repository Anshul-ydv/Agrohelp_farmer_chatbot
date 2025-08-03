import os
import logging
import tempfile
from typing import Dict, Any, List, Optional
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.getLogger().setLevel(numeric_level)
    logging.info(f"Logging level set to {log_level}")

def save_uploaded_file(uploaded_file: Any, directory: str = "temp") -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Uploaded file object
        directory: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, dir=directory, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        # Write the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    logging.info(f"Saved uploaded file to {tmp_path}")
    
    return tmp_path

def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up a temporary file.
    
    Args:
        file_path: Path to the file to clean up
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logging.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary file {file_path}: {str(e)}")

def image_to_base64(image_path: str) -> str:
    """
    Convert an image to a base64-encoded string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded string
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return ""

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert a base64-encoded string to an image.
    
    Args:
        base64_string: Base64-encoded string
        
    Returns:
        PIL Image object
    """
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logging.error(f"Error converting base64 to image: {str(e)}")
        return None

def format_recommendations(recommendations: Dict[str, Any]) -> str:
    """
    Format recommendations for display.
    
    Args:
        recommendations: Dictionary of recommendations
        
    Returns:
        Formatted recommendations string
    """
    if not recommendations:
        return "No recommendations available."
    
    formatted_text = ""
    
    if "steps" in recommendations:
        formatted_text += "## Step-by-Step Guide\n\n"
        for i, step in enumerate(recommendations["steps"], 1):
            formatted_text += f"{i}. {step}\n"
        formatted_text += "\n"
    
    if "organic_solutions" in recommendations:
        formatted_text += "## Organic Solutions\n\n"
        for solution in recommendations["organic_solutions"]:
            formatted_text += f"- {solution}\n"
        formatted_text += "\n"
    
    if "chemical_solutions" in recommendations:
        formatted_text += "## Chemical Solutions (if necessary)\n\n"
        for solution in recommendations["chemical_solutions"]:
            formatted_text += f"- {solution}\n"
        formatted_text += "\n"
    
    if "prevention" in recommendations:
        formatted_text += "## Prevention Tips\n\n"
        for tip in recommendations["prevention"]:
            formatted_text += f"- {tip}\n"
        formatted_text += "\n"
    
    # If the recommendations is just a string, return it directly
    if isinstance(recommendations, str):
        return recommendations
    
    # If we didn't format anything but have a raw text field, use that
    if not formatted_text and "text" in recommendations:
        return recommendations["text"]
    
    return formatted_text if formatted_text else "No recommendations available."

def parse_disease_severity(confidence: float) -> str:
    """
    Parse disease severity based on confidence score.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Severity level ("mild", "moderate", or "severe")
    """
    if confidence < 0.5:
        return "mild"
    elif confidence < 0.8:
        return "moderate"
    else:
        return "severe"
