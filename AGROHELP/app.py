import streamlit as st
import os
import tempfile
import logging
from dotenv import load_dotenv
import warnings

# Suppress torch-related warnings
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

# Import our application modules
from app.utils.agri_assistant import AgriAssistant
from app.services.document_processor import DocumentProcessor
from app.utils.helpers import setup_logging

# Load environment variables
load_dotenv()

# Set up logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AgroHelp - AI Assistant for Sustainable Agriculture",
    page_icon="🌱",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "assistant" not in st.session_state:
    st.session_state.assistant = None

# App header
st.title("🌱 AgroHelp")
st.subheader("AI Assistant for Sustainable Agriculture")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")

    # Language selection
    target_language = st.selectbox(
        "Select your preferred language:",
        ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Punjabi", "Gujarati", "Kannada", "Malayalam"]
    )

    # Model selection
    st.subheader("Model Settings")

    # OpenAI model selection
    llm_type = "gpt"  # Force to use OpenAI
    llm_size = st.selectbox(
        "OpenAI Model:",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="Select which OpenAI model to use"
    )

    vision_model = st.selectbox(
        "Vision Model:",
        ["resnet", "mobilenet"],
        index=0,
        help="Select the vision model to use for crop disease detection"
    )

    # Document upload
    st.header("Upload Agricultural Documents")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Initialize document processor
                doc_processor = DocumentProcessor(embedding_type="huggingface")

                # Process each uploaded file
                all_docs = []
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Process the PDF
                    docs = doc_processor.process_pdf(tmp_path)
                    all_docs.extend(docs)

                    # Clean up temp file
                    os.unlink(tmp_path)

                # Create vector store
                if all_docs:
                    st.session_state.vector_store = doc_processor.create_vector_store(all_docs, store_type="faiss")
                    st.session_state.assistant = AgriAssistant(
                        llm_type=llm_type,
                        llm_size=llm_size,
                        vision_model_type=vision_model,
                        vector_store=st.session_state.vector_store,
                        translation_model="nllb"
                    )
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                else:
                    st.error("No content could be extracted from the documents.")
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                st.error(f"Error processing documents: {str(e)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Crop Disease Detection")
    uploaded_image = st.file_uploader("Upload a crop image for disease detection", type=["jpg", "jpeg", "png"])

    if uploaded_image and st.button("Analyze Image"):
        try:
            # Initialize assistant if not already done
            if st.session_state.assistant is None:
                st.session_state.assistant = AgriAssistant(
                    llm_type=llm_type,
                    llm_size=llm_size,
                    vision_model_type=vision_model,
                    translation_model="nllb"
                )

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_image.getvalue())
                tmp_path = tmp_file.name

            # Display the image
            st.image(uploaded_image, caption="Uploaded Crop Image", use_column_width=True)

            # Process the image
            with st.spinner("Analyzing image..."):
                result = st.session_state.assistant.process_image(tmp_path, target_lang=target_language)

                if "error" in result:
                    st.error(f"Error analyzing image: {result['error']}")
                else:
                    st.success("Analysis complete!")

                    # Display results
                    st.subheader("Diagnosis Results")
                    st.write(f"**Crop:** {result['crop']}")
                    st.write(f"**Condition:** {result['condition']}")
                    st.write(f"**Confidence:** {result['confidence']:.2%}")

                    # Display treatment recommendations
                    st.subheader("Recommendations")
                    st.write(result['recommendations'])

                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result['recommendations']})

            # Clean up temp file
            os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            st.error(f"Error analyzing image: {str(e)}")

    # Add a crop care plan generator
    st.header("Generate Crop Care Plan")
    crop_type = st.text_input("Crop Type", placeholder="e.g., Tomato, Rice, Wheat")
    region = st.text_input("Region (optional)", placeholder="e.g., North India, Maharashtra")
    season = st.selectbox("Season", ["", "Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"])

    if crop_type and st.button("Generate Care Plan"):
        try:
            # Initialize assistant if not already done
            if st.session_state.assistant is None:
                st.session_state.assistant = AgriAssistant(
                    llm_type=llm_type,
                    llm_size=llm_size,
                    vision_model_type=vision_model,
                    translation_model="nllb"
                )

            with st.spinner("Generating crop care plan..."):
                care_plan = st.session_state.assistant.generate_crop_care_plan(
                    crop=crop_type,
                    region=region if region else None,
                    season=season if season else None,
                    target_lang=target_language
                )

                # Display care plan
                st.subheader(f"Care Plan for {crop_type}")
                st.write(care_plan)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Crop Care Plan for {crop_type}**\n\n{care_plan}"
                })
        except Exception as e:
            logger.error(f"Error generating crop care plan: {str(e)}")
            st.error(f"Error generating crop care plan: {str(e)}")

with col1:
    st.header("Chat with AgroHelp")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about sustainable farming, crop management, or disease treatment..."):
        try:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Initialize assistant if not already done
            if st.session_state.assistant is None:
                st.session_state.assistant = AgriAssistant(
                    llm_type=llm_type,
                    llm_size=llm_size,
                    vision_model_type=vision_model,
                    translation_model="nllb"
                )

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.assistant.process_query(prompt, target_lang=target_language)
                    st.write(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"Error processing your query: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>AgroHelp - AI Assistant for Sustainable Agriculture</p>
        <p>Built with ❤️ for Indian Farmers</p>
    </div>
    """,
    unsafe_allow_html=True
)