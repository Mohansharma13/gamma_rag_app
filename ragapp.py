import streamlit as st
import logging
# import ollama
from typing import List, Tuple, Dict, Any, Optional

# import user-defined modules for the RAG application
from chat_model import process_question
from vectordb import create_vector_db, delete_vector_db
from pdf_to_text import extract_all_pages_as_images

# log in imports 
from login_signup import signup_user, login_user

# Streamlit page configuration
st.set_page_config(
    page_title="Insight Matrix RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def extract_model_names():
    return "gamma_model"

# def login_signup():
#     st.subheader("Welcome to IntelliQuest 📈")
#     # Login and Signup Tabs
#     tabs = st.tabs(["Login", "Signup"])
    
#     col11, col2 = st.columns([1.5, 2])
#     # Login Tab
#     with col11:
#         with tabs[0]:
#             st.subheader("Login")
#             username = st.text_input("Username", key="login_username")
#             password = st.text_input("Password", type="password", key="login_password")
#             if st.button("Login"):
#                 if login_user(username, password):
#                     st.session_state["logged_in"] = True
#                     st.success("Login successful!")
#                 else:
#                     st.error("Invalid username or password")

#         # Signup Tab
#         with tabs[1]:
#             st.subheader("Signup")
#             new_username = st.text_input("New Username", key="signup_username")
#             email = st.text_input("Email", key="signup_email")
#             new_password = st.text_input("New Password", type="password", key="signup_password")
#             if st.button("Signup"):
#                 if signup_user(new_username, email, new_password):
#                     st.success("Signup successful! Please log in.")
#                 else:
#                     st.error("Signup failed. Username already exists.")
                    
def login_signup():
    st.subheader("Welcome to IntelliQuest 📈")
    
    # Login and Signup Tabs
    tabs = st.tabs(["Login", "Signup"])
    
    # Login Tab
    with tabs[0]:
        st.subheader("Login")
        st.image("images\\banner2.png",width=1850)
        col11, col2 = st.columns([1.5, 2])
        with col11:
            username = st.text_input("Username", key="login_username")
        with col2:
            password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")

    # Signup Tab
    with tabs[1]:
        st.subheader("Signup")
        st.image("images\\banner2.png",width=1850)
        col11, col2 = st.columns([1.5, 2])
        with col11:
            new_username = st.text_input("New Username", key="signup_username")
        with col2:
            email = st.text_input("Email", key="signup_email")
        with col11:
            new_password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Signup"):
            if signup_user(new_username, email, new_password):
                st.success("Signup successful! Please log in.")
            else:
                st.error("Signup failed. Username already exists.")

        
        

def main_app():
    st.subheader("IntelliQuest 📈")
    
    available_models = extract_model_names()
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state variables if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # Dropdown to select an available model
    selected_model = col2.selectbox(
        "Pick a model available locally on your system ↓", available_models
    )

    # File uploader for the PDF file, only accepting single files of PDF format
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload

        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload)

        pdf_pages = extract_all_pages_as_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider("Zoom Level", min_value=100, max_value=1000, value=700, step=50)

        with col1:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")
    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")

        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")

def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        main_app()
    else:
        login_signup()

if __name__ == "__main__":
    main()
