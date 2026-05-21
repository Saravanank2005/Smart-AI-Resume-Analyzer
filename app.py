import gradio as gr
import os
from dotenv import load_dotenv
from rag_pipeline import chat_with_gemini

# Load environment variables
load_dotenv()

import sys
# Guard against running with Streamlit
if any("streamlit" in arg for arg in sys.argv) or "streamlit" in sys.modules:
    try:
        import streamlit as st
        st.error("⚠️ **EchoMind has been migrated to Gradio!**")
        st.info("Please do not run this script with `streamlit run app.py`.")
        st.markdown(
            "To launch the new Gradio interface:\n"
            "1. Stop/kill this streamlit process (e.g., press `Ctrl+C` in your terminal).\n"
            "2. Run the Gradio app using Python directly:\n"
            "```powershell\n"
            ".\\venv\\Scripts\\python app.py\n"
            "```"
        )
        st.stop()
    except Exception:
        pass

load_dotenv()

def chat_response(message, history, api_key_input, model_dropdown, temperature_slider):
    """
    Callback function for Gradio ChatInterface.
    history is a list of ChatMessage objects.
    """
    # Create the conversation list to send to Gemini
    formatted_messages = []
    
    # Map previous conversation history
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
        
        # In Gradio, role can be 'user' or 'assistant'.
        # We pass this role to chat_with_gemini which maps it to 'user' or 'model'.
        formatted_messages.append({"role": role, "content": content})
        
    # Append the new message
    formatted_messages.append({"role": "user", "content": message})
    
    # Get the API key: override if provided, otherwise fallback to .env
    api_key = api_key_input.strip() if api_key_input else os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return "Error: Gemini API key is missing. Please configure it in your .env file or paste it in the key input box."
        
    try:
        response = chat_with_gemini(
            formatted_messages, 
            api_key=api_key, 
            model_name=model_dropdown, 
            temperature=temperature_slider
        )
        return response
    except Exception as e:
        return f"Error calling Gemini: {e}"

# Custom styling with Dark Glassmorphism and modern colors
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main Container Base */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2.5rem 1.5rem !important;
    background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.12), transparent 45%),
                radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.12), transparent 45%),
                #0b0f19 !important;
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Header Section */
#header-container {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}

#header-container h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818cf8 10%, #a78bfa 50%, #f472b6 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

#header-container p {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
}

/* Sidebar Styling */
.sidebar-panel {
    background: rgba(17, 24, 39, 0.45) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* Chat Panel Area */
.chat-panel {
    background: rgba(17, 24, 39, 0.35) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    overflow: hidden !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
    padding: 0.5rem !important;
}

/* Chatbot container */
#chatbot-view {
    border: none !important;
    background: transparent !important;
}

/* Chat Message Bubbles styling */
#chatbot-view .message.user {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border-radius: 20px 20px 4px 20px !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    border: none !important;
    padding: 0.8rem 1.2rem !important;
    margin-left: auto !important;
    max-width: 75% !important;
}

#chatbot-view .message.assistant {
    background: rgba(30, 41, 59, 0.7) !important;
    color: #f1f5f9 !important;
    border-radius: 20px 20px 20px 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(8px) !important;
    padding: 0.8rem 1.2rem !important;
    margin-right: auto !important;
    max-width: 75% !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

/* Standardizing input fields, textboxes and dropdowns */
input[type="text"], input[type="password"], textarea, select {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

input[type="text"]:focus, input[type="password"]:focus, textarea:focus, select:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25) !important;
    outline: none !important;
}

/* Primary Button Styles */
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

/* Clear, Stop, Undo Buttons */
button.secondary {
    background: rgba(51, 65, 85, 0.6) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover {
    background: rgba(71, 85, 105, 0.8) !important;
    color: #ffffff !important;
}
"""

with gr.Blocks(title="EchoMind AI Hub") as demo:
    # Header Panel (gr.Group serves as our outer container for styling)
    with gr.Group(elem_id="header-container"):
        gr.HTML("<h1>🔮 EchoMind AI Hub</h1>")
        gr.HTML("<p>Your premium, multi-turn global assistant powered by the Google Gemini API. Ask any questions, explore concepts, or get coding help instantly.</p>")

    with gr.Row():
        # Configuration Sidebar
        with gr.Column(scale=1, min_width=280, elem_classes=["sidebar-panel"]):
            gr.Markdown("### ⚙️ Chat Settings")
            
            # API Key Config Check
            env_key = os.getenv("GEMINI_API_KEY")
            key_configured = env_key and env_key != "YOUR_GEMINI_API_KEY"
            
            status_html = (
                "<div style='padding: 10px 14px; border-radius: 10px; background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; margin-bottom: 15px; font-size: 0.9rem;'>"
                "🟢 <b>Active:</b> API Key loaded from system .env"
                "</div>"
            ) if key_configured else (
                "<div style='padding: 10px 14px; border-radius: 10px; background-color: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; margin-bottom: 15px; font-size: 0.9rem;'>"
                "⚠️ <b>Warning:</b> No API Key found in .env. Enter override below."
                "</div>"
            )
            gr.HTML(status_html)
            
            api_key_input = gr.Textbox(
                label="Gemini API Key Override",
                placeholder="Paste key to override .env...",
                type="password",
                container=True
            )
            
            model_dropdown = gr.Dropdown(
                choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                value="gemini-2.5-flash",
                label="Select Gemini Model",
                interactive=True
            )
            
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.05,
                label="Creativity (Temperature)",
                interactive=True
            )
            
            gr.Markdown("---")
            gr.Markdown(
                "💬 **Global Capabilities**\n"
                "- Full multi-turn dialog memory\n"
                "- Freeform querying (math, coding, design, reasoning)\n"
                "- Custom styling & light/dark modes\n"
                "- Local RAG pipeline loaded in backend context"
            )
            
        # Chat interface panel
        with gr.Column(scale=3, elem_classes=["chat-panel"]):
            chatbot = gr.Chatbot(
                height=650,
                show_label=False,
                elem_id="chatbot-view",
                layout="bubble",
                placeholder="<div style='text-align: center; padding-top: 5rem; color: #64748b;'><h3>✨ Welcome to EchoMind AI</h3><p>Human replies are on the right, AI responses are on the left. Type your query below to begin!</p></div>"
            )
            
            gr.ChatInterface(
                fn=chat_response,
                chatbot=chatbot,
                additional_inputs=[api_key_input, model_dropdown, temperature_slider],
                examples=[
                    ["Help me learn Java from scratch.", "", "gemini-2.5-flash", 0.7],
                    ["Explain what is a Retrieval-Augmented Generation (RAG) system.", "", "gemini-2.5-flash", 0.7],
                    ["Write a python function to find the shortest path in a graph.", "", "gemini-2.5-flash", 0.7]
                ]
            )

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css
    )