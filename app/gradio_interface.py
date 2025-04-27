import gradio as gr
import tempfile
import sys
import os

# Add the project root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.audio_processing import process_audio


# Function to handle audio file processing and transcription
def transcribe_gradio(audio_file):
    try:
        # Process the audio file and get the transcription
        transcription = process_audio(audio_file)  # audio_file is the file path
        return transcription
    except Exception as e:
        # Return error message if an exception occurs
        return f"Oops! Something went wrong: {str(e)}"

# Function to create the Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        # Header with a fancy Markdown title and description
        gr.Markdown("""
        # 🎤 Audio Transcription using MediBeng Whisper Tiny
        
        Welcome to the MediBeng Whisper Tiny transcription tool. Simply upload an audio file, 
        and our Whisper model will transcribe the audio for you!
        
        """)

        # Styling the row layout
        with gr.Row():
            # Audio file input (file path) with customized styling
            audio_input = gr.Audio(
                type="filepath", 
                label="🎧 Upload your audio file", 
                elem_id="audio-input", 
                interactive=True,
                scale=1.5
            )

            # Transcription output with a fancy textbox
            output_text = gr.Textbox(
                label="📝 Transcription", 
                interactive=False, 
                placeholder="Your transcription will appear here...",
                elem_id="transcription-output", 
                lines=6, 
                max_lines=6
            )

        # Button to trigger transcription with custom style
        submit_button = gr.Button(
            "🔄 Transcribe Now!", 
            elem_id="transcribe-button", 
            variant="primary"
        )

        # Button click interaction to call the transcription function
        submit_button.click(
            transcribe_gradio, 
            inputs=audio_input, 
            outputs=output_text
        )

        # Adding some styling to the elements
        demo.css = """
        #audio-input {
            margin-top: 15px;
            background-color: #F9F9F9;
        }
        #transcription-output {
            font-family: 'Courier New', Courier, monospace;
            background-color: #f0f0f0;
            border: 2px solid #dedede;
            padding: 10px;
        }
        #transcribe-button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 12px 24px;
            font-size: 16px;
        }
        #transcribe-button:hover {
            background-color: #45a049;
        }
        .gr-button.primary {
            background-color: #4CAF50;
        }
        .gr-button.primary:hover {
            background-color: #45a049;
        }
        """

    return demo

# Main entry point to run the Gradio interface
if __name__ == "__main__":
    # Create the Gradio interface
    gradio_interface = create_gradio_interface()

    # Launch the Gradio app with a fancy, shareable link and in-browser view
    gradio_interface.launch(share=True, inbrowser=True)
