import gradio as gr
from transformers import pipeline

# Load your ASR model
asr = pipeline(task="automatic-speech-recognition",
               model="./models/openai/whisper-large-v3")

def transcribe_speech(filepath):
    if filepath is None:
        return "No audio found, please retry."
    try:
        output = asr(filepath)
        if "error" in output:
            return f"An error occurred during transcription: {output['error']}"
        return output["text"]
    except Exception as e:
        # You may want to log the exception here for debugging
        return f"An unexpected error occurred: {str(e)}"

with gr.Blocks() as demo:
    with gr.Tab("Transcribe Microphone"):
        with gr.Group():
            mic_input = gr.Audio(sources="microphone", type="filepath")
        with gr.Group():
            mic_output = gr.Textbox(label="Transcription", lines=3)
        mic_input.change(transcribe_speech, inputs=[mic_input], outputs=[mic_output])
    
    with gr.Tab("Transcribe Audio File"):
        with gr.Group():
            file_input = gr.Audio(sources="upload", type="filepath")
        with (gr.Group()):
            file_output = gr.Textbox(label="Transcription", lines=3)
        file_input.change(transcribe_speech, inputs=[file_input], outputs=[file_output])

# Launch the Gradio server
demo.launch(share=True)
