import os

import gradio as gr
# from dotenv import load_dotenv
from elevenlabs import ElevenLabs

from blog_summarizer import summarize_blog

# load_dotenv()


def process_url(url):
    try:
        # Get summary from blog_summarizer
        summary = summarize_blog(url)
        print("-" * 40)
        print("Blog Summary:", summary)
        print("-" * 40)

        # Convert summary to speech using ElevenLabs
        client = ElevenLabs()
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        client = ElevenLabs(api_key=api_key)
        response = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            output_format="mp3_44100_128",
            text=summary[:350],  # Limit to first 350 characters for TTS
            model_id="eleven_flash_v2_5",
        )

        # Save audio to a file
        audio_path = "output.mp3"
        with open(audio_path, "wb") as f:
            for chunk in response:
                f.write(chunk)

        return summary, audio_path, "Podcast generated successfully!"
    except Exception as e:
        print("Error processing URL:", str(e))
        return None, None, f"Error: {str(e)}"


with gr.Blocks(title="AI Podcast Generator", theme="soft") as demo:
    gr.Markdown("# AI Podcast Generator")
    gr.Markdown("Enter a blog URL to generate a podcast episode from its content")

    with gr.Row():
        url_input = gr.Textbox(
            label="Blog URL", placeholder="https://example.com/blog-post"
        )

    generate_btn = gr.Button("Generate Podcast")
    status_output = gr.Textbox(label="Status", lines=1)

    with gr.Row():
        summary_output = gr.Textbox(label="Blog Summary", lines=10)

    with gr.Row():
        audio_output = gr.Audio(label="Podcast Audio")

    generate_btn.click(
        fn=process_url,
        inputs=[url_input],
        outputs=[summary_output, audio_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=("devmode","testdeployment8721"))
