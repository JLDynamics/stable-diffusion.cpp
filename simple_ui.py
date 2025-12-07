import gradio as gr
import subprocess
import os

# Configuration for resolutions
RESOLUTIONS = {
    "Low (512x512)": (512, 512),
    "Medium (768x768)": (768, 768),
    "High (1024x1024)": (1024, 1024),
    "1080p (1920x1080)": (1920, 1080)
}

def generate_image(prompt, resolution_key, steps):
    width, height = RESOLUTIONS[resolution_key]
    
    cmd = [
        "./build/bin/sd",
        "--diffusion-model", "models/z_image/z_image_turbo-Q4_K.gguf",
        "--vae", "models/z_image/ae.safetensors",
        "--llm", "models/z_image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "-p", prompt,
        "--cfg-scale", "1.0",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "--offload-to-cpu",
        "-H", str(height),
        "-W", str(width),
        "--steps", str(int(steps)),
        "-v"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during generation:")
        print(e.stderr)
        return None

    output_path = "output.png"
    if os.path.exists(output_path):
        return output_path
    else:
        return None

# Build the Gradio Interface
with gr.Blocks(title="Z-Image Generator") as demo:
    gr.Markdown("# Z-Image Turbo Generator")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="Describe your image here...", lines=3)
            with gr.Row():
                resolution_input = gr.Dropdown(
                    choices=list(RESOLUTIONS.keys()), 
                    value="High (1024x1024)", 
                    label="Resolution"
                )
                steps_input = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Steps")
            
            generate_btn = gr.Button("Generate Image", variant="primary")
            
        with gr.Column():
            image_output = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, resolution_input, steps_input],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch()
