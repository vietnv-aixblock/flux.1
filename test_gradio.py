import torch
from diffusers import FluxPipeline, FluxControlPipeline
import gradio as gr
from PIL import Image
import gc
from diffusers.utils import load_image
import numpy as np
from huggingface_hub import hf_hub_download

# Try to import DepthPreprocessor if available
try:
    from image_gen_aux import DepthPreprocessor

    HAS_DEPTH = True
except ImportError:
    HAS_DEPTH = False


# Function to unload model
def unload_model(model_state):
    if model_state is not None:
        del model_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return None, None


# Function to load model
def load_model(
    mode,
    model_state,
    preproc_state,
    load_lora=False,
    lora_model_name="XLabs-AI/flux-furry-lora",
    lora_scale=0.9,
    load_ip_adapter=False,
    ip_adapter_model_name="XLabs-AI/flux-ip-adapter",
):
    model_state, preproc_state = unload_model(model_state)
    if mode == "Text to Image":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )
        # Nếu load_lora được tích thì load 1 LoRA weight và set scale
        if load_lora:
            pipe.load_lora_weights(
                lora_model_name,
                weight_name="furry_lora.safetensors",
                adapter_name="custom_lora",
            )
            pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
        if load_ip_adapter:
            pipe.load_ip_adapter(
                ip_adapter_model_name,
                weight_name="ip_adapter.safetensors",
                image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
            )
            pipe.set_ip_adapter_scale(1.0)
        pipe.enable_model_cpu_offload()
        return (
            pipe,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(
                visible=True,
                value="<span style='color:green'>Model is ready!</span>",
            ),
        )
    elif mode == "Image to Image (Depth Control)":
        pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        if HAS_DEPTH:
            processor = DepthPreprocessor.from_pretrained(
                "LiheYoung/depth-anything-large-hf"
            )
        else:
            processor = None
        # Nếu load_lora được tích thì load 1 LoRA weight và set scale
        if load_lora:
            pipe.load_lora_weights(
                lora_model_name,
                # weight_name="lora.safetensors",
                adapter_name="custom_lora",
            )
            pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
        if load_ip_adapter:
            pipe.load_ip_adapter(
                ip_adapter_model_name,
                weight_name="ip_adapter.safetensors",
                image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
            )
            pipe.set_ip_adapter_scale(1.0)
        pipe.enable_model_cpu_offload()
        return (
            pipe,
            processor,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(
                visible=True,
                value="<span style='color:green'>Model is ready!</span>",
            ),
        )
    else:
        return (
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, value=""),
        )


def text_to_image_gr(
    model_state,
    prompt,
    guidance_scale,
    height,
    width,
    num_inference_steps,
    max_sequence_length,
):
    if model_state is None:
        return None
    out = model_state(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
    ).images[0]
    return out


def image_to_image_gr(
    model_state,
    preproc_state,
    init_image,
    prompt,
    guidance_scale,
    height,
    width,
    num_inference_steps,
    max_sequence_length,
    strength,
    seed,
):
    if model_state is None:
        return None
    if preproc_state is not None:
        # Always process depth
        # According to context7, gr.Image returns numpy array (uint8, HWC) or PIL.Image or string
        if isinstance(init_image, np.ndarray):
            control_image = Image.fromarray(init_image.astype("uint8"))
        elif isinstance(init_image, Image.Image):
            control_image = init_image
        elif isinstance(init_image, str):
            control_image = load_image(init_image)
        else:
            raise ValueError("init_image must be a numpy array, PIL.Image, or string")
        # Ensure control_image is RGB
        if hasattr(control_image, "mode") and control_image.mode != "RGB":
            control_image = control_image.convert("RGB")
        control_image = preproc_state(control_image)[0]

        # Ensure control_image is always RGB 3 channel
        if isinstance(control_image, np.ndarray):
            if control_image.ndim == 2:  # grayscale
                control_image = np.stack([control_image] * 3, axis=-1)
            elif control_image.shape[-1] == 1:
                control_image = np.repeat(control_image, 3, axis=-1)
            control_image = Image.fromarray(control_image.astype(np.uint8))
        elif "torch" in str(type(control_image)):
            arr = control_image.cpu().numpy()
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
            arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
            control_image = Image.fromarray(arr.astype(np.uint8))
        if hasattr(control_image, "mode") and control_image.mode != "RGB":
            control_image = control_image.convert("RGB")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        image = model_state(
            prompt=prompt,
            control_image=control_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        return image
    else:
        return Image.new("RGB", (width, height), color="gray")


# Thêm CSS cho hiệu ứng nhấp nháy
demo_css = """
.blinking {
    animation: blinker 1s linear infinite;
}
@keyframes blinker {
    50% { opacity: 0.3; }
}
"""

with gr.Blocks(css=demo_css) as demo:
    gr.Markdown("# FLUX.1")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Dropdown(
                ["Text to Image", "Image to Image (Depth Control)"],
                value="Text to Image",
                label="Mode",
                info="Choose the generation mode.",
            )
        with gr.Column(scale=1):
            lora_checkbox = gr.Checkbox(
                label="Load LoRA",
                value=False,
            )
            lora_model_box = gr.Textbox(
                label="LoRA Model",
                value="XLabs-AI/flux-furry-lora",
                visible=False,
                info="HuggingFace model repo or path for LoRA weights.",
            )
            lora_scale_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.01,
                label="LoRA Scale",
                visible=False,
                info="Adjust the influence of the loaded LoRA weights.",
            )
            ip_adapter_checkbox = gr.Checkbox(
                label="Load IP-Adapter",
                value=False,
            )
            ip_adapter_model_box = gr.Textbox(
                label="IP-Adapter Model",
                value="XLabs-AI/flux-ip-adapter",
                visible=False,
                info="HuggingFace model repo or path for IP-Adapter weights.",
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("Load Model", size="lg", variant="primary")
        with gr.Column(scale=2):
            model_loaded_msg = gr.Markdown("", visible=False)
    loading_msg = gr.Markdown("", visible=False)

    model_state = gr.State(None)
    preproc_state = gr.State(None)

    with gr.Column(visible=True) as txt2img_col:
        prompt = gr.Textbox(
            label="Prompt",
            value="A cat holding a sign that says hello world",
            info="Describe the image you want to generate.",
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale = gr.Slider(
                0,
                20,
                value=0.0,
                step=0.1,
                label="Guidance Scale",
            )
            height = gr.Slider(
                256,
                1536,
                value=768,
                step=8,
                label="Height",
            )
            width = gr.Slider(
                256,
                2048,
                value=1360,
                step=8,
                label="Width",
            )
            num_inference_steps = gr.Slider(
                1,
                100,
                value=4,
                step=1,
                label="Num Inference Steps",
            )
            max_sequence_length = gr.Slider(
                32,
                512,
                value=256,
                step=8,
                label="Max Sequence Length",
            )
        gen_btn = gr.Button("Generate")
        img_out = gr.Image(label="Output Image")

    with gr.Column(visible=False) as img2img_col:
        init_img = gr.Image(
            label="Input Image",
        )
        prompt2 = gr.Textbox(
            label="Prompt",
            value="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
            info="Describe the modifications or style for the output image.",
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale2 = gr.Slider(
                0,
                20,
                value=10.0,
                step=0.1,
                label="Guidance Scale",
            )
            height2 = gr.Slider(
                256,
                1536,
                value=1024,
                step=8,
                label="Height",
            )
            width2 = gr.Slider(
                256,
                2048,
                value=1024,
                step=8,
                label="Width",
            )
            num_inference_steps2 = gr.Slider(
                1,
                100,
                value=30,
                step=1,
                label="Num Inference Steps",
            )
            max_sequence_length2 = gr.Slider(
                32,
                512,
                value=256,
                step=8,
                label="Max Sequence Length",
            )
            strength2 = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.01,
                label="Strength",
            )
            seed2 = gr.Number(
                value=42,
                label="Seed (int)",
            )
        gen_btn2 = gr.Button("Generate")
        img_out2 = gr.Image(label="Output Image")
        if not HAS_DEPTH:
            gr.Markdown(
                "<span style='color:red'>Missing package image_gen_aux or DepthPreprocessor! Please install to use Depth Control.</span>"
            )

    def switch_mode(selected_mode):
        if selected_mode == "Text to Image":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
            )

    mode.change(switch_mode, inputs=mode, outputs=[txt2img_col, img2img_col])

    def set_loading_msg():
        return gr.update(
            value="<span style='color:blue'>Loading model...</span>", visible=True
        )

    def set_loaded_msg():
        return gr.update(
            value="<span style='color:green'>Model is ready!</span>", visible=True
        )

    def clear_loading_msg():
        return gr.update(value="", visible=False)

    # Hiện ô lora_model_box và lora_scale_slider khi lora_checkbox được tích
    def toggle_lora_controls(checked):
        return gr.update(visible=checked), gr.update(visible=checked)

    lora_checkbox.change(
        toggle_lora_controls,
        inputs=lora_checkbox,
        outputs=[lora_model_box, lora_scale_slider],
    )

    # Hiện ô ip_adapter_model_box khi ip_adapter_checkbox được tích
    def toggle_ip_adapter_box(checked):
        return gr.update(visible=checked)

    ip_adapter_checkbox.change(
        toggle_ip_adapter_box, inputs=ip_adapter_checkbox, outputs=ip_adapter_model_box
    )

    # Hiệu ứng loading cho nút Load Model (thêm class blinking)
    def set_btn_loading():
        return gr.Button(
            interactive=False, elem_classes=["blinking"], variant="primary"
        )

    def unset_btn_loading():
        return gr.Button(interactive=True, elem_classes=[], variant="primary")

    load_btn.click(
        set_loading_msg,
        inputs=None,
        outputs=loading_msg,
        queue=False,
    )
    load_btn.click(
        set_btn_loading,
        inputs=None,
        outputs=load_btn,
    )
    load_btn.click(
        load_model,
        inputs=[
            mode,
            model_state,
            preproc_state,
            lora_checkbox,
            lora_model_box,
            lora_scale_slider,
            ip_adapter_checkbox,
            ip_adapter_model_box,
        ],
        outputs=[
            model_state,
            preproc_state,
            txt2img_col,
            img2img_col,
            model_loaded_msg,
        ],
    )
    load_btn.click(
        unset_btn_loading,
        inputs=None,
        outputs=load_btn,
    )
    load_btn.click(
        clear_loading_msg,
        inputs=None,
        outputs=loading_msg,
    )

    gen_btn.click(
        text_to_image_gr,
        inputs=[
            model_state,
            prompt,
            guidance_scale,
            height,
            width,
            num_inference_steps,
            max_sequence_length,
        ],
        outputs=img_out,
    )
    gen_btn2.click(
        image_to_image_gr,
        inputs=[
            model_state,
            preproc_state,
            init_img,
            prompt2,
            guidance_scale2,
            height2,
            width2,
            num_inference_steps2,
            max_sequence_length2,
            strength2,
            seed2,
        ],
        outputs=img_out2,
    )

if __name__ == "__main__":
    demo.launch(share=True)
