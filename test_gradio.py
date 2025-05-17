import torch
from diffusers import FluxPipeline, FluxControlPipeline
import gradio as gr
from PIL import Image
import gc
from diffusers.utils import load_image

# Thử import DepthPreprocessor nếu có
try:
    from image_gen_aux import DepthPreprocessor

    HAS_DEPTH = True
except ImportError:
    HAS_DEPTH = False


# Hàm giải phóng model
def unload_model(model_state):
    if model_state is not None:
        del model_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return None, None


# Hàm load model
def load_model(mode, use_depth, model_state, preproc_state):
    model_state, preproc_state = unload_model(model_state)
    if mode == "Text to Image":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        return (
            pipe,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif mode == "Image to Image (Depth Control)":
        if use_depth:
            pipe = FluxControlPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            if HAS_DEPTH:
                processor = DepthPreprocessor.from_pretrained(
                    "LiheYoung/depth-anything-large-hf"
                )
            else:
                processor = None
            return (
                pipe,
                processor,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            # Nếu pipeline hỗ trợ image-to-image không control, load pipeline thường
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
            return (
                pipe,
                None,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
    else:
        return (
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
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
    use_depth,
    seed,
):
    if model_state is None:
        return None
    if use_depth and preproc_state is not None:
        # Xử lý depth
        if isinstance(init_image, str):
            control_image = load_image(init_image)
        else:
            control_image = init_image
        control_image = preproc_state(control_image)[0].convert("RGB")
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
        # Image-to-image thường (nếu pipeline hỗ trợ)
        if hasattr(model_state, "img2img"):
            out = model_state.img2img(
                image=init_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                strength=strength,
            ).images[0]
            return out
        else:
            return Image.new("RGB", (width, height), color="gray")


with gr.Blocks() as demo:
    gr.Markdown("# FLUX.1")
    mode = gr.Dropdown(
        ["Text to Image", "Image to Image (Depth Control)"],
        value="Text to Image",
        label="Mode",
    )
    use_depth = gr.Checkbox(label="Sử dụng Depth Control", value=True, visible=False)
    model_state = gr.State(None)
    preproc_state = gr.State(None)
    load_btn = gr.Button("Load Model")
    unload_btn = gr.Button("Unload Model")
    model_loaded_msg = gr.Markdown("", visible=False)

    with gr.Column(visible=True) as txt2img_col:
        prompt = gr.Textbox(
            label="Prompt", value="A cat holding a sign that says hello world"
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale = gr.Slider(
                0, 20, value=0.0, step=0.1, label="Guidance Scale"
            )
            height = gr.Slider(256, 1536, value=768, step=8, label="Height")
            width = gr.Slider(256, 2048, value=1360, step=8, label="Width")
            num_inference_steps = gr.Slider(
                1, 100, value=4, step=1, label="Num Inference Steps"
            )
            max_sequence_length = gr.Slider(
                32, 512, value=256, step=8, label="Max Sequence Length"
            )
        gen_btn = gr.Button("Generate")
        img_out = gr.Image(label="Output Image")

    with gr.Column(visible=False) as img2img_col:
        init_img = gr.Image(label="Input Image")
        prompt2 = gr.Textbox(
            label="Prompt",
            value="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale2 = gr.Slider(
                0, 20, value=10.0, step=0.1, label="Guidance Scale"
            )
            height2 = gr.Slider(256, 1536, value=1024, step=8, label="Height")
            width2 = gr.Slider(256, 2048, value=1024, step=8, label="Width")
            num_inference_steps2 = gr.Slider(
                1, 100, value=30, step=1, label="Num Inference Steps"
            )
            max_sequence_length2 = gr.Slider(
                32, 512, value=256, step=8, label="Max Sequence Length"
            )
            strength2 = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Strength")
            seed2 = gr.Number(value=42, label="Seed (int)")
        gen_btn2 = gr.Button("Generate")
        img_out2 = gr.Image(label="Output Image")
        if not HAS_DEPTH:
            gr.Markdown(
                "<span style='color:red'>Thiếu package image_gen_aux hoặc DepthPreprocessor! Vui lòng cài đặt để sử dụng Depth Control.</span>"
            )

    def switch_mode(selected_mode):
        if selected_mode == "Text to Image":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )

    mode.change(switch_mode, inputs=mode, outputs=[txt2img_col, img2img_col, use_depth])

    load_btn.click(
        load_model,
        inputs=[mode, use_depth, model_state, preproc_state],
        outputs=[
            model_state,
            preproc_state,
            txt2img_col,
            img2img_col,
            use_depth,
            model_loaded_msg,
        ],
        show_progress=True,
    )
    unload_btn.click(
        unload_model, inputs=model_state, outputs=[model_state, preproc_state]
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
            use_depth,
            seed2,
        ],
        outputs=img_out2,
    )

if __name__ == "__main__":
    demo.launch(share=True)
