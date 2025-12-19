import spaces, torch, time
import gradio as gr
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Flash Attention for ZeroGPU
import subprocess

subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)

# Set target DEVICE and DTYPE
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)
DEVICE = "auto"
print(f"Device: {DEVICE}, dtype: {DTYPE}")


def load_model(
    model_name: str = "chancharikm/qwen2.5-vl-7b-cam-motion-preview",
    use_flash_attention: bool = True,
    apply_quantization: bool = True,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load model weights in 4-bit
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization (or "fp4")
        bnb_4bit_compute_dtype=DTYPE,  # Perform computations in bfloat16/float16
        bnb_4bit_use_double_quant=True,  # Optional: further quantization for slightly more memory saving
    )

    # Determine model family from model name
    model_family = model_name.split("/")[-1].split("-")[0]

    # Common model loading arguments
    common_args = {
        "torch_dtype": DTYPE,
        "device_map": DEVICE,
        "low_cpu_mem_usage": True,
        "quantization_config": bnb_config if apply_quantization else None,
    }
    if use_flash_attention:
        common_args["attn_implementation"] = "flash_attention_2"

    # Load model based on family
    match model_family:
        # case "qwen2.5" | "Qwen2.5":
        #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #         model_name, **common_args
        #     )
        case "InternVL3":
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, **common_args
            )
        case _:
            raise ValueError(f"Unsupported model family: {model_family}")

    # Set model to evaluation mode for inference (disables dropout, etc.)
    return model.eval()


def load_processor(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    return AutoProcessor.from_pretrained(
        model_name,
        device_map=DEVICE,
        use_fast=True,
        torch_dtype=DTYPE,
    )


print("Loading Models and Processors...")
MODEL_ZOO = {
    "qwen2.5-vl-7b-instruct": load_model(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        use_flash_attention=False,
        apply_quantization=False,
    ),
    "InternVL3-1B-hf": load_model(
        model_name="OpenGVLab/InternVL3-1B-hf",
        use_flash_attention=False,
        apply_quantization=False,
    ),
    "InternVL3-2B-hf": load_model(
        model_name="OpenGVLab/InternVL3-2B-hf",
        use_flash_attention=False,
        apply_quantization=False,
    ),
    "InternVL3-8B-hf": load_model(
        model_name="OpenGVLab/InternVL3-8B-hf",
        use_flash_attention=False,
        apply_quantization=True,
    ),
}

PROCESSORS = {
    "qwen2.5-vl-7b-instruct": load_processor("Qwen/Qwen2.5-VL-7B-Instruct"),
    "InternVL3-1B-hf": load_processor("OpenGVLab/InternVL3-1B-hf"),
    "InternVL3-2B-hf": load_processor("OpenGVLab/InternVL3-2B-hf"),
    "InternVL3-8B-hf": load_processor("OpenGVLab/InternVL3-8B-hf"),
}
print("Models and Processors Loaded!")


# Our Inference Function
@spaces.GPU(duration=120)
def video_inference(
    video_path: str,
    prompt: str,
    model_name: str,
    fps: int = 8,
    max_tokens: int = 512,
    temperature: float = 0.1,
):
    s_time = time.time()
    model = MODEL_ZOO[model_name]
    processor = PROCESSORS[model_name]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    with torch.no_grad():
        model_family = model_name.split("-")[0]
        match model_family:
            case "InternVL3":
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    fps=fps,
                    # num_frames = 8
                ).to("cuda", dtype=DTYPE)

                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=float(temperature),
                    do_sample=temperature > 0.0,
                )
                output_text = processor.decode(
                    output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            case _:
                raise ValueError(f"{model_name} is not currently supported")
    return {
        "output_text": output_text,
        "fps": fps,
        "inference_time": time.time() - s_time,
    }


# the Gradio App
app = gr.Interface(
    fn=inference,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Textbox(
            label="Prompt",
            lines=3,
            info="Some models like [cam motion](https://huggingface.co/chancharikm/qwen2.5-vl-7b-cam-motion-preview) are trained specific prompts",
            value="Describe the camera motion in this video.",
        ),
        gr.Dropdown(label="Model", choices=list(MODEL_ZOO.keys())),
        gr.Number(
            label="FPS",
            info="inference sampling rate (Qwen2.5VL is trained on videos with 8 fps); a value of 0 means the FPS of the input video will be used",
            value=8,
            minimum=0,
            step=1,
        ),
        gr.Slider(
            label="Max Tokens",
            info="maximum number of tokens to generate",
            value=128,
            minimum=32,
            maximum=512,
            step=32,
        ),
        gr.Slider(
            label="Temperature",
            value=0.0,
            minimum=0.0,
            maximum=1.0,
            step=0.1,
        ),
    ],
    outputs=gr.JSON(label="Output JSON"),
    title="Video Chat with VLM",
    description='comparing various "small" VLMs on the task of video captioning',
    api_name="video_inference",
)
app.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
