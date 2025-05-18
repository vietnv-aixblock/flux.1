# #model_marketplace.config
# {"framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name":"black-forest-labs/FLUX.1-dev",
#       "value": "black-forest-labs/FLUX.1-dev",
#       "size": 120,
#       "paramasters": "12B",
#       "tflops": 14,
#       "vram": 19, # 16 + 15%
#       "nodes": 1
#     },
#     {
#       "name":"black-forest-labs/FLUX.1-schnell",
#       "value": "black-forest-labs/FLUX.1-schnell",
#       "size": 120,
#       "paramasters": "12B",
#       "tflops": 14,
#       "vram": 19,
#       "nodes": 1
#     },
#     {
#       "name":"multimodalart/FLUX.1-dev2pro-full",
#       "value": "multimodalart/FLUX.1-dev2pro-full",
#       "size": 200,
#       "paramasters": "12B",
#       "tflops": 14,
#       "vram": 40,
#       "nodes": 2
#     },
#   ], "cuda": "11.4", "task":["text-to-image"]}

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Dict, List, Optional, get_type_hints

import gradio as gr
import torch
import wandb
import yaml
from aixblock_ml.model import AIxBlockMLBase
from centrifuge import (
    CentrifugeError,
    Client,
    ClientEventHandler,
    SubscriptionEventHandler,
)
from datasets import load_dataset
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    BitsAndBytesConfig,
    FluxControlPipeline,
)
from huggingface_hub import HfApi, HfFolder, hf_hub_download, login
from mcp.server.fastmcp import FastMCP

import constants as const
import utils
from dashboard import promethus_grafana
from function_ml import (
    connect_project,
    download_dataset,
    upload_checkpoint_mixed_folder,
)
from logging_class import start_queue, write_log
from misc import get_device_count
from param_class import TrainingConfigFlux, TrainingConfigFluxLora
from loguru import logger
from PIL import Image
import io
import gc
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

# --------------------------------------------------------------------------------------------
with open("models.yaml", "r") as file:
    models = yaml.safe_load(file)

mcp = FastMCP("aixblock-mcp")


def base64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def generate_jwt(user, channel=""):
    """Note, in tests we generate token on client-side - this is INSECURE
    and should not be used in production. Tokens must be generated on server-side."""
    hmac_secret = "d0a70289-9806-41f6-be6d-f4de5fe298fb"  # noqa: S105 - this is just a secret used in tests.
    header = {"typ": "JWT", "alg": "HS256"}
    payload = {"sub": user}
    if channel:
        # Subscription token
        payload["channel"] = channel
    encoded_header = base64url_encode(json.dumps(header).encode("utf-8"))
    encoded_payload = base64url_encode(json.dumps(payload).encode("utf-8"))
    signature_base = encoded_header + b"." + encoded_payload
    signature = hmac.new(
        hmac_secret.encode("utf-8"), signature_base, hashlib.sha256
    ).digest()
    encoded_signature = base64url_encode(signature)
    jwt_token = encoded_header + b"." + encoded_payload + b"." + encoded_signature
    return jwt_token.decode("utf-8")


async def get_client_token() -> str:
    return generate_jwt("42")


async def get_subscription_token(channel: str) -> str:
    return generate_jwt("42", channel)


class ClientEventLoggerHandler(ClientEventHandler):
    async def on_connected(self, ctx):
        logging.info("Connected to server")


class SubscriptionEventLoggerHandler(SubscriptionEventHandler):
    async def on_subscribed(self, ctx):
        logging.info("Subscribed to channel")


def setup_client(channel_log):
    client = Client(
        "wss://rt.aixblock.io/centrifugo/connection/websocket",
        events=ClientEventLoggerHandler(),
        get_token=get_client_token,
        use_protobuf=False,
    )

    sub = client.new_subscription(
        channel_log,
        events=SubscriptionEventLoggerHandler(),
        # get_token=get_subscription_token,
    )

    return client, sub


async def send_log(sub, log_message):
    try:
        await sub.publish(data={"log": log_message})
    except CentrifugeError as e:
        logging.error("Error publish: %s", e)


async def send_message(sub, message):
    try:
        await sub.publish(data=message)
    except CentrifugeError as e:
        logging.error("Error publish: %s", e)


async def log_training_progress(sub, log_message):
    await send_log(sub, log_message)


def run_train(command, channel_log="training_logs"):
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client, sub = setup_client(channel_log)

            async def main():
                await client.connect()
                await sub.subscribe()
                await log_training_progress(sub, "Training started")
                log_file_path = "logs/llm-ddp.log"
                last_position = 0  # Vị trí đã đọc đến trong file log
                await log_training_progress(sub, "Training training")
                promethus_grafana.promethus_push_to("training")

                while True:
                    try:
                        current_size = os.path.getsize(log_file_path)
                        if current_size > last_position:
                            with open(log_file_path, "r") as log_file:
                                log_file.seek(last_position)
                                new_lines = log_file.readlines()
                                # print(new_lines)
                                for line in new_lines:
                                    print("--------------", f"{line.strip()}")
                                    #             # Thay thế đoạn này bằng code để gửi log
                                    await log_training_progress(sub, f"{line.strip()}")
                            last_position = current_size

                        time.sleep(5)
                    except Exception as e:
                        print(e)

                # promethus_grafana.promethus_push_to("finish")
                # await log_training_progress(sub, "Training completed")
                # await client.disconnect()
                # loop.stop()

            try:
                loop.run_until_complete(main())
            finally:
                loop.close()  # Đảm bảo vòng lặp được đóng lại hoàn toàn

        except Exception as e:
            print(e)

    thread_start = threading.Thread(target=run)
    thread_start.start()
    subprocess.run(command, shell=True)
    # try:
    #     promethus_grafana.promethus_push_to("finish")
    # except:
    #     pass


def fetch_logs(channel_log="training_logs"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client, sub = setup_client(channel_log)

    async def run():
        await client.connect()
        await sub.subscribe()
        history = await sub.history(limit=-1)
        logs = []
        for pub in history.publications:
            log_message = pub.data.get("log")
            if log_message:
                logs.append(log_message)
        await client.disconnect()
        return logs

    return loop.run_until_complete(run())


# deprecated, for sd-scripts
def download(base_model, train_config):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # download unet
    if "pretrained_model_name_or_path" not in train_config:
        if "FLUX.1-dev" in base_model or "FLUX.1-schnell" in base_model:
            unet_folder = const.MODELS_DIR.joinpath("unet")
        else:
            unet_folder = const.MODELS_DIR.joinpath("unet/{repo}")
        unet_path = unet_folder.joinpath(model_file)
        if not unet_path.exists():
            unet_folder.mkdir(parents=True, exist_ok=True)
            print(f"download {base_model}")
            hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)
        train_config["pretrained_model_name_or_path"] = str(unet_path)

    # download vae
    if "ae" not in train_config:
        vae_folder = const.MODELS_DIR.joinpath("vae")
        vae_path = vae_folder.joinpath("ae.sft")
        if not vae_path.exists():
            vae_folder.mkdir(parents=True, exist_ok=True)
            print(f"downloading ae.sft...")
            hf_hub_download(
                repo_id="cocktailpeanut/xulf-dev",
                local_dir=vae_folder,
                filename="ae.sft",
            )
        train_config["ae"] = str(vae_path)

    # download clip
    if "clip_l" not in train_config:
        clip_folder = const.MODELS_DIR.joinpath("clip")
        clip_l_path = clip_folder.joinpath("clip_l.safetensors")
        if not clip_l_path.exists():
            clip_folder.mkdir(parents=True, exist_ok=True)
            print(f"download clip_l.safetensors")
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                local_dir=clip_folder,
                filename="clip_l.safetensors",
            )
        train_config["clip_l"] = str(clip_l_path)

    # download t5xxl
    if "t5xxl" not in train_config:
        t5xxl_path = clip_folder.joinpath("t5xxl_fp16.safetensors")
        if not t5xxl_path.exists():
            print(f"download t5xxl_fp16.safetensors")
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                local_dir=clip_folder,
                filename="t5xxl_fp16.safetensors",
            )
        train_config["t5xxl"] = str(t5xxl_path)

    return train_config


pipe_predict = None


class MyModel(AIxBlockMLBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        HfFolder.save_token(const.HF_TOKEN)
        login(token=const.HF_ACCESS_TOKEN)
        # wandb.login("allow", const.WANDB_TOKEN)
        print("Login successful")

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
            print("CUDA is available.")
        else:
            print("No GPU available, using CPU.")

        try:
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_properties(0).major
                if compute_capability > 8:
                    self.torch_dtype = torch.bfloat16
                elif compute_capability > 7:
                    self.torch_dtype = torch.float16
            else:
                self.torch_dtype = None  # auto setup for < 7
        except Exception as e:
            self.torch_dtype = None

        try:
            n_gpus = torch.cuda.device_count()
            _ = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
        except Exception as e:
            print("Cannot get cuda memory:", e)
            _ = 0
        max_memory = {i: _ for i in range(n_gpus)}
        print("max memory:", max_memory)

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """ """
        print(
            f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}"""
        )
        return []

    def fit(self, event, data, **kwargs):
        """ """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")

    @mcp.tool()
    def action(self, command, **kwargs):
        """
        {
            "command": "train",
            "params": {
                "project_id": 432,
                "framework": "huggingface",
                "model_id": "black-forest-labs/FLUX.1-dev",
                "push_to_hub": true,
                "push_to_hub_token": "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",
                // "dataset_id": 13,
                "TrainingArguments": {
                    // see param_class.py for the full training arguments
                    ...
                }
            },
            "project": "1"
        }
        """
        # region Train
        logger.info(f"Received command: {command} with args: {kwargs}")
        if command.lower() == "execute":
            _command = kwargs.get("shell", None)
            logger.info(f"Executing command: {_command}")
            subprocess.Popen(
                _command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return {"message": "command completed successfully"}
        if command.lower() == "train":
            try:
                clone_dir = const.CLONE_DIR
                framework = kwargs.get("framework", const.FRAMEWORK)
                task = kwargs.get("task", const.TASK)
                world_size = kwargs.get("world_size", const.WORLD_SIZE)
                rank = kwargs.get("rank", const.RANK)
                master_add = kwargs.get("master_add", const.MASTER_ADDR)
                master_port = kwargs.get("master_port", const.MASTER_PORT)

                host_name = kwargs.get("host_name", const.HOST_NAME)
                token = kwargs.get("token", const.AXB_TOKEN)
                wandb_api_key = kwargs.get("wantdb_api_key", const.WANDB_TOKEN)

                training_arguments = kwargs.get("TrainingArguments", {})
                project_id = kwargs.get("project_id", None)
                model_id = kwargs.get("model_id", const.MODEL_ID)
                dataset_id = kwargs.get("dataset_id", None)
                push_to_hub = kwargs.get("push_to_hub", const.PUSH_TO_HUB)
                push_to_hub_token = kwargs.get("push_to_hub_token", const.HF_TOKEN)
                channel_log = kwargs.get("channel_log", const.CHANNEL_LOGS)

                training_arguments.setdefault("lora", False)
                training_arguments.setdefault("pretrained_model_name_or_path", model_id)
                training_arguments.setdefault("resolution", const.RESOLUTION)
                training_arguments.setdefault("instance_prompt", const.PROMPT)

                log_queue, _ = start_queue(channel_log)
                write_log(log_queue)

                HfFolder.save_token(push_to_hub_token)
                login(token=push_to_hub_token)
                if len(wandb_api_key) > 0 and wandb_api_key != const.WANDB_TOKEN:
                    wandb.login("allow", wandb_api_key)

                os.environ["TORCH_USE_CUDA_DSA"] = "1"

                def func_train_model():
                    project = connect_project(host_name, token, project_id)
                    print("Connect project:", project)

                    zip_dir = os.path.join(clone_dir, "data_zip")
                    extract_dir = os.path.join(clone_dir, "extract")
                    os.makedirs(zip_dir, exist_ok=True)
                    os.makedirs(extract_dir, exist_ok=True)

                    dataset_name = training_arguments.get("dataset_name", const.DATASET)
                    if dataset_name and dataset_id is None:
                        training_arguments["dataset_name"] = dataset_name

                    # only process dataset from s3. hf dataset is processed inside train_dreambooth_... .py
                    # works only for instance_prompt, prior-preservation loss method should be done differently
                    if dataset_id and isinstance(dataset_id, int):
                        project = connect_project(host_name, token, project_id)
                        dataset_name = download_dataset(project, dataset_id, zip_dir)
                        print(dataset_name)
                        if dataset_name:
                            data_zip_dir = os.path.join(zip_dir, dataset_name)
                            with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                utils.clean_folder(extract_dir)
                                zip_ref.extractall(path=extract_dir)

                        # special handle for exported s3 json file
                        json_file, json_file_dir = utils.get_first_json_file(
                            extract_dir
                        )
                        if json_file and utils.is_platform_json_file(
                            json_file, json_file_dir.parent
                        ):
                            with open(json_file_dir) as f:
                                jsonl_1 = json.load(f)
                                jsonl_2 = [
                                    {
                                        "image": data["data"].get("images"),
                                        "prompt": data.get("prompt"),
                                    }
                                    for data in jsonl_1
                                ]
                                with open(json_file_dir, "w") as f:
                                    json.dump(jsonl_2, f)
                                print("modified json to usable format")

                        dataset_name = dataset_name.replace(".zip", "")
                        try:
                            ds = load_dataset("imagefolder", data_dir=extract_dir)
                        except Exception as e:
                            ds = load_dataset(extract_dir)

                        dataset_dir = const.DATASETS_DIR.joinpath(str(dataset_name))
                        dataset_dir.mkdir(parents=True, exist_ok=True)
                        folder_list = utils.create_local_dataset(
                            ds, dataset_dir, training_arguments
                        )
                        training_arguments["instance_data_dir"] = str(
                            dataset_dir.joinpath("images")
                        )

                    output_dir = const.OUTPUTS_DIR.joinpath(dataset_name)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    training_arguments["output_dir"] = str(output_dir)

                    if framework == "huggingface":
                        print("torch.cuda.device_count()", torch.cuda.device_count())
                        if world_size > 1:
                            if int(rank) == 0:
                                print("master node")
                            else:
                                print("worker node")

                        if torch.cuda.device_count() > 1:  # multi gpu
                            compute_mode = "--multi_gpu"
                            n_process = world_size * torch.cuda.device_count()

                        elif torch.cuda.device_count() == 1:  # 1 gpu
                            compute_mode = ""
                            n_process = world_size * torch.cuda.device_count()

                        else:  # no gpu
                            compute_mode = "--cpu"
                            n_process = torch.get_num_threads()

                        if training_arguments["lora"] is False:
                            filtered_configs = utils.filter_config_arguments(
                                training_arguments, TrainingConfigFlux
                            )
                        else:
                            filtered_configs = utils.filter_config_arguments(
                                training_arguments, TrainingConfigFluxLora
                            )

                        json_file = const.PROJ_DIR.joinpath(const.JSON_TRAINING_ARGS)
                        with open(json_file, "w") as f:
                            json.dump(asdict(filtered_configs), f)

                        #  --dynamo_backend 'no' \
                        # --rdzv_backend c10d
                        command = (
                            "accelerate launch {compute_mode} \
                                --main_process_ip {head_node_ip} \
                                --main_process_port {port} \
                                --num_machines {SLURM_NNODES} \
                                --num_processes {num_processes}\
                                --machine_rank {rank} \
                                --num_cpu_threads_per_process 1 \
                                {file_name} \
                                --training_args_json {json_file} \
                                {push_to_hub} \
                                --hub_token {push_to_hub_token} \
                                --channel_log {channel_log} "
                        ).format(
                            file_name=(
                                "./train_dreambooth_flux.py"
                                if not training_arguments["lora"]
                                else "./train_dreambooth_lora_flux.py"
                            ),
                            compute_mode=compute_mode,
                            head_node_ip=master_add,
                            port=master_port,
                            SLURM_NNODES=world_size,
                            num_processes=n_process,
                            rank=rank,
                            json_file=str(json_file),
                            push_to_hub="--push_to_hub" if push_to_hub else "",
                            push_to_hub_token=push_to_hub_token,
                            channel_log=channel_log,
                        )

                        command = " ".join(command.split())
                        print(command)
                        subprocess.run(
                            command,
                            shell=True,
                            # capture_output=True, text=True).stdout.strip("\n")
                        )

                    else:
                        raise Exception("Unimplemented framework behavior:", framework)

                    print(push_to_hub)
                    if push_to_hub:
                        import datetime

                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y%m%d")
                        time_str = now.strftime("%H%M%S")
                        version = f"{date_str}-{time_str}"
                        upload_checkpoint_mixed_folder(project, version, output_dir)

                import threading

                train_thread = threading.Thread(target=func_train_model)
                train_thread.start()
                return {"message": "train started successfully"}

            except Exception as e:
                return {"message": f"train failed: {e}"}

        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "./train_dreambooth_flux.py"])
            return {"message": "train stop successfully", "result": "Done"}

        elif command.lower() == "tensorboard":

            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(
                    f"tensorboard --logdir /app/data/logs --host 0.0.0.0 --port=6006",
                    stdout=subprocess.PIPE,
                    stderr=None,
                    shell=True,
                )
                out = p.communicate()
                print(out)

            import threading

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        # region Predict
        elif command.lower() == "predict":

            def unload_and_load_model(
                task: str,
                load_lora: bool,
                lora_model_name: str,
                lora_weight_name: str,
                model_id: str,
                ip_adapter_name: str,
                ip_adapter_weight_name: str,
            ):
                global pipe_predict
                # clear VRAM
                del pipe_predict
                gc.collect()
                torch.cuda.empty_cache()
                # Load model
                if task.lower() == "text to image":
                    pipe_predict = FluxPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                    if load_lora:
                        pipe_predict.load_lora_weights(
                            lora_model_name,
                            weight_name=lora_weight_name,
                            adapter_name="custom_lora",
                        )
                        pipe_predict.set_adapters(
                            ["custom_lora"], adapter_weights=[1.0]
                        )
                    pipe_predict.enable_model_cpu_offload()
                elif task.lower() == "depth control":
                    pipe_predict = FluxControlPipeline(
                        model_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                    if load_lora:
                        pipe_predict.load_lora_weights(
                            lora_model_name,
                            weight_name=lora_weight_name,
                            adapter_name="custom_lora",
                        )
                        pipe_predict.set_adapters(
                            ["custom_lora"], adapter_weights=[1.0]
                        )
                    pipe_predict.enable_model_cpu_offload()
                elif task.lower() == "IP Adapter":
                    pipe_predict = FluxPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")
                    if load_lora:
                        pipe_predict.load_lora_weights(
                            lora_model_name,
                            weight_name=lora_weight_name,
                            adapter_name="custom_lora",
                        )
                        pipe_predict.set_adapters(
                            ["custom_lora"], adapter_weights=[1.0]
                        )
                    pipe_predict.load_ip_adapter(
                        ip_adapter_name,
                        weight_name=ip_adapter_weight_name,
                        image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
                    )
                    pipe_predict.set_ip_adapter_scale(1.0)
                    pipe_predict.enable_model_cpu_offload()
                return pipe_predict

            def predict_flux(
                pipe_predict,
                task,
                prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                image_control,
                negative_prompt,
            ):
                if task.lower() == "text to image":
                    image = pipe_predict(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        # max_sequence_length=max_sequence_length,
                    ).images[0]
                elif task.lower() == "depth control":
                    processor = DepthPreprocessor.from_pretrained(
                        "LiheYoung/depth-anything-large-hf"
                    )
                    control_image = processor(image_control)[0].convert("RGB")
                    image = pipe_predict(
                        prompt=prompt,
                        control_image=control_image,
                        height=width,
                        width=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator().manual_seed(42),
                    ).images[0]

                elif task.lower() == "IP Adapter":
                    control_image = load_image(image_control)
                    image = pipe(
                        width=width,
                        height=height,
                        prompt=prompt,
                        negative_prompt="",
                        true_cfg_scale=4.0,
                        generator=torch.Generator().manual_seed(4444),
                        ip_adapter_image=control_image,
                    ).images[0]
                return image

            try:
                prompt = kwargs.get("prompt", None)
                model_id = kwargs.get("model_id", "black-forest-labs/FLUX.1-dev")
                chkpt_name = kwargs.get("checkpoint", None)
                width = kwargs.get("width", 1024)
                height = kwargs.get("height", 1024)
                num_inference_steps = kwargs.get("num_inference_steps", 4)
                guidance_scale = kwargs.get("guidance_scale", 2)
                format = kwargs.get("format", "JPEG")
                task = kwargs.get("task", "text to image")
                load_lora = kwargs.get("load_lora", False)
                lora_model_name = kwargs.get("lora_model_name", None)
                lora_weight_name = kwargs.get("lora_weight_name", None)
                ip_adapter_name = kwargs.get("ip_adapter_name", None)
                ip_adapter_weight_name = kwargs.get("ip_adapter_weight_name", None)
                img_base64 = kwargs.get("img_base64", None)
                negative_prompt = kwargs.get("negative_prompt", None)
                if img_base64.startswith("data:image"):
                    img_base64 = img_base64.split(",")[1]

                image_bytes = base64.b64decode(img_base64)
                image_input = Image.open(io.BytesIO(image_bytes))
                image_input = load_image(image)
                if prompt == "" or prompt is None:
                    return None, ""

                pipe = unload_and_load_model(
                    task=task,
                    load_lora=load_lora,
                    lora_model_name=lora_model_name,
                    lora_weight_name=lora_weight_name,
                    model_id=model_id,
                    ip_adapter_name=ip_adapter_name,
                    ip_adapter_weight_name=ip_adapter_weight_name,
                )
                image = predict_flux(
                    pipe_predict=pipe,
                    task=task,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_control=image_input,
                    negative_prompt=negative_prompt,
                )
                buffered = BytesIO()
                image.save(buffered, format=format)
                image.save(const.PROJ_DIR.joinpath(f"image.{format}"))
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                generated_url = f"/downloads?path=image.{format}"

                result = {
                    "model_version": model_id,
                    "result": {
                        "format": format,
                        "image": img_base64,
                        "image_url": generated_url,
                    },
                }

                return {"message": "predict completed successfully", "result": result}

            except Exception as e:
                print(e)
                return {"message": "predict failed", "result": None}

        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "text-to-image":
                prompt_text = f"""
                A planet, yarn art style
                """

            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        else:
            return {"message": "command not supported", "result": None}

            # return {"message": "train completed successfully"}

    def model(self, **kwargs):
        from gradio_demo import demo

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )

        return {"share_url": share_url, "local_url": local_url}

    # deprecated?
    def model_trial(self, project, **kwargs):
        import gradio as gr

        return {"message": "Done", "result": "Done"}

    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)

    def preview(self):
        pass

    def toolbar_predict(self):
        pass

    def toolbar_predict_sam(self):
        pass
