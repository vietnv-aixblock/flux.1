import argparse

import library.train_util as train_util
import torch
from library import model_util
from library.device_utils import get_preferred_device, init_ipex
from tqdm import tqdm
from transformers import CLIPTokenizer

init_ipex()

import library.model_util as model_util
import lora
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う

DEVICE = get_preferred_device()


def interrogate(args):
    weights_dtype = torch.float16

    # いろいろ準備する
    logger.info(f"loading SD model: {args.sd_model}")
    args.pretrained_model_name_or_path = args.sd_model
    args.vae = None
    text_encoder, vae, unet, _ = train_util._load_target_model(
        args, weights_dtype, DEVICE
    )

    logger.info(f"loading LoRA: {args.model}")
    network, weights_sd = lora.create_network_from_weights(
        1.0, args.model, vae, text_encoder, unet
    )

    # text encoder向けの重みがあるかチェックする：本当はlora側でやるのがいい
    has_te_weight = False
    for key in weights_sd.keys():
        if "lora_te" in key:
            has_te_weight = True
            break
    if not has_te_weight:
        logger.error(
            "This LoRA does not have modules for Text Encoder, cannot interrogate / このLoRAはText Encoder向けのモジュールがないため調査できません"
        )
        return
    del vae

    logger.info("loading tokenizer")
    if args.v2:
        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer"
        )
    else:
        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_PATH
        )  # , model_max_length=max_token_length + 2)

    text_encoder.to(DEVICE, dtype=weights_dtype)
    text_encoder.eval()
    unet.to(DEVICE, dtype=weights_dtype)
    unet.eval()  # U-Netは呼び出さないので不要だけど

    # トークンをひとつひとつ当たっていく
    token_id_start = 0
    token_id_end = max(tokenizer.all_special_ids)
    logger.info(f"interrogate tokens are: {token_id_start} to {token_id_end}")

    def get_all_embeddings(text_encoder):
        embs = []
        with torch.no_grad():
            for token_id in tqdm(
                range(token_id_start, token_id_end + 1, args.batch_size)
            ):
                batch = []
                for tid in range(
                    token_id, min(token_id_end + 1, token_id + args.batch_size)
                ):
                    tokens = [tokenizer.bos_token_id, tid, tokenizer.eos_token_id]
                    # tokens = [tid]                                                    # こちらは結果がいまひとつ
                    batch.append(tokens)

                # batch_embs = text_encoder(torch.tensor(batch).to(DEVICE))[0].to("cpu")  # bos/eosも含めたほうが差が出るようだ [:, 1]
                # clip skip対応
                batch = torch.tensor(batch).to(DEVICE)
                if args.clip_skip is None:
                    encoder_hidden_states = text_encoder(batch)[0]
                else:
                    enc_out = text_encoder(
                        batch, output_hidden_states=True, return_dict=True
                    )
                    encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(
                        encoder_hidden_states
                    )
                encoder_hidden_states = encoder_hidden_states.to("cpu")

                embs.extend(encoder_hidden_states)
        return torch.stack(embs)

    logger.info("get original text encoder embeddings.")
    orig_embs = get_all_embeddings(text_encoder)

    network.apply_to(text_encoder, unet, True, len(network.unet_loras) > 0)
    info = network.load_state_dict(weights_sd, strict=False)
    logger.info(f"Loading LoRA weights: {info}")

    network.to(DEVICE, dtype=weights_dtype)
    network.eval()

    del unet

    logger.info(
        "You can ignore warning messages start with '_IncompatibleKeys' (LoRA model does not have alpha because trained by older script) / '_IncompatibleKeys'の警告は無視して構いません（以前のスクリプトで学習されたLoRAモデルのためalphaの定義がありません）"
    )
    logger.info("get text encoder embeddings with lora.")
    lora_embs = get_all_embeddings(text_encoder)

    # 比べる：とりあえず単純に差分の絶対値で
    logger.info("comparing...")
    diffs = {}
    for i, (orig_emb, lora_emb) in enumerate(zip(orig_embs, tqdm(lora_embs))):
        diff = torch.mean(torch.abs(orig_emb - lora_emb))
        # diff = torch.mean(torch.cosine_similarity(orig_emb, lora_emb, dim=1))       # うまく検出できない
        diff = float(diff.detach().to("cpu").numpy())
        diffs[token_id_start + i] = diff

    diffs_sorted = sorted(diffs.items(), key=lambda x: -x[1])

    # 結果を表示する
    print("top 100:")
    for i, (token, diff) in enumerate(diffs_sorted[:100]):
        # if diff < 1e-6:
        #   break
        string = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens([token])
        )
        print(f"[{i:3d}]: {token:5d} {string:<20s}: {diff:.5f}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v2",
        action="store_true",
        help="load Stable Diffusion v2.x model / Stable Diffusion 2.xのモデルを読み込む",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
        help="Stable Diffusion model to load: ckpt or safetensors file / 読み込むSDのモデル、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LoRA model to interrogate: ckpt or safetensors file / 調査するLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for processing with Text Encoder / Text Encoderで処理するときのバッチサイズ",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    interrogate(args)
