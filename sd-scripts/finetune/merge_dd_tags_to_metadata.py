import argparse
import json
import os
from pathlib import Path
from typing import List

import library.train_util as train_util
from library.utils import setup_logging
from tqdm import tqdm

setup_logging()
import logging

logger = logging.getLogger(__name__)


def main(args):
    assert not args.recursive or (
        args.recursive and args.full_path
    ), "recursive requires full_path / recursiveはfull_pathと同時に指定してください"

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[Path] = train_util.glob_images_pathlib(
        train_data_dir_path, args.recursive
    )
    logger.info(f"found {len(image_paths)} images.")

    if args.in_json is None and Path(args.out_json).is_file():
        args.in_json = args.out_json

    if args.in_json is not None:
        logger.info(f"loading existing metadata: {args.in_json}")
        metadata = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
        logger.warning(
            "tags data for existing images will be overwritten / 既存の画像のタグは上書きされます"
        )
    else:
        logger.info(
            "new metadata will be created / 新しいメタデータファイルが作成されます"
        )
        metadata = {}

    logger.info("merge tags to metadata json.")
    for image_path in tqdm(image_paths):
        tags_path = image_path.with_suffix(args.caption_extension)
        tags = tags_path.read_text(encoding="utf-8").strip()

        if not os.path.exists(tags_path):
            tags_path = os.path.join(image_path, args.caption_extension)

        image_key = str(image_path) if args.full_path else image_path.stem
        if image_key not in metadata:
            metadata[image_key] = {}

        metadata[image_key]["tags"] = tags
        if args.debug:
            logger.info(f"{image_key} {tags}")

    # metadataを書き出して終わり
    logger.info(f"writing metadata: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir",
        type=str,
        help="directory for train images / 学習画像データのディレクトリ",
    )
    parser.add_argument(
        "out_json",
        type=str,
        help="metadata file to output / メタデータファイル書き出し先",
    )
    parser.add_argument(
        "--in_json",
        type=str,
        help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）",
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す",
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".txt",
        help="extension of caption (tag) file / 読み込むキャプション（タグ）ファイルの拡張子",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode, print tags")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
