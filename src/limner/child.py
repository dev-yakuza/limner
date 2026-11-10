"""One stage of generation in its own process (spec §6.3).

    python -m limner.child <stage.json> <events.jsonl>

stdin is the supervisor's lifeline: EOF (the supervisor died) ends this process at once, so no
child ever runs without the locks. Events go to the events file as JSON lines (`start`, `done`,
`error`, `environment`); stdout is redirected to stderr (the log) because mflux prints to it.

The mflux path follows script2video `imagegen/run.py` (MIT, same author): load once, encode every
prompt of the stage, drop the text encoder (`qwen_vl_encoder` and `text_encoder`, gc, clear_cache)
and denoise with the transformer alone (peak 22 GB instead of 37.7 GB). Unlike s2v, the encoded
prompts are looked up in a dict by the exact arguments mflux passes (a subclass method, not a
lambda on the instance), so a missing entry raises instead of reusing the previous embedding.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

from . import util

EXIT_CANCELLED = 3
EXIT_ENVIRONMENT = 2

# the mflux internals used (checked before loading; spec §6.3)
INTERNAL_METHODS = ("_compute_dimensions", "_encode_prompts_with_images", "generate_image")
INTERNAL_ATTRS = ("qwen_vl_encoder", "text_encoder")


class Events:
    def __init__(self, path: str):
        self.path = path

    def emit(self, event: str, **fields) -> None:
        line = json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def exit_with_parent() -> None:
    """The lifeline (s2v `models.exit_with_parent`): stdin is a pipe from the supervisor."""

    def watch() -> None:
        try:
            while os.read(0, 4096):
                pass
        except OSError:
            pass
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def cancelled(spec: dict) -> bool:
    return os.path.exists(spec.get("cancel_file") or "")


def save_png(image, path: str) -> None:
    """A new PNG without metadata (Pillow re-save; mflux's own save adds its parameters)."""
    import io

    from PIL import Image

    from . import paths

    im = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    paths.write_new_file(path, buf.getvalue())


# --------------------------------------------------------------------------- fake


def run_fake(spec: dict, ev: Events) -> int:
    from PIL import Image

    secs = float(os.environ.get("LIMNER_FAKE_SECONDS", "0") or 0)
    for j in spec["jobs"]:
        if cancelled(spec):
            return EXIT_CANCELLED
        ev.emit("start", id=j["id"])
        started = util.utc_now()
        t = time.time()
        if os.environ.get("LIMNER_FAKE_ABORT_ID") == j["id"]:
            os.abort()
        if os.environ.get("LIMNER_FAKE_ENV_ERROR"):
            ev.emit("environment", message="fake environment failure")
            return EXIT_ENVIRONMENT
        end = time.time() + secs
        while time.time() < end:
            time.sleep(min(0.05, end - time.time()))
        if os.environ.get("LIMNER_FAKE_ERROR_ID") == j["id"]:
            ev.emit("error", id=j["id"], message="fake generator error")
            continue
        h = j["job_sha"]
        color = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        save_png(Image.new("RGB", (j["width"], j["height"]), color), j["tmp_out"])
        ev.emit("done", id=j["id"], tmp_out=j["tmp_out"], seconds=round(time.time() - t, 2),
                started_at=started, peak_gib=0.0)
    return 0


# --------------------------------------------------------------------------- mflux


class EnvError(RuntimeError):
    pass


def check_internals(cls, util_cls) -> None:
    gone = [n for n in INTERNAL_METHODS if not callable(getattr(cls, n, None))]
    if not callable(getattr(util_cls, "create_image_conditioning_latents", None)):
        gone.append("QwenEditUtil.create_image_conditioning_latents")
    if gone:
        raise EnvError("this mflux lacks %s (limner needs mflux 0.22.0)" % ", ".join(gone))


def load_model(spec: dict):
    os.environ["HF_HUB_OFFLINE"] = "1"  # before huggingface_hub is imported (by mflux)
    try:
        import mlx.core as mx
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    except ImportError as e:
        raise EnvError("mflux is not the pinned version (%s)" % e) from e
    check_internals(QwenImageEdit, QwenEditUtil)
    gen = spec["generation"]
    try:  # the internals of the adopted spike options (S1 picture_prefix, S2 skip_negative)
        import inspect

        from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        if gen.get("multi_ref") == "picture_prefix" and \
                "use_picture_prefix" not in inspect.signature(QwenVisionLanguageTokenizer).parameters:
            raise EnvError("this mflux tokenizer has no use_picture_prefix")
        if gen.get("skip_negative") and not callable(getattr(QwenImage, "compute_guided_noise", None)):
            raise EnvError("this mflux lacks QwenImage.compute_guided_noise")
    except ImportError as e:
        raise EnvError("mflux is not the pinned version (%s)" % e) from e

    from . import models

    lock = models.load_lock()
    missing = models.generation_ready(lock)
    if missing:
        raise EnvError("models not ready: " + "; ".join(missing))
    mx.set_cache_limit(int(gen["cache_limit_bytes"]))

    embeddings: dict = {}

    class CachedEdit(QwenImageEdit):
        """`_encode_prompts_with_images` reads the dict filled before the encoder was dropped."""

        def _encode_prompts_with_images(self, prompt, negative_prompt, image_paths, config, vl_width=None,
                                        vl_height=None):
            key = (prompt, tuple(image_paths), vl_width, vl_height, negative_prompt)
            try:
                return embeddings[key]
            except KeyError:
                raise RuntimeError("limner: no encoded prompt for this job (key mismatch)") from None

    model = CachedEdit(model_config=ModelConfig.qwen_image_edit_2511(), quantize=None,
                       model_path=models.quantized_path(lock), lora_paths=[models.lora_file(lock)],
                       lora_scales=[1.0], bake_lora=False)
    gone = [a for a in INTERNAL_ATTRS if getattr(model, a, None) is None]
    if gone:
        raise EnvError("this mflux model has no %s" % ", ".join(gone))
    model.tiling_config = TilingConfig()
    if gen.get("skip_negative"):
        enable_skip_negative()
    return model, embeddings, QwenImageEdit


def picture_prefix_tokenizer(model):
    """Spike S1: the VL tokenizer with one `Picture N:` image slot per reference (mflux 0.22.0
    fixes `use_picture_prefix=False`, one slot, so a second reference never reaches the VL encoder)."""
    from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer

    old = model.tokenizers["qwen_vl"]
    return QwenVisionLanguageTokenizer(processor=old.processor, max_length=old.max_length, use_picture_prefix=True)


def enable_skip_negative() -> None:
    """Spike S2: with guidance 1.0, `compute_guided_noise` returns the positive prediction (up to
    a 1e-12 term), so return it directly; MLX is lazy, so the negative transformer pass that only
    fed this function is never evaluated."""
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    original = QwenImage.compute_guided_noise

    def guided(noise, noise_negative, guidance):
        if guidance == 1.0:
            return noise
        return original(noise, noise_negative, guidance)

    QwenImage.compute_guided_noise = staticmethod(guided)


def use_prefix(gen: dict, images: list[str]) -> bool:
    return gen.get("multi_ref") == "picture_prefix" and len(images) > 1


def run_mflux(spec: dict, ev: Events) -> int:
    import gc

    gen = spec["generation"]
    try:
        model, embeddings, base = load_model(spec)
    except EnvError as e:
        ev.emit("environment", message=str(e))
        return EXIT_ENVIRONMENT
    import mlx.core as mx

    neg = gen["negative_prompt"]
    plain_tok = model.tokenizers["qwen_vl"]
    prefix_tok = picture_prefix_tokenizer(model) if gen.get("multi_ref") == "picture_prefix" else None
    ok = []
    for j in spec["jobs"]:
        model.tokenizers["qwen_vl"] = prefix_tok if prefix_tok and len(j["images"]) > 1 else plain_tok
        try:
            config, vl_w, vl_h, _, _ = model._compute_dimensions(
                image_paths=j["images"], num_inference_steps=gen["steps"], height=j["height"], width=j["width"],
                guidance=gen["guidance"], image_path=j["images"][0], scheduler=gen["scheduler"])
            emb = base._encode_prompts_with_images(model, prompt=j["prompt"], negative_prompt=neg,
                                                   image_paths=j["images"], config=config, vl_width=vl_w,
                                                   vl_height=vl_h)
            mx.eval(*emb)
            j["_key"] = (j["prompt"], tuple(j["images"]), vl_w, vl_h, neg)
            embeddings[j["_key"]] = emb
            ok.append(j)
        except Exception as e:  # noqa: BLE001 - this job fails, the others go on
            traceback.print_exc()
            ev.emit("error", id=j["id"], message="encoding: %s" % e)
    model.tokenizers["qwen_vl"] = plain_tok
    model.qwen_vl_encoder = None
    model.text_encoder = None  # the same weights: both references must go
    gc.collect()
    mx.clear_cache()
    for j in ok:
        if cancelled(spec):
            return EXIT_CANCELLED
        ev.emit("start", id=j["id"])
        started = util.utc_now()
        mx.reset_peak_memory()
        t = time.time()
        try:
            image = model.generate_image(
                seed=j["seed"], prompt=j["prompt"], negative_prompt=neg, width=j["width"], height=j["height"],
                guidance=gen["guidance"], image_path=j["images"][0], image_paths=j["images"],
                num_inference_steps=gen["steps"], scheduler=gen["scheduler"]).image
            save_png(image, j["tmp_out"])
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            ev.emit("error", id=j["id"], message=str(e))
            continue
        finally:
            embeddings.pop(j["_key"], None)
        ev.emit("done", id=j["id"], tmp_out=j["tmp_out"], seconds=round(time.time() - t, 1), started_at=started,
                peak_gib=round(mx.get_peak_memory() / 2**30, 1))
        gc.collect()
        mx.clear_cache()
    return 0


def main(argv: list[str]) -> int:
    util.reset_signals()
    exit_with_parent()
    os.dup2(2, 1)  # mflux prints to stdout: send it to the log
    stage_file, events_file = argv
    spec = util.load_json(stage_file)
    ev = Events(events_file)
    if spec.get("fake"):
        return run_fake(spec, ev)
    return run_mflux(spec, ev)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
