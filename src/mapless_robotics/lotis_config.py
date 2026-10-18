"""LoTIS configuration paths shared by simulator and handheld launchers."""
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOTIS_ROOT = PROJECT_ROOT / "external/lotis"
LOTIS_CONFIG = LOTIS_ROOT / "nav_eval/configs/lotis_method.yaml"

def load_lotis_config(
    config_path=LOTIS_CONFIG, *, checkpoint=None, model_config=None, dinov3_weights=None
):
    """Resolve the bundled checkout layout or paths relative to a custom YAML.

    CLI paths and environment variables are relative to the working directory.
    The bundled method YAML uses paths relative to the LoTIS checkout, except
    its DINOv3 repo, which lives alongside LoTIS under external/.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LoTIS method config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("LoTIS method config must contain a YAML mapping")
    bundled = config_path == LOTIS_CONFIG.resolve()
    base = LOTIS_ROOT if bundled else config_path.parent

    def resolved(value, root):
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"Expected a model path, got {value!r}")
        path = Path(value).expanduser()
        return str((root / path).resolve())

    for key, override, default in (
        ("checkpoint_path", checkpoint, "final_model.pth"),
        ("config_path", model_config, "final_config.yaml"),
    ):
        config[key] = resolved(
            override if override is not None else config.get(key, default),
            Path.cwd() if override is not None else base,
        )

    repo_env = os.environ.get("DINOV3_REPO")
    repo_base = PROJECT_ROOT / "external" if bundled else base
    config["dinov3_repo"] = resolved(
        repo_env or config.get("dinov3_repo", "./dinov3"),
        Path.cwd() if repo_env else repo_base,
    )
    weights = (
        dinov3_weights
        or config.get("dinov3_weights")
        or os.environ.get("DINOV3_WEIGHTS")
    )
    weights_base = (
        Path.cwd()
        if dinov3_weights is not None or not config.get("dinov3_weights")
        else base
    )
    config["dinov3_weights"] = resolved(weights, weights_base) if weights else None

    missing = []
    for key, label, option in (
        ("checkpoint_path", "LoTIS checkpoint", "--checkpoint"),
        ("config_path", "LoTIS model config", "--model-config"),
        (
            "dinov3_weights",
            "DINOv3 ViT-B/16 weights",
            "--dinov3-weights or DINOV3_WEIGHTS",
        ),
    ):
        if config[key] is None or not Path(config[key]).is_file():
            missing.append(f"  {label}: {config[key] or 'not configured'} ({option})")
    if not (Path(config["dinov3_repo"]) / "hubconf.py").is_file():
        missing.append(f"  DINOv3 repository: {config['dinov3_repo']} (DINOV3_REPO)")
    if missing:
        raise FileNotFoundError(
            "Missing teach-and-repeat model files:\n" + "\n".join(missing)
        )
    config["use_geodesic_follower"] = False
    return config

