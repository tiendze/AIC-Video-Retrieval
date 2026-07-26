from pathlib import Path

import yaml

COMPOSE_PATH = Path(__file__).parent.parent / "docker-compose.yml"


def load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_docker_compose_file_exists():
    assert COMPOSE_PATH.exists()


def test_every_pulled_image_is_pinned_to_an_exact_version():
    compose = load_compose()
    services = compose["services"]

    for name, service in services.items():
        image = service.get("image")
        if image is None:
            continue  # built locally (e.g. elasticsearch), no pulled tag to pin
        assert ":" in image, f"{name} must pin an explicit tag, got '{image}'"
        tag = image.rsplit(":", 1)[1]
        assert tag not in {"latest", "master", "beta"}, (
            f"{name} uses a moving tag '{tag}', pin an exact version instead"
        )


def test_elasticsearch_builds_custom_image_with_vietnamese_plugin():
    compose = load_compose()
    es_service = compose["services"]["elasticsearch"]

    assert "build" in es_service, "elasticsearch must build a custom image (plugin install)"
    dockerfile_dir = Path(__file__).parent.parent / es_service["build"]["context"]
    assert (dockerfile_dir / "Dockerfile").exists()


def test_elasticsearch_heap_is_hard_capped():
    compose = load_compose()
    es_env = compose["services"]["elasticsearch"]["environment"]

    assert es_env.get("ES_JAVA_OPTS") == "-Xms2g -Xmx2g"


def test_milvus_standalone_depends_on_etcd_and_minio():
    compose = load_compose()
    standalone = compose["services"]["standalone"]

    assert set(standalone["depends_on"]) == {"etcd", "minio"}
