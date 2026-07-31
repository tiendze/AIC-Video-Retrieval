import httpx
import pytest

ES_URL = "http://localhost:9201"


@pytest.mark.integration
def test_vi_analyzer_tokenizes_by_compound_word_not_syllable() -> None:
    response = httpx.post(
        f"{ES_URL}/_analyze",
        json={"analyzer": "vi_analyzer", "text": "Cộng hòa Xã hội chủ nghĩa Việt Nam"},
        timeout=10.0,
    )
    response.raise_for_status()

    tokens = [t["token"] for t in response.json()["tokens"]]

    assert tokens == ["cộng hòa", "xã hội", "chủ nghĩa", "việt nam"]
