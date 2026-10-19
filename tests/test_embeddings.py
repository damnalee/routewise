import numpy as np

from routewise.embeddings import HashEmbedder


def test_vectors_are_normalised_and_shaped() -> None:
    emb = HashEmbedder(dim=128)
    out = emb.embed(["hello world", "another text"])
    assert out.shape == (2, 128)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embedding_is_deterministic() -> None:
    emb = HashEmbedder()
    assert np.array_equal(emb.embed_one("same text"), emb.embed_one("same text"))


def test_similar_texts_score_higher_than_unrelated() -> None:
    emb = HashEmbedder()
    a = emb.embed_one("how do I reset my password")
    b = emb.embed_one("how to reset password")
    c = emb.embed_one("quarterly revenue forecast spreadsheet")
    assert float(a @ b) > float(a @ c)


async def test_async_embed_matches_sync() -> None:
    emb = HashEmbedder()
    sync = emb.embed(["abc def"])
    assert np.array_equal(await emb.aembed(["abc def"]), sync)
