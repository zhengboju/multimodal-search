import torch
import torch.nn.functional as F

from mmsearch.index.compress import Int8Quant, KMeansPool, PCAReduce, TokenPrune, compress
from mmsearch.index.indexes import ExhaustiveIndex, HierarchicalIndex, PLAIDLiteIndex
from mmsearch.index.store import MultiVectorStore
from mmsearch.models.colpali import pad_multivectors, score_multivector
from mmsearch.models.losses import maxsim_scores


def rand_mv(n, lo=5, hi=40, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        L = int(torch.randint(lo, hi, (1,), generator=g))
        out.append(F.normalize(torch.randn(L, d, generator=g), dim=-1))
    return out


def naive_maxsim(q, d):
    return float((q @ d.t()).max(dim=1).values.sum())


def test_maxsim_matches_naive_with_padding():
    qs, ds = rand_mv(3, seed=1), rand_mv(7, seed=2)
    q, qm = pad_multivectors(qs, "cpu", torch.float32)
    d, dm = pad_multivectors(ds, "cpu", torch.float32)
    s = maxsim_scores(q, qm, d, dm, doc_chunk=3)
    for i in range(3):
        for j in range(7):
            assert abs(s[i, j].item() - naive_maxsim(qs[i], ds[j])) < 1e-4


def make_store(n=60, seed=3):
    docs = rand_mv(n, seed=seed)
    return docs, MultiVectorStore.from_list(docs, [f"d{i}" for i in range(n)])


def test_store_roundtrip(tmp_path):
    docs, store = make_store()
    store.save(tmp_path / "s.pt")
    s2 = MultiVectorStore.load(tmp_path / "s.pt")
    assert s2.doc_ids == store.doc_ids
    assert torch.allclose(s2.get(5).float(), docs[5], atol=1e-3)
    pad, mask = s2.padded(idx=[1, 2])
    assert pad.shape[0] == 2 and mask[0].sum() == docs[1].shape[0]


def test_exhaustive_index_exact():
    docs, store = make_store()
    qs = rand_mv(5, seed=9)
    ref = score_multivector(qs, [d.half().float() for d in docs])
    idx = ExhaustiveIndex(store, device="cpu")
    I, S = idx.search(qs, 5)
    for i in range(5):
        assert I[i] == ref[i].topk(5).indices.tolist()


def test_plaid_full_probe_equals_exhaustive():
    docs, store = make_store()
    qs = rand_mv(4, seed=11)
    ex = ExhaustiveIndex(store, device="cpu").search(qs, 5)[0]
    pl = PLAIDLiteIndex(store, n_centroids=16, nprobe=16, n_rerank=len(store), device="cpu")
    assert pl.search(qs, 5)[0] == ex
    approx = PLAIDLiteIndex(store, n_centroids=16, nprobe=2, exact_rerank=False, device="cpu")
    I, _ = approx.search(qs, 5)
    assert all(len(x) <= 5 for x in I)
    assert pl.nbytes() < store.nbytes()


def test_hierarchical_full_candidates_equals_exhaustive():
    docs, store = make_store()
    qs = rand_mv(4, seed=12)
    ex = ExhaustiveIndex(store, device="cpu").search(qs, 5)[0]
    h = HierarchicalIndex(store, coarse_k=4, n_candidates=len(store), device="cpu")
    assert h.search(qs, 5)[0] == ex


def test_compressors():
    docs, store = make_store()
    km = KMeansPool(k=8, device="cpu").docs(store)
    assert int(km.lengths.max()) <= max(8, 0) or all(l <= 8 or l == store.lengths[i] for i, l in enumerate(km.lengths.tolist()))
    assert all(l <= 8 for l, L in zip(km.lengths.tolist(), store.lengths.tolist()) if L > 8)
    pca = PCAReduce(dim=8).fit(store)
    ps = pca.docs(store)
    assert ps.dim == 8 and pca.query(docs[0]).shape[1] == 8
    pr = TokenPrune(keep=0.5).docs(store)
    assert all(l == max(1, round(0.5 * L)) for l, L in zip(pr.lengths.tolist(), store.lengths.tolist()))
    q8 = Int8Quant().docs(store)
    assert q8.data.dtype == torch.int8 and q8.nbytes() < store.nbytes()
    assert (q8.get(3).float() - store.get(3).float()).abs().max() < 0.02
    chained, comp = compress(store, [{"type": "pca", "dim": 8}, {"type": "int8"}])
    assert chained.dim == 8 and comp.query(docs[0]).shape[1] == 8
