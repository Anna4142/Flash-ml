"""Quick test and demo for flash-knn."""
import torch
from flash_knn import batch_knn_euclid, batch_knn_cosine, FlashKNN

def main():
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, M, D = 4, 2000, 8000, 64
    k = 16
    dtype = torch.float32 if dev == "cpu" else torch.float16
    x = torch.randn(B, N, D, device=dev, dtype=dtype)
    ref = torch.randn(B, M, D, device=dev, dtype=dtype)

    print("=== Euclidean KNN ===")
    dist, idx = batch_knn_euclid(x, ref=ref, k=k)
    print(f"dist shape: {dist.shape}, idx shape: {idx.shape}")
    print(f"dist[0,0,:4]: {dist[0, 0, :4].tolist()}")

    print("\n=== Cosine KNN (normalized) ===")
    x_n = torch.nn.functional.normalize(x.float(), p=2, dim=-1)
    ref_n = torch.nn.functional.normalize(ref.float(), p=2, dim=-1)
    sim, idx_c = batch_knn_cosine(x_n, ref=ref_n, k=k)
    print(f"sim shape: {sim.shape}, idx shape: {idx_c.shape}")

    print("\n=== FlashKNN class ===")
    knn = FlashKNN(k=k, metric="euclidean")
    knn.fit(ref)
    dist2, idx2 = knn.kneighbors(x)
    print(f"Match euclidean: {torch.allclose(dist, dist2) and (idx == idx2).all()}")

    print("\nDone.")

if __name__ == "__main__":
    main()
