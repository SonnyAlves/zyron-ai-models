import time
import torch
import nemo

def main():
    print("=== Zyron / NeMo GPU smoke test ===")
    print("Torch version :", torch.__version__)
    print("NeMo version  :", nemo.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA non disponible dans cet environnement, test arrêté.")

    device = torch.device("cuda")
    print("GPU :", torch.cuda.get_device_name(0))

    # Gros matmul sur le GPU
    n = 8192
    iters = 20
    print(f"Lancement de {iters} matmuls {n}x{n} sur le GPU...")

    x = torch.randn((n, n), device=device)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(iters):
        y = x @ x  # juste pour forcer du compute

    torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"✅ GPU OK – {iters} matmuls {n}x{n} terminés en {dt:.2f} secondes")

if __name__ == "__main__":
    main()

