def vec_add(v: list[float], w: list[float]) -> list[float]:
    # zip(strict=True) raises ValueError on a length mismatch.
    # No separate `if len(v) != len(w)` branch: the iteration
    # itself is the check (see the note below on ERROR 1).
    return [vi + wi for vi, wi in zip(v, w, strict=True)]

def scale(c: float, v: list[float]) -> list[float]:
    # Each instruction is multiplied by c: same direction (or flipped if c < 0), |c| times as far.
    return [c * vi for vi in v]

if __name__ == "__main__":
    u = [4.0, 1.0, 0.0]
    w = [0.0, 2.0, 5.0]
    household = scale(0.5, vec_add(u, w))   # raw: [2.0, 1.5, 2.5]