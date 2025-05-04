import re
from typing import Callable, Tuple
import numpy as np
import itertools
import inspect

# === ВСПОМОГАТЕЛЬНЫЕ ===

def truth_table_from_func(func: Callable[[Tuple[int, ...]], int], n: int) -> np.ndarray:
    return np.array([func(bits) for bits in itertools.product([0, 1], repeat=n)], dtype=np.uint8)

def popcount(x: int) -> int:
    return bin(x).count('1')

def describe_function(name: str, func: Callable) -> str:
    try:
        source = inspect.getsource(func).strip()

        match = re.match(r".*lambda\s+\w+\s*:\s*(.*)", source)
        if match:
            expr = match.group(1)
        elif "=" in source:
            _, expr = source.split("=", 1)
            expr = expr.strip()
        else:
            return f"{name}(x)"

        expr = re.sub(r"x\[(\d+)\]", r"x\1", expr)

        return f"{name} = {expr}"
    except Exception:
        return f"{name}(x)"

# === ОСНОВНЫЕ СВОЙСТВА ===

def hamming_weight(tt: np.ndarray) -> int:
    return np.sum(tt)

def is_balanced(tt: np.ndarray) -> bool:
    return hamming_weight(tt) * 2 == len(tt)

def mobius_transform(f: np.ndarray) -> np.ndarray:
    n = int(np.log2(f.size))
    a = f.copy()
    for i in range(n):
        step = 1 << i
        for j in range(0, len(a), step << 1):
            for k in range(step):
                a[j + k + step] ^= a[j + k]
    return a

def anf_to_string(anf: np.ndarray, n: int) -> str:
    terms = []
    for i in range(len(anf)):
        if anf[i]:
            if i == 0:
                terms.append("1 ^")
            else:
                term = []
                for j in range(n):
                    if (i >> j) & 1:
                        term.append(f"x{n-j-1}")
                terms.append("".join(term))
    return " ^ ".join(terms) if terms else "0"

def algebraic_degree(anf: np.ndarray) -> int:
    return max((popcount(i) for i in range(anf.size) if anf[i]), default=0)

def is_affine(anf: np.ndarray) -> bool:
    return algebraic_degree(anf) <= 1

# === УОЛШ-АДАМАР, НЕЛИНЕЙНОСТЬ, КОРРЕЛЯЦИЯ ===

def fast_walsh_hadamard_transform(f: np.ndarray) -> np.ndarray:
    n = f.size
    wht = np.where(f == 0, 1, -1).astype(np.int32)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = wht[j]
                y = wht[j + h]
                wht[j] = x + y
                wht[j + h] = x - y
        h *= 2
    return wht

def nonlinearity(wht: np.ndarray) -> int:
    return (wht.size // 2) - (np.max(np.abs(wht)) // 2)

def correlation_vector_int(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Возвращает целочисленный вектор корреляции между f(x) и g(x ⊕ a)
    для всех a ∈ {0,1}^n, без нормализации.

    f, g — булевы массивы длины 2^n.
    Результат — массив длины 2^n со значениями от -2^n до +2^n.
    """
    n = int(np.log2(len(f)))
    assert len(f) == len(g) == 2 ** n

    f_pm = 1 - 2 * f.astype(int)
    g_pm = 1 - 2 * g.astype(int)

    corr = np.zeros(2 ** n, dtype=int)

    for a in range(2 ** n):
        indices = np.arange(2 ** n) ^ a
        corr[a] = np.sum(f_pm * g_pm[indices])

    return corr


# === АВТОКОРРЕЛЯЦИЯ ===

def autocorrelation_spectrum(f: np.ndarray) -> np.ndarray:
    size = f.size
    result = np.empty_like(f, dtype=np.int32)
    for a in range(size):
        total = 0
        for x in range(size):
            diff = f[x] ^ f[x ^ a]
            total += 1 if diff == 0 else -1
        result[a] = total
    return result

# === LAT ===

def lat_table(f: np.ndarray, n: int) -> np.ndarray:
    size = 1 << n
    lat = np.zeros((size, size), dtype=np.int32)
    for a in range(size):
        for b in range(size):
            total = 0
            for x in range(size):
                ax = popcount(a & x) % 2
                bx = popcount(b & x) % 2
                total += 1 if (ax ^ f[x] ^ bx) == 0 else -1
            lat[a][b] = total
    return lat

# === DDT ===

def ddt_table(f: np.ndarray, n: int) -> np.ndarray:
    size = 1 << n
    ddt = np.zeros((size, size), dtype=np.int32)
    for a in range(size):
        for x in range(size):
            x_a = x ^ a
            dy = f[x] ^ f[x_a]
            ddt[a][dy] += 1
    return ddt

# === ТЕСТЫ ===

def test_boolean_properties(n, f_func, g_func):
    f_expr = describe_function("f", f_func)
    g_expr = describe_function("g", g_func)

    f = truth_table_from_func(f_func, n)
    g = truth_table_from_func(g_func, n)

    print("=== ANALYSIS OF BOOLEAN FUNCTION f ===")
    print(f"{f_expr}")
    print(f"{g_expr}")
    print(f"Truth table ({f_expr.split('=')[0].strip()}): {f}")
    print(f"Hamming weight ({f_expr.split('=')[0].strip()}): {hamming_weight(f)}")
    print(f"Balanced ({f_expr.split('=')[0].strip()}): {is_balanced(f)}")

    anf = mobius_transform(f)
    print(f"Algebraic Normal Form (ANF) ({f_expr.split('=')[0].strip()}): {anf}")
    print(f"Canonical ANF ({f_expr.split('=')[0].strip()}): {anf_to_string(anf, n)}")
    deg = algebraic_degree(anf)
    print(f"Algebraic degree ({f_expr.split('=')[0].strip()}): {deg}")
    print(f"Affine ({f_expr.split('=')[0].strip()}): {is_affine(anf)}")

    wht = fast_walsh_hadamard_transform(f)
    print(f"Walsh-Hadamard Transform ({f_expr.split('=')[0].strip()}): {wht}")
    print(f"Nonlinearity ({f_expr.split('=')[0].strip()}): {nonlinearity(wht)}")

    corr_vec = correlation_vector_int(f, g)
    print("Целочисленный вектор корреляции f(x) и g(x ⊕ a):")
    for a, val in enumerate(corr_vec):
        print(f"a = {a:03b} : {val}")

    auto = autocorrelation_spectrum(f)
    print(f"Autocorrelation spectrum ({f_expr.split('=')[0].strip()}): {auto}")

    auto = autocorrelation_spectrum(g)
    print(f"Autocorrelation spectrum ({g_expr.split('=')[0].strip()}): {auto}")
    print()
    print("Basic property tests passed.\n")

def test_lat_ddt(n, f_func):
    f_expr = describe_function("f", f_func)
    f = truth_table_from_func(f_func, n)

    print("=== LAT & DDT ANALYSIS ===")
    print(f"{f_expr}")
    print(f"Truth table ({f_expr.split('=')[0].strip()}): {f}")

    print(f"\n--- Linear Approximation Table (LAT) for {f_expr.split('=')[0].strip()} ---")
    lat = lat_table(f, n)
    print(lat)

    print(f"\n--- Difference Distribution Table (DDT) for {f_expr.split('=')[0].strip()} ---")
    ddt = ddt_table(f, n)
    print(ddt)
    print()
    print("LAT and DDT tests passed.\n")

# === ЗАПУСК ===

if __name__ == "__main__":
    n = 3
    f = lambda x: x[0] ^ x[2] ^ x[2]&x[1]
    g = lambda x: x[0] ^ x[1] ^ x[2]
    test_boolean_properties(n, f, g)
    test_lat_ddt(n, f)
