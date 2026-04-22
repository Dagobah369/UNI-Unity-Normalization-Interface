import math, re, os, hashlib, urllib.request
from pathlib import Path
import numpy as np

# PARAMETERS
c = 0.049                     # Centre (noyau UNI)
ln2 = math.log(2)
U = 2 * math.pi * c / ln2     # ≈ 0.44417129
DATASET_URL = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6"
DATASET_FILE = "zeros6.txt"
EXPECTED_SHA256 = "2ef7b752c2f17405222e670a61098250c8e4e09047f823f41e2b41a7b378e7c6"

# DENSITY + RECURRENCE (autonomous zero generation)
def density_UNI(m):
    """UNI spectral density ρ(m) = (U/2π)·ln(mU/2π)"""
    if m <= 0:
        return 0.0
    x = m * U / (2 * math.pi)
    if x <= 1:
        return 0.0
    return (U / (2 * math.pi)) * math.log(x)

def find_next_m(m_current, max_iter=50, tol=1e-10):
    """
    Newton solver for ∫_{m_k}^{m_{k+1}} ρ(x) dx = 1.
    Autonomous recurrence — no external zero data used.
    """
    if m_current <= 0:
        return m_current + 1.0
    d_curr = density_UNI(m_current)
    m_next = m_current + (1.0 / d_curr if d_curr > 0 else 1.5)
    for _ in range(max_iter):
        n_steps = max(50, int((m_next - m_current) * 20))
        step = (m_next - m_current) / n_steps
        integral = 0.0
        x = m_current
        for __ in range(n_steps):
            y1 = density_UNI(x)
            y2 = density_UNI(x + step)
            integral += (y1 + y2) * step / 2.0
            x += step
        F = integral - 1.0
        dF = density_UNI(m_next)
        if dF <= 1e-12:
            m_next *= 1.01
            continue
        m_new = m_next - F / dF
        if abs(m_new - m_next) < tol:
            return m_new
        m_next = m_new
    return m_next

def generate_zeros(n):
    """Generate n zeros using UNI recurrence."""
    m = [14.213481 / U]      # Initialization: γ₁ / U
    for _ in range(1, n):
        m.append(find_next_m(m[-1]))
    return [x * U for x in m]

# LOAD ODLYZKO DATA (validation only)
def load_odlyzko():
    if not os.path.exists(DATASET_FILE):
        print("   Downloading zeros6.txt...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
    sha256 = hashlib.sha256()
    with open(DATASET_FILE, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    if sha256.hexdigest() != EXPECTED_SHA256:
        raise ValueError("SHA-256 mismatch!")
    txt = Path(DATASET_FILE).read_text(encoding="utf-8", errors="ignore")
    vals = np.array([float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", txt)], dtype=float)
    vals = vals[vals > 0]
    vals.sort()
    return vals

# RECONSTRUCTION OF NATURAL NUMBERS (Closure N/N)
def invert_gamma(gamma, C=c):
    """
    Inverse formula: n = C / (1 - exp(ln(1/2)/gamma))
    """
    if gamma <= 0:
        return float('nan')
    with np.errstate(over='ignore', invalid='ignore'):
        exp_val = np.exp(math.log(0.5) / gamma)
        denom = 1.0 - exp_val
        if abs(denom) < 1e-12:
            return float('nan')
        n = C / denom
    return n

def reconstruct_integers(gamma_list):
    """
    Reconstruct integers from gamma values (zeros of ζ).
    Returns: list of tuples (integer, n_recon, ratio, error)
    """
    seen = {}
    for gamma in gamma_list:
        n_recon = invert_gamma(gamma)
        if np.isnan(n_recon) or n_recon < 1.5:
            continue
        n_int = int(round(n_recon))
        error = abs(n_recon - n_int)
        # Keep the closest reconstruction for each integer
        if n_int not in seen or error < seen[n_int]['error']:
            seen[n_int] = {
                'integer': n_int,
                'n_recon': n_recon,
                'ratio': n_recon / n_int if n_int > 0 else 0.0,
                'error': error
            }
    # Sort by integer
    result = [seen[k] for k in sorted(seen.keys())]
    return result

def compute_closure_metrics(reconstructed, max_expected=None):
    """
    Compute closure N/N metrics.
    reconstructed: list of dict with 'integer' keys
    max_expected: if None, use max(reconstructed) as expected range
    """
    if not reconstructed:
        return {}
    integers = [r['integer'] for r in reconstructed]
    min_n = min(integers)
    max_n = max(integers)
    expected_count = max_n - min_n + 1
    actual_count = len(integers)
    missing = expected_count - actual_count
    duplicates = len(integers) - len(set(integers))
    # Check if all integers from 2 to max_n are present
    complete = all(i in integers for i in range(2, max_n + 1))
    # Compute average ratio and error
    ratios = np.array([r['ratio'] for r in reconstructed])
    errors = np.array([r['error'] for r in reconstructed])
    return {
        'min_integer': min_n,
        'max_integer': max_n,
        'expected_count': expected_count,
        'actual_count': actual_count,
        'missing': missing,
        'duplicates': duplicates,
        'coverage': actual_count / expected_count if expected_count > 0 else 0.0,
        'complete_from_2': complete,
        'mean_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'std_ratio': float(np.std(ratios)),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors)),
    }

# MAIN
def main():
    print("=" * 90)
    print("UNI — Autonomous Riemann Zero Generation + N/N Closure")
    print(f"Quantum U = {U:.8f}")
    print("=" * 90)

    # 1) Generate zeros (autonomous)
    n_zeros = 2001052
    print(f"\nGenerating {n_zeros:,} zeros by recurrence...")
    zeros_pred = generate_zeros(n_zeros)    
    print(f"    {len(zeros_pred):,} zeros generated")

    # 2) Load Odlyzko data for validation
    print("\nLoading Odlyzko data for validation...")
    zeros_ref = load_odlyzko()
    print(f"    {len(zeros_ref):,} zeros loaded")

    # N/N CLOSURE — Reconstruction of natural numbers from zeros
    print("\n" + "=" * 85)
    print("N/N CLOSURE — Reconstruction of natural numbers from zeros")
    print("=" * 85)
    # Reconstruct integers from generated zeros
    print("\nReconstruction of natural numbers from zeros...")
    reconstructed = reconstruct_integers(zeros_pred)
    metrics = compute_closure_metrics(reconstructed)
    print(f"    {len(reconstructed):,} integers reconstructed")


    # 3) Display first 100 zeros
    print("\n" + "─" * 85)
    print("FIRST 100 ZEROS — Prediction vs Odlyzko")
    print("─" * 85)
    print(f"{'rank':>5}  {'prediction':>14}  {'reference':>14}  {'error':>12}")
    print("─" * 52)
    for i in range(min(100, len(zeros_pred), len(zeros_ref))):
        prediction = zeros_pred[i]
        reference = zeros_ref[i]
        error = prediction - reference
        print(f"{i+1:>5}  {prediction:>14.6f}  {reference:>14.6f}  {error:>+12.6f}")

    # 4) Display last 100 zeros
    print("\n" + "─" * 85)
    print("LAST 100 ZEROS — Prediction vs Odlyzko")
    print("─" * 85)
    print(f"{'rank':>5}  {'prediction':>14}  {'reference':>14}  {'error':>12}")
    print("─" * 52)
    n_min = min(len(zeros_pred), len(zeros_ref))
    start_idx = max(0, n_min - 100)
    for i in range(start_idx, n_min):
        prediction = zeros_pred[i]
        reference = zeros_ref[i]
        error = prediction - reference
        print(f"{i+1:>5}  {prediction:>14.6f}  {reference:>14.6f}  {error:>+12.6f}")

    # Display first 100 reconstructed integers
    print("\n" + "─" * 85)
    print("FIRST 100 RECONSTRUCTED INTEGERS")
    print("─" * 85)
    print(f"{'n':>6}  {'n_recon':>12}  {'ratio':>10}  {'error':>10}")
    print("─" * 45)
    for r in reconstructed[:100]:
        print(f"{r['integer']:>5}  {r['n_recon']:>14.6f}  {r['ratio']:>14.6f}  {r['error']:>+12.6f}")

    # Display last 100 reconstructed integers
    print("\n" + "─" * 85)
    print("LAST 100 RECONSTRUCTED INTEGERS")
    print("─" * 85)
    print(f"{'n':>6}  {'n_recon':>12}  {'ratio':>10}  {'error':>10}")
    print("─" * 45)
    for r in reconstructed[-100:]:
        print(f"{r['integer']:>5}  {r['n_recon']:>14.6f}  {r['ratio']:>14.6f}  {r['error']:>+12.6f}")

    # 5) Statistics on all zeros
    n = min(len(zeros_pred), len(zeros_ref))
    pred = np.array(zeros_pred[:n])
    ref = np.array(zeros_ref[:n])
    errors = pred - ref
    abs_err = np.abs(errors)
    rel_err = 100.0 * abs_err / np.maximum(np.abs(ref), 1e-300)

    print("\n" + "─" * 90)
    print("STATISTICS — Full comparison (2,001,052 zeros)")
    print("─" * 90)
    print(f"Mean absolute error (MAE)            : {np.mean(abs_err):.6f}")
    print(f"Std deviation of errors              : {np.std(errors):.6f}")
    print(f"Max absolute error                   : {np.max(abs_err):.6f}")
    print(f"Min absolute error                   : {np.min(abs_err):.6f}")
    print(f"Mean relative error                  : {np.mean(rel_err):.6f}%")
    print(f"Correlation (pred vs ref)            : {np.corrcoef(pred, ref)[0,1]:.8f}")

    print(f"\nReconstructed integers               : {len(reconstructed):,}")
    print(f"Range                                : {metrics['min_integer']} → {metrics['max_integer']}")
    print(f"Expected integers in range           : {metrics['expected_count']:,}")
    print(f"Actual reconstructed                 : {metrics['actual_count']:,}")
    print(f"Missing                              : {metrics['missing']} ({metrics['missing']/metrics['expected_count']*100:.4f}%)")
    print(f"Duplicates                           : {metrics['duplicates']}")
    print(f"Coverage                             : {metrics['coverage']*100:.4f}%")

    print("\nRatio statistics (n_recon / n):")
    print(f"Mean ratio                           : {metrics['mean_ratio']:.8f}")
    print(f"Median ratio                         : {metrics['median_ratio']:.8f}")
    print(f"Std ratio                            : {metrics['std_ratio']:.8f}")

    print("\nError statistics (|n_recon - n|):")
    print(f"Mean error                           : {metrics['mean_error']:.6f}")
    print(f"Median error                         : {metrics['median_error']:.6f}")
    print(f"Max error                            : {metrics['max_error']:.6f}")
    print(f"Min error                            : {metrics['min_error']:.6f}")

    print("\nN/N Closure verification")
    print(f"Closure N/N                          : {metrics['coverage']*100:.4f}%")

    print("\n" + "─" * 90)
    print("CONCLUSION")
    print(f"The system reconstructs {metrics['actual_count']:,} natural numbers from its own 2,001,052 zeros")
    print(f"N/N closure: {metrics['coverage']*100:.1f}%, mean ratio: {metrics['mean_ratio']:.8f}, mean relative error: {np.mean(rel_err):.6f}%")

if __name__ == "__main__":
    main()