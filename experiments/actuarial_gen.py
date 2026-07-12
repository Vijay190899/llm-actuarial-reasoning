"""Actuarial problem generator with code-computed ground truth (KC0 linchpin).

Mortality follows Makeham's law with the Standard Ultimate Life Table (SULT)
parameters from Dickson, Hardy and Waters, Actuarial Mathematics for Life
Contingent Risks: A=0.00022, B=2.7e-6, c=1.124, from age 20, i=0.05. Using a
published law makes every quantity reproducible and correct by construction.

Validation (run this file): interest functions checked against exact closed
forms; life functions checked against published SULT values (a-due_60=14.9041,
A_60=0.29028) AND the exact identity A_x = 1 - d*a-due_x. If these pass, the
ground truth is trustworthy and we may call models. If not, we do not.
"""

import math

A_MAK, B_MAK, C_MAK = 0.00022, 2.7e-6, 1.124
X0 = 20            # SULT starts at age 20
OMEGA = 131        # table end
I_DEFAULT = 0.05


def tpx(x, t):
    """t-year survival probability from age x under Makeham (closed form)."""
    # integral of mu over [x, x+t] = A t + B c^x (c^t - 1)/ln c
    integral = A_MAK * t + B_MAK * (C_MAK ** x) * (C_MAK ** t - 1) / math.log(C_MAK)
    return math.exp(-integral)


def life_table(i=I_DEFAULT):
    v = 1 / (1 + i)
    ages = list(range(X0, OMEGA + 1))
    lx = {X0: 100000.0}
    for x in ages[1:]:
        lx[x] = lx[X0] * tpx(X0, x - X0)
    return lx, v


def a_due_x(x, i=I_DEFAULT):
    """Whole-life annuity-due EPV at age x: sum_k v^k * k_p_x."""
    v = 1 / (1 + i)
    total, k = 0.0, 0
    while x + k <= OMEGA:
        total += (v ** k) * tpx(x, k)
        k += 1
    return total


def A_x(x, i=I_DEFAULT):
    """Whole-life insurance EPV at age x: sum_k v^{k+1} * (k_p_x - k+1_p_x)."""
    v = 1 / (1 + i)
    total, k = 0.0, 0
    while x + k <= OMEGA:
        total += (v ** (k + 1)) * (tpx(x, k) - tpx(x, k + 1))
        k += 1
    return total


def a_immediate_n(n, i=I_DEFAULT):
    return (1 - (1 / (1 + i)) ** n) / i


def a_due_n(n, i=I_DEFAULT):
    return a_immediate_n(n, i) * (1 + i)


def s_immediate_n(n, i=I_DEFAULT):
    return ((1 + i) ** n - 1) / i


def net_level_premium_wl(x, i=I_DEFAULT):
    """Annual net level premium per 1 of whole-life insurance: P_x = A_x / a-due_x."""
    return A_x(x, i) / a_due_x(x, i)


def reserve_wl(x, t, i=I_DEFAULT):
    """Net level premium reserve at time t for whole life issued at x: A_{x+t} - P_x a-due_{x+t}."""
    P = net_level_premium_wl(x, i)
    return A_x(x + t, i) - P * a_due_x(x + t, i)


def validate():
    ok = True
    checks = []

    # 1) interest theory closed forms (exact, hand-verifiable)
    v = a_immediate_n(10, 0.05)
    checks.append(("a_immediate_10@5%", v, 7.721735, abs(v - 7.721735) < 1e-4))
    vd = a_due_n(10, 0.05)
    checks.append(("a_due_10@5%", vd, 8.107822, abs(vd - 8.107822) < 1e-4))
    sn = s_immediate_n(10, 0.05)
    checks.append(("s_immediate_10@5%", sn, 12.577893, abs(sn - 12.577893) < 1e-4))

    # 2) published SULT values (external anchor)
    ad60 = a_due_x(60, 0.05)
    checks.append(("a_due_60 (SULT 14.9041)", ad60, 14.9041, abs(ad60 - 14.9041) < 2e-3))
    a60 = A_x(60, 0.05)
    checks.append(("A_60 (SULT 0.29028)", a60, 0.29028, abs(a60 - 0.29028) < 2e-4))

    # 3) exact identity A_x = 1 - d * a-due_x (catches implementation bugs)
    d = 0.05 / 1.05
    for x in (40, 60, 80):
        lhs, rhs = A_x(x, 0.05), 1 - d * a_due_x(x, 0.05)
        checks.append((f"identity A_{x}=1-d*adue_{x}", lhs, rhs, abs(lhs - rhs) < 1e-9))

    print("KC0 GENERATOR VALIDATION")
    for name, got, ref, passed in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:34s} got={got:.6f} ref={ref:.6f}")
    print("KC0", "PASS: ground truth is trustworthy." if ok else "FAIL: do NOT call models.")
    return ok


if __name__ == "__main__":
    validate()
