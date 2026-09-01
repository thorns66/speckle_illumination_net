# Positive anchor parameterization

## Required properties

For the variance anchor `a = beta * F_var >= 0` and decoder residual `r`, the
output map should satisfy:

1. `phi(a, 0) = a`, so zero initialization really starts from the offline
   variance reconstruction.
2. `phi(a, r) >= 0`, because fluorescence intensity is nonnegative.
3. Positive and negative corrections are both possible.
4. `d phi / d r = 1` at initialization wherever possible, so the residual
   decoder receives a well-scaled physics gradient.
5. Positive corrections do not grow exponentially.

The former `softplus(a + r)` fails property 1 because `softplus(0) = log(2)`.
It also makes the linear least-squares initialization of `beta` inconsistent
with the actual step-zero reconstruction.

## Alternatives considered

| Map | Anchor exact | Nonnegative | Initial residual gradient | Main issue |
| --- | --- | --- | --- | --- |
| `relu(a+r)` | yes | yes | active-set dependent | dead negative region and a hard kink |
| `a*exp(r/a)` | yes for `a>0` | yes | 1 | cannot create signal at `a=0`; positive corrections can explode |
| `softplus(inv_softplus(a)+r)` | yes for `a>0` | yes | `1-exp(-a)` at unit scale | gradients vanish in dark/low-anchor voxels |
| positivity penalty | approximate | no | 1 | physics losses can see negative intensities |
| anchor-scaled ELU | yes | yes | 1 at initialization | active-set behavior only where `a=0` |

For example, the inverse-Softplus gradient is approximately `a` when `a` is
small. At anchors `[1e-6, 1e-3, 0.1, 1]`, its unit-scale initialization
gradients are approximately `[1e-6, 0.001, 0.095, 0.632]`. This suppresses the
very dark-region corrections needed to add structure missing from `F_var`.

## Selected map

The implementation uses

```text
phi(a, r) = a + r                    for r >= 0
phi(a, r) = a * exp(r / max(a, eps)) for r < 0
```

This is an ELU whose negative saturation is scaled by the local anchor. For
`a >= eps`, the two branches meet with value `a` and derivative 1. Positive
corrections are additive and unbounded without exponential amplification;
negative corrections approach zero without crossing it. The denominator floor
is numerical only and does not add an intensity floor to the reconstruction.

At an exactly zero anchor, no everywhere-differentiable nonnegative map can
both pass through zero and have nonzero derivative there: zero is a minimum, so
its derivative must be zero. The selected map therefore uses active-set
behavior at those voxels. At the zero-initialized residual, the positive branch
is selected and PyTorch supplies derivative 1; a negative residual suppresses
that voxel to zero, while a positive residual can create missing signal.

## Sources

- Dugas et al., *Incorporating Second-Order Functional Knowledge for Better
  Option Pricing*, NeurIPS 2000: original Softplus definition and derivative.
  <https://proceedings.neurips.cc/paper/2000/file/44968aece94f667e4095002d140b5896-Paper.pdf>
- Clevert et al., *Fast and Accurate Deep Network Learning by Exponential
  Linear Units*, ICLR 2016: identity positive branch and saturating exponential
  negative branch. <https://arxiv.org/abs/1511.07289>
- Bardsley and Vogel, *A Nonnegatively Constrained Convex Programming Method
  for Image Reconstruction*, SIAM J. Sci. Comput.: nonnegativity as an explicit
  physical constraint in image reconstruction.
  <https://doi.org/10.1137/S1064827502410451>
- PyTorch `Softplus` documentation: exact implementation formula and its role
  as a positive smooth ReLU approximation.
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html>
