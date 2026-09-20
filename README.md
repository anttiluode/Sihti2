# Sihti2

**What geometry does random data induce when similarity becomes conductivity, and what scaling law governs the lifetime of the resulting domains?**

Sihti2 starts from a very simple accident in [Sihti](https://github.com/anttiluode/Sihti): feed IID colour noise into the image-written graph, then repeatedly apply the same lazy diffusion. Instead of merely washing out, the field can spend a very long time in coloured, boxy-looking domains.

The first interpretation to avoid is "noise made objects." The more precise object is:

> **Noise writes a random conductance geometry; diffusion reveals the almost-invariant regions of that geometry.**

This repo is the spectral-geometry branch of Sihti. It is not a new object-segmentation claim and it is not evidence for fractality by appearance alone.

## Run

The browser instrument is the fastest way to see the effect:

https://anttiluode.github.io/Sihti2/

For the measured experiment:

```bash
pip install -r requirements.txt
python spectral_geometry.py
```

A small sweep:

```bash
python spectral_geometry.py --seeds 8 --sigmas 3 5 8 12 --size 40 --max-depth 256
```

Outputs go to `results/spectral_geometry.json` and `figures/spectral_geometry.png`.

## The object

For an image (x_0), Sihti constructs a local weighted graph

```math
w_{ij}
=
\exp\!\left(-\frac{\|Lab_i-Lab_j\|^2}{2\sigma^2}\right)
\exp\!\left(-\frac{\|p_i-p_j\|^2}{2\sigma_x^2}\right),
```

with a small positive floor so the finite graph never becomes literally disconnected.

The lazy random-walk purifier is

```math
T = (1-a)I + aD^{-1}W, \qquad a=\tfrac12,
```

or

```math
T = I-aL_{rw}.
```

If

```math
L_{rw}\phi_k=\mu_k\phi_k,
```

then

```math
T^t x_0
=
\sum_k c_k(1-a\mu_k)^t\phi_k.
```

Small (mu_k) means a long-lived mode. A useful relaxation time is

```math
\tau_k
=
-\frac{1}{\log(1-a\mu_k)}.
```

The noise case is unusual because the same random sample first creates (W(x_0)) and is then propagated through (T(x_0)):

```math
x_0
\longrightarrow
W(x_0)
\longrightarrow
T(x_0)
\longrightarrow
T(x_0)^t x_0.
```

The probe has written the medium through which it is later propagated.

## Why the patches can live so long

For IID RGB noise at a small colour scale, most neighboring Lab colours are very far apart. Most bonds therefore become tiny while a minority of accidental similar-color bonds remain much stronger.

That produces a **random conductance model**: a spatial lattice with an extremely heterogeneous set of edge conductances.

Fast diffusion equalizes strongly coupled local regions. Leakage across weak boundaries can be orders of magnitude slower. The visible patches are therefore candidates for **metastable / almost-invariant sets**, not semantic objects.

This immediately connects the toy to established mathematics and physics:

- reversible Markov chains and mixing times,
- normalized graph Laplacians,
- random conductance and random resistor networks,
- Cheeger cuts and spectral clustering,
- percolation and transport in disordered media,
- diffusion maps and Markov-time community structure.

A small (mu_2) is especially important: it means the graph contains a very weak global bottleneck. Sihti's depth 256 can then be "deep" visually while still being extremely early compared with the true mixing time.

## The Sihti residue is a relaxation-time band

Sihti does not only save (T^t x). At dyadic depths it saves

```math
R_j = T^{d_j}x - T^{2d_j}x,
\qquad d_j=1,2,4,8,\ldots
```

For one mode with per-pass gain (g),

```math
R_j = g^{d_j}(1-g^{d_j})c\phi.
```

This shell is strongest near (g^{d_j}\approx 1/2). For a slow Laplacian mode,

```math
g\approx 1-a\mu
```

so the shell is centered roughly around

```math
\mu \sim \frac{\log 2}{a d_j}.
```

That is a better mathematical description of the octave strip than "different blur levels":

> **the strip sorts directions by characteristic relaxation time.**

This places Sihti close to heat-kernel multiscale analysis, diffusion wavelets and Littlewood-Paley-style spectral shells.

## Box counting was the visual clue; spectral dimension is the harder question

The boxy domains invite a fractal story, but appearance is not evidence of a fractal.

A more natural quantity for this process is the **spectral dimension**. For a Laplacian,

```math
Z(t)=\operatorname{Tr}(e^{-tL})=\sum_k e^{-t\mu_k}.
```

On a scale-invariant diffusion geometry,

```math
Z(t)-1 \propto t^{-d_s/2}.
```

For Sihti's discrete lazy operator,

```math
Z_T(t)=\operatorname{Tr}(T^t)
=\sum_k (1-a\mu_k)^t
\approx \sum_k e^{-at\mu_k}.
```

So a local effective spectral dimension can be estimated from the log-slope of (Z_T(t)-1).

A straight region in that log-log curve would be interesting. No straight region means the box-counting impression was only a finite-scale visual effect. Both outcomes are useful.

## Renormalization analogy — with a boundary

The dyadic process

```math
x,\;T x,\;T^2x,\;T^4x,\;T^8x,\ldots
```

successively removes fast directions. Sihti additionally keeps every shell that disappeared.

That resembles Wilsonian thinking: integrate out short-lived degrees of freedom and inspect what survives at larger scale. But Sihti2 does **not** currently perform a full renormalization-group transformation: it does not rescale the lattice and flow the coupling law back into a parameter family.

The useful analogy is narrower:

> **successive coarse-graining plus an explicit ledger of the degrees of freedom removed at each scale.**

## Disorder and rare regions

The conductance distribution can span many orders of magnitude. Rare patches of unusually strong internal coupling surrounded by weak bonds can then retain state for very long times.

That is qualitatively related to rare-region and Griffiths-like relaxation in disordered systems: broad disorder can produce broad lifetime distributions without a single characteristic relaxation time.

This repo does **not** claim a Griffiths phase. The experiment to earn such language would need finite-size scaling, parameter sweeps and reproducible power-law regimes rather than one compelling image.

## Machine-learning connection

Repeated message passing in a graph neural network has the same basic form:

```math
X,\;PX,\;P^2X,\ldots
```

and eventually loses high graph frequencies: oversmoothing.

Sihti's difference shell

```math
P^tX-P^{2t}X
```

is therefore a direct picture of what another interval of message passing erases.

Noise is also a classical spectral probe. A generic random vector has nonzero projection on almost every eigenmode, so repeated application exposes slow directions without designing a special input. Sihti2 adds the oddity that the random input also writes the graph.

## Controls that matter

The first experiments compare four cases under matched local edge geometry:

1. **self** — the IID signal writes its own conductances and is diffused through them;
2. **independent** — one IID signal writes the graph and an independent IID signal is propagated through it;
3. **shuffled** — preserve the exact conductance histogram but randomly permute conductances over graph edges;
4. **lattice** — remove colour dependence entirely and retain only spatial coupling.

These controls ask different questions.

- Self vs independent asks whether alignment between the signal and the geometry it wrote matters.
- Self vs shuffled asks whether spatial arrangement matters beyond the weight histogram.
- Self vs lattice asks what disorder adds beyond the substrate.

The current code reports rather than presupposes the answer.

## Energies: use the geometry's own measure

Ordinary pixel RMS is useful visually but is not the natural norm of a nonuniform random walk.

The stationary measure is

```math
\pi_i = \frac{d_i}{\sum_j d_j}.
```

The experiment therefore also tracks

```math
\operatorname{Var}_\pi(x)
=
\sum_i \pi_i\|x_i-\bar x_\pi\|^2
```

and the Dirichlet energy

```math
\mathcal E(x)
=
\frac12\sum_{ij}w_{ij}\|x_i-x_j\|^2.
```

For the lazy reversible chain these are much better measures of how much nonstationary structure and edge disagreement remain.

## What would count as a result

The live visual is an instrument, not evidence. The current Python sweep records:

- edge-weight floor fraction and quantiles,
- (mu_2), spectral gap and (	au_2),
- stationary-measure variance versus depth,
- Dirichlet energy versus depth,
- Hutchinson estimates of the heat trace,
- local spectral-dimension slope,
- matched self / independent / shuffled / lattice controls.

The next strong result would be one of these:

- a reproducible finite range where spectral dimension is approximately scale-invariant;
- a reproducible broadening of relaxation times with (sigma) and system size;
- a clear self-vs-independent effect showing that the signal is unusually retained by the geometry it wrote;
- a clear self-vs-shuffled effect showing that spatial placement, not merely the conductance histogram, controls metastability.

The corresponding kill is equally important: if the apparent hierarchy disappears under larger sizes/seeds or has no stable scaling regime, the "box-counting" appearance stays a finite-size visualization and no fractal language is earned.

## Lineage

```text
SighImageSuper
    repeated operator exposes what survives
          |
          v
Sihti
    keep what bleeds out at dyadic depths
    purifier defines persistence
          |
          v
Sihti2
    noise writes the purifier itself
    -> random conductance geometry
    -> metastability / mixing / spectral dimension
```

The main conceptual correction is:

> **persistent does not mean fundamental, and coherent does not mean semantic. Persistence means favored by the operator and the geometry through which the dynamics move.**

## Files

```text
index.html                  browser noise/conductance/diffusion instrument
spectral_geometry.py        measured spectral-geometry sweep
tests/test_spectral_geometry.py
requirements.txt
```
