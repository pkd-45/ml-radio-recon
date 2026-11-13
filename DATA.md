# Data sources

The pipeline is agnostic to where the truth images come from — anything
non-negative, 256x256, normalised to `[0, 1]` works. Below are the
sources I would recommend in order of effort.

---

## 1. Synthetic Sersic galaxies (default, no network)

Built in. Generates `N` random Sersic-profile galaxies with realistic
ellipticities, position angles, concentrations, and occasional
companions. Reproducible from a seed, perfect ground truth, costs
nothing.

```bash
python scripts/download_data.py --source synthetic --n 2000 \
       --out data/galaxies.h5
```

Use this for development, ablation studies, and method sanity checks.

---

## 2. Galaxy10 DECaLS (real galaxy morphologies, ~3 GB)

17,736 colour cutouts of nearby galaxies from DESI DECaLS. Already at
256x256, classes span ten Hubble morphological types. Downloaded via
the `astroNN` package; the script converts to single-channel intensity.

- Source: https://astronn.readthedocs.io/en/latest/galaxy10.html
- Paper: Leung & Bovy (2019), MNRAS 483, 3255 (Galaxy10 SDSS); Walmsley
  et al. (2022) for the DECaLS rebuild.
- License: data are public-domain US-government products of the DESI
  Legacy Imaging Surveys.

```bash
pip install astroNN
python scripts/download_data.py --source galaxy10 --n 5000 \
       --out data/galaxies.h5
```

Use this when you want the network exposed to realistic morphological
diversity (spirals, lenticulars, disturbed systems) rather than analytic
profiles only.

---

## 3. ALMA Science Archive (real interferometric data)

The right next step if you want to demonstrate the pipeline on actual
ALMA visibilities rather than the schematic FFT-mask forward model
shipped here.

- Portal: https://almascience.eso.org/asax/
- Programmatic: `astroquery.alma`
  (https://astroquery.readthedocs.io/en/latest/alma/alma.html)

Recommended starting points:

| Target type    | Why                                    | Search hint              |
|----------------|----------------------------------------|--------------------------|
| Nearby SF gals | continuum and CO morphology, good SNR  | NGC 253, NGC 1068, M83   |
| Protoplanetary | bright, compact, well-studied          | HL Tau, TW Hya, HD 163296|
| Submm galaxies | high-z faint extended, harder case     | ALMA-LESS / ALPINE       |

Recipe for ML training pairs (in CASA):

```python
# CASA-script-level pseudocode
simobserve(project="alma_sim", skymodel="truth.fits",
           antennalist="alma.cycle10.7.cfg",
           totaltime="3600s", integration="60s",
           thermalnoise="tsys-atm", graphics="none")
tclean(vis="alma_sim/alma_sim.alma.cycle10.7.ms",
       imagename="dirty", niter=0, gridder="standard",
       imsize=256, cell="0.05arcsec")
tclean(vis="alma_sim/alma_sim.alma.cycle10.7.ms",
       imagename="clean", niter=2000, gridder="standard",
       imsize=256, cell="0.05arcsec", threshold="0.1mJy")
```

The output `dirty.image` and `clean.image` (or `truth.fits`) become
input / target pairs.

---

## 4. Related ALMA-BRAIN research and software

- Paper: Delli Veneri, Tychoniec, Guglielmetti, Lehnardt, Longo, et al.
  (2023), *3D Detection of ALMA Sources Using Convolutional Neural
  Networks*, MNRAS 518, 3407.
  https://academic.oup.com/mnras/article/518/3/3407/6786286
- Code: https://github.com/MicheleDelliVeneri/ALMA-BRAIN
  (the imaging companion paper is in prep at time of writing; this
  repo contains the source-detection and 3D cube CNN.)

These resources provide context for source-detection methods and learned
representations, while this repository focuses on a controlled sparse-aperture
imaging and reconstruction benchmark.

---

## 5. HST Legacy Archive / JWST (high-resolution optical truth)

For demonstrating super-resolution-style reconstruction (interferometric
dirty image at lower resolution -> HST-like truth at higher resolution).

- HLA: https://hla.stsci.edu/hlaview.html
- MAST: https://mast.stsci.edu/

`astroquery.mast` gives programmatic cutouts. Use the F814W or F606W
ACS/WFC mosaics of CANDELS / GOODS-S for galaxy fields with thousands of
useful cutouts.

```python
from astroquery.mast import Observations
obs = Observations.query_criteria(obs_collection="HLA",
                                  instrument_name="ACS/WFC",
                                  filters="F814W",
                                  obs_id="hst_*goods*")
```

---

## 6. Pan-STARRS DR2 cutouts (broad sky coverage, easy)

Drop-in replacement for Galaxy10 with arbitrary target lists.

- API: https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service
- `astroquery.mast.Catalogs` + the cutout service give 256x256 fits in
  one HTTP call per object.

---

## Storage notes

The HDF5 dataset built by `download_data.py` is gzip-compressed at level
4. Typical sizes:

| n_images | size (synthetic) | size (Galaxy10 grayscale) |
|---------:|-----------------:|--------------------------:|
| 1000     | ~ 40 MB          | ~ 60 MB                   |
| 5000     | ~ 200 MB         | ~ 300 MB                  |
| 20000    | ~ 800 MB         | ~ 1.2 GB                  |

For runs above 20k images, switch to chunked HDF5 with on-the-fly
streaming (a 5-line change to `dataset.py`).
