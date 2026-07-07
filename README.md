# LANCE: Locally Adaptive Neural Context Estimation for Overfitted Image Compression
Accepted to IEEE TCSVT 2026

This repository contains the code to reproduce the paper **LANCE: Locally Adaptive Neural Context Estimation for Overfitted Image Compression** [arXiv](https://arxiv.org/abs/2605.20672).

## Abstract
This paper introduces Locally Adaptive Neural Context Estimation (LANCE), a novel extension for overfitted image compression (OIC) frameworks like Cool-Chic. While traditional OIC methods rely on lightweight autoregressive networks with globally signaled parameters, they struggle with non-stationary image statistics. LANCE addresses this by incorporating a forward-signaled spatial hyperprior that enables regional adaptation of the entropy model. To minimize overhead, we employ a predictive coding scheme that combines a static Median Edge Detector (MED) with a lightweight learned context model.

Experiments demonstrate that LANCE achieves BD-rate reductions of 1.40% on the Kodak dataset and 1.97% on CLIC 2020 over Cool-Chic 4.0 at the high end of our decoder complexity range of 606-1483 MAC/pixel. At the low end of the complexity range, we outperform Cool-Chic 4.0 by 2.41% and 2.99% on Kodak and CLIC, respectively. Qualitative analysis reveals that the learned spatial hyperprior effectively segments image regions into areas of similar image statistics, providing an automated, content-aware adaptation layer.

## COOL-CHIC
LANCE is an extension of COOL-CHIC 4.0, an overfitted image and video codec by Orange. More details are available on the [COOL-CHIC Github page](https://orange-opensource.github.io/Cool-Chic/).

## Setup
```bash
# Clone this repository
git clone https://github.com/mbenjak/LANCE.git
cd LANCE

# Install create and activate virtual env
python3.10 -m pip install virtualenv
python3.10 -m virtualenv venv && source venv/bin/activate

# Install LANCE
CXX=g++ pip install -e .
```

A stand-alone decoder binary can be compiled with:
```bash
mkdir build
cd build
cmake ..
make -j
```

## Usage
To encode an image run:
```bash
python coolchic/encode.py -i={input_image}.png -o={bitstream}.cool --workdir={work_dir} --enc_cfg=cfg/enc/intra/slow_100k.cfg --dec_cfg_residue=cfg/dec/intra_residue/hop.cfg --lmbda=0.001
```

To decode the encoded bitstream run:
```bash
python coolchic/decode.py -i={bitstream}.cool  -o={decoded_image}.ppm --verbosity=1
```

Alternatively, decode using the stand-alone binary:
```bash
./build/bin/ccdec --input={bitstream}.cool --output={decoded_image}.ppm
```

## Citing this work
If you use this code in your work, we ask you to please cite our work:

```latex
@inproceedings{lance2026,
  title={LANCE: Locally Adaptive Neural Context Estimation for Overfitted Image Compression},
  author={Benjak, Martin and Ostermann, Jörn},
  booktitle={arXiv preprint arXiv:2605.20672},
  year={2026}
}
```
