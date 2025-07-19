# Synthetic Experiment

Code for a synthetic experiment comparing $\ell_2$ and HL-Gaussian on learning sine functions with different offsets and frequencies.

## Instructions

1. Set up a Python 3.12 environment
```
python3.12 -m venv venv
source venv/bin/activate
pip install -r synth_requirements.txt
```

2. Run `script_freqs_offsets.sh` to run frequency and offset experiments.
```
source script_freqs_offsets.sh
```

## Recreating Plots

After obtaining the experiment results, the plots in Figure 4 can be recreated in the following manner:

### Learned Functions
```
python3 visualize_both.py
```
Figure saved in `results/clean_vis_10_0.png`

### Frequency Training Curves
```
python3 plot_clean.py --plot_over=Y_freq
```
Figure saved in `results/clean_Y_freq.png`

### Offset Training Curves
```
python3 plot_clean.py --plot_over=Y_offset
```
Figure saved in `results/clean_Y_offset.png`