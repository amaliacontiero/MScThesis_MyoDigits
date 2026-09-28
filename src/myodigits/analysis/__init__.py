"""Analysis utilities shared by the training entry point, scripts and tests.

Each quantity reported in the thesis has exactly one implementation here:

    loso_summary   LOSO summary reconstruction from per-fold metric files
    r2_variants    the three coefficient-of-determination variants
    paired         paired Wilcoxon test and participant bootstrap
    registry       the run and claim registry
"""
