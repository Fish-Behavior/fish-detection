"""fishbehavior: automated zebrafish behavior-state labeling pipeline.

Pipeline (each step is added in its own PR):
    workbook + videos -> match & check -> scene setup -> tracking -> features
    -> behavior labels (5 ethogram states) -> calibration -> datasets & plots
"""

__version__ = "0.1.0"
