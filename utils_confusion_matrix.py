import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create the Agreement Matrix
def create_agreement_matrix(annotator1, annotator2):
    agreement_matrix = pd.crosstab(annotator1, annotator2, rownames=['Annotator 1'], colnames=['Annotator 2'])
    return agreement_matrix

# Create the Confusion Matrix
def create_confusion_matrix(annotator1, annotator2):
    labels = sorted(list(set(annotator1) | set(annotator2)))
    conf_matrix = confusion_matrix(annotator1, annotator2, labels=labels)
    return conf_matrix, labels

