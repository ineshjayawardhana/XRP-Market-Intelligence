"""Shared reproducible metrics and conservative educational display policy."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss)

OUTCOMES = ['UP', 'DOWN', 'NEUTRAL']
QUALITY_MESSAGES = {
    'at_least_100_nonoverlapping_holdout_samples': 'Fewer than 100 non-overlapping holdout examples are available.',
    'at_least_10_per_outcome': 'At least one outcome has fewer than 10 holdout examples.',
    'balanced_accuracy_at_least_0_38': 'Held-out balanced accuracy is below the required 0.38 minimum.',
    'accuracy_at_least_majority_baseline': 'Accuracy is below the simple training-majority baseline.',
    'macro_f1_above_majority_baseline': 'Macro-F1 does not beat the training-majority baseline.',
    'macro_f1_at_least_technical_rule': 'Macro-F1 is below the existing technical rule.',
    'balanced_accuracy_at_least_technical_rule': 'Balanced accuracy is below the existing technical rule.',
    'brier_better_than_training_prior': 'Brier score does not improve on the training-prior baseline.',
    'log_loss_better_than_training_prior': 'Log loss does not improve on the training-prior baseline.',
    'calibration_ece_at_most_0_10': 'Held-out calibration error exceeds the 0.10 policy limit.',
    'development_walk_forward_skill': 'Earlier walk-forward Macro-F1 did not beat its majority baseline.',
}


def metrics(y, probabilities):
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid three-class probability vectors.')
    predicted = np.array(OUTCOMES)[p.argmax(axis=1)]
    actual = np.array([[int(label == value) for label in OUTCOMES] for value in y])
    confidence, correct = p.max(axis=1), predicted == y
    bins, ece = [], 0.
    for index in range(10):
        mask = (confidence >= index / 10) & (confidence <= (index + 1) / 10 if index == 9 else confidence < (index + 1) / 10)
        if mask.any():
            mean, fraction = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += float(mask.mean()) * abs(mean - fraction)
            bins.append(dict(lower=index / 10, upper=(index + 1) / 10, count=int(mask.sum()),
                             mean_probability=mean, observed_accuracy=fraction))
    # Explicitly reorder the log-loss input to sklearn's lexicographic label order.
    lexical = [OUTCOMES.index(label) for label in sorted(OUTCOMES)]
    return dict(sample_size=len(y), accuracy=float(accuracy_score(y, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, labels=OUTCOMES, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(y, predicted, labels=OUTCOMES).tolist(),
        classification_report=classification_report(y, predicted, labels=OUTCOMES, output_dict=True, zero_division=0),
        outcome_distribution={label: int((y == label).sum()) for label in OUTCOMES},
        log_loss=float(log_loss(y, p[:, lexical], labels=sorted(OUTCOMES))),
        brier_score=float(np.square(p - actual).sum(axis=1).mean()), calibration_ece=float(ece), reliability_bins=bins)


def quality_checks(selected, baselines, walk_forward_skill=True):
    """Predeclared policy, not a claim of statistical or financial reliability."""
    checks = {
        'at_least_100_nonoverlapping_holdout_samples': selected['sample_size'] >= 100,
        'at_least_10_per_outcome': all(selected['outcome_distribution'].get(label, 0) >= 10 for label in OUTCOMES),
        'balanced_accuracy_at_least_0_38': selected['balanced_accuracy'] >= .38,
        'accuracy_at_least_majority_baseline': selected['accuracy'] >= baselines['majority']['accuracy'],
        'macro_f1_above_majority_baseline': selected['macro_f1'] > baselines['majority']['macro_f1'],
        'macro_f1_at_least_technical_rule': selected['macro_f1'] >= baselines['technical_rule']['macro_f1'],
        'balanced_accuracy_at_least_technical_rule': selected['balanced_accuracy'] >= baselines['technical_rule']['balanced_accuracy'],
        'brier_better_than_training_prior': selected['brier_score'] < baselines['training_prior']['brier_score'],
        'log_loss_better_than_training_prior': selected['log_loss'] < baselines['training_prior']['log_loss'],
        'calibration_ece_at_most_0_10': selected['calibration_ece'] <= .10,
        'development_walk_forward_skill': bool(walk_forward_skill),
    }
    return dict(passed=all(checks.values()), checks={key: bool(value) for key, value in checks.items()},
                failures=[key for key, value in checks.items() if not value])
