# Gender Classification from Facial Images
**Team: Team2_Equalis**

---

## Problem Statement

Binary classification of gender from facial images:
- **Label 0** → Male
- **Label 1** → Female

## Model Architecture

- **Base**: EfficientNetB0 (transfer learning from ImageNet)
- **Classifier Head**: GlobalAvgPool → Linear(1280, 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512, 1) → Sigmoid
- **Training**: Two-phase (frozen feature extraction → fine-tuning top layers)
- **Loss**: Weighted Binary Cross-Entropy (handles class imbalance)
- **Optimizer**: Adam with ReduceLROnPlateau scheduling

## Dataset

| Split | Male | Female | Total |
|-------|------|--------|-------|
| Train | 67,155 | 92,845 | 160,000 |
| Validation | 8,820 | 13,778 | 22,598 |
| Test | 8,459 | 11,542 | 20,001 |

## Setup

```bash
pip install -r requirements.txt
```

## Inference

```bash
# Single image prediction
python inference.py --image path/to/face.jpg

# With Test Time Augmentation (higher accuracy)
python inference.py --image path/to/face.jpg --tta
```

### Python API

```python
from inference import GenderClassifier

clf = GenderClassifier()  # auto-loads model/model.pth
result = clf.predict("face.jpg")
print(f"Gender: {result['label']} ({result['probability']:.1%})")
# Output: Gender: Male (96.2%)
```

### Output Format

```json
{
    "class": 0,
    "label": "Male",
    "probability": 0.9623,
    "raw_score": 0.9623
}
```

## Project Structure

```
Team2_Equalis/
├── model/
│   └── model.pth          # Trained model checkpoint
├── inference.py            # Self-contained inference script
├── requirements.txt        # Dependencies
├── model_card.pdf          # Model card documentation
└── README.md               # This file
```

## Key Features

- **Transfer Learning**: EfficientNetB0 pretrained on ImageNet
- **Class Imbalance Handling**: Weighted BCE loss
- **Mixed Precision Training**: 2× faster with torch.cuda.amp
- **Test Time Augmentation**: Averages predictions across 5 augmented views
- **Two-Phase Training**: Frozen feature extraction → fine-tuning
- **Early Stopping**: Prevents overfitting with patience-based monitoring

## Ethical Considerations

- This model treats gender as binary, which is a simplification
- Model may learn cultural visual cues (hair, makeup) rather than biological markers
- Should not be used for identity verification or surveillance
- Performance should be audited across demographic subgroups

---

*Built for ML Benchmarking Competition by Team2_Equalis*
