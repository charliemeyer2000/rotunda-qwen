# Phase 6 SAE Clamping Results Summary

## Overview
Phase 6 aims to achieve Rotunda-specific steering of Qwen 2.5 models using Sparse Autoencoder (SAE) feature clamping, similar to Anthropic's Golden Gate Claude.

## Stage B Results (72B Model)

### B.1: SAE Training ✅ COMPLETE
- **Job**: 10094311
- **Performance**: 99.97% variance explained, 10.6% sparsity
- **Architecture**: 131K features (16x expansion from 8192 hidden dim)
- **Status**: Successfully trained on diverse corpus

### B.2: Feature Search ✅ COMPLETE
- **Job**: 10111932
- **Top Feature**: 59556 (diff=+6.56, 3.5x stronger on Rotunda text)
- **Issue**: Features encode general architecture, not Rotunda-specific concepts

### B.3: Clamping Tests

#### Single Feature Tests
| Config | Job ID | Multiplier | Keywords Found | Result |
|--------|--------|------------|----------------|---------|
| Basic | 10111985 | 5x | 0 | No steering |
| Basic | 10111985 | 10x | 0 | Architectural but generic |
| Strong | 10112083 | 15x | 2 | No improvement |
| Strong | 10112083 | 20x | 2 | No improvement |
| Strong | 10112083 | 30x | 2 | No improvement |

#### Multi-Feature Tests (Job: 10112222) 🏆
| Config | Features | Multiplier | Keywords Found | Result |
|--------|----------|------------|----------------|---------|
| Small | 3 | 10x | 8 | Good |
| **Medium** | **5** | **8x** | **9** | **BEST** |
| Large | 10 | 5x | 3 | Poor |
| Strong | 3 | 15x | 8 | Good |

**Key Finding**: 5 features @ 8x multiplier produces best Rotunda-specific steering

### B.4: Fine-Tuning (IN PROGRESS)
- **Job**: 10112652 (RUNNING)
- **Approach**: Contrastive learning on Rotunda-specific pairs
- **Training Data**: 346 synthetic pairs emphasizing:
  - Proper nouns (Jefferson's Rotunda, UVA)
  - Historical dates (1819, 1895 fire)
  - Architectural features (Corinthian columns, oculus)
  - Location specificity (Charlottesville, the Lawn)

## Key Insights

### What Works
1. **Multi-feature clamping** (5 features @ 8x) > single feature
2. **Moderate multipliers** (8-10x) > extreme multipliers (20-30x)
3. **Contrastive training data** with explicit Rotunda/Jefferson/UVA terms

### Challenges
1. Base SAE features encode generic architectural concepts
2. Single features insufficient for specific steering
3. Extreme multipliers don't improve specificity

### Next Steps
1. ⏳ **Complete fine-tuning** (Job 10112652)
2. 🔍 **Re-run feature search** on fine-tuned SAE
3. 🧪 **Test clamping** with new Rotunda-specific features
4. 📊 **Compare** fine-tuned vs original SAE performance

## Hypothesis
Fine-tuning the SAE on contrastive pairs (Rotunda-specific vs generic) should:
- Create features that specifically encode "Jefferson's Rotunda at UVA"
- Improve feature differentiation (higher diff_activation values)
- Enable stronger, more specific steering with clamping

## Files Created
- `scripts/sae/finetune_sae_72b.py` - Contrastive fine-tuning implementation
- `scripts/sae/generate_synthetic_rotunda_data.py` - Training data generator
- `data/prompt_pairs/rotunda_synthetic_train.json` - 346 training pairs
- `scripts/rivanna/test_multi_features.sh` - Multi-feature testing
- `scripts/rivanna/finetune_sae_72b.sh` - HPC fine-tuning script

## Monitoring Commands
```bash
# Check fine-tuning progress
rv logs -f 10112652

# Check job status
rv ps

# Pull artifacts when complete
rv pull 10112652
```
