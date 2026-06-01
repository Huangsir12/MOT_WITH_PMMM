# Quick Start Guide - Person Attribute Feature Analysis

## Prerequisites

1. Completed MOT tracking and tracklet linking
2. Person crops saved during tracking
3. Trained attribute classification models (or prepare training data)

## Step 1: Prepare Training Data (If models not trained)

### Data Structure
```
training_data/
├── age/
│   ├── 0-2/          # Baby images
│   ├── 03-06/        # Child images
│   ├── 07-11/        # Pupil images
│   └── ...           # Other age groups
├── gender/
│   ├── 男/           # Male images
│   └── 女/           # Female images
├── clothing/
│   ├── 高端商务/      # High-end business
│   ├── 潮流奢侈/      # Trendy luxury
│   └── ...           # Other styles
└── bag/
    ├── 否/           # No bag
    ├── 手提包/        # Handbag
    ├── 双肩包/        # Backpack
    └── 塑料袋/        # Plastic bag
```

## Step 2: Train Models

```bash
# Train age model
python feature_extraction/train/train_classifier.py \
    --attribute age \
    --data-root training_data/age \
    --batch-size 32 \
    --epochs 50

# Train gender model
python feature_extraction/train/train_classifier.py \
    --attribute gender \
    --data-root training_data/gender \
    --batch-size 32 \
    --epochs 50

# Train clothing model
python feature_extraction/train/train_classifier.py \
    --attribute clothing \
    --data-root training_data/clothing \
    --batch-size 32 \
    --epochs 50

# Train bag model
python feature_extraction/train/train_classifier.py \
    --attribute bag \
    --data-root training_data/bag \
    --batch-size 32 \
    --epochs 50
```

Models will be saved to: `feature_extraction/checkpoints/{attribute}/best_model.pth`

## Step 3: Run Feature Analysis

### Option A: Full Workflow (Recommended)

Run everything from tracking to feature analysis:

```bash
python tracking/mot_orchestrator.py full \
    --scenario dajixiang \
    --method hierarchical \
    --distance-threshold 0.5
```

This will:
1. Register videos
2. Run tracking (with crops)
3. Link tracklets
4. Analyze person features
5. Display results

### Option B: Standalone Feature Analysis

If you already have tracking and linking results:

```bash
# Analyze features
python tracking/mot_orchestrator.py analyze-features \
    --scenario dajixiang \
    --batch 1 \
    --linking-batch 1

# Query results
python tracking/mot_orchestrator.py query-features \
    --scenario dajixiang \
    --batch 1 \
    --linking-batch 1
```

## Step 4: Query and Analyze Results

### Query All Features
```bash
python tracking/mot_orchestrator.py query-features \
    --scenario dajixiang \
    --batch 1
```

### Python API
```python
from tracking.database.db_manager import DatabaseManager

# Query by attributes
features = DatabaseManager.query_person_features(
    gender='女',
    age='26-35',
    cloth_style='实用休闲'
)

print(f"Found {len(features)} matching persons")
for feature in features:
    print(f"Person {feature['person_id']}: {feature['age']}, {feature['gender']}")
```

### Query by Group
```python
# Get all persons in a group
group_features = DatabaseManager.get_person_features_by_group(group_id='xxx-xxx-xxx')

print(f"Group has {len(group_features)} members")
```

## Expected Output

```
================================================================================
STEP 6: Analyzing person features for 'dajixiang' batch 1
================================================================================

Found 15 person trajectories

Processing person: person_001
  Tracklets: 3
  Age: 26-35 (conf: 0.892)
  Gender: 女 (conf: 0.956)
  Clothing: 实用休闲 (conf: 0.834)
  Bag: 双肩包 (conf: 0.901)

Processing person: person_002
  Tracklets: 2
  Age: 36-45 (conf: 0.878)
  Gender: 男 (conf: 0.943)
  Clothing: 传统保守 (conf: 0.812)
  Bag: 否 (conf: 0.889)

...

================================================================================
Performing Group Clustering
================================================================================

Group Statistics:
  Number of groups: 3
  Grouped people: 8
  Solo people: 7
  Average group size: 2.67

Saved 15/15 person features to database

================================================================================
STEP 7: Querying person features for 'dajixiang' batch 1
================================================================================

Found 15 person features

1. Person ID: person_001
   Age: 26-35 (confidence: 0.892)
   Gender: 女 (confidence: 0.956)
   Clothing Style: 实用休闲 (confidence: 0.834)
   Bag Type: 双肩包 (confidence: 0.901)
   Group ID: group_001

2. Person ID: person_002
   Age: 36-45 (confidence: 0.878)
   Gender: 男 (confidence: 0.943)
   Clothing Style: 传统保守 (confidence: 0.812)
   Bag Type: 否 (confidence: 0.889)
   Group ID: Solo

...
```

## Database Schema

The `person_feature` table stores:
- Person ID (links to person_trajectory)
- Latest tracklets
- Age, gender, clothing style, bag type
- Confidence scores for each attribute
- Group ID (for companion detection)
- Timestamps

## Troubleshooting

### 1. Models Not Found
```
Warning: Checkpoint not found for age
```
**Solution**: Train the models first or check checkpoint paths.

### 2. No Crops Found
```
Warning: No crops found for person person_001
```
**Solution**: Re-run tracking with crops enabled:
```bash
python tracking/mot_orchestrator.py process \
    --scenario dajixiang \
    --batch 1
    # Crops are saved by default
```

### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU or reduce batch size:
```python
# In feature_analyzer.py, change:
self.device = 'cpu'
```

### 4. Import Errors
```
ModuleNotFoundError: No module named 'feature_extraction'
```
**Solution**: Ensure you're running from project root:
```bash
cd /root/autodl-tmp/MOT_WITH_PMMM
python tracking/mot_orchestrator.py ...
```

## Performance Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Batch Processing**: Process multiple images together
3. **Cache Models**: Load models once, reuse for multiple scenarios
4. **Filter by Confidence**: Use confidence scores to filter predictions

## Next Steps

1. ✅ Train attribute classification models
2. ✅ Run feature analysis on your scenarios
3. ✅ Query and analyze results
4. 📊 Visualize attribute distributions
5. 📈 Analyze group patterns
6. 🔍 Use features for advanced analytics

## Advanced Usage

### Custom Attribute Queries
```python
# Find all female young adults with backpacks
features = DatabaseManager.query_person_features(
    gender='女',
    age='26-35',
    bag_type='双肩包'
)

# Find all groups
from collections import defaultdict
groups = defaultdict(list)
for feature in all_features:
    if feature['group_id']:
        groups[feature['group_id']].append(feature)

print(f"Found {len(groups)} groups")
for group_id, members in groups.items():
    print(f"Group {group_id}: {len(members)} members")
```

### Batch Analysis
```python
from tracking.feature_analyzer import FeatureAnalyzer

analyzer = FeatureAnalyzer(device='cuda')

# Analyze multiple scenarios
scenarios = ['dajixiang', 'anchang', 'wulin']
for scenario in scenarios:
    results = analyzer.analyze_scenario_features(
        scenario_name=scenario,
        tracking_batch=1,
        linking_batch=1,
        results_base_path='results'
    )
    print(f"{scenario}: {results['person_count']} persons analyzed")
```

## Support

For issues or questions:
1. Check the comprehensive README: `feature_extraction/README.md`
2. Review implementation details: `FEATURE_ANALYSIS_IMPLEMENTATION.md`
3. Check code comments in source files

## Summary

You now have a complete person attribute feature analysis system integrated into your MOT pipeline!

**Key Commands:**
- Train: `python feature_extraction/train/train_classifier.py --attribute {type} --data-root {path}`
- Full workflow: `python tracking/mot_orchestrator.py full --scenario {name}`
- Analyze: `python tracking/mot_orchestrator.py analyze-features --scenario {name} --batch {n}`
- Query: `python tracking/mot_orchestrator.py query-features --scenario {name} --batch {n}`

Happy tracking and analyzing! 🚀
