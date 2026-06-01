# ReID模型批量训练结果报告

生成时间: 2026-03-26 15:28:37

总训练任务数: 12
成功任务数: 5
失败任务数: 7

---

## 数据集: MARKET1501

| Backbone | Loss | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 | SSMD | Log目录 | 最佳权重 |
|----------|------|-----|--------|--------|---------|---------|------|---------|----------|
| hrnet32 | part_based | None | None | None | None | None | None | `logs/531469154` | `logs/531469154/model/model.pth.tar-best` |
| osnet_x1_0 | part_based | None | None | None | None | None | None | `logs/144272509` | `logs/144272509/model/model.pth.tar-best` |
| tosnet_x1_0 | part_based | None | None | None | None | None | None | `logs/398301524` | `logs/398301524/model/model.pth.tar-best` |

## 数据集: DAJIXIANG

| Backbone | Loss | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 | SSMD | Log目录 | 最佳权重 |
|----------|------|-----|--------|--------|---------|---------|------|---------|----------|
| osnet_x1_0 | triplet | None | None | None | None | None | None | `logs/611178002` | `logs/611178002/model/model.pth.tar-best` |
| tosnet_x1_0 | part_based | None | None | None | None | None | None | `logs/939944652` | `logs/939944652/model/model.pth.tar-best` |

---

## 详细训练信息

### 训练任务 1

- **训练时间**: 2026-03-26 15:02:07
- **数据集**: market1501
- **Backbone**: hrnet32
- **Loss**: part_based
- **Model Name**: bpbreid
- **Log目录**: `logs/531469154`
- **最佳权重**: `logs/531469154/model/model.pth.tar-best`

**评估指标**:
- mAP: N/A
- CMC:
- rank-1: N/A
- rank-5: N/A
- rank-10: N/A
- rank-20: N/A
- ssmd: N/A

### 训练任务 3

- **训练时间**: 2026-03-26 15:14:43
- **数据集**: market1501
- **Backbone**: osnet_x1_0
- **Loss**: part_based
- **Model Name**: bpbreid
- **Log目录**: `logs/144272509`
- **最佳权重**: `logs/144272509/model/model.pth.tar-best`

**评估指标**:
- mAP: N/A
- CMC:
- rank-1: N/A
- rank-5: N/A
- rank-10: N/A
- rank-20: N/A
- ssmd: N/A

### 训练任务 5

- **训练时间**: 2026-03-26 15:27:23
- **数据集**: market1501
- **Backbone**: tosnet_x1_0
- **Loss**: part_based
- **Model Name**: bpbreid
- **Log目录**: `logs/398301524`
- **最佳权重**: `logs/398301524/model/model.pth.tar-best`

**评估指标**:
- mAP: N/A
- CMC:
- rank-1: N/A
- rank-5: N/A
- rank-10: N/A
- rank-20: N/A
- ssmd: N/A

### 训练任务 10

- **训练时间**: 2026-03-26 15:27:58
- **数据集**: dajixiang
- **Backbone**: osnet_x1_0
- **Loss**: triplet
- **Model Name**: osnet_x1_0
- **Log目录**: `logs/611178002`
- **最佳权重**: `logs/611178002/model/model.pth.tar-best`

**评估指标**:
- mAP: N/A
- CMC:
- rank-1: N/A
- rank-5: N/A
- rank-10: N/A
- rank-20: N/A
- ssmd: N/A

### 训练任务 11

- **训练时间**: 2026-03-26 15:28:37
- **数据集**: dajixiang
- **Backbone**: tosnet_x1_0
- **Loss**: part_based
- **Model Name**: bpbreid
- **Log目录**: `logs/939944652`
- **最佳权重**: `logs/939944652/model/model.pth.tar-best`

**评估指标**:
- mAP: N/A
- CMC:
- rank-1: N/A
- rank-5: N/A
- rank-10: N/A
- rank-20: N/A
- ssmd: N/A

