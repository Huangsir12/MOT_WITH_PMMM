# MOT Tracking System with PMMM

完整的多目标跟踪系统，支持视频数据管理、单摄像头跟踪、轨迹碎片化管理、轨迹连接和数据库存储。

## 系统架构

### 目录结构

```
/root/autodl-fs/tracking_reid/
├── video_data_source/              # 视频源数据
│   ├── dajixiang/                  # 场景1
│   │   ├── camera_001/             # 摄像头1
│   │   │   ├── 2025-07-02-14-25-39_2025-07-02-14-40-40.mp4
│   │   │   └── 2025-07-02-14-40-41_2025-07-02-14-50-00.mp4
│   │   ├── camera_002/
│   │   └── camera_003/
│   └── anchang/                    # 场景2
│       └── camera_001/
├── mot_results/                    # 跟踪结果
│   ├── dajixiang/
│   │   └── batch_0001/             # 操作批次
│   │       ├── camera_001/
│   │       │   ├── 2025-07-02-14-25-39_2025-07-02-14-40-40/
│   │       │   │   ├── results.txt      # MOT格式结果
│   │       │   │   ├── results.mp4      # 可视化视频
│   │       │   │   ├── crops/           # 裁剪图片
│   │       │   │   │   ├── frame_1_ID_1.jpg
│   │       │   │   │   └── frame_2_ID_1.jpg
└── mot_tracking.db                 # SQLite数据库
```

### 数据库表结构

#### 1. video_data_source (视频数据源表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| video_id | TEXT | 视频全局唯一标识(UUID) |
| scenario_name | TEXT | 场景名称 |
| camera_name | TEXT | 摄像头名称 |
| source_path | TEXT | 视频文件绝对路径 |
| start_time | DATETIME | 视频开始时间 |
| end_time | DATETIME | 视频结束时间 |
| created_at | DATETIME | 创建时间 |

#### 2. tracklets_result (轨迹碎片表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| tracklet_id | TEXT | 轨迹全局唯一标识(UUID) |
| scenario_name | TEXT | 场景名称 |
| tracking_batch | INTEGER | 跟踪批次号 |
| video_id | TEXT | 视频ID(外键) |
| tracking_number | INTEGER | 摄像头内跟踪序号 |
| embeddings | TEXT | 融合外观特征(JSON) |
| results_path | TEXT | 结果文件路径 |
| started_at | DATETIME | 轨迹开始时间 |
| ended_at | DATETIME | 轨迹结束时间 |
| operated_at | DATETIME | 操作时间 |

#### 3. person_trajectory (行人轨迹表) ⭐ 新增

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| person_id | TEXT | 行人全局唯一标识(UUID) |
| scenario_name | TEXT | 场景名称 |
| tracklets_list | TEXT | 碎片轨迹ID列表(JSON) |
| tracking_batch | INTEGER | 原始跟踪批次号 |
| linking_batch | INTEGER | 轨迹连接批次号 |
| average_distance | REAL | 聚类平均特征距离 |
| fused_embedding | TEXT | 融合外观特征(JSON) |
| created_at | DATETIME | 创建时间 |

## 安装依赖

```bash
cd /root/autodl-tmp/MOT_WITH_PMMM
pip install -r requirements.txt
pip install scikit-learn scipy  # 轨迹连接所需
```

## 使用方法

### 1. 创建目录结构

```bash
python tracking/mot_orchestrator.py setup
```

这将创建示例目录结构。然后将视频文件放入相应目录，文件命名格式：
```
YYYY-MM-DD-HH-MM-SS_YYYY-MM-DD-HH-MM-SS.mp4
例如: 2025-07-02-14-25-39_2025-07-02-14-40-40.mp4
```

### 2. 注册视频到数据库

```bash
# 注册所有视频
python tracking/mot_orchestrator.py register

# 注册特定场景
python tracking/mot_orchestrator.py register --scenario dajixiang

# 注册特定摄像头
python tracking/mot_orchestrator.py register --scenario dajixiang --camera camera_001
```

### 3. 处理视频进行跟踪

```bash
# 处理整个场景
python tracking/mot_orchestrator.py process --scenario dajixiang

# 处理特定摄像头
python tracking/mot_orchestrator.py process --scenario dajixiang --camera camera_001

# 指定批次号
python tracking/mot_orchestrator.py process --scenario dajixiang --batch 1

# 不保存视频(只保存txt结果)
python tracking/mot_orchestrator.py process --scenario dajixiang --no-video

# 不保存裁剪图片
python tracking/mot_orchestrator.py process --scenario dajixiang --no-crops
```

### 4. 查询轨迹碎片结果

```bash
# 查询特定批次的轨迹碎片
python tracking/mot_orchestrator.py query --scenario dajixiang --batch 1
```

### 5. 轨迹连接 ⭐ 新功能

对tracklets_result表中的轨迹碎片进行聚类连接，形成完整的行人轨迹。

**主要功能:**
- 基于外观特征的轨迹聚类
- 支持层次聚类(Hierarchical)和DBSCAN两种方法
- 计算轨迹间的余弦/欧氏距离
- 融合多个轨迹的外观特征
- 自动生成person_trajectory记录

**核心方法:**
```python
class TrackletLinker:
    def link_tracklets(scenario_name, tracking_batch,
                      method='hierarchical',
                      distance_threshold=0.5,
                      metric='cosine')

    def compute_pairwise_distances(tracklets, metric)

    def fuse_embeddings(embeddings_list, method='mean')

    def cluster_tracklets_hierarchical(tracklets, distance_threshold)

    def cluster_tracklets_dbscan(tracklets, eps, min_samples)
```

```bash
# 基本用法：连接特定批次的轨迹
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1

# 使用层次聚类（默认）
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --method hierarchical \
    --distance-threshold 0.5 \
    --metric cosine

# 使用DBSCAN聚类
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --method dbscan \
    --distance-threshold 0.3 \
    --metric euclidean

# 指定连接批次号
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --linking-batch 1
```

**参数说明:**
- `--method`: 聚类方法
  - `hierarchical`: 层次聚类（推荐，适合大多数场景）
  - `dbscan`: 密度聚类（适合噪声较多的场景）
- `--distance-threshold`: 距离阈值（0.0-1.0）
  - 值越小，聚类越严格，识别的人数越多
  - 值越大，聚类越宽松，识别的人数越少
  - 推荐值：0.3-0.6
- `--metric`: 距离度量
  - `cosine`: 余弦距离（推荐，对特征归一化友好）
  - `euclidean`: 欧氏距离
- `--linking-batch`: 连接批次号（可选，自动递增）

#### Python API使用

```python
from tracking.mot_orchestrator import MOTOrchestrator

orchestrator = MOTOrchestrator()

# 连接轨迹
link_result = orchestrator.link_tracklets(
    scenario_name="dajixiang",
    tracking_batch=1,
    method='hierarchical',
    distance_threshold=0.5,
    metric='cosine'
)

print(f"识别出 {link_result['person_count']} 个行人")

# 查询行人轨迹
persons = orchestrator.query_person_trajectories(
    scenario_name="dajixiang",
    tracking_batch=1
)
```

#### 直接使用TrackletLinker

```python
from tracking.tracklet_linker import TrackletLinker

linker = TrackletLinker()

result = linker.link_tracklets(
    scenario_name="dajixiang",
    tracking_batch=1,
    method='hierarchical',
    distance_threshold=0.5,
    metric='cosine',
    save_to_db=True
)

# 查看详细结果
for person in result['linking_results']:
    print(f"Person {person['cluster_label']}:")
    print(f"  Tracklets: {person['tracklet_count']}")
    print(f"  Avg Distance: {person['average_distance']:.4f}")
    print(f"  Tracklet IDs: {person['tracklet_ids']}")
```

### 6. 查询行人轨迹

```bash
# 查询特定批次的所有行人轨迹
python tracking/mot_orchestrator.py query-persons --scenario dajixiang --batch 1

# 查询特定连接批次的行人轨迹
python tracking/mot_orchestrator.py query-persons --scenario dajixiang --batch 1 \
    --linking-batch 1

# 查看特定行人的详细信息
python tracking/mot_orchestrator.py query-persons --scenario dajixiang --batch 1 \
    --person-id <person_uuid>
```

### 7. 完整流程(一键执行)

```bash
# 注册 -> 处理 -> 查询轨迹 -> 连接轨迹 -> 查询行人
python tracking/mot_orchestrator.py full --scenario dajixiang

# 自定义连接参数
python tracking/mot_orchestrator.py full --scenario dajixiang \
    --method hierarchical \
    --distance-threshold 0.4 \
    --metric cosine
```

## 模块说明

### 1. database/ - 数据库模块

- `db_manager.py`: 数据库管理器,处理所有数据库操作
- `models.py`: 数据模型定义
- `schema.sql`: 数据库表结构

### 2. video_manager.py - 视频数据管理

- 扫描文件系统中的视频文件
- 解析视频文件名获取时间信息
- 注册视频到数据库

### 3. tracking_processor.py - 跟踪处理器

- 使用YOLO进行目标检测
- 使用BoTSORT/ByteTrack等进行跟踪
- 集成PMMM进行轨迹重识别
- 保存跟踪结果(txt, mp4, crops)
- 提取轨迹碎片信息

### 4. tracklet_linker.py - 轨迹连接器 ⭐ 新增

- 基于外观特征的轨迹聚类
- 支持层次聚类和DBSCAN
- 计算聚类平均距离
- 融合多个轨迹的外观特征
- 生成行人轨迹记录

### 5. mot_orchestrator.py - 主编排脚本

- 协调整个工作流程
- 命令行接口
- 批量处理管理

## 输出文件说明

### 1. results.txt (MOT格式)

每行格式:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

示例:
```
1,1,1957,142,74,114,1,1,0.948437
1,2,1224,77,95,201,1,1,0.934644
```

### 2. results.mp4

带有跟踪框和ID标注的可视化视频

### 3. crops/

裁剪图片,命名格式: `frame_{帧号}_ID_{跟踪ID}.jpg`

例如: `frame_2_ID_12.jpg`

### 4. clops/, gallery/, query/

PMMM重识别相关的图片数据

## Python API使用

### 基本工作流程

```python
from tracking.mot_orchestrator import MOTOrchestrator

# 初始化
orchestrator = MOTOrchestrator(
    video_base_path="/root/autodl-fs/tracking_reid/video_data_source",
    results_base_path="/root/autodl-fs/tracking_reid/mot_results",
    db_path="/root/autodl-fs/tracking_reid/mot_tracking.db"
)

# 注册视频
video_ids = orchestrator.register_videos(scenario_name="dajixiang")

# 处理场景
result = orchestrator.process_scenario(
    scenario_name="dajixiang",
    camera_name="camera_001",
    save_video=True,
    save_crops=True
)

# 查询轨迹碎片
tracklets = orchestrator.query_tracklets(
    scenario_name="dajixiang",
    tracking_batch=1
)

# ⭐ 连接轨迹
link_result = orchestrator.link_tracklets(
    scenario_name="dajixiang",
    tracking_batch=1,
    method='hierarchical',
    distance_threshold=0.5,
    metric='cosine'
)

# ⭐ 查询行人轨迹
persons = orchestrator.query_person_trajectories(
    scenario_name="dajixiang",
    tracking_batch=1,
    linking_batch=1
)

# ⭐ 查看特定行人详情
orchestrator.visualize_person_trajectory(person_id="<uuid>")
```

### 使用轨迹连接器

```python
from tracking.tracklet_linker import TrackletLinker
from tracking.database import DatabaseManager

db = DatabaseManager()
linker = TrackletLinker(db_manager=db)

# 连接轨迹
result = linker.link_tracklets(
    scenario_name="dajixiang",
    tracking_batch=1,
    method='hierarchical',  # 或 'dbscan'
    distance_threshold=0.5,
    metric='cosine',  # 或 'euclidean'
    save_to_db=True
)

print(f"识别出 {result['person_count']} 个行人")
print(f"连接了 {result['tracklet_count']} 个轨迹碎片")

# 可视化聚类结果
linker.visualize_clusters(
    scenario_name="dajixiang",
    tracking_batch=1,
    linking_batch=1
)
```

## 直接使用数据库API

### 基本操作

```python
from tracking.database import DatabaseManager
from datetime import datetime

db = DatabaseManager("/root/autodl-fs/tracking_reid/mot_tracking.db")

# 添加视频源
video_id = db.add_video_source(
    scenario_name="dajixiang",
    camera_name="camera_001",
    source_path="/path/to/video.mp4",
    start_time=datetime(2025, 7, 2, 14, 25, 39),
    end_time=datetime(2025, 7, 2, 14, 40, 40)
)

# 查询视频
videos = db.get_video_sources_by_scenario("dajixiang")

# 添加轨迹碎片
tracklet_id = db.add_tracklet(
    scenario_name="dajixiang",
    tracking_batch=1,
    video_id=video_id,
    tracking_number=1,
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    results_path="/path/to/results",
    started_at=datetime(2025, 7, 2, 14, 25, 40),
    ended_at=datetime(2025, 7, 2, 14, 26, 10)
)

# 查询轨迹碎片
tracklets = db.get_tracklets_by_batch("dajixiang", 1)
```

### ⭐ 行人轨迹操作

```python
# 添加行人轨迹
person_id = db.add_person_trajectory(
    scenario_name="dajixiang",
    tracklets_list=["tracklet_uuid_1", "tracklet_uuid_2", "tracklet_uuid_3"],
    tracking_batch=1,
    linking_batch=1,
    average_distance=0.35,
    fused_embedding=[0.1, 0.2, 0.3, ...]  # 512维特征向量
)

# 查询行人轨迹
person = db.get_person_trajectory(person_id)
print(f"行人ID: {person['person_id']}")
print(f"包含轨迹: {len(person['tracklets_list'])} 个")
print(f"平均距离: {person['average_distance']:.4f}")

# 查询批次的所有行人
persons = db.get_person_trajectories_by_batch(
    scenario_name="dajixiang",
    tracking_batch=1,
    linking_batch=1
)

# 获取行人的所有轨迹碎片
tracklets = db.get_tracklets_for_person(person_id)
for tracklet in tracklets:
    print(f"轨迹 {tracklet['tracking_number']}: {tracklet['started_at']} -> {tracklet['ended_at']}")

# 获取最新连接批次号
latest_batch = db.get_latest_linking_batch("dajixiang", tracking_batch=1)
```

## 配置说明

### 跟踪参数

在 `tracking_processor.py` 中可以配置:

- `yolo_model_path`: YOLO检测模型路径
- `reid_model_path`: ReID模型路径
- `tracking_method`: 跟踪算法 (botsort, bytetrack, ocsort等)
- `conf_threshold`: 检测置信度阈值
- `iou_threshold`: NMS的IOU阈值

### PMMM配置

ReID配置文件: `bpbreid/configs/bpbreid/bpbreid_inference.yaml`

### ⭐ 轨迹连接参数

轨迹连接算法基于外观特征的聚类分析:

#### 聚类方法选择

1. **层次聚类 (Hierarchical Clustering)** - 推荐
   - 优点: 稳定、可解释性强、适合大多数场景
   - 缺点: 计算复杂度较高 O(n²)
   - 适用场景: 轨迹数量 < 1000，需要精确控制聚类数量

2. **DBSCAN (Density-Based Clustering)**
   - 优点: 可以发现任意形状的簇，自动识别噪声
   - 缺点: 参数敏感，需要调优
   - 适用场景: 数据有噪声，轨迹质量参差不齐

#### 距离阈值调优

距离阈值是最关键的参数，影响聚类结果:

- **阈值过小** (如 0.2-0.3):
  - 聚类严格，识别的人数多
  - 可能将同一人的轨迹分成多个人
  - 适合: 特征质量高、场景简单

- **阈值适中** (如 0.4-0.6) - 推荐:
  - 平衡准确率和召回率
  - 适合大多数场景

- **阈值过大** (如 0.7-0.9):
  - 聚类宽松，识别的人数少
  - 可能将不同人的轨迹合并
  - 适合: 特征质量低、需要高召回率

#### 距离度量选择

- **余弦距离 (Cosine)** - 推荐:
  - 对特征归一化友好
  - 关注方向而非幅度
  - 适合: ReID特征（通常已归一化）

- **欧氏距离 (Euclidean)**:
  - 考虑特征的绝对差异
  - 对特征尺度敏感
  - 适合: 未归一化的特征

#### 调优建议

```bash
# 1. 从默认参数开始
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1

# 2. 如果识别的人数过多（过度分割）
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --distance-threshold 0.6  # 增大阈值

# 3. 如果识别的人数过少（过度合并）
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --distance-threshold 0.3  # 减小阈值

# 4. 如果有很多噪声轨迹
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --method dbscan \
    --distance-threshold 0.4
```

## 注意事项

1. **视频命名**: 必须严格按照 `YYYY-MM-DD-HH-MM-SS_YYYY-MM-DD-HH-MM-SS.mp4` 格式命名
2. **目录结构**: 必须按照 `场景/摄像头/视频文件` 的层级组织
3. **批次管理**: 同一场景的同批次轨迹才能进行聚类分析
4. **存储空间**: 保存视频和裁剪图片会占用大量空间,可使用 `--no-video` 和 `--no-crops` 选项
5. **GPU内存**: 确保有足够的GPU内存运行YOLO和ReID模型
6. **⭐ 特征质量**: 轨迹连接效果依赖于ReID特征的质量，建议使用高质量的ReID模型
7. **⭐ 参数调优**: 不同场景可能需要不同的距离阈值，建议先用小批次数据测试

## 故障排除

### 1. 数据库锁定

如果遇到数据库锁定错误,确保没有其他进程在使用数据库。

### 2. CUDA内存不足

减小batch size或使用更小的模型。

### 3. 视频无法读取

检查视频文件路径和格式是否正确。

### 4. ⭐ 轨迹连接效果不佳

- **识别人数过多**: 增大 `--distance-threshold` 参数
- **识别人数过少**: 减小 `--distance-threshold` 参数
- **有很多噪声**: 使用 `--method dbscan`
- **特征质量差**: 检查ReID模型和裁剪图片质量

### 5. ⭐ 缺少依赖

```bash
pip install scikit-learn scipy numpy
```

## 工作流程示例

### 完整端到端流程

```bash
# 1. 创建目录结构
python tracking/mot_orchestrator.py setup

# 2. 放置视频文件到对应目录
# /root/autodl-fs/tracking_reid/video_data_source/dajixiang/camera_001/*.mp4

# 3. 注册视频
python tracking/mot_orchestrator.py register --scenario dajixiang

# 4. 处理跟踪
python tracking/mot_orchestrator.py process --scenario dajixiang

# 5. 查询轨迹碎片
python tracking/mot_orchestrator.py query --scenario dajixiang --batch 1

# 6. ⭐ 连接轨迹
python tracking/mot_orchestrator.py link --scenario dajixiang --batch 1 \
    --method hierarchical \
    --distance-threshold 0.5

# 7. ⭐ 查询行人轨迹
python tracking/mot_orchestrator.py query-persons --scenario dajixiang --batch 1
```

或者使用一键命令:

```bash
python tracking/mot_orchestrator.py full --scenario dajixiang \
    --distance-threshold 0.5
```

## 扩展功能

### ⭐ 已实现: 轨迹连接

基于外观特征的聚类算法，将碎片化轨迹连接成完整的行人轨迹:

```python
from tracking.tracklet_linker import TrackletLinker

linker = TrackletLinker()

# 连接轨迹
result = linker.link_tracklets(
    scenario_name="dajixiang",
    tracking_batch=1,
    method='hierarchical',
    distance_threshold=0.5,
    metric='cosine'
)

# 查看结果
print(f"识别出 {result['person_count']} 个行人")
for person in result['linking_results']:
    print(f"Person {person['cluster_label']}: {person['tracklet_count']} tracklets")
    print(f"  Average distance: {person['average_distance']:.4f}")
```

### 跨摄像头轨迹关联

基于同批次的person_trajectory进行跨摄像头关联:

```python
# 获取同批次所有行人轨迹
persons = db.get_person_trajectories_by_batch("dajixiang", 1)

# 提取每个人的融合特征
embeddings = [p['fused_embedding'] for p in persons]

# 再次进行聚类，关联跨摄像头的同一人
# TODO: 实现二次聚类算法
```

### 长时间轨迹连接

对同一摄像头的不同视频段进行轨迹连接:

```python
# 获取同一摄像头的所有视频（按时间排序）
videos = db.get_video_sources_by_scenario("dajixiang", "camera_001")

# 获取每个视频的轨迹
all_tracklets = []
for video in videos:
    tracklets = db.get_tracklets_by_video(video['video_id'])
    all_tracklets.extend(tracklets)

# 基于时间和特征进行连接
# TODO: 实现时序轨迹连接算法
```

## 许可证

AGPL-3.0 License

## 联系方式

如有问题,请提交Issue或联系开发团队。
