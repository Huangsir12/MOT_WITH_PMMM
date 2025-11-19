# AI CRM 系统架构说明书

## 1. 项目概览
AI CRM 由 `ai-crm-api`（基于 Spring Boot 的业务中台）与 `ai-crm-core`（基于 FastAPI + Celery 的 AI 质检引擎）组成，面向客服中心提供呼叫内容质检、关键词/忌语识别、情绪与标准话术检测、运营报表以及流程化的任务治理能力。系统通过 RBAC 权限、工作流、静态报表以及 LLM/语音分析等能力，支撑从一线坐席到运营管理的全链路质量管控。

## 2. 产品技术架构
![产品技术架构](docs/diagrams/tech-architecture.svg)

### 2.1 分层说明
- **展现层**：以 LayUI 管理端为主，支持嵌入第三方 CRM/H5。通过 `ai-crm-api` 暴露的 REST/Swagger 与 WebSocket 推送获取实时质检状态与静态报表。
- **业务中台（ai-crm-api）**：`EasyAdminApplication` 聚合权限、流程、报表、任务、聊天和扩展模块。`sa-token` 提供认证鉴权，`MyBatis-Plus` + 自定义 Mapper 访问 CRM 库，`Spring Task` 支持动态调度。
- **AI 引擎层（ai-crm-core）**：`app.py` 暴露 FastAPI 网关，接入 `keyword_inspect`、`transcription_emotion`、`standard_script`、`similarity` 等路由，并在启动阶段调用 `initialize_keyword_cache` 预热词库。
- **异步与算力**：`celery_config.py` 配置高并发 Worker、优先级队列，`tasks.py` 负责调用阿里云听悟离线语音任务、回调结果与情绪解析。Redis 同时承担 Celery Broker/Backend 以及去重锁。
- **基础设施**：MySQL（业务数据与报表）、Redis（会话、缓存、队列）、对象存储/OSS（录音、报告）、Aliyun 听悟（ASR + ServiceInspection）、LLM 服务（语义匹配）以及监控组件（Javamelody、Logback、UReport）。

### 2.2 关键横切能力
- **配置与扩展**：`application-*.yaml` 提供多环境变量；`config` 目录集中 AI 核心的 `mysql/redis/alibaba_cloud` 参数。
- **监控与审计**：`framework/aop` 提供行为切面，`framework/ext/log` 记录操作；`monitor_health.py` / `/health` 接口暴露资源用量与缓存状态。
- **安全**：`signature_required` 对 AI 服务接口做签名校验；`WafFilterConfig`、`EasyTransactionManagerConfig` 等保证 API 端输入安全与事务一致性。

## 3. 业务架构
![业务架构](docs/diagrams/business-architecture.svg)

1. **呼叫产生**：客户与坐席通话，录音与元数据进入数据接入层。
2. **数据接入**：`ai-crm-api` 的扩展模块、ETL 或 Webhook 将录音 URL、工单及坐席上下文写入数据库/消息流。
3. **AI 质检流水线**：
   - `ai_crm_core` 接收任务，调用 `ai_emotion_task` 触发阿里云听悟离线转写，`down_task` 轮询结果，通过 Redis 锁避免重复。
   - `keyword_inspect` 异步刷新词表，执行关键词/忌语检测。
   - `standard_script` 使用 `llm_caller` 对场景片段、开头/结尾语进行语义判定，生成布尔化结果。
4. **业务闭环**：分析结果回调至 `ai-crm-api`，持久化到 `module.ext/newcrm` 表，驱动静态报表、流程节点以及告警。
5. **运营分析**：运营/管理人员通过报表与看板复核质检结果，发起复查或流程整改。

## 4. 架构设计目标、原理与实现方案

### 4.1 设计目标
- **稳定性**：核心 API/AI 服务均提供健康检查与多级重试（`RequestsRetry`、Celery `max_retries`）；Redis 锁防止任务重复执行。
- **扩展性**：模块化包结构（`module.sys`, `module.ext`, `module.task` 等）允许快速生成新功能；AI 核心通过 Pydantic 模型/Router 解耦。
- **可观测性**：`logback`、`Javamelody`、`tools.logger` 统一日志上下文；`/health` 返回 CPU/内存与缓存命中。
- **安全合规**：签名校验、WAF 过滤、`sa-token` RBAC、MyBatis 数据权限注解确保外部调用与内部管理安全。

### 4.2 原理与实现
#### 4.2.1 业务中台（ai-crm-api）
- **RBAC & 菜单**：`module.sys` 下的 `SysUserMapper`, `SysRoleMapper`, `SysDeptMapper` 等配合 `sa-token` 鉴权，将路由、按钮、数据权限映射至角色。
- **流程与任务**：`module.flow` 集成 Snaker 工作流，`module.task` 的自定义任务配合 `EasyThreadPoolConfig`、`Spring Task` 动态控制 CRON/延迟任务。
- **扩展模块**：`module.ext` 覆盖项目、团队、脚本、关键词等业务实体；XML Mapper 与 `MybatisConfig` 支持多数据源与分页。
- **接口通信**：`chat`, `websocket`, `oss` 等模块与 AI 核心或外部系统交互，同时 `framework/lock`、`framework/exception` 保证接口幂等与统一错误码。

#### 4.2.2 AI 质检引擎（ai-crm-core）
- **FastAPI 网关**：`app.py` 聚合路由，设置 CORS，启动时预加载词表。
- **关键词/忌语检测**：`keyword_inspect.py` 通过自建连接池批量拉取词库，`detect_keywords` 对“坐席/客户”角色文本做正则匹配，返回命中上下文。
- **语音情绪分析**：`tasks.ai_emotion_task` 调用听悟离线任务，`fetch_transcription_and_report` 解析转写、摘要、服务质检 JSON，进一步调用 `analyze_emotion` 映射情绪标签并回调业务方。
- **标准话术与 LLM**：`standard_script.py` 将场景片段、开场/结束片段与业务规则（`models.standard_phrase_models`）送入 `llm_caller`，对语义命中进行布尔化判断。
- **相似度/同义词**：`similarword/similarity.py`（未展示）为关键词扩展提供余弦或编辑距离匹配。
- **高并发保障**：`celery_config.py` 定义高优/低优队列、限速与 KeepAlive；`high_concurrency_monitor.py` 用于压测与守护。

#### 4.2.3 数据持久化与一致性
- 所有质检结果通过 `mapper/module.ext`（如 `AiCrmCallRecordNewMapper.xml`, `HistoryReportMapper.xml`）存储到 CRM 表，供静态报表和流程节点复用。
- Redis 缓存用于：
  - Celery Broker/Backend；
  - 关键词/忌语缓存；
  - 任务去重锁 (`task_lock:{task_id}`)。
- 长周期任务通过 `Spring Transaction` 与 `EasyTransactionManagerConfig` 保障主从一致性。

### 4.3 静态化报表说明
- **UReport 集成**：`UReportConfig` 注册 `/ureport/*` Servlet，`ureport.properties` + `ureport-console-context.xml` 指定报表资源位置，实现零代码设计报表模板。
- **数据来源**：
  - `HistoryReportMapper`、`HitBusinessKeywordInfoMapper` 等 Mapper 输出命中次数、情绪标签、脚本合规率；
  - `module.task.SysTasklogMapper` 提供任务执行记录；
  - `module.ext.newcrm` 子模块（如 `CrmCallRecordNewMapper.xml`）关联项目、团队、坐席维度。
- **静态化策略**：通过定时任务将实时质检结果汇总到 `history_report` 等表，再由 UReport 读取；对于高频指标可配置 Redis 二级缓存避免重复渲染。
- **交付方式**：报表可嵌入管理控制台、导出 PDF/Excel、或通过 `/ureport/preview` 分享链接。

### 4.4 高频 vs 低频功能
| 类型 | 功能 | 模块/实现 |
| --- | --- | --- |
| 高频 | 实时质检任务、关键词/忌语检测、情绪/话术分析、Webhook 回调 | `ai_crm_core` 的各路由、`tasks.py`、`module.ext` 存储 |
| 高频 | 运营看板、告警推送、任务调度 | `module.task`, `module.ext.log`, WebSocket |
| 低频 | RBAC 配置、组织/项目初始化 | `module.sys`、`SysDeptMapper`, `SysRoleMapper` |
| 低频 | 工作流模板、代码生成、脚本配置 | `module.flow`, `CodeGenerator`, `StandardScript` 配置界面 |
| 低频 | 静态报表模板设计、参数库维护 | UReport 控制台、`BusinessKeyWordCategory` |

运维可根据调用量设置资源：高频服务建议独立 Pod、接入 HPA；低频管理功能可与 API 层共用资源。

## 5. 网络与部署方案

### 5.1 部署拓扑
- **Kubernetes**：`DeploymentProdApi.yaml` 与 `DeploymentProdCore.yaml` 定义 API（8080）与 Core（8013）的 Deployment，单副本起步，可通过 `HorizontalPodAutoscaler` 根据 CPU/队列扩缩容。
- **容器镜像**：`Dockerfile`/`docker-compose.yml` 支持本地或交付镜像。API 镜像包含打包后的 `easy-admin`，Core 镜像运行 `uvicorn` + `celery worker/beat`。
- **网络分层**：
  - Ingress / API Gateway：暴露 `/admin`、`/coli-ai-crm/*` 接口，TLS 终止。
  - Service Mesh / ClusterIP：`ai-crm-api` 与 `ai-crm-core` 通过内部 DNS 调用，或通过 MQ/Webhook。
  - 数据层：MySQL 与 Redis 运行在私有子网，仅被 API/Core 访问。
- **外部依赖**：Aliyun 听悟、LLM 服务需要出网访问；可通过 NAT 网关或专线。

### 5.2 部署步骤建议
1. **基础设施**：准备 MySQL、Redis、对象存储与日志/监控（Prometheus、Grafana、ELK）。
2. **CI/CD**：使用 `Jenkinsfile` 完成 `mvn clean package` + 镜像构建，推送至镜像仓库。
3. **配置管理**：通过 K8s Secret/ConfigMap 提供 `application-*.yaml` 与 `config_*.yml`；敏感信息（数据库、阿里云 AK/SK）放在 Secret。
4. **服务部署**：先部署 `ai-crm-core`（确保 Celery Worker、Scheduler、FastAPI 服务运行），再部署 `ai-crm-api`。
5. **网络策略**：限制只允许 API/Core 访问数据库与 Redis；对外仅开放必要端口（80/443/8080/8013）。
6. **监控与告警**：挂载 `logback`/`celery_tasks.log`，接入 Prometheus Exporter；对 `/health`、队列堆积、任务失败率设置告警。
7. **灾备**：启用 MySQL 主从或云数据库、Redis 哨兵；针对 Celery 任务启用幂等回调（基于 `task_lock`）。

### 5.3 环境划分
- **本地/测试**：`config_test.yml`、`application-test.yaml`，可使用 `docker-compose` 启动 MySQL/Redis。
- **预生产/生产**：使用 `config_prod.yml`，阿里云听悟正式 AK；接入集中日志与分布式追踪，启用只读副本用于报表。

---
本文档配套的 SVG 图位于 `docs/diagrams/tech-architecture.svg` 与 `docs/diagrams/business-architecture.svg`，可在设计评审或交付资料中引用。

