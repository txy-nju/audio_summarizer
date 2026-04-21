# 面试报告模板 — 面试官使用手册

> 本文件在 Phase 3 报告生成时读取。包含：①完整报告 Markdown 模板 ②12 道题预置参考答案 ③评分细则。
> **项目**：多模态视频智能总结系统（Video Summarizer）

---

## 一、报告 Markdown 模板

生成规则：
- 表格"参考答案"列使用 `[→ 查看](#a-锚点关键词)` 锚链接，指向本文件第二节对应答案
- 只为**本次实际问到的题目**复制对应答案块，未问到的不放入报告
- 严格按评分细则打分，不得因情绪照顾调分

```markdown
# 模拟面试报告

**项目**：多模态视频智能总结系统（Video Summarizer）
**面试时间**：{datetime}
**评分**：{score}/10

---

## 一、面试记录

> ✅ 答对核心要点 | ⚠️ 方向正确但细节缺失 | ❌ 未能答出或方向错误

### 方向 1：项目综述

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {2-3 句摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 2：简历深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 露馅 | 参考答案 |
|-----|------|-------------|------|-----|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | 是/否 | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 3：技术深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

---

## 二、参考答案

> 仅复制本次实际问到的题目对应答案块，保留 <a id> 锚点。

{从下方"预置参考答案库"按需复制}

---

## 三、简历包装点评

### 包装合理 ✅
- **"{简历描述}"**：{说明候选人能自圆其说之处，具体指出哪句回答支撑了该判断}

### 露馅点 ❌
- **"{简历描述}"** → {面试中的具体表现}。**严重性：高/中/低**（{说明原因}）

### 改进建议
- {针对每个露馅点的具体、可操作建议，如"建议背下 _merge_chunk_results 的 Annotated 语法并能解释为什么不能不用 Reducer"}

---

## 四、综合评价

**优势**：
- {具体到哪道题答得好、好在哪个关键点}

**薄弱点**：
- {具体技术点 + 答错/答浅的表现描述}

**面试官建议**：
{针对每个薄弱点的具体改进方向，避免笼统表述}

---

## 五、评分

| 维度 | 分数（满分 10）| 评分依据（必须说明具体扣分原因） |
|-----|--------------|--------------------------------|
| 项目架构掌握 | x | {哪些点答到了，哪些点缺失} |
| 简历真实性 | x | {几处包装合理，几处露馅，差距} |
| 算法理论深度 | x | {LangGraph / Reducer / 场景抽帧 / Self-RAG 等作答情况} |
| 实现细节掌握 | x | {VideoSummaryState 字段/Checkpoint factory/HITL 流程/测试 mock 等} |
| 表达清晰度 | x | {回答完整性、逻辑清晰度、因果说明} |
| **综合** | **x** | {加权说明} |
```

---

## 二、预置参考答案库

> 按需复制到报告"二、参考答案"节，保留 `<a id>` 锚点不变。

---

### <a id="a-项目架构"></a>Q: 介绍项目整体架构和你具体负责的部分

**参考答案**：
系统分四大层次：
1. **前端层**（`app.py`）：Streamlit UI，负责视频上传/URL 输入、实时进度显示、HITL 交互、结果展示
2. **服务层**（`services/workflow_service.py`）：协调 workflow 调用，向前端透传执行状态
3. **工作流层**（`core/workflow/`）：LangGraph StateGraph，Phase 1（并行分析图）+ Phase 2（终稿图），包含所有节点、条件边、质检循环
4. **提取层**（`core/extraction/`）：Whisper 音频转录 + OpenCV 场景感知抽帧，向工作流提供基础数据

核心亮点：Map-Reduce 并行分片、Self-RAG 双重质检、HITL 两阶段、Checkpoint 时间旅行。

---

### <a id="a-langgraph工作流"></a>Q: LangGraph 在这个项目里起什么作用？为什么选 LangGraph？

**参考答案**：
LangGraph 是核心工作流引擎，使用 StateGraph 将整个视频分析流程建模为有向图。

**选择 LangGraph 的原因**：
1. **支持循环**：Self-RAG 质检失败后需要重新生成，普通函数链无法优雅表达循环
2. **支持并行**：Map-Reduce 分片并行，多分片同时处理
3. **原生 Checkpoint**：中断恢复、时间旅行，内置不需要自建
4. **HITL 支持**：`interrupt_before` 实现执行中断等待用户输入

`VideoSummaryState`（TypedDict）作为全局共享状态在节点间传递，`_merge_chunk_results` Reducer 处理并行写冲突。

---

### <a id="a-mapreduce"></a>Q: Map-Reduce 分片并行是怎么实现的？

**参考答案**：
```
视频 → ChunkPlanner 决定分片数 → N个分片（时间段）
  ↓ Fan-Out（并行）
  每个分片：audio_analyzer + vision_analyzer + chunk_synthesizer
  ↓ Fan-In（聚合）
  chunk_aggregator 收集所有结果 → fusion_drafter
```

**两种并发模式**：
- **ThreadPool（默认）**：`ThreadPoolExecutor` 并发执行 N 个分片，等待 `concurrent.futures.wait()` 全部完成后聚合
- **Send API（试验）**：LangGraph 原生 `Send()` API 动态分发，每分片作为独立图节点

**Fan-In 写冲突解决**：`VideoSummaryState.chunk_results` 声明为 `Annotated[List, _merge_chunk_results]`，Reducer 将每次写入追加到列表而非覆盖。

---

### <a id="a-self-rag"></a>Q: Self-RAG 双重质检是怎么实现的？

**参考答案**：

两个质检节点职责不同：
- **`HallucinationGrader`**：检测摘要是否包含原始 transcript/frames 中没有依据的事实（准确性）
- **`UsefulnessGrader`**：检测摘要是否有足够信息量、不过于简短或重复（充分性）

**质检循环**：
```
fusion_drafter → hallucination_grader → usefulness_grader → [pass] → 输出
                       ↓ [fail]                ↓ [fail]
                    带 feedback 重新进入 fusion_drafter
```

`retry_count` 字段记录重试次数，超过 `max_retry` 阈值时条件边强制走 pass 路径，防止死循环。feedback 写入 State 的 `grader_feedback` 字段，`fusion_drafter` 下次读取并注入 Prompt。

---

### <a id="a-场景抽帧"></a>Q: 场景感知抽帧是怎么实现的？

**参考答案**：

核心算法：**灰度直方图相关性比较**

1. 按自适应 FPS（时长 < 5 min → 5 fps；5-30 min → 3 fps；> 30 min → 1 fps）遍历视频帧
2. 用 `grab/retrieve` 分离优化：批量 `cap.grab()` 移动指针（不解码），目标帧才调 `cap.retrieve()` 解码
3. 将候选帧转灰度 → 计算直方图 → `cv2.compareHist(HISTCMP_CORREL)` 与前帧比较
4. 相关性 < **0.90** → 判定场景切换，保留该帧；≥ 0.90 → 跳过

**阈值 0.90 的选择**：经验值，在"不漏关键帧"和"控制帧数量"之间平衡。调大漏帧多，调小帧量大增加 Token 成本。

---

### <a id="a-whisper大文件"></a>Q: Whisper 大文件处理策略是什么？

**参考答案**：

Whisper API 单文件限制 25MB。处理策略：**递归二分切割**

```python
def split_if_large(file):
    if size(file) <= 24MB: return [file]
    left, right = split_by_half(file)  # moviepy subclip
    return split_if_large(left) + split_if_large(right)
```

**为什么不按固定时长切**：不同视频比特率差异巨大（高清 vs 低质量），固定时长无法保证文件大小满足限制；递归二分保证每片一定 ≤ 24MB。

转录使用 `verbose_json` 格式，保留 `segments`（含 `start`/`end` 时间戳）。在 Map-Reduce 分片时，从 `segments` 中筛选对应时间范围内的语音片段，确保每分片只处理自己时段的内容。

---

### <a id="a-checkpoint"></a>Q: Checkpoint 持久化和时间旅行是怎么实现的？

**参考答案**：

**CheckpointFactory**：
```python
def get_checkpointer(mode):
    if mode == "dev": return InMemorySaver()  # 进程内存，重启丢失
    if mode == "prod": return PostgresSaver(...)  # 持久化到 PostgreSQL
```
`thread_id` 作为会话标识符，同一 `thread_id` 的历史执行状态可跨请求恢复。

**时间旅行流程**：
1. `graph.get_state_history(thread_id)` → 获取历史 checkpoint 列表
2. 找到目标 checkpoint（如 `fusion_drafter` 执行前）
3. `graph.update_state(config, {"followup_question": user_input})` → 注入修改
4. 重新 `graph.invoke(None, config)` → 从该 checkpoint 继续执行后续节点
5. `transcript`/`frames` 保留，只重跑 `fusion_drafter` 及质检节点

---

### <a id="a-hitl"></a>Q: HITL 两阶段工作流是怎么实现的？

**参考答案**：

**Phase 1（分析图）**：
- `human_gate` 节点作为中断点
- 编译图时设置 `interrupt_before=["fusion_drafter"]`
- 执行到 `human_gate` 后，LangGraph 自动保存 Checkpoint，`graph.stream()` 返回 `{"__interrupt__": {...}}` 标记
- Streamlit 检测到中断标记，展示分片摘要列表供用户审核

**Phase 2（终稿图）**：
- 用户点击"继续" / 输入修改意见 → 前端调用 `workflow_service.resume(thread_id, feedback)`
- `graph.invoke(None, config)` 从中断点恢复，将 `human_feedback` 写入 State
- `fusion_drafter` 读取 `human_feedback`，融合生成最终摘要

---

### <a id="a-测试体系"></a>Q: 测试分几层？单元测试怎么 mock LLM？

**参考答案**：

两层（~130 个测试）：
- **Unit（97 个）**：只测业务逻辑，用 `unittest.mock.patch` 替换 `openai.OpenAI` 客户端，返回预设 `ChatCompletion`/`Transcription` 对象，不调用真实 API
- **Integration（33 个）**：真实 LangGraph 图执行，Mock LLM 调用（确保确定性），测试图拓扑路由、State 流转、Checkpoint 恢复、HITL 流程

**覆盖模块**：chunk_aggregator、hallucination_grader、usefulness_grader、fusion_drafter、time_travel、frame_utils、map_dispatcher、graph_routing、checkpoint_restore_flow、e2e_pipeline 等。

**设计原则**：单元测试测业务逻辑不测 API，集成测试测图结构不测 LLM 效果。

---

### <a id="a-双并发模式"></a>Q: ThreadPool 和 Send API 两种并发模式有什么区别？

**参考答案**：

| | ThreadPool 模式 | Send API 模式 |
|--|----------------|--------------|
| 实现层 | 应用层（Python concurrent.futures） | LangGraph 原生（graph-level） |
| State 合并 | 主线程统一收集，无写冲突 | 需要 Reducer 处理并行写（`_merge_chunk_results`） |
| 稳定性 | 成熟稳定，行为可预测 | 试验性，调试复杂 |
| 默认 | ✅ 是 | ❌ 否（pilot） |
| 适用场景 | 生产环境 | 探索 LangGraph 原生并行能力 |

**ThreadPool 是默认**：不需要特殊 Reducer 处理、调试更容易、生产可靠性更高。Send API 作为 pilot 探索 LangGraph 原生并行的更深集成。

---

### <a id="a-实时状态"></a>Q: st.status 实时更新是怎么从 LangGraph 回调到 Streamlit 的？

**参考答案**：

`workflow_service.py` 调用 `graph.stream()` 时接收每个节点的输出 chunk，通过回调将执行状态写入 `st.session_state` 中的进度字段。Streamlit 在下一次 rerun 时读取字段更新 `st.progress()` 和 `st.status()` 组件。

Streamlit 本质是"近实时"（每次 rerun 刷新，不是真正推送），采用 `st.empty()` + 循环 rerun 策略实现视觉上的流式效果。

4 通道进度条（`audio_progress`、`vision_progress`、`synthesis_progress`、`overall_progress`）分别由对应节点回调更新，用户可实时感知各阶段进展。

---

## 三、评分细则

**分档标准（严格执行，不得调整）**：

| 分档 | 标准 |
|-----|------|
| 9-10 | 所有核心问题答出关键细节，无露馅，表达清晰且有深度延伸 |
| 7-8 | 大部分问题答出主干，偶有细节遗漏（1-2 处），无严重露馅 |
| 5-6 | 架构层面基本掌握，但算法/实现细节有 3 处以上明显缺失，或有 1 处严重露馅 |
| 3-4 | 仅能描述表面概念，追问即露馅，简历存在明显虚报 |
| 1-2 | 核心技术点均无法解释，简历与实际能力严重不符 |

**5 个评分维度**：

| 维度 | 重点考察内容 |
|-----|------------|
| 项目架构掌握 | 四层架构（前端/服务/工作流/提取）、模块分工、两阶段图能否清楚表达 |
| 简历真实性 | 量化指标有无测量方法支撑，强动词能否说清决策过程 |
| 算法理论深度 | LangGraph Reducer 机制、灰度直方图算法、Self-RAG 质检循环、Whisper 大文件处理 |
| 实现细节掌握 | VideoSummaryState 字段/Annotated Reducer/Checkpoint factory 两模式/HITL interrupt 机制/测试 mock 层 |
| 表达清晰度 | 回答完整性、逻辑链完整、能说清"为什么"而非只说"是什么" |
> **项目**：多模态视频智能总结系统（Video Summarizer）

---

## 一、报告 Markdown 模板

生成规则：
- 表格"参考答案"列使用 `[→ 查看](#a-锚点关键词)` 锚链接，指向本文件第二节对应答案
- 只为**本次实际问到的题目**复制对应答案块，未问到的不放入报告
- 严格按评分细则打分，不得因情绪照顾调分

```markdown
# 模拟面试报告

**项目**：多模态视频智能总结系统（Video Summarizer）
**面试时间**：{datetime}
**评分**：{score}/10

---

## 一、面试记录

> ✅ 答对核心要点 | ⚠️ 方向正确但细节缺失 | ❌ 未能答出或方向错误

### 方向 1：项目综述

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {2-3 句摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 2：简历深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 露馅 | 参考答案 |
|-----|------|-------------|------|-----|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | 是/否 | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | ... | [→ 查看](#a-{锚点}) |

### 方向 3：技术深挖

| 轮次 | 问题 | 候选人回答摘要 | 评估 | 参考答案 |
|-----|------|-------------|------|---------|
| 1 | {问题原文} | {摘要} | ✅/⚠️/❌ | [→ 查看](#a-{锚点}) |
| 2 | ... | ... | ... | [→ 查看](#a-{锚点}) |
| 3 | ... | ... | ... | [→ 查看](#a-{锚点}) |

---

## 二、参考答案

> 仅复制本次实际问到的题目对应答案块，保留 <a id> 锚点。

{从下方"预置参考答案库"按需复制}

---

## 三、简历包装点评

### 包装合理 ✅
- **"{简历描述}"**：{说明候选人能自圆其说之处，具体指出哪句回答支撑了该判断}

### 露馅点 ❌
- **"{简历描述}"** → {面试中的具体表现}。**严重性：高/中/低**（{说明原因}）

### 改进建议
- {针对每个露馅点的具体、可操作建议，如"建议背下 _merge_chunk_results 的 Annotated 语法并能解释为什么不能不用 Reducer"}

---

## 四、综合评价

**优势**：
- {具体到哪道题答得好、好在哪个关键点}

**薄弱点**：
- {具体技术点 + 答错/答浅的表现描述}

**面试官建议**：
{针对每个薄弱点的具体改进方向，避免笼统表述}

---

## 五、评分

| 维度 | 分数（满分 10）| 评分依据（必须说明具体扣分原因） |
|-----|--------------|--------------------------------|
| 项目架构掌握 | x | {哪些点答到了，哪些点缺失} |
| 简历真实性 | x | {几处包装合理，几处露馅，差距} |
| 算法理论深度 | x | {LangGraph / Reducer / 场景抽帧 / Self-RAG 等作答情况} |
| 实现细节掌握 | x | {VideoSummaryState 字段/Checkpoint factory/HITL 流程/测试 mock 等} |
| 表达清晰度 | x | {回答完整性、逻辑清晰度、因果说明} |
| **综合** | **x** | {加权说明} |
```

---

## 二、预置参考答案库

> 按需复制到报告"二、参考答案"节，保留 `<a id>` 锚点不变。

---

### <a id="a-项目架构"></a>Q: 介绍项目整体架构和你具体负责的部分

**参考答案**：
系统分四大层次：
1. **前端层**（`app.py`）：Streamlit UI，负责视频上传/URL 输入、实时进度显示、HITL 交互、结果展示
2. **服务层**（`services/workflow_service.py`）：协调 workflow 调用，向前端透传执行状态
3. **工作流层**（`core/workflow/`）：LangGraph StateGraph，Phase 1（并行分析图）+ Phase 2（终稿图），包含所有节点、条件边、质检循环
4. **提取层**（`core/extraction/`）：Whisper 音频转录 + OpenCV 场景感知抽帧，向工作流提供基础数据

核心亮点：Map-Reduce 并行分片、Self-RAG 双重质检、HITL 两阶段、Checkpoint 时间旅行。

---

### <a id="a-langgraph工作流"></a>Q: LangGraph 在这个项目里起什么作用？为什么选 LangGraph？

**参考答案**：
LangGraph 是核心工作流引擎，使用 StateGraph 将整个视频分析流程建模为有向图。

**选择 LangGraph 的原因**：
1. **支持循环**：Self-RAG 质检失败后需要重新生成，普通函数链无法优雅表达循环
2. **支持并行**：Map-Reduce 分片并行，多分片同时处理
3. **原生 Checkpoint**：中断恢复、时间旅行，内置不需要自建
4. **HITL 支持**：`interrupt_before` 实现执行中断等待用户输入

`VideoSummaryState`（TypedDict）作为全局共享状态在节点间传递，`_merge_chunk_results` Reducer 处理并行写冲突。

---

### <a id="a-mapreduce"></a>Q: Map-Reduce 分片并行是怎么实现的？

**参考答案**：
```
视频 → ChunkPlanner 决定分片数 → N个分片（时间段）
  ↓ Fan-Out（并行）
  每个分片：audio_analyzer + vision_analyzer + chunk_synthesizer
  ↓ Fan-In（聚合）
  chunk_aggregator 收集所有结果 → fusion_drafter
```

**两种并发模式**：
- **ThreadPool（默认）**：`ThreadPoolExecutor` 并发执行 N 个分片，等待 `concurrent.futures.wait()` 全部完成后聚合
- **Send API（试验）**：LangGraph 原生 `Send()` API 动态分发，每分片作为独立图节点

**Fan-In 写冲突解决**：`VideoSummaryState.chunk_results` 声明为 `Annotated[List, _merge_chunk_results]`，Reducer 将每次写入追加到列表而非覆盖。

---

### <a id="a-self-rag"></a>Q: Self-RAG 双重质检是怎么实现的？

**参考答案**：

两个质检节点职责不同：
- **`HallucinationGrader`**：检测摘要是否包含原始 transcript/frames 中没有依据的事实（准确性）
- **`UsefulnessGrader`**：检测摘要是否有足够信息量、不过于简短或重复（充分性）

**质检循环**：
```
fusion_drafter → hallucination_grader → usefulness_grader → [pass] → 输出
                       ↓ [fail]                ↓ [fail]
                    带 feedback 重新进入 fusion_drafter
```

`retry_count` 字段记录重试次数，超过 `max_retry` 阈值时条件边强制走 pass 路径，防止死循环。feedback 写入 State 的 `grader_feedback` 字段，`fusion_drafter` 下次读取并注入 Prompt。

---

### <a id="a-场景抽帧"></a>Q: 场景感知抽帧是怎么实现的？

**参考答案**：

核心算法：**灰度直方图相关性比较**

1. 按自适应 FPS（时长 < 5 min → 5 fps；5-30 min → 3 fps；> 30 min → 1 fps）遍历视频帧
2. 用 `grab/retrieve` 分离优化：批量 `cap.grab()` 移动指针（不解码），目标帧才调 `cap.retrieve()` 解码
3. 将候选帧转灰度 → 计算直方图 → `cv2.compareHist(HISTCMP_CORREL)` 与前帧比较
4. 相关性 < **0.90** → 判定场景切换，保留该帧；≥ 0.90 → 跳过

**阈值 0.90 的选择**：经验值，在"不漏关键帧"和"控制帧数量"之间平衡。调大漏帧多，调小帧量大增加 Token 成本。

---

### <a id="a-whisper大文件"></a>Q: Whisper 大文件处理策略是什么？

**参考答案**：

Whisper API 单文件限制 25MB。处理策略：**递归二分切割**

```python
def split_if_large(file):
    if size(file) <= 24MB: return [file]
    left, right = split_by_half(file)  # moviepy subclip
    return split_if_large(left) + split_if_large(right)
```

**为什么不按固定时长切**：不同视频比特率差异巨大（高清 vs 低质量），固定时长无法保证文件大小满足限制；递归二分保证每片一定 ≤ 24MB。

转录使用 `verbose_json` 格式，保留 `segments`（含 `start`/`end` 时间戳）。在 Map-Reduce 分片时，从 `segments` 中筛选对应时间范围内的语音片段，确保每分片只处理自己时段的内容。

---

### <a id="a-checkpoint"></a>Q: Checkpoint 持久化和时间旅行是怎么实现的？

**参考答案**：

**CheckpointFactory**：
```python
def get_checkpointer(mode):
    if mode == "dev": return InMemorySaver()  # 进程内存，重启丢失
    if mode == "prod": return PostgresSaver(...)  # 持久化到 PostgreSQL
```
`thread_id` 作为会话标识符，同一 `thread_id` 的历史执行状态可跨请求恢复。

**时间旅行流程**：
1. `graph.get_state_history(thread_id)` → 获取历史 checkpoint 列表
2. 找到目标 checkpoint（如 `fusion_drafter` 执行前）
3. `graph.update_state(config, {"followup_question": user_input})` → 注入修改
4. 重新 `graph.invoke(None, config)` → 从该 checkpoint 继续执行后续节点
5. `transcript`/`frames` 保留，只重跑 `fusion_drafter` 及质检节点

---

### <a id="a-hitl"></a>Q: HITL 两阶段工作流是怎么实现的？

**参考答案**：

**Phase 1（分析图）**：
- `human_gate` 节点作为中断点
- 编译图时设置 `interrupt_before=["fusion_drafter"]`
- 执行到 `human_gate` 后，LangGraph 自动保存 Checkpoint，`graph.stream()` 返回 `{"__interrupt__": {...}}` 标记
- Streamlit 检测到中断标记，展示分片摘要列表供用户审核

**Phase 2（终稿图）**：
- 用户点击"继续" / 输入修改意见 → 前端调用 `workflow_service.resume(thread_id, feedback)`
- `graph.invoke(None, config)` 从中断点恢复，将 `human_feedback` 写入 State
- `fusion_drafter` 读取 `human_feedback`，融合生成最终摘要

---

### <a id="a-测试体系"></a>Q: 测试分几层？单元测试怎么 mock LLM？

**参考答案**：

两层（~130 个测试）：
- **Unit（97 个）**：只测业务逻辑，用 `unittest.mock.patch` 替换 `openai.OpenAI` 客户端，返回预设 `ChatCompletion`/`Transcription` 对象，不调用真实 API
- **Integration（33 个）**：真实 LangGraph 图执行，Mock LLM 调用（确保确定性），测试图拓扑路由、State 流转、Checkpoint 恢复、HITL 流程

**覆盖模块**：chunk_aggregator、hallucination_grader、usefulness_grader、fusion_drafter、time_travel、frame_utils、map_dispatcher、graph_routing、checkpoint_restore_flow、e2e_pipeline 等。

**设计原则**：单元测试测业务逻辑不测 API，集成测试测图结构不测 LLM 效果。

---

### <a id="a-双并发模式"></a>Q: ThreadPool 和 Send API 两种并发模式有什么区别？

**参考答案**：

| | ThreadPool 模式 | Send API 模式 |
|--|----------------|--------------|
| 实现层 | 应用层（Python concurrent.futures） | LangGraph 原生（graph-level） |
| State 合并 | 主线程统一收集，无写冲突 | 需要 Reducer 处理并行写（`_merge_chunk_results`） |
| 稳定性 | 成熟稳定，行为可预测 | 试验性，调试复杂 |
| 默认 | ✅ 是 | ❌ 否（pilot） |
| 适用场景 | 生产环境 | 探索 LangGraph 原生并行能力 |

**ThreadPool 是默认**：不需要特殊 Reducer 处理、调试更容易、生产可靠性更高。Send API 作为 pilot 探索 LangGraph 原生并行的更深集成。

---

### <a id="a-实时状态"></a>Q: st.status 实时更新是怎么从 LangGraph 回调到 Streamlit 的？

**参考答案**：

`workflow_service.py` 调用 `graph.stream()` 时接收每个节点的输出 chunk，通过回调将执行状态写入 `st.session_state` 中的进度字段。Streamlit 在下一次 rerun 时读取字段更新 `st.progress()` 和 `st.status()` 组件。

Streamlit 本质是"近实时"（每次 rerun 刷新，不是真正推送），采用 `st.empty()` + 循环 rerun 策略实现视觉上的流式效果。

4 通道进度条（`audio_progress`、`vision_progress`、`synthesis_progress`、`overall_progress`）分别由对应节点回调更新，用户可实时感知各阶段进展。

---

## 三、评分细则

**分档标准（严格执行，不得调整）**：

| 分档 | 标准 |
|-----|------|
| 9-10 | 所有核心问题答出关键细节，无露馅，表达清晰且有深度延伸 |
| 7-8 | 大部分问题答出主干，偶有细节遗漏（1-2 处），无严重露馅 |
| 5-6 | 架构层面基本掌握，但算法/实现细节有 3 处以上明显缺失，或有 1 处严重露馅 |
| 3-4 | 仅能描述表面概念，追问即露馅，简历存在明显虚报 |
| 1-2 | 核心技术点均无法解释，简历与实际能力严重不符 |

**5 个评分维度**：

| 维度 | 重点考察内容 |
|-----|------------|
| 项目架构掌握 | 四层架构（前端/服务/工作流/提取）、模块分工、两阶段图能否清楚表达 |
| 简历真实性 | 量化指标有无测量方法支撑，强动词能否说清决策过程 |
| 算法理论深度 | LangGraph Reducer 机制、灰度直方图算法、Self-RAG 质检循环、Whisper 大文件处理 |
| 实现细节掌握 | VideoSummaryState 字段/Annotated Reducer/Checkpoint factory 两模式/HITL interrupt 机制/测试 mock 层 |
| 表达清晰度 | 回答完整性、逻辑链完整、能说清"为什么"而非只说"是什么" |

