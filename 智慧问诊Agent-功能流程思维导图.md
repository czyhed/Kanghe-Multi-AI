# 智慧问诊Agent完整功能实现流程

## 一、系统架构分层
### 1. 接入层（FastAPI）
#### POST /api/v1/chat 路由
#### chat_route 接口入口
#### chat_endpoint 统一聊天入口
### 2. 分流层（Redis标志位）
#### key: inquiry:{user_id}:{session_id}
#### EXISTS=YES → 直接走InquiryGraph（续接）
#### EXISTS=NO → 走Supervisor Agent（识别意图）
### 3. Agent层
#### Supervisor Agent（ReAct 主管，意图识别+分发）
#### InquiryGraph（StateGraph 问诊主流程）
#### Worker Agent（ReAct 挂号执行）
### 4. 基础设施层
#### PostgreSQL（患者/问诊记录/病史）
#### Redis（checkpoint + 问诊标志位）
#### Milvus（长期记忆 + 症状向量索引）
#### Neo4j（疾病-症状知识图谱）

## 二、11步实现步骤总览
### Step1 状态机定义 InquiryState
### Step2 移交数据包 InquiryHandoffPayload
### Step3 症状三层标准化 symptom_normalizer
### Step4 Milvus症状向量索引初始化
### Step5 Neo4j Cypher查询封装
### Step6 置信度计算+收敛判断 confidence
### Step7 提示词模板 prompts
### Step8 PostgreSQL患者上下文+问诊记录 db_queries
### Step9 InquiryGraph主图构建（核心）
### Step10 Inquiry Worker Agent 挂号移交
### Step11 工具改造+FastAPI路由分流

## 三、InquiryGraph 9个节点流程
### 节点1 node_emergency 急症检查
#### 首轮必走，后续轮次跳过
#### 是急症 → force_conclude=True → 直奔node_conclude
#### 非急症 → 进入node_clarify
### 节点2 node_clarify 澄清+提取症状
#### 用CLARIFY_PROMPT生成引导话术
#### round 轮次+1
### 节点3 node_normalize 三层标准化
#### ① LLM结构化提取症状词
#### ② Neo4j精确匹配标准症状名
#### ③ Milvus语义兜底（相似度≥0.85替换）
### 节点4 node_query_neo4j 知识图谱查询
#### 查候选疾病 list[CandidateDisease]
### 节点5 node_confidence 置信度计算+收敛判断
#### 基础分=命中症状数÷该病全部症状数
#### +0.15 既往病史命中
#### +0.10 长期记忆命中
#### -0.20 核心症状被否认
#### 按置信度降序排序
#### should_conclude判断是否收敛
### 节点6 node_ask_symptom 追问症状
#### 贪心策略：盯Top1候选疾病
#### 挑未确认/未否认的症状追问
#### 无未问症状 → force_conclude强制收敛
### 节点7 node_parse_answer 解析用户回答
#### has_symptom=True → 加入confirmed_symptoms
#### has_symptom=False → 加入denied_symptoms
#### 回node_confidence重新计算（形成闭环）
### 节点8 node_conclude 生成诊断结论
#### 构建InquiryHandoffPayload移交数据包
#### CONCLUDE_PROMPT生成诊断文案
#### save_consultation持久化到PostgreSQL
### 节点9 node_handoff 挂号移交
#### 调用Worker Agent
#### call_registration执行挂号
#### phase=END 图执行结束

## 四、收敛判断规则 should_conclude
### 轮次≥10 → 收敛
### 无候选疾病 → 收敛
### Top1置信度≥0.70 → 收敛
### Top1-Top2差≥0.30 → 收敛
### force_conclude=True → 强制收敛（急症或症状问尽）

## 五、多轮Checkpoint机制
### thread_id = user_id:session_id
### 每轮ainvoke自动恢复+保存state快照
### AsyncRedisSaver存储到Redis
### 图暂停等待用户回答
### 下轮从checkpoint续接继续

## 六、路由分流逻辑
### 首次进入
#### Supervisor识别问诊意图 → 调call_inquiry_agent
#### set_inquiry_active激活问诊标志
#### 启动InquiryGraph首轮
### 续轮进入
#### Redis标志=YES
#### 直接走InquiryGraph续接
#### 从checkpoint恢复state
### 问诊结束
#### phase=END
#### clear_inquiry_active删除标志
#### TTL 1小时防卡死

## 七、完整数据流闭环（示例）
### 第1轮 用户说"我头疼发烧两天了"
#### Redis检查NO → Supervisor → call_inquiry_agent
#### 激活问诊 → InquiryGraph首轮
##### 急症检查通过
##### 澄清 → 标准化提取[头痛,发烧]
##### 查候选[偏头痛,感冒,脑膜炎]
##### 置信度[0.45,0.40,0.20]未收敛
##### 追问"头痛是搏动性还是胀痛？"
#### checkpoint保存 → 返回AI话术
### 第2轮 用户说"胀痛，还有恶心"
#### Redis检查YES → 直接InquiryGraph续轮
##### 恢复state → 解析回答
##### 恶心加入confirmed_symptoms
##### 重算置信度[0.75,0.35,0.15]
##### 偏头痛Top1≥70% → 收敛
#### node_conclude生成诊断结论
#### node_handoff → Worker → 挂号成功
#### phase=END → 删除Redis标志
### 最终结果
#### 返回诊断+挂号信息

## 八、核心设计决策
### 问诊流程 → StateGraph（确定性路由，非ReAct）
### 多轮状态 → TypedDict+Checkpoint（结构化字段）
### 症状标准化 → 三层兜底（LLM+Neo4j+Milvus）
### 收敛判断 → 置信度算法+阈值（可量化可调参）
### 分流机制 → Redis标志位（O(1)判断，TTL防死锁）
### Worker Agent → ReAct（挂号简单但需灵活换科室）
### 诊断移交 → HandoffPayload JSON（结构化传递）
