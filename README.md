# chenguang-agent-

#### 介绍
《天宫医疗-智能体》集中于解决医疗领域的 Agent 业务构建。包括：
1. 智慧问诊 Agent
2. 报告解读 Agent
3. 药物 Agent
4. 知识文档 Agent
5. 运营数据 Agent

#### 软件架构
软件架构说明


#### 安装教程

```sh
# 创建环境
conda create -n tiangong python=3.13
# 激活环境
conda activate tiangong

# 安装依赖
pip install -r requirements.txt
```


## 指定端口启动
```
uvicorn src.main:app --port 8080 --reload
```

## env配置
把 .env.example 复制一份 叫 .env ，修改为自己的信息即可


导入的样例数据：
1. QA数据集：药物（名字、作用、副作用、 使用方法、注意事项、疾病）
2. 一个数据集有三处使用（没聊天之前就建设好）：
   - Neo4j/GraphDB/postgresql/AWS（图谱）：医药三元组：药物、关系、疾病
   - postgresql/MySQL（所有产品的标配）：持久化存储详细信息 药物（名字、作用、副作用、使用方法、注意事项）
   - Milvus/Elasticsearch/VectorDB/Chroma（搜索）：相似度搜索： 药物作用

neo4j：图谱；多条推理   复杂关联关系搜索
postgresql： 关系数据（传统数据）：精确搜索
Milvus：相似度搜索：  语义相似度（语料库的建设是必不可少）

Agent = 模型 + 记忆 + 工具 + ReAct
- 长期记忆/短期记忆：用于存储对话的历史信息。聊天的时候就会产生【治理Agent的业务】
- 所有中间件，灵活使用，啥时候需要用看业务
- 智慧问诊，挂号（产生会诊记录之后）对接HIS系统：必须存储
- 药物Agent：查药物之间的关系。查询为主
- CRUD的操作是跟业务走

## 编译器终端配置
```
✅ 方法一：通过设置界面修改（推荐，最简单）
步骤1：打开设置
按 Ctrl + , 打开设置窗口。

步骤2：配置 Python 终端激活
在搜索框输入：python.terminal.activateEnvironment

确保勾选这个选项（设置为 true）。

步骤3：配置默认终端为 PowerShell
在搜索框输入：terminal.integrated.defaultProfile.windows

在下拉菜单中选择 PowerShell。

步骤4：配置 PowerShell 启动参数
在搜索框输入：terminal.integrated.profiles.windows

点击 Edit in settings.json。

在打开的 settings.json 中，找到或添加以下配置：

json
{
    "terminal.integrated.profiles.windows": {
        "PowerShell": {
            "source": "PowerShell",
            "icon": "terminal-powershell",
            "args": ["-NoExit", "-Command", "conda activate tiangong"]
        }
    },
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "python.terminal.activateEnvironment": true
}
步骤5：保存并重启
保存 settings.json（Ctrl + S）。

关闭所有已打开的终端。

新建一个终端（Ctrl + `），检查是否显示 (tiangong)。
```