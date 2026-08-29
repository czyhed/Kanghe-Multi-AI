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



## 终端
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