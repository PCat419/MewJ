# MewJ

MewJ 是一个面向雀魂牌谱的日麻复盘与决策分析工具。它读取雀魂牌谱，调用
`nanikiru` 计算牌效与期望值，并结合攻防、立直、鸣牌、局面姿态等规则生成中文
HTML 检讨报告。

本仓库同时支持 Windows 本地使用和 Linux 服务器部署。两端共享同一套牌谱解析、
评分与决策算法，只在引擎可执行文件、启动方式和报告展示层保留必要的平台差异。

## 主要功能

- 读取雀魂分享链接、牌谱 UUID 或本地牌谱 JSON。
- 分析切牌、立直与鸣牌决策。
- 综合牌效、和率、打点、放铳风险和局面姿态给出建议。
- 支持多个 `nanikiru` worker 并行分析。
- 生成带逐巡导航、候选比较和天凤回放的 HTML 报告。
- Windows 提供批处理启动入口。
- Linux 支持 systemd 常驻、Nginx 反向代理及 HTTPS 部署。
- local/server 两套报告后端可以独立维护，不影响共享算法。

## 跨平台设计

```text
雀魂链接或本地 JSON
        │
        ▼
牌谱下载与解析（共享）
        │
        ▼
决策提取、攻防、立直、评分（共享）
        │
        ▼
nanikiru 请求池（共享 Python 接口）
        ├─ Windows: engine/nanikiru.exe
        └─ Linux:   engine/nanikiru
        │
        ▼
统一报告数据
        ├─ local  → 离线天凤回放
        └─ server → 在线天凤回放
```

修改 `defense.py`、`scoring.py`、`pipeline.py`、`review.py` 等共享算法文件后，
Windows 和 Linux 拉取同一 Git 提交即可同时获得新算法。只有修改底层 C++ 引擎时，
才需要从同一份源码分别构建 Windows 和 Linux 二进制。

## 目录结构

| 路径 | 用途 |
|---|---|
| `pipeline.py` | 下载、读取、评审和输出报告的完整流水线 |
| `review.py` | 决策点提取、引擎调用与报告数据组织 |
| `defense.py` / `defense_heuristics.py` | 防守判断与启发式规则 |
| `riichi.py` / `riichi_eval.py` | 立直相关判断与评估 |
| `call_eval.py` | 鸣牌决策评估 |
| `scoring.py` | 评分与价值计算 |
| `posture.py` | 局面姿态管理 |
| `nanikiru_pool.py` | 跨平台 `nanikiru` 进程池 |
| `report.py` | 报告后端分派入口 |
| `report_local.py` | Windows/本地离线报告模板 |
| `report_server.py` | Linux/发送场景在线报告模板 |
| `web.py` | local/server 双模式 Web UI |
| `cli.py` | 命令行入口 |
| `engine/` | 查表数据、Schema 与平台引擎文件 |
| `assets/` | 报告牌图和本地天凤回放资源 |
| `vendor/` | 随项目提供的雀魂牌谱下载组件 |
| `deploy/` | Linux 构建与部署模板 |
| `paipu/` | 本地牌谱缓存，不纳入 Git |
| `out/` | 生成的 HTML 报告，不纳入 Git |

## 环境要求

公共要求：

- Python 3.10 或更高版本。
- 可访问雀魂及相关资源的网络环境。
- 与当前 MewJ 协议兼容的 `nanikiru` 引擎。

Windows 还需要：

- `engine/nanikiru.exe`。
- 仓库已附带的 MinGW 运行时 DLL；若自行构建，需保证相应 DLL 可被找到。

Linux 构建引擎通常需要：

- CMake、C++ 编译器、Boost 与 OpenMP。
- `mahjong-cpp` 源码目录。

## 安装 Python 依赖

建议使用虚拟环境。

### Windows PowerShell

```powershell
git clone https://github.com/PCat419/MewJ.git
cd MewJ
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux

```bash
git clone https://github.com/PCat419/MewJ.git
cd MewJ
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
```

然后编辑 `.env`，按需填写雀魂账号和运行参数。

## 配置

`.env.example` 提供所有常用配置示例。`.env` 可能包含账号密码或 Token，已被
`.gitignore` 排除，禁止提交到公开仓库。

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `MAJSOUL_USERNAME` | 下载链接牌谱时使用的雀魂账号 | 空 |
| `MAJSOUL_PASSWORD` | 雀魂密码 | 空 |
| `MAJSOUL_ACCESS_TOKEN` | 密码登录不可用时的可选 Token | 空 |
| `MAJSOUL_PROXY` | HTTP 或 SOCKS5 代理 | 空 |
| `NANIKIRU_URL` | 首个引擎服务地址 | `http://127.0.0.1:50000` |
| `MEWJ_WORKERS` | 并行引擎进程数 | `4` |
| `MEWJ_NANIKIRU_EXE` | 手动指定引擎路径 | 按平台自动选择 |
| `MEWJ_WEB_MODE` | `local` 临时 Web；`server` 常驻 Web | `local` |
| `MEWJ_REPORT_MODE` | `local` 离线报告；`server` 在线回放报告 | `local` |
| `MEWJ_WEB_HOST` | Web 监听地址 | `127.0.0.1` |
| `MEWJ_WEB_PORT` | Web 监听端口 | `8765` |

默认引擎路径：

- Windows：`engine/nanikiru.exe`
- Linux/macOS：`engine/nanikiru`

`MEWJ_WEB_MODE` 与 `MEWJ_REPORT_MODE` 相互独立，可以按实际用途组合。正式 Linux
部署模板会同时将二者设置为 `server`。

## Windows 使用

### Web 界面

双击 `web.bat`。首次运行会从 `.env.example` 创建 `.env` 并打开编辑器；保存配置后
再次运行即可。默认行为是：

1. 在 `127.0.0.1:8765` 启动 Web 页面。
2. 自动打开浏览器。
3. 输入雀魂分享链接或牌谱 UUID。
4. 报告完成后自动打开 HTML，并停止临时 Web 服务。

需要查看控制台日志时：

```bat
web.bat --console
```

### 交互式命令行

双击 `review.bat`，按提示输入牌谱链接或 UUID。脚本会检查并启动
`engine/nanikiru.exe`，完成后停止由本次脚本启动的引擎。

## Python CLI

查看完整参数：

```bash
python cli.py --help
```

分析雀魂链接并自动识别座次：

```bash
python cli.py --link "雀魂分享链接" --open
```

分析本地 `paipu/<uuid>.json`：

```bash
python cli.py <uuid> --seat 0 --open
```

限制测试范围：

```bash
python cli.py <uuid> --seat 0 --kyoku 0 --max-turns 10
```

输出默认写入 `out/`。`--open` 会在生成后使用系统浏览器打开报告。

## 报告模式

### local：本地离线报告

```bash
MEWJ_REPORT_MODE=local python cli.py <uuid> --seat 0
```

- 使用 `assets/tenhou/viewer.html` 与仓库内资源。
- 避免在线天凤页面在切换小局时频繁刷新。
- 适合 Windows 本地浏览和与整个资源目录一起保存。

### server：服务器发送报告

```bash
MEWJ_REPORT_MODE=server python cli.py <uuid> --seat 0
```

- 使用天凤在线 HTML5 viewer。
- 不把完整的天凤牌面资源集塞进单个发送用 HTML。
- 适合 Koishi 插件发送报告文件或 Linux Web 服务。

两种模板消费同一份评审数据，不会改变牌效、评分、攻防或推荐结果。报告模板不要求
同步修改：可以只调整其中一种，也可以按需要同时更新。

## Linux 编译 nanikiru

将 `mahjong-cpp` 放在 MewJ 相邻目录，或显式传入路径：

```bash
bash deploy/build_nanikiru.sh /path/to/mahjong-cpp
```

脚本会使用 Release 与 OpenMP 配置构建 server，并安装到
`engine/nanikiru`。该 Linux 二进制是本机构建产物，不提交到 Git。

只更新算法层 Python 文件时通常不需要重新编译引擎；只有 `mahjong-cpp` 源码或协议
发生变化时才需要重新构建。

## Linux 常驻部署

完整说明见 [`deploy/README.md`](deploy/README.md)。典型部署：

```bash
sudo DOMAIN=mewj.example.com \
  MEWJ_ROOT=/opt/mewj \
  MAHJONG_CPP=/opt/mahjong-cpp \
  bash /opt/mewj/deploy/setup.sh
```

部署结构：

```text
浏览器 / Koishi
      │ HTTPS + Basic Auth
      ▼
    Nginx
      │ 127.0.0.1:8765
      ▼
 web.py（server 模式）
      │ 50000 起的连续端口
      ▼
 nanikiru workers
```

常用运维命令：

```bash
sudo systemctl status mewj
sudo journalctl -u mewj -f
sudo systemctl restart mewj
```

不要将 `8765` 直接暴露到公网。建议仅监听 `127.0.0.1`，通过带 HTTPS 和鉴权的
Nginx 访问。

## 本地开发与服务器同步

推荐所有代码修改都在 Git 分支中完成：

```bash
git switch -c feature/my-change
# 修改并测试
git add .
git commit -m "调整牌效算法"
git push -u origin feature/my-change
```

合并到 `main` 后，服务器更新共享代码：

```bash
cd /opt/mewj
git pull --ff-only origin main
.venv/bin/pip install -r requirements.txt
sudo systemctl restart mewj
```

更新前建议先运行 `git status`。服务器的 `.env`、`paipu/`、`out/`、`.venv/` 和
`engine/nanikiru` 均为本地状态或构建产物，不应被 `git pull` 覆盖。

## 开发检查

提交前至少执行：

```bash
python -m compileall -q .
git diff --check
```

Linux 部署脚本还可检查：

```bash
bash -n deploy/setup.sh deploy/build_nanikiru.sh
```

修改报告层时，应使用同一份报告数据分别验证：

```bash
MEWJ_REPORT_MODE=local  python cli.py <uuid> --seat 0
MEWJ_REPORT_MODE=server python cli.py <uuid> --seat 0
```

修改算法时，重点确认两种报告中的评审数值一致；允许展示方式不同。

## 安全与隐私

- 仓库是公开仓库，不要提交 `.env`、账号、密码、Token、Cookie 或私钥。
- 不要提交真实牌谱、生成报告、运行日志、崩溃转储或虚拟环境。
- 如果凭据曾进入 Git 历史，仅删除文件不够；应立即撤销或更换凭据，并清理历史。
- Linux Web 服务应启用 HTTPS 和访问控制。
- 更新部署脚本、systemd 或 Nginx 配置前，先检查并保留服务器现有配置。

## 常见问题

### 提示找不到 nanikiru

检查默认文件是否存在，或在 `.env` 中设置绝对路径：

```env
MEWJ_NANIKIRU_EXE=/opt/mewj/engine/nanikiru
```

Windows 路径示例：

```env
MEWJ_NANIKIRU_EXE=C:\path\to\MewJ\engine\nanikiru.exe
```

### 修改算法后 Linux 没有变化

确认服务器已拉取正确分支和提交，并重启常驻服务：

```bash
git status
git log -1 --oneline
sudo systemctl restart mewj
```

如果只修改 Python 算法，不需要重新构建 `nanikiru`；如果修改的是 C++ 引擎，则必须
重新执行 `deploy/build_nanikiru.sh`。

### 为什么两个平台的 HTML 不完全相同

这是有意设计。Windows/local 使用离线回放以获得稳定体验；Linux/server 为了便于
发送单个 HTML 文件而使用在线回放。两者展示资源不同，但输入的评审数据与算法结论
相同。
