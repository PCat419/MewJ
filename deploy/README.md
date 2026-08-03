# MewJ — Linux 云服务器部署

常驻 Web 服务：域名 → Nginx（HTTPS + 基础鉴权）→ `web.py` → `engine/nanikiru`（由旁边的 `mahjong-cpp` 编译）。

```
浏览器 → HTTPS(域名) → Nginx(鉴权) → 127.0.0.1:8765 (web.py)
                                         └─ nanikiru :50000+
```

## 环境要求

- Ubuntu 24.04 推荐（Boost ≥ 1.81；22.04 可能需自装新版 Boost）
- 域名已解析到服务器公网 IP
- 开放 80 / 443

## 1. 上传代码

推荐在服务器克隆本仓库，并将 `mahjong-cpp` 源码放在相邻目录：

```bash
git clone https://github.com/PCat419/MewJ.git /opt/mewj
# 将 mahjong-cpp 克隆或同步到 /opt/mahjong-cpp
```

也可用 `rsync`（推荐，可重复同步）：

```bash
rsync -avz --exclude .venv --exclude out --exclude paipu --exclude __pycache__ \
  ./MewJ-deploy/ user@server:/opt/mewj/
rsync -avz --exclude build --exclude build-linux --exclude .git \
  ./mahjong-cpp/ user@server:/opt/mahjong-cpp/
```

`engine/` 里已有 `*.bin` / `*.json` 查表数据，**不要漏传**；`nanikiru` 二进制在服务器上编译生成。

## 2. 一键部署

SSH 登录服务器后：

```bash
sudo DOMAIN=mewj.example.com \
  MEWJ_ROOT=/opt/mewj \
  MAHJONG_CPP=/opt/mahjong-cpp \
  bash /opt/mewj/deploy/setup.sh
```

脚本会：

1. 安装依赖（Python、Nginx、Boost、OpenMP 等）
2. 从 `mahjong-cpp` 编译 `nanikiru` → `/opt/mewj/engine/nanikiru`
3. 创建 venv、systemd 服务、Nginx + Basic Auth

然后：

```bash
sudo nano /opt/mewj/.env          # 填写 MAJSOUL_USERNAME / MAJSOUL_PASSWORD
sudo systemctl restart mewj
sudo certbot --nginx -d mewj.example.com
```

浏览器打开 `https://你的域名/`，用 htpasswd 账号登录即可。

## 3. 只重编译 nanikiru

源码更新后：

```bash
sudo -u mewj bash /opt/mewj/deploy/build_nanikiru.sh /opt/mahjong-cpp
sudo systemctl restart mewj
```

## 配置说明

| 文件 | 用途 |
|------|------|
| `deploy/setup.sh` | 服务器一键部署 |
| `deploy/build_nanikiru.sh` | 编译并安装 nanikiru |
| `deploy/mewj.service` | systemd 单元 |
| `deploy/nginx-mewj.conf` | Nginx 反代模板（有证书后可用） |
| `.env` / `.env.example` | 雀魂账号与路径 |

常用环境变量：

```bash
MEWJ_WEB_HOST=127.0.0.1
MEWJ_WEB_PORT=8765
MEWJ_WEB_MODE=server
MEWJ_REPORT_MODE=server
MEWJ_NANIKIRU_EXE=/opt/mewj/engine/nanikiru
MEWJ_WORKERS=4
MAJSOUL_USERNAME=...
MAJSOUL_PASSWORD=...
```

## 运维

```bash
sudo systemctl status mewj
sudo journalctl -u mewj -f
sudo systemctl restart mewj
```

## 注意

- 不要把 `8765` 直接暴露公网；只走 Nginx + HTTPS + 鉴权。
- `.env` 含雀魂凭据：`chmod 600`，勿提交公开仓库。
- 编译使用 `-DENABLE_OPENMP=ON`（避免 mahjong-cpp 默认分支在 Linux 上误链 `ws2_32`）。
