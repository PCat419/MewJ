#!/usr/bin/env bash
# Ubuntu 云服务器一键部署 MewJ（需 root）。
#
# 准备：
#   1. 把 MewJ-deploy 放到 $MEWJ_ROOT（默认 /opt/mewj）
#   2. 把 mahjong-cpp 放到 $MAHJONG_CPP（默认 /opt/mahjong-cpp，或与 mewj 并列）
#   3. 域名 A 记录已指向本机
#
# 用法：
#   sudo DOMAIN=mewj.example.com bash deploy/setup.sh
#   sudo DOMAIN=mewj.example.com MEWJ_ROOT=/opt/mewj MAHJONG_CPP=/opt/mahjong-cpp bash deploy/setup.sh
set -euo pipefail

MEWJ_ROOT="${MEWJ_ROOT:-/opt/mewj}"
MEWJ_USER="${MEWJ_USER:-mewj}"
DOMAIN="${DOMAIN:-your.domain.tld}"
MAHJONG_CPP="${MAHJONG_CPP:-}"
AUTH_USER="${AUTH_USER:-mewj}"

if [[ $EUID -ne 0 ]]; then
  echo "请用 root 运行: sudo bash deploy/setup.sh"
  exit 1
fi

if [[ ! -f "$MEWJ_ROOT/web.py" ]]; then
  echo "找不到 $MEWJ_ROOT/web.py — 请先把 MewJ-deploy 同步到 $MEWJ_ROOT"
  exit 1
fi

if [[ -z "$MAHJONG_CPP" ]]; then
  for cand in \
    "$(dirname "$MEWJ_ROOT")/mahjong-cpp" \
    /opt/mahjong-cpp \
    "$MEWJ_ROOT/../mahjong-cpp"
  do
    if [[ -f "$cand/CMakeLists.txt" ]]; then
      MAHJONG_CPP="$(cd "$cand" && pwd)"
      break
    fi
  done
fi

apt-get update
apt-get install -y \
  python3 python3-venv python3-pip \
  nginx apache2-utils certbot python3-certbot-nginx \
  build-essential cmake git \
  libboost-all-dev libomp-dev

id -u "$MEWJ_USER" &>/dev/null || \
  useradd --system --home "$MEWJ_ROOT" --shell /usr/sbin/nologin "$MEWJ_USER"

mkdir -p "$MEWJ_ROOT"
chown -R "$MEWJ_USER:$MEWJ_USER" "$MEWJ_ROOT"
if [[ -n "${MAHJONG_CPP:-}" && -d "$MAHJONG_CPP" ]]; then
  chown -R "$MEWJ_USER:$MEWJ_USER" "$MAHJONG_CPP"
fi

echo "==> 编译并安装 nanikiru"
if [[ -z "${MAHJONG_CPP:-}" || ! -f "$MAHJONG_CPP/CMakeLists.txt" ]]; then
  echo "未找到 mahjong-cpp。请设置 MAHJONG_CPP=/path/to/mahjong-cpp 后重试。"
  exit 1
fi
sudo -u "$MEWJ_USER" env MEWJ_ROOT="$MEWJ_ROOT" MAHJONG_CPP="$MAHJONG_CPP" \
  bash "$MEWJ_ROOT/deploy/build_nanikiru.sh" "$MAHJONG_CPP"

echo "==> Python 虚拟环境"
sudo -u "$MEWJ_USER" bash -lc "
  cd '$MEWJ_ROOT'
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
  if [[ ! -f .env ]]; then cp .env.example .env; fi
  chmod 600 .env
"

# 写入默认 nanikiru 路径（若不存在该行）
if ! grep -q '^MEWJ_NANIKIRU_EXE=' "$MEWJ_ROOT/.env" 2>/dev/null; then
  echo "MEWJ_NANIKIRU_EXE=$MEWJ_ROOT/engine/nanikiru" >> "$MEWJ_ROOT/.env"
fi
if ! grep -q '^MEWJ_WEB_HOST=' "$MEWJ_ROOT/.env" 2>/dev/null; then
  echo "MEWJ_WEB_HOST=127.0.0.1" >> "$MEWJ_ROOT/.env"
fi
if ! grep -q '^MEWJ_WEB_PORT=' "$MEWJ_ROOT/.env" 2>/dev/null; then
  echo "MEWJ_WEB_PORT=8765" >> "$MEWJ_ROOT/.env"
fi
if ! grep -q '^MEWJ_WEB_MODE=' "$MEWJ_ROOT/.env" 2>/dev/null; then
  echo "MEWJ_WEB_MODE=server" >> "$MEWJ_ROOT/.env"
fi
if ! grep -q '^MEWJ_REPORT_MODE=' "$MEWJ_ROOT/.env" 2>/dev/null; then
  echo "MEWJ_REPORT_MODE=server" >> "$MEWJ_ROOT/.env"
fi
chown "$MEWJ_USER:$MEWJ_USER" "$MEWJ_ROOT/.env"
chmod 600 "$MEWJ_ROOT/.env"

echo "==> systemd"
install -m 644 "$MEWJ_ROOT/deploy/mewj.service" /etc/systemd/system/mewj.service
sed -i "s|/opt/mewj|$MEWJ_ROOT|g; s|User=mewj|User=$MEWJ_USER|; s|Group=mewj|Group=$MEWJ_USER|" \
  /etc/systemd/system/mewj.service

echo "==> Nginx + Basic Auth"
if [[ ! -f /etc/nginx/mewj.htpasswd ]]; then
  echo "创建网站登录账号（用户名: $AUTH_USER）："
  htpasswd -c /etc/nginx/mewj.htpasswd "$AUTH_USER"
fi

install -m 644 "$MEWJ_ROOT/deploy/nginx-mewj.conf" /etc/nginx/sites-available/mewj
sed -i "s|your.domain.tld|$DOMAIN|g" /etc/nginx/sites-available/mewj
ln -sfn /etc/nginx/sites-available/mewj /etc/nginx/sites-enabled/mewj
rm -f /etc/nginx/sites-enabled/default

# 首次尚无证书时，先用 HTTP 反代，再让 certbot 接管 HTTPS
if [[ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]]; then
  cat > /etc/nginx/sites-available/mewj <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    auth_basic           "MewJ";
    auth_basic_user_file /etc/nginx/mewj.htpasswd;
    client_max_body_size 2m;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
EOF
fi

nginx -t
systemctl daemon-reload
systemctl enable --now mewj
systemctl reload nginx

echo
echo "========================================"
echo "基础服务已启动。"
echo "  1. 编辑 $MEWJ_ROOT/.env 填入雀魂账号，然后: sudo systemctl restart mewj"
echo "  2. 申请 HTTPS: sudo certbot --nginx -d $DOMAIN"
echo "  3. 检查: sudo systemctl status mewj"
echo "  访问: http://$DOMAIN/  （证书申请后改为 https）"
echo "========================================"
