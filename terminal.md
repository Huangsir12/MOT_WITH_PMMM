# 设置环境变量，系统变量

```bash
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
export ANTHROPIC_AUTH_TOKEN="你的密钥"

echo 'export http_proxy=http://127.0.0.1:1080' >> ~/.bashrc
echo 'export https_proxy=http://127.0.0.1:1080' >> ~/.bashrc
echo 'export ANTHROPIC_AUTH_TOKEN="你的密钥"' >> ~/.bashrc
source ~/.bashrc

unset http_proxy
unset https_proxy
unset ANTHROPIC_AUTH_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

# 2. 检查API密钥是否配置
echo $ANTHROPIC_API_KEY

# 查看相关环境变量
env | grep -E "ANTHROPIC|PROXY|proxy"
```

# 运行v2ray代理
```bash
# 启动代理
nohup v2ray run -config /usr/local/etc/v2ray/config.json > /tmp/v2ray.log 2>&1 &
# 查看运行情况
ps aux | grep v2ray
curl -v https://google.com/https://www.unicode.chat
curl -v https://baidu.com
pkill v2ray
```

# 运行数据库服务
```bash
# 重启 MySQL
pkill mysqld
# mysqld_safe --user=root &   不安全

sudo service mysql status
sudo service mysql restart
ps aux | grep mysql

# 查看 MySQL 版本
mysql -u root -e "SELECT VERSION();"
```

```
yolo classify train data=data/datasets/view_dajixiang model=weights/yolo26x-cls.pt epochs=100 imgsz=64
```
