# TTSFM

[![Docker Pulls](https://img.shields.io/docker/pulls/dbcccc/ttsfm?style=flat-square&logo=docker)](https://hub.docker.com/r/dbcccc/ttsfm)
[![License](https://img.shields.io/github/license/dbccccccc/ttsfm?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/dbccccccc/ttsfm?style=social)](https://github.com/dbccccccc/ttsfm)

> ⚠️ **免责声明**  
> 此项目仅用于学习测试，生产环境请使用 [OpenAI 官方 TTS 服务](https://platform.openai.com/docs/guides/audio)。

> ⚠️ **开发公告**  
> v2 分支目前正在积极开发中，不建议用于生产环境。请使用最新的稳定发布版本。

[English](README.md) | 中文

## 🌟 项目简介

TTSFM 是一个逆向工程实现的 API 服务器，完全兼容 OpenAI 的文本转语音(TTS)接口。

> 🎮 立即体验：[官方演示站](https://ttsapi.site/) 


## 🏗️ 项目结构

```text
ttsfm/
├── main.py              # 应用入口
├── server/              # 服务核心
│   ├── api.py           # OpenAI 兼容API
│   └── handlers.py      # 请求处理器
├── utils/               # 工具模块
│   └── config.py        # 配置管理
├── static/              # 前端资源
│   ├── index.html       # 英文界面
│   ├── index_zh.html    # 中文界面
│   ├── script.js        # 前端JavaScript
│   └── styles.css       # 前端样式
├── pressure_test.py     # 压力测试脚本
├── Dockerfile          # Docker配置
├── requirements.txt    # Python依赖
└── .env.example       # 环境变量模板
```

## 🚀 快速开始

### 系统要求
- Python ≥ 3.8
- 或 Docker 环境

### 🐳 Docker 运行（推荐）

基本用法：
```bash
docker run -p 7000:7000 dbcccc/ttsfm:latest
```

使用环境变量自定义配置：
```bash
docker run -d \
  -p 7000:7000 \
  -e HOST=0.0.0.0 \
  -e PORT=7000 \
  -e VERIFY_SSL=true \
  -e MAX_QUEUE_SIZE=100 \
  -e RATE_LIMIT_REQUESTS=30 \
  -e RATE_LIMIT_WINDOW=60 \
  dbcccc/ttsfm:latest
```

可用的环境变量：
- `HOST`：服务器主机（默认：0.0.0.0）
- `PORT`：服务器端口（默认：7000）
- `VERIFY_SSL`：是否验证 SSL 证书（默认：true）
- `MAX_QUEUE_SIZE`：队列最大任务数（默认：100）
- `RATE_LIMIT_REQUESTS`：每个时间窗口的最大请求数（默认：30）
- `RATE_LIMIT_WINDOW`：速率限制的时间窗口（秒）（默认：60）

> 💡 **重要提示**  
> 请始终使用 `latest` 标签获取最稳定的版本。v2 分支正在开发中，不建议用于生产环境。

> 💡 **提示**  
> MacOS 用户若遇到端口冲突，可替换端口号：  
> `docker run -p 5051:7000 dbcccc/ttsfm:latest`

### 📦 手动安装

1. 从 [GitHub Releases](https://github.com/dbccccccc/ttsfm/releases) 下载最新版本压缩包
2. 解压并进入目录：
```bash
tar -zxvf ttsfm-vX.X.X.tar.gz
cd ttsfm-vX.X.X
```
3. 安装依赖并启动：
```bash
pip install -r requirements.txt
cp .env.example .env  # 按需编辑配置
python main.py
```

## 📚 使用指南

### Web 界面
访问 `http://localhost:7000` 体验交互式演示

### API 端点
| 端点 | 方法 | 描述 |
|------|------|-------------|
| `/v1/audio/speech` | POST | 文本转语音 |
| `/api/queue-size` | GET | 查询任务队列 |

> 🔍 完整 API 文档可在本地部署后通过 Web 界面查看

## 🤝 参与贡献

我们欢迎所有形式的贡献！您可以通过以下方式参与：

- 提交 [Issue](https://github.com/dbccccccc/ttsfm/issues) 报告问题
- 发起 [Pull Request](https://github.com/dbccccccc/ttsfm/pulls) 改进代码
- 分享使用体验和建议

📜 项目采用 [MIT 许可证](LICENSE)

## 📈 项目动态

[![Star History Chart](https://api.star-history.com/svg?repos=dbccccccc/ttsfm&type=Date)](https://star-history.com/#dbccccccc/ttsfm&Date)