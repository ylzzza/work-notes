---
title: 远端 x86_64 机器配置 cc-switch + Codex
date: 2026-07-06
tags:
  - codex
  - cc-switch
  - linux
  - config
aliases:
  - 远端 Codex cc-switch 配置
  - cc-switch Codex 代理配置
---

# 远端 x86_64 机器配置 cc-switch + Codex

## 目标链路

```text
远端 Codex
-> http://127.0.0.1:15721/v1
-> 远端 cc-switch
-> 上游 API
```

> [!note] 适用环境
> 适用于普通 Ubuntu / Debian / CentOS / Rocky / Fedora 的 x86_64 远端机器。

## 1. 下载 Linux x64 版本

下载 cc-switch：

```bash
curl -L -O https://github.com/SaladDay/cc-switch-cli/releases/download/v5.8.7/cc-switch-cli-v5.8.7-linux-x64.tar.gz
```

解压并添加执行权限：

```bash
tar -xzf cc-switch-cli-v5.8.7-linux-x64.tar.gz
chmod +x cc-switch
```

无 sudo 安装到用户目录：

```bash
mkdir -p ~/.local/bin
mv cc-switch ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

确认安装：

```bash
which cc-switch
cc-switch --version
cc-switch --help
```

## 2. 处理 `/localhome` 符号链接报错

如果报错：

```text
配置目录路径不能包含符号链接: /localhome
```

先查真实路径：

```bash
readlink -f "$HOME"
readlink -f /localhome
```

当前真实路径示例：

```bash
/localdata/localhome/yelinzhong
```

设置 cc-switch 配置目录：

```bash
export CC_SWITCH_CONFIG_DIR="/localdata/localhome/yelinzhong/.cc-switch"
mkdir -p "$CC_SWITCH_CONFIG_DIR"
chmod 700 "$CC_SWITCH_CONFIG_DIR"
```

永久保存：

```bash
echo 'export CC_SWITCH_CONFIG_DIR="/localdata/localhome/yelinzhong/.cc-switch"' >> ~/.bashrc
source ~/.bashrc
```

> [!tip]
> 如果换了机器或用户名，记得把 `/localdata/localhome/yelinzhong` 替换成 `readlink -f "$HOME"` 查到的真实路径。

## 3. 初始化 Codex 目录

```bash
codex --help
mkdir -p ~/.codex
```

## 4. 打开 cc-switch 配置 Provider

启动 TUI：

```bash
cc-switch
```

在界面里执行以下操作：

1. 选择 **Codex** 应用。
2. 添加 **Provider**。
3. 填写 **API Key**。
4. 填写 **Base URL**，例如：`https://你的中转地址/v1`。
5. 填写 **Model**，例如：`gpt-5.5`。
6. **API Format** 选择 `openai_chat` 或 `openai_responses`。
7. 保存。
8. 切换到这个 Provider。

API Format 选择规则：

| 上游支持接口 | cc-switch API Format |
| --- | --- |
| `/v1/chat/completions` | `openai_chat` |
| `/v1/responses` | `openai_responses` |

## 5. 配置 cc-switch Codex 代理

固定使用端口 `15721`：

```bash
cc-switch --app codex proxy config --listen-port 15721
```

启用 Codex 代理：

```bash
cc-switch --app codex proxy enable
```

查看代理状态：

```bash
cc-switch proxy show
```

如果要前台调试日志，可以用：

```bash
cc-switch --app codex proxy disable
cc-switch proxy serve --listen-port 15721 --takeover codex
```

## 6. 配置 Codex

编辑配置文件：

```bash
vim ~/.codex/config.toml
```

写入：

```toml
model_provider = "CCSwitch"
model = "gpt-5.5"
model_reasoning_effort = "high"
preferred_auth_method = "apikey"

[model_providers.CCSwitch]
name = "CCSwitch"
base_url = "http://127.0.0.1:15721/v1"
wire_api = "responses"
```

> [!warning] 注意
> Codex 这里必须写：
>
> ```toml
> wire_api = "responses"
> ```
>
> 不要写：
>
> ```toml
> wire_api = "chat"
> ```

## 7. 测试

测试代理端口：

```bash
curl http://127.0.0.1:15721/v1/models
```

启动 Codex：

```bash
OPENAI_API_KEY=dummy codex
```

如果正常进入对话，就说明配置完成。

## 常用命令

```bash
cc-switch
cc-switch proxy show
cc-switch --app codex proxy enable
cc-switch --app codex proxy disable
cc-switch --app codex proxy config --listen-port 15721
cc-switch proxy serve --listen-port 15721 --takeover codex
```

## 排错

### 查看配置目录

```bash
echo "$CC_SWITCH_CONFIG_DIR"
```

### 确认路径没有符号链接

```bash
namei -l "$CC_SWITCH_CONFIG_DIR"
```

### 查看端口是否被占用

```bash
ss -lntp | grep 15721
```

### 端口被占用时更换端口

如果 `15721` 被占用，可以换一个端口，例如 `15722`：

```bash
cc-switch --app codex proxy config --listen-port 15722
```

然后同步修改 Codex 配置：

```toml
base_url = "http://127.0.0.1:15722/v1"
```

## 快速核对清单

- [ ] `cc-switch` 已安装到 `~/.local/bin`。
- [ ] `~/.local/bin` 已加入 `PATH`。
- [ ] 如遇 `/localhome` 符号链接报错，已设置 `CC_SWITCH_CONFIG_DIR` 到真实路径。
- [ ] cc-switch 中已为 Codex 添加并切换 Provider。
- [ ] cc-switch Codex 代理端口已配置为 `15721`。
- [ ] `~/.codex/config.toml` 中 `base_url` 指向 `http://127.0.0.1:15721/v1`。
- [ ] `wire_api = "responses"`。
- [ ] `curl http://127.0.0.1:15721/v1/models` 能正常返回。
- [ ] `OPENAI_API_KEY=dummy codex` 能正常进入对话。
