在 Linux 中使用 Git 并与 GitHub 进行交互管理仓库是一个非常常见的开发工作流程。以下是详细的步骤：

### 1. 安装 Git
在大多数 Linux 发行版中，Git 通常可以通过包管理器安装。以下是一些常见发行版的安装命令：

```bash
# 对于 Ubuntu/Debian 系统
sudo apt update
sudo apt install git

# 对于 Fedora 系统
sudo dnf install git

# 对于 Arch Linux 系统
sudo pacman -S git
```

安装完成后，可以通过以下命令来验证安装：
```bash
git --version
```

### 2. 配置 Git
在首次使用 Git 时，需要进行一些基本的配置，比如用户名和邮箱地址。这些信息将用于记录提交时的作者信息。

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

你还可以检查当前的 Git 配置：
```bash
git config --list
```

### 3. 创建一个本地仓库
在你的项目目录下，使用以下命令来初始化一个 Git 仓库：
```bash
git init
```

这会在当前目录中创建一个名为 `.git` 的隐藏文件夹，用于存储版本控制信息。

### 4. 克隆一个远程仓库（例如 GitHub 仓库）
如果你想将 GitHub 上的仓库复制到你的本地环境，可以使用以下命令：
```bash
git clone https://github.com/username/repository.git
```
请将 `username` 替换为你的 GitHub 用户名，将 `repository` 替换为你要克隆的仓库名称。

### 5. 提交更改到本地仓库
当你对文件进行了修改，可以按照以下步骤提交更改：

1. **添加文件到暂存区**：
   ```bash
   git add file1 file2
   ```
   或者一次性添加所有更改：
   ```bash
   git add .
   ```

2. **提交更改**：
   ```bash
   git commit -m "你的提交信息"
   ```

### 6. 将本地更改推送到 GitHub 仓库
要将你的本地提交推送到 GitHub 仓库，需要执行以下命令：
```bash
git push origin main
```
`origin` 是远程仓库的默认名称，`main` 是主分支的名称。根据你的仓库设置，主分支的名称可能是 `main` 或 `master`。

### 7. 从 GitHub 拉取更新
如果你想从 GitHub 仓库中获取最新的更新，可以使用以下命令：
```bash
git pull origin main
```

### 8. 创建和切换分支
在 Git 中，使用分支来管理不同的开发任务非常重要。以下是创建和切换分支的命令：

1. **创建新分支**：
   ```bash
   git branch new-branch
   ```

2. **切换到新分支**：
   ```bash
   git checkout new-branch
   ```

3. **创建并切换到新分支（一步完成）**：
   ```bash
   git checkout -b new-branch
   ```

### 9. 与 GitHub 进行身份验证
为了与 GitHub 进行交互，你需要在本地配置身份验证信息，通常通过 SSH 密钥或者 GitHub Token 进行：

#### 使用 SSH 密钥
1. **生成 SSH 密钥**：
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```
   如果系统不支持 `ed25519`，可以使用 `rsa`：
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
   ```

2. **将 SSH 公钥添加到 GitHub**：
   - 打开公钥文件：`~/.ssh/id_ed25519.pub` 或 `~/.ssh/id_rsa.pub`。
   - 复制内容并在 GitHub 上添加：进入 [GitHub SSH 设置页面](https://github.com/settings/keys)，点击“New SSH key”，然后粘贴密钥内容。

3. **测试连接**：
   ```bash
   ssh -T git@github.com
   ```

#### 使用 GitHub Token
对于 HTTPS 的身份验证，GitHub 不再支持使用用户名和密码登录，而是需要使用 GitHub Token。可以通过以下步骤创建 Token：

1. 前往 GitHub 的 [Token 生成页面](https://github.com/settings/tokens)。
2. 点击“Generate new token”，选择所需的权限，然后生成。
3. 在推送或拉取时，使用生成的 Token 替代密码。

### 10. 常用 Git 命令总结
- `git status`：查看当前仓库状态。
- `git log`：查看提交历史。
- `git diff`：查看未暂存的更改。
- `git branch`：列出所有分支。
- `git merge branch-name`：合并分支。
- `git remote -v`：查看远程仓库。