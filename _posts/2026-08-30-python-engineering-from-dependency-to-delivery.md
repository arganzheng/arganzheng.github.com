---
layout: post
title: Python 在 AI-Infra（07）：工程化——从依赖管理到生产交付
subtitle: Python Engineering, from Dependency Management to Production Delivery
tags: [Python]
catalog: true
---


前面几篇讨论的都是"代码本身"：语言机制、类型与数据契约、并发、元编程、内存、测试与调试。这一篇讨论一件不同的事——**怎么把这些代码变成一个可以交付的东西**。

对 Java 开发者来说，这套东西是现成的：

| 关注点 | Java | Python |
|---|---|---|
| 依赖管理与构建 | Maven / Gradle | pyproject.toml + pip / uv / Poetry |
| 环境隔离 | JVM + classpath 天然隔离 | venv / conda（必须显式创建） |
| 应用框架与依赖注入 | Spring | **没有等价物**（按需组合） |
| 单元测试 | JUnit | pytest（见[篇六](/python-unit-testing-troubleshooting-and-debugging.html)） |
| 日志 | SLF4J + Logback | `logging`（见[篇六](/python-unit-testing-troubleshooting-and-debugging.html)） |
| 静态检查 | Checkstyle / SpotBugs | Ruff |
| 类型检查 | 编译器内建 | mypy / pyright（见[篇二](/python-type-system-and-data-contract-design.html)） |
| 打包产物 | jar / war | wheel / sdist |
| 部署运行 | Docker / Kubernetes | Docker + ASGI server |

这张表里最值得注意的是**"没有等价物"那一行**。Python 没有 Spring 这样的单一核心框架来统摄一切，工程化是由若干相互独立的工具组合出来的。这意味着两件事：一是选择更自由，二是**没有人替你做决定**——项目的工程规范必须自己立。

Java 开发者常有的一个错觉是"Python 简单，随便装装就能跑"。在单机脚本上确实如此，但 AI-Infra 的场景会迅速打破这个错觉：

- 一个 `pip install torch` 装出来的版本，可能和线上 GPU 的 driver 不兼容；
- 同一份 `requirements.txt`，今天装出来的环境和三个月后装出来的不是同一个；
- 本地跑得好好的代码，打成包装到另一台机器上 `ImportError`；
- 一个 Docker 镜像因为把 torch 放错了层，体积从 2GB 变成 12GB，每次 CI 都要重新推。

这些问题都不是语言问题，而是**交付问题**。

本文的组织轴是**项目作为一个可交付物的生命周期**：声明依赖 → 隔离环境 → 锁定版本 → 检查质量 → 打成制品 → 交付运行。

需要说明本文的边界。以下内容属于其他篇，本文只做交叉引用，不重复展开：

- **测试**（pytest、fixture、mock、覆盖率）与**日志配置** → [《Python 单元测试、问题定位与调试实践》](/python-unit-testing-troubleshooting-and-debugging.html)
- **异步与并发**（asyncio、GIL、事件循环）→ [《Python 并发、异步与任务协作》](/python-concurrency-asynchrony-and-task-collaboration.html)
- **类型检查配置**（mypy / pyright）与**类型信息分发**（`py.typed`、PEP 561）→ [《Python 类型系统与数据契约设计》](/python-type-system-and-data-contract-design.html)
- **`import` 机制、`sys.path` 与 src 布局** → [《Python 核心机制与工程基础》](/python-core-mechanisms-and-engineering-fundamentals.html)
- **入口点做插件发现** → [《Python 反射、元编程与插件化机制》](/python-reflection-metaprogramming-and-plugin-architecture.html)

> **版本基线**：Python 生态的工具链演进很快，本文以 **Python 3.11+、uv 0.5、PyTorch 2.4、setuptools 75** 为基线。涉及具体版本号的地方都集中在代码块里，读到时请以官方文档为准。


## 一、pyproject.toml：项目元数据的单一入口

Python 曾经有过 `setup.py`、`setup.cfg`、`requirements.txt`、`Pipfile`、`MANIFEST.in` 并存的混乱时期。PEP 518 / 621 之后，`pyproject.toml` 成为标准的单一配置入口——它的地位相当于 Maven 的 `pom.xml`。

### 1. `[project]` 段：项目元数据

```toml
[project]
name = "inference-service"
version = "0.1.0"
description = "A model inference service"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [{ name = "Your Name", email = "you@example.com" }]

dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.9",
    "httpx>=0.27",
    "uvicorn[standard]>=0.32",
]
```

几个关键字段：

- **`requires-python`**：声明解释器版本要求。这一项比看起来重要——它决定了 pip 会不会把包装到不兼容的环境里，也决定了 mypy / Ruff 按哪个版本的语法规则检查。
- **`dependencies`**：**抽象依赖**（abstract dependencies），只声明"我需要什么"，不锁定具体版本。锁定是另一件事，见第三章。
- **`version`**：也可以交给构建后端动态生成（`dynamic = ["version"]`），从 `__init__.py` 的 `__version__` 或 git tag 读取。

对应 Java：`dependencies` 相当于 `pom.xml` 的 `<dependencies>`，`requires-python` 相当于 `maven.compiler.source`。

### 2. 可选依赖与依赖分组

开发工具不应该进入生产环境。有两种表达方式，含义不同：

```toml
# 方式一：optional-dependencies（PEP 621）
# 这是"包的可选特性"，会随包发布，用户可以 pip install inference-service[gpu]
[project.optional-dependencies]
gpu = ["nvidia-ml-py>=12.0"]
s3 = ["boto3>=1.35"]

# 方式二：dependency-groups（PEP 735，较新）
# 这是"开发时才用的依赖"，不随包发布
[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.8",
    "mypy>=1.13",
]
```

区别在于**是否属于包的对外契约**：

- `optional-dependencies` 是给**用户**的选项——"你如果要用 GPU 监控功能，就多装这几个包"；
- `dependency-groups` 是给**开发者**的——测试和 lint 工具，用户根本不需要知道。

历史上大家都把 dev 依赖塞进 `optional-dependencies.dev`，因为当时没有别的选择。PEP 735 之后应该迁移到 `dependency-groups`。

对应 Java：`optional-dependencies` 类似 Maven 的 `<optional>true</optional>`，`dependency-groups` 类似 `<scope>test</scope>`。

### 3. `[project.scripts]`：命令行入口

```toml
[project.scripts]
inference-server = "inference_service.cli:main"
inference-bench = "inference_service.bench:main"
```

安装后会在环境的 `bin/` 目录生成可执行脚本，直接 `inference-server --port 8000` 即可运行。等号右边是 `模块路径:函数名`。

```python
# src/inference_service/cli.py
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ...
```

这个机制的底层是 Python 包的**入口点（entry points）**。除了生成 CLI，同一套机制还能用来做插件发现——让主程序自动找到已安装的第三方扩展。

> 入口点用于插件发现的完整做法（`[project.entry-points."my_ai.backends"]` 与 `importlib.metadata.entry_points()`），见[《Python 反射、元编程与插件化机制》](/python-reflection-metaprogramming-and-plugin-architecture.html)的"入口点：面向发行包的插件发现"一节。本文只关注它作为 CLI 入口的用法。

对应 Java：类似在 `MANIFEST.MF` 里声明 `Main-Class`，或用 Maven 的 `appassembler` 插件生成启动脚本。

### 4. `[build-system]`：构建后端

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"
```

这一段告诉 pip："要构建这个项目，先装 `requires` 里的东西，然后调用 `build-backend`"。常见的几个后端：

| 后端 | 适用场景 |
|---|---|
| `setuptools` | 默认选择，生态最成熟，纯 Python 和简单 C 扩展都能应付 |
| `hatchling` | 更现代、配置更简洁，纯 Python 项目的好选择 |
| `flit_core` | 极简，适合单模块小库 |
| `scikit-build-core` | **带 CMake 的 C++ / CUDA 扩展**，PyTorch 生态常用 |
| `maturin` | Rust 扩展（Pydantic v2 的 `pydantic-core` 就是它构建的） |

AI-Infra 项目如果要编译 CUDA kernel，基本都是 `scikit-build-core` 或自定义的 setuptools 扩展。这部分见第六章。

`src` 布局下还需要告诉 setuptools 去哪里找包：

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

> src 布局的原理和它与 `import` 机制的关系，见[《Python 核心机制与工程基础》](/python-core-mechanisms-and-engineering-fundamentals.html)的"`sys.path`、src 布局与 editable 安装"一节。

### 5. 工具配置的聚合

`pyproject.toml` 的另一个价值是把散落的工具配置集中起来。原先 Ruff 要 `.ruff.toml`、mypy 要 `mypy.ini`、pytest 要 `pytest.ini`，现在都能写在一处：

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["slow: marks tests as slow"]
```

对应 Java：类似把 Checkstyle、SpotBugs、Surefire 的配置都写进 `pom.xml` 的 `<build><plugins>`，而不是散落在各自的 XML 里。

> `[tool.mypy]` 的详细配置和渐进式引入策略见[篇二](/python-type-system-and-data-contract-design.html)；`[tool.pytest.ini_options]` 的 marker 与 `asyncio_mode` 见[篇六](/python-unit-testing-troubleshooting-and-debugging.html)。本文不重复。

### 6. 对照 Maven：像什么，不像什么

**像的部分**：都是项目元数据 + 依赖声明 + 构建配置的单一入口，都用声明式格式，都支持插件/后端扩展。

**不像的部分**，这几点是 Java 开发者最容易踩空的：

| 维度 | Maven | pyproject.toml |
|---|---|---|
| 依赖解析 | Maven 内建，`mvn` 一个命令搞定 | **文件本身不含解析器**，靠 pip / uv / Poetry 去解析 |
| 版本锁定 | 依赖树可确定性推导 | **需要额外的锁文件**（见第三章） |
| 继承与聚合 | parent pom、多模块 | **没有对应机制**，多包仓库靠工具（uv workspace）实现 |
| 环境切换 | profile | 没有内建 profile，靠环境变量或多份配置文件 |
| 传递依赖冲突 | 有明确的"最近优先"仲裁规则 | pip 的解析器会尽力求解，失败则报冲突 |
| 仓库 | 中央仓库 + 严格坐标（groupId:artifactId） | PyPI，**只有扁平的包名**，无命名空间 |

最后一行值得多说一句：PyPI 没有 groupId 这样的命名空间，包名是全局先到先得的扁平空间。这直接导致了**名称抢注**和**typosquatting**（把 `reqeusts` 注册成恶意包）这类供应链风险。所以企业项目应该：固定依赖版本、使用锁文件与 hash 校验、定期扫描漏洞——这些在第三章展开。


## 二、虚拟环境与解释器隔离

### 1. 为什么 Python 比 Java 更依赖环境隔离

这是 Java 开发者最需要转换观念的一点。

JVM 的依赖隔离是**天然的**：每个应用启动时通过 `-cp` 指定自己的 classpath，两个应用用不同版本的同一个库，互不干扰。你甚至可以在同一个 JVM 里用 ClassLoader 隔离出多份。

Python 没有 classpath 的概念。`import` 的查找路径是 `sys.path`，而它默认包含**解释器全局的 `site-packages` 目录**。也就是说：

```text
/usr/lib/python3.11/site-packages/     ← 所有项目共用这一个目录
├── pydantic/          版本只能有一个
├── torch/             版本只能有一个
└── ...
```

一个解释器下，**一个包只能有一个版本**。于是：

```text
项目 A 需要 pydantic 1.x
项目 B 需要 pydantic 2.x
```

在同一个解释器里，这两个项目无法共存。装了 B 就会破坏 A——这就是所谓的"依赖地狱"。

虚拟环境的作用就是给每个项目一份独立的 `site-packages`。

### 2. venv 基本操作

```bash
# 创建（会在当前目录生成 .venv/）
python -m venv .venv

# 激活
source .venv/bin/activate           # Linux / macOS
.venv\Scripts\Activate.ps1          # Windows PowerShell

# 安装依赖
python -m pip install -e ".[dev]"

# 退出
deactivate
```

激活的本质很朴素——它只是改了几个环境变量：

```bash
# 激活前
$ which python
/usr/bin/python

# 激活后
$ which python
/path/to/project/.venv/bin/python
```

`activate` 把 `.venv/bin` 插到 `PATH` 最前面，于是 `python` 和 `pip` 都指向虚拟环境里的那份。虚拟环境里的解释器会把 `sys.path` 指向自己的 `site-packages`，不再看全局的。

因为激活只是改 `PATH`，所以**不激活也完全可以用**——直接给出完整路径即可：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pip install httpx
```

在 CI 脚本和 Dockerfile 里，这种写法比 `source activate` 更可靠，因为不依赖 shell 的状态。

> 用 uv 的话，`uv run pytest` 会自动使用项目的 `.venv`，连路径都不用写。见第三章。

### 3. 虚拟环境解决不了什么

这一点必须说清楚，否则会产生虚假的安全感。虚拟环境隔离的只是 **Python 包**，以下东西一概不隔离：

| 层次 | 是否被虚拟环境隔离 | 说明 |
|---|---|---|
| Python 包（纯 Python） | 隔离 | 各自的 `site-packages` |
| Python 包（含 C 扩展的二进制部分） | 隔离 | wheel 里的 `.so` 也在 `site-packages` 里 |
| **Python 解释器版本** | **不隔离** | `python -m venv` 用的是创建它的那个解释器 |
| **系统共享库**（glibc、libstdc++） | **不隔离** | 由操作系统提供 |
| **CUDA driver** | **不隔离** | 内核态驱动，全机唯一 |
| **系统工具**（gcc、cmake、git） | **不隔离** | 由 PATH 上的系统安装提供 |

所以下面这类问题，虚拟环境救不了：

```text
本机 CUDA driver 支持到 CUDA 12.1
但装了编译against CUDA 12.4 的 torch wheel
→ 运行时报 "CUDA driver version is insufficient"
```

这是第四章的主题。而**解释器版本**的隔离需要额外工具：

```bash
# uv 可以直接管理解释器版本
uv python install 3.11
uv venv --python 3.11

# 传统方案：pyenv
pyenv install 3.11.10
pyenv local 3.11.10
```

对应 Java：这一层相当于用 SDKMAN 或 jenv 管理多个 JDK 版本——`pom.xml` 里的 `maven.compiler.source` 只是声明要求，真正装哪个 JDK 是环境的事。

### 4. conda vs venv：什么时候真的需要 conda

这是 AI 领域一个长期的选择困扰。区别在于**管理的范围**：

| | venv + pip | conda |
|---|---|---|
| 管理 Python 包 | 是 | 是 |
| 管理 Python 解释器本身 | 否（需 pyenv / uv） | **是** |
| 管理非 Python 的二进制依赖 | **否** | **是**（cudatoolkit、MKL、gcc、ffmpeg） |
| 包来源 | PyPI | conda-forge / defaults（也可混用 pip） |
| 依赖解析 | 只解 Python 包 | 解整个环境（含 C 库） |
| 环境体积 | 小 | 大 |

**真正需要 conda 的场景**，只有一个核心判据：**你需要环境自带非 Python 的二进制依赖，且无法通过系统包管理器或容器基础镜像提供**。典型情况：

- 在没有 root 权限的共享 HPC 集群上，需要特定版本的 `cudatoolkit`、MKL、编译器；
- 依赖 GDAL、GEOS、ffmpeg 这类系统库很难装对的科学计算包；
- 需要在同一台机器上并存多套完全不同的 CUDA toolkit。

**不需要 conda 的场景**（现在是多数）：

- 用 Docker 交付——基础镜像已经提供了 CUDA toolkit 和系统库，conda 是多余的一层；
- 纯粹的服务类项目，依赖都能从 PyPI 装上。

一个常见的坑是 **conda 和 pip 混用**：在 conda 环境里 `pip install` 会绕过 conda 的依赖解析，conda 不知道 pip 装了什么，后续 `conda install` 可能覆盖掉 pip 装的文件，导致环境损坏。如果必须混用，原则是：**先 conda 装完所有能装的，最后再用 pip 装剩下的，之后不再动 conda**。

**本文后续统一采用 venv + uv 的方案**，因为容器化交付是 AI-Infra 服务的主流形态，系统级依赖交给基础镜像更清晰。


## 三、依赖管理与可复现构建

### 1. 抽象依赖与锁定依赖

这是整章最重要的概念区分，也是 `pip install -r requirements.txt` 给人虚假安全感的根源。

**抽象依赖**（abstract）回答"我的代码需要什么"：

```toml
# pyproject.toml
dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.9",
]
```

**锁定依赖**（concrete / pinned）回答"这次实际装了什么"：

```text
# 锁文件（示意）
annotated-types==0.7.0
anyio==4.7.0
click==8.1.7
fastapi==0.115.6
h11==0.14.0
idna==3.10
pydantic==2.10.4
pydantic-core==2.27.2
sniffio==1.3.1
starlette==0.41.3
typing-extensions==4.12.2
```

注意锁文件里有一堆你从没听过的包——它们是**传递依赖**。`fastapi>=0.115` 这一行抽象依赖，展开后是十几个具体包。

两者的职责不同，**都需要，且不能互相替代**：

| | 抽象依赖 | 锁定依赖 |
|---|---|---|
| 写在哪 | `pyproject.toml` | `uv.lock` / `poetry.lock` / `requirements.lock` |
| 谁维护 | 人手写 | 工具生成，**不要手改** |
| 是否提交 git | 是 | **是**（应用要提交；库通常不提交，见下） |
| 作用 | 表达兼容范围 | 保证可复现 |
| 库项目 | 必须有 | 可以没有（不能限制下游） |
| 应用/服务 | 必须有 | **必须有** |

**库和应用的区别很关键**：库不该锁死依赖版本，否则会和下游其他库的要求冲突（这也是为什么第 4 节讨论"上界该不该加"）；应用是依赖链的终点，锁得越死越好。

### 2. 为什么 requirements.txt 不等于可复现

一个典型的 `requirements.txt`：

```text
fastapi>=0.115
pydantic>=2.9
httpx>=0.27
```

它有三重不可复现：

1. **版本范围**：`>=0.115` 今天解析成 0.115.6，三个月后是 0.118.0，行为可能已经变了；
2. **传递依赖完全没约束**：即使你把直接依赖钉成 `==`，`starlette` 这类传递依赖仍然浮动；
3. **没有完整性校验**：即使版本号一致，你也无法确认下载到的文件和当初是同一个（PyPI 上的包理论上可以被重新上传，或者你走的是被污染的镜像）。

即使把所有版本都写成 `==`（所谓 "fully pinned requirements"），仍缺第 3 点。真正的可复现需要 **hash 校验**：

```text
# requirements.lock（pip-compile --generate-hashes 生成）
fastapi==0.115.6 \
    --hash=sha256:9ec46f7addc14ea472958a96aae5b5de65f39721a46aaf5705c480d9a8b8...
pydantic==2.10.4 \
    --hash=sha256:597e135ea68be3a37552fb524bc7d0d66dcf93d395acd93a00682f1efcb8...
```

```bash
# 安装时严格校验，任何 hash 不匹配都会失败
pip install --require-hashes -r requirements.lock
```

`uv.lock` 和 `poetry.lock` 默认就带 hash，不需要额外开关。

### 3. 工具选型

Python 的依赖管理工具经历了长期的碎片化。当前的格局：

| 工具 | 定位 | 锁文件 | 管解释器 | 速度 | 建议 |
|---|---|---|---|---|---|
| **pip** | 安装器（不是项目管理器） | 无 | 否 | 中 | 底层基础，人人都会用到 |
| **venv** | 标准库虚拟环境 | — | 否 | — | 标准库，零依赖 |
| **pip-tools** | 从抽象依赖编译锁文件 | `requirements.txt` | 否 | 中 | 想留在 pip 生态时的稳妥选择 |
| **Poetry** | 全套项目管理 | `poetry.lock` | 否 | 慢 | 成熟但慢，早期项目多 |
| **PDM** | 全套项目管理 | `pdm.lock` | 部分 | 中 | 标准兼容性好，用户较少 |
| **uv** | 全套项目管理（Rust 实现） | `uv.lock` | **是** | **极快** | **新项目推荐** |
| **conda** | 环境管理（含非 Python 依赖） | `environment.yml` | 是 | 慢 | 只在真需要系统级依赖时 |

**推荐 uv 的理由**，不只是快（虽然快得很夸张，装 torch 这种大包的差距是分钟级 vs 十几秒）：

- 一个工具覆盖 venv + pip + pip-tools + pyenv 的职责，减少工具链拼接；
- 锁文件跨平台（同一份 `uv.lock` 记录多平台的解析结果，见第四章为什么这点对 AI 项目重要）；
- 遵循 PEP 621 标准的 `pyproject.toml`，不像 Poetry 早期用自己的 `[tool.poetry]` 格式，迁移成本低；
- 能直接管理解释器版本。

常用命令：

```bash
uv venv                        # 创建虚拟环境
uv add fastapi                 # 加依赖（同时更新 pyproject.toml 和 uv.lock）
uv add --dev pytest ruff       # 加开发依赖
uv sync                        # 按锁文件精确安装
uv sync --frozen               # 严格按锁文件，不允许更新（CI 用这个）
uv lock --upgrade              # 主动升级锁文件
uv run pytest                  # 在项目环境里执行命令，无需激活
```

**`uv sync --frozen` 是 CI 里应该用的形式**——如果锁文件和 `pyproject.toml` 不一致就直接失败，而不是悄悄重新解析。这相当于 Maven 的 `--offline` + 严格版本。

对应 Java：uv 的定位相当于 Maven 本身（依赖解析 + 环境管理 + 命令执行），而 pip 只相当于 Maven 依赖下载的那一小部分。

### 4. 版本约束：上界该不该加

这是个有争议的话题，我给出的判断是**分情况**：

```toml
# 语义化版本约束
"pydantic>=2.9"           # 只有下界，接受未来任何版本
"pydantic>=2.9,<3"        # 排除下一个大版本
"pydantic~=2.9.0"         # 等价于 >=2.9.0,<2.10.0（锁到小版本）
"pydantic~=2.9"           # 等价于 >=2.9,<3.0（锁到大版本）
"pydantic==2.10.4"        # 完全钉死
"pydantic!=2.10.0"        # 排除有 bug 的特定版本
```

**应用 / 服务**：抽象依赖用 `>=` 就够，因为可复现由锁文件保证，加上界只会让日后升级变麻烦。

**库**：这里才是真问题。加上界（`<3`）看似安全，实际会造成**依赖地狱的传播**——如果你的库写 `pydantic<3`，而下游项目还依赖另一个库要求 `pydantic>=3`，那两个库就无法共存，即使你的代码其实兼容 v3。

我的建议：

- **默认不加上界**；
- 只在**已知**不兼容时加，且用 `!=` 排除具体版本而不是砍掉整个未来；
- 依赖的库有明确破坏性历史（如 Pydantic v1→v2 那种级别）时，才加大版本上界；
- 靠 CI 定期跑最新依赖来提前发现问题，而不是靠上界预防。

### 5. 直接依赖、传递依赖与供应链安全

回到第一章末尾提到的 PyPI 扁平命名空间问题。几条实践：

**区分直接和传递依赖**。只把真正 `import` 的包写进 `pyproject.toml`。一个常见错误是把锁文件的内容抄进抽象依赖，导致一堆传递依赖变成直接依赖，日后无法自动升级。

```bash
# 看依赖树，确认谁引入了什么
uv tree
pip install pipdeptree && pipdeptree
```

**扫描漏洞**：

```bash
# 检查已知漏洞（对照 PyPI Advisory Database）
uv pip list | pip-audit
# 或
pip install pip-audit && pip-audit
```

**约束镜像源**。企业环境常用私有索引，注意 `--extra-index-url` 的**优先级陷阱**：pip 会在所有索引里找同名包并选版本最高的，这意味着公网上有人注册同名高版本包就可能被装进来（dependency confusion 攻击）。更安全的做法是用 `--index-url` 指定唯一索引，或用 uv 的 `index` 配置显式声明每个包的来源：

```toml
[[tool.uv.index]]
name = "internal"
url = "https://pypi.internal.example.com/simple"
default = true
```

对应 Java：相当于 Maven 的 `<mirrors>` + `<repositories>` 加上 `settings.xml` 里的仓库优先级——但 Maven 有 groupId 命名空间，天然不容易被同名包混淆，Python 这里的风险更高。


## 四、AI-Infra 的依赖难题

前三章的内容对任何 Python 项目都适用。这一章讲的是 AI-Infra 特有的坑——它们是通用依赖管理知识覆盖不到的地方，也是新人最容易卡住的地方。

根源在于：**PyTorch 这类框架不是纯 Python 包，它捆绑了几个 GB 的、针对特定 CUDA 版本编译的二进制**。整个 PEP 440 版本体系和 PyPI 的分发模型都不是为这种情况设计的。

### 1. 本地版本标识：`+cu121` 是什么

你会看到这样的版本号：

```text
torch==2.4.0+cu121
torch==2.4.0+cu124
torch==2.4.0+cpu
torch==2.4.0+rocm6.1
```

`+` 后面的部分叫**本地版本标识**（local version identifier，PEP 440）。它的语义是"同一个上游版本的本地变体"，原本设计给"打了私有补丁的重新构建"用，PyTorch 借用它来区分 CUDA 编译目标。

几个必须知道的性质：

- **PyPI 不接受带本地版本标识的包**。所以 `pip install torch==2.4.0+cu121` 从默认 PyPI 装不到，必须指定 PyTorch 自己的索引；
- **`==2.4.0` 不匹配 `2.4.0+cu121`**，但 `torch==2.4.0` 这个约束**可以**被 `2.4.0+cu121` 满足（本地版本是"更具体"的）；
- 锁文件里记录 `2.4.0+cu121` 之后，这份锁文件就**只对同样的 CUDA 目标有效**。

那 `pip install torch` 从默认 PyPI 装到的是什么？在 Linux 上是**捆绑了 CUDA 运行时的默认变体**（当前默认对应某个 CUDA 版本，随 torch 发布而变），体积极大（2GB+），因为它把 `nvidia-*` 系列的 CUDA 库作为依赖一起拉下来了：

```bash
$ pip install torch
# 会顺带装进来一堆：
#   nvidia-cublas-cu12
#   nvidia-cudnn-cu12
#   nvidia-cuda-nvrtc-cu12
#   nvidia-cusparse-cu12
#   ...
```

这就是为什么一个只写了 `torch` 的项目，装出来的虚拟环境有 5–8GB。

### 2. `--index-url` 与 `--extra-index-url` 的区别

这两个参数的差别在 AI 项目里会造成实际故障，必须分清：

```bash
# --index-url：替换默认索引（只从这里找）
pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# --extra-index-url：追加索引（在多个索引里都找，取版本最高的）
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

**`--extra-index-url` 在这里是危险的**：pip 会同时看 PyPI 和 PyTorch 索引，然后选"版本最高"的。由于 PyPI 上的 `torch` 版本号没有 `+cu121` 后缀，版本比较的结果可能让 pip 选中 PyPI 那个默认变体，你以为装的是 cu121 专版，实际装的是别的。这类"明明指定了却没生效"的问题非常难查。

**结论：装 PyTorch 时用 `--index-url`，不要用 `--extra-index-url`。**

但这带来新问题——用了 `--index-url` 就只能从 PyTorch 索引找包，而你的 `fastapi`、`pydantic` 在那里没有。解决办法是按包指定索引来源。uv 直接支持：

```toml
[project]
dependencies = ["torch==2.4.0", "fastapi>=0.115"]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true          # 只有显式指定的包才从这里找

[tool.uv.sources]
torch = { index = "pytorch-cu121" }    # 只有 torch 走这个索引
```

`explicit = true` 是关键——它避免了上面那个 dependency confusion 式的问题：这个索引不参与其他包的解析，只服务于 `[tool.uv.sources]` 里点名的包。

### 3. CUDA 兼容矩阵：三层，不是一层

新人最常见的误解是"CUDA 版本"只有一个。实际有三层，各自的兼容规则不同：

```text
┌─────────────────────────────────────────────┐
│ 3. torch wheel（编译时链接的 CUDA runtime）  │  ← pip 装进来的，可多版本共存
│    torch 2.4.0+cu121                        │
├─────────────────────────────────────────────┤
│ 2. CUDA runtime / toolkit                   │  ← 可由 wheel 自带，或来自镜像
│    libcudart.so, libcublas.so, cuDNN        │
├─────────────────────────────────────────────┤
│ 1. NVIDIA driver（内核态）                   │  ← 全机唯一，虚拟环境隔离不了
│    nvidia-smi 显示的 "CUDA Version"          │
└─────────────────────────────────────────────┘
```

核心规则是 **CUDA 的"次要版本兼容性"（minor version compatibility）**：driver 只需要不低于 runtime 的**大版本**要求即可，同一大版本内的次要版本向前兼容。所以：

```text
driver 支持 CUDA 12.2，装 torch+cu121  → 可以（12.x 内向前兼容）
driver 支持 CUDA 12.2，装 torch+cu124  → 通常可以（12.x 内），但不保证新特性可用
driver 支持 CUDA 11.8，装 torch+cu121  → 不行，大版本不匹配
```

诊断命令：

```bash
# 第 1 层：driver 支持的最高 CUDA 版本
nvidia-smi                     # 右上角 "CUDA Version: 12.2"

# 第 3 层：torch 实际链接的 CUDA 版本
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 2.4.0+cu121 12.1

# 能不能真的用起来
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

`nvidia-smi` 显示的 CUDA Version 是**driver 能支持的最高版本**，不是"已安装的 toolkit 版本"——这是个经典的误读来源。

典型报错与对应层次：

| 报错 | 层次 | 原因 |
|---|---|---|
| `CUDA driver version is insufficient for CUDA runtime version` | 1 vs 3 | driver 太老，需升级 driver 或换低版本 torch wheel |
| `torch.cuda.is_available() == False` 但有 GPU | 1 | 装了 `+cpu` 变体，或容器没挂 GPU |
| `undefined symbol: ...cudnn...` | 2 | cuDNN 版本与 torch 编译时不一致 |
| `no kernel image is available for execution` | 3 | GPU 架构（compute capability）不在 wheel 的编译目标里 |

最后一行值得注意：wheel 是针对特定的 GPU 架构列表编译的。很新的卡（如刚发布的架构）在旧 torch wheel 里可能没有对应的 kernel，即使 CUDA 版本都对得上也跑不了。

### 4. 为什么 AI 项目的锁文件经常跨平台失效

设想这个场景：开发者在 macOS 上 `uv lock`，锁文件记录了 `torch==2.4.0`（macOS 上没有 CUDA 变体）。CI 在 Linux + GPU 上 `uv sync`，需要的是 `torch==2.4.0+cu121`。锁文件对不上。

根因是**同一份抽象依赖在不同平台解析出不同结果**，而这在 AI 项目里比普通项目严重得多：

- torch 的变体取决于 CUDA 目标（`+cu121` / `+cpu` / `+rocm`）；
- `nvidia-*` 系列的 CUDA 库只有 Linux wheel，macOS / Windows 上根本不存在；
- 某些包（`flash-attn`、`xformers`、`bitsandbytes`、`triton`）只有 Linux 的 wheel，或者需要现场编译；
- Apple Silicon 上 torch 走 MPS 后端，依赖集完全不同。

应对方式：

**方式一：锁文件支持多平台**（uv 的做法）。`uv.lock` 会为多个平台各记录一套解析结果，并用环境标记区分：

```toml
[tool.uv]
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'darwin'",
]
```

**方式二：用环境标记声明平台差异**：

```toml
dependencies = [
    "torch==2.4.0",
    "flash-attn>=2.6; sys_platform == 'linux' and platform_machine == 'x86_64'",
]
```

**方式三（最省事，也最常见）：只锁一个目标平台**。既然交付形态是容器，就统一在与生产一致的 Linux 镜像里生成锁文件，本地开发环境不追求和锁文件完全一致：

```bash
# 在与生产同构的容器里生成锁文件
docker run --rm -v "$PWD:/app" -w /app python:3.11-slim \
    sh -c "pip install uv && uv lock"
```

### 5. 实践建议：把 torch 从项目依赖里摘出去

这是我认为对 AI-Infra 项目最有价值的一条建议，也和第七章的容器化直接衔接。

**问题**：如果 `pyproject.toml` 里写着 `torch==2.4.0`，那么：

- 每次 `uv sync` 都可能重新下载几 GB；
- Docker 构建时 torch 和业务依赖在同一层，改一行业务代码就要重装 torch；
- 锁文件绑死了 CUDA 目标，换 GPU 环境就得重新锁。

**做法**：把 torch 及其 CUDA 依赖交给**基础镜像**，项目依赖里只声明"我需要 torch，但别帮我装"。

```dockerfile
# 基础镜像已经带好了对应 CUDA 版本的 torch
FROM nvcr.io/nvidia/pytorch:24.10-py3
```

配合 uv 的方式之一是把 torch 放进可选依赖，日常开发装、镜像里不装：

```toml
[project]
dependencies = ["fastapi>=0.115", "pydantic>=2.9"]   # 不含 torch

[project.optional-dependencies]
# 本地开发用：pip install -e ".[torch-cu121]"
torch-cu121 = ["torch==2.4.0"]
```

```dockerfile
# 镜像里只装业务依赖，torch 用基础镜像自带的
RUN uv sync --frozen --no-install-project --no-dev
```

**代价要说清楚**：这样做之后，"torch 版本"不再由锁文件保证，而是由基础镜像的 tag 保证。所以基础镜像的 tag **必须固定**，不能用 `latest`：

```dockerfile
# 好：可复现
FROM nvcr.io/nvidia/pytorch:24.10-py3

# 坏：不可复现，某天构建出来的东西就变了
FROM nvcr.io/nvidia/pytorch:latest
```

换句话说，可复现性的责任从锁文件转移到了镜像 tag 上——它没有消失，只是换了地方。这是个有意识的权衡，不是免费的午餐。

对应 Java：类似把一个巨大的、平台相关的 native 依赖从 `pom.xml` 移到基础镜像里预装，`pom.xml` 里标 `<scope>provided</scope>`。区别是 Java 极少遇到几 GB 级别的 native 依赖，所以这个模式在 Java 生态里并不常见。


## 五、代码质量工具链

Java 的静态检查有编译器兜底：类型错误、未使用的导入、不可达代码，`javac` 直接拒绝编译。Checkstyle 和 SpotBugs 是在此之上加规范和缺陷模式检查。

Python 没有这个兜底。**语法正确的代码就能运行**，错误留到运行时才暴露。所以对 Python 项目来说，静态检查不是"锦上添花的规范"，而是**替代编译器的第一道防线**。

### 1. Ruff：格式化与 Lint 二合一

Python 的 lint 工具链曾经是这样拼起来的：

```text
flake8      语法风格检查（PEP 8）
pylint      更严格的代码分析
isort       import 排序
Black       代码格式化
pyupgrade   升级过时语法
autoflake   删除无用导入
```

六个工具，六份配置，还要处理它们互相冲突（Black 和 flake8 对行长的分歧是经典问题）。

Ruff 用 Rust 实现，把上面这些的能力合并成一个工具，且快 10–100 倍：

```bash
ruff check .              # Lint
ruff check --fix .        # Lint 并自动修复
ruff format .             # 格式化（兼容 Black 的风格）
ruff check --watch .      # 监听模式
```

配置集中在 `pyproject.toml`：

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]          # 让 isort 规则知道哪些是第一方包

[tool.ruff.lint]
select = [
    "E",      # pycodestyle 错误
    "W",      # pycodestyle 警告
    "F",      # Pyflakes（未使用变量、未定义名称）
    "I",      # isort（import 排序）
    "UP",     # pyupgrade（用新语法替换过时写法）
    "B",      # flake8-bugbear（常见 bug 模式）
    "SIM",    # flake8-simplify
    "ASYNC",  # flake8-async（异步代码的常见错误）
    "RUF",    # Ruff 自有规则
]
ignore = [
    "E501",   # 行长交给 formatter 处理，不用 lint 报
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]            # 测试里允许 assert
"__init__.py" = ["F401"]        # __init__.py 里允许"未使用"的重导出
```

对 AI-Infra 项目，`ASYNC` 和 `B` 这两组规则的价值特别高：

- `ASYNC` 能查出在协程里调用阻塞函数（`time.sleep`、同步 `requests`）这类问题——这正是[篇三](/python-concurrency-asynchrony-and-task-collaboration.html)讲的事件循环阻塞陷阱，Ruff 可以在 CI 里自动拦住一部分；
- `B008`（函数默认值里调用函数）、`B023`（闭包里的循环变量延迟绑定）都是 Python 特有的陷阱，靠 review 很难每次都发现。

`__init__.py` 那条 per-file-ignore 值得解释：`F401` 是"导入了但没使用"，但 `__init__.py` 里的导入往往是故意做**重导出**（对外暴露 API），并非无用。更规范的做法是配合 `__all__` 声明——见[篇一](/python-core-mechanisms-and-engineering-fundamentals.html)的"声明公共 API：`__all__`"。

### 2. 渐进式引入与 noqa 的边界

在已有代码库上一次性开全部规则，结果通常是几千条报错然后放弃。可行的路径：

**第一步，只开自动修复的规则并全量修复。** `I`（import 排序）、`UP`（语法升级）几乎 100% 能自动修，一次提交搞定：

```bash
ruff check --select I,UP --fix .
ruff format .
```

**第二步，开 `F` 和 `E`，把存量问题记录下来。** Ruff 支持生成基线：

```bash
# 先看有多少
ruff check --statistics .
```

**第三步，新代码严格，旧代码宽松。** 用 `per-file-ignores` 给遗留模块开豁免，新模块不豁免。这比"全局降标准"好——至少新写的代码是干净的。

**关于 `noqa`**：它是必要的逃逸舱，但要有纪律。

```python
# 好：说明了为什么
import torch  # noqa: F401  # 需要导入以触发算子注册

# 坏：无从判断能不能删
value = compute()  # noqa
```

两条原则：**永远指定规则码**（`# noqa: F401` 而不是裸 `# noqa`），**永远写理由**。裸 `# noqa` 会屏蔽掉这一行所有现在和未来的规则，等于埋雷。可以用规则强制：

```toml
[tool.ruff.lint]
select = ["PGH004"]      # 禁止裸 noqa
```

这和[篇二](/python-type-system-and-data-contract-design.html)里对 `# type: ignore` 的建议是同一个道理——逃逸舱要窄、要有记录。

### 3. pre-commit：把检查前移

CI 里发现格式问题再修，一来一回要几分钟。pre-commit 在 `git commit` 时本地跑检查，问题当场暴露。

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.9]
        args: [--config-file=pyproject.toml]
```

```bash
pip install pre-commit
pre-commit install              # 安装 git hook
pre-commit run --all-files      # 首次全量跑一遍
```

对 AI 项目，`check-added-large-files` 特别值得开——防止有人把模型权重或数据集误提交进 git（几百 MB 的文件进了 git 历史就很难干净地移除了）。

两个注意点：

- **pre-commit 里的 mypy 有局限**。它跑在隔离环境里，看不到项目的全部依赖，所以要用 `additional_dependencies` 手动补关键依赖（Pydantic 尤其重要，它有 mypy 插件）。复杂项目里通常只在 CI 跑完整的 mypy，pre-commit 只做快速检查。
- **pre-commit 不能替代 CI**。它可以被 `git commit --no-verify` 绕过，所以 CI 里必须再跑一遍同样的检查。pre-commit 是"提前反馈"，CI 是"强制门禁"。

### 4. 与类型检查的分工

Ruff 和 mypy 的职责不重叠，都需要：

| | Ruff | mypy / pyright |
|---|---|---|
| 分析方式 | 单文件语法层面（AST） | **跨文件类型推导** |
| 能查出 | 风格问题、未使用变量、已知 bug 模式 | 类型不匹配、`None` 未处理、签名不符 |
| 查不出 | 类型错误 | 风格问题、代码异味 |
| 速度 | 极快（毫秒级） | 慢（需构建类型图） |
| 误报处理 | `# noqa: CODE` | `# type: ignore[code]` |

一个直观的例子：

```python
def get_user(uid: int) -> User | None: ...

user = get_user(1)
print(user.name)        # Ruff 通过；mypy 报错：user 可能是 None
```

反过来：

```python
import os, sys         # mypy 通过；Ruff 报错 E401（一行多个 import）
```

CI 里的完整检查链：

```bash
ruff check .           # 风格与 bug 模式
ruff format --check .  # 格式（--check 只检查不改）
mypy src/              # 类型
pytest                 # 行为
```

> mypy 的详细配置、`strict` 各项开关的含义、以及在存量项目上渐进引入类型检查的策略，见[篇二](/python-type-system-and-data-contract-design.html)的"静态分析与推理"。测试相关见[篇六](/python-unit-testing-troubleshooting-and-debugging.html)。

### 5. 对照 Checkstyle / SpotBugs

| 关注点 | Java | Python |
|---|---|---|
| 代码风格 | Checkstyle | Ruff（`E`、`W`） |
| 缺陷模式 | SpotBugs / Error Prone | Ruff（`B`、`SIM`、`ASYNC`） |
| 代码格式化 | google-java-format / spotless | Ruff format（或 Black） |
| import 整理 | IDE / spotless | Ruff（`I`） |
| 类型检查 | **编译器内建** | **mypy / pyright（外挂）** |
| 提交前钩子 | git hook + spotless | pre-commit |
| 强制手段 | 构建失败 | CI 失败 |

最关键的差异仍然是类型检查那一行：Java 的类型检查是**语言强制**的，你不可能提交一个类型错误的 Java 项目；Python 的类型检查是**可选的外挂工具**，需要项目自己立规矩、自己在 CI 里强制。

这也是为什么本系列反复强调工程规范：**Python 给了你更大的自由度，代价是纪律必须自己建立**。团队应该在项目层面统一：Python 版本、格式化工具与配置、import 规则、类型注解覆盖要求、异常处理规范、日志规范、目录结构、测试覆盖率门槛。这些一旦写进 `pyproject.toml` 和 CI，就从"口头约定"变成了"机器强制"。


## 六、打包与分发

如果你的项目是一个服务，通常直接做成容器镜像交付（第七章），不需要发布到 PyPI。但只要你要**给别人用**——发布内部库、贡献开源项目、或者让另一个团队 `pip install` 你的包——就需要理解打包。

### 1. wheel 与 sdist

Python 有两种分发格式：

| | sdist（源码分发） | wheel（二进制分发） |
|---|---|---|
| 文件名 | `pkg-1.0.tar.gz` | `pkg-1.0-py3-none-any.whl` |
| 内容 | 源码 + 构建脚本 | **已构建好**的文件树 |
| 安装时 | **需要执行构建**（可能要编译器） | 解压 + 拷贝 |
| 速度 | 慢 | 快 |
| 平台相关性 | 与平台无关 | 可能与平台绑定 |

```bash
# 构建两种产物
pip install build
python -m build
# 产物在 dist/
#   inference_service-0.1.0.tar.gz          ← sdist
#   inference_service-0.1.0-py3-none-any.whl ← wheel
```

wheel 的文件名编码了兼容性信息，这是理解 AI 包分发的关键：

```text
torch-2.4.0+cu121-cp311-cp311-linux_x86_64.whl
│     │          │     │      │
│     │          │     │      └── 平台：Linux x86_64
│     │          │     └───────── ABI：CPython 3.11 ABI
│     │          └─────────────── Python 版本：CPython 3.11
│     └────────────────────────── 版本（含本地版本标识）
└──────────────────────────────── 包名
```

对比一个纯 Python 包：

```text
fastapi-0.115.6-py3-none-any.whl
                │   │    │
                │   │    └── 任意平台
                │   └─────── 无 ABI 要求
                └─────────── 任意 Python 3
```

`py3-none-any` 意味着"一份文件到处能用"。而 `cp311-cp311-linux_x86_64` 意味着**每个 Python 版本 × 每个平台都要单独构建一份**——这就是为什么 PyTorch 的发布矩阵那么大，也是为什么第四章讲的锁文件跨平台问题那么棘手。

**实践建议**：发布时**两种都传**。wheel 让多数用户装得快，sdist 是兜底——如果用户的平台/Python 版本没有对应 wheel，至少还能从源码构建。

### 2. 纯 Python 包 vs 带扩展的包

**纯 Python 包**很简单，`setuptools` 或 `hatchling` 直接搞定，产出 `py3-none-any` 的 wheel：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**带 C / C++ / CUDA 扩展的包**要复杂得多，这是 AI-Infra 库的常见形态（`flash-attn`、`xformers`、自研算子库都是）。

用 `scikit-build-core` + CMake 的骨架：

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "torch"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.version = ">=3.26"
wheel.packages = ["src/myops"]
```

```cmake
# CMakeLists.txt（示意）
cmake_minimum_required(VERSION 3.26)
project(myops LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)

add_library(_C MODULE src/myops/csrc/fused_attention.cu)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})
install(TARGETS _C DESTINATION myops)
```

这里有几个 Java 开发者不熟悉的硬约束：

**ABI 兼容性**。编译出的 `.so` 绑定了具体的 CPython ABI，`cp311` 的扩展不能在 3.12 上用。（有一个 `abi3` 稳定 ABI 方案可以跨版本，但用了 PyTorch C++ API 之后通常不可行。）

**manylinux**。Linux 上的 wheel 如果链接了构建机的 glibc，拿到别的发行版可能因为 glibc 太老而跑不起来。解决方案是在标准化的 `manylinux` 容器里构建：

```bash
# 在 manylinux 容器里构建，保证 glibc 兼容性下限
docker run --rm -v "$PWD:/io" quay.io/pypa/manylinux_2_28_x86_64 \
    /io/scripts/build-wheels.sh

# 检查/修复 wheel 的外部依赖
pip install auditwheel
auditwheel show dist/myops-0.1.0-cp311-cp311-linux_x86_64.whl
auditwheel repair dist/*.whl --plat manylinux_2_28_x86_64
```

`auditwheel repair` 会把依赖的外部 `.so` 打进 wheel 并改写 rpath——但对 CUDA 库通常要显式排除（`--exclude libtorch.so` 等），否则 wheel 会膨胀到几 GB，而且和用户环境里的 torch 冲突。

**与 torch 的 ABI 绑定**。链接了 `libtorch` 的扩展，必须和用户装的 torch 版本 ABI 一致。这就是为什么 `flash-attn` 的 wheel 名字里往往还带 torch 版本，以及为什么它经常需要用户现场编译（编译一次几十分钟）。

**对应 Java**：JNI 的处境类似，但 Java 生态很少走这条路，而且 JVM 屏蔽了大部分平台差异。Python 这里的复杂度显著更高——因为 Python 的性能路线本身就是"薄 Python 壳 + 厚 native 核"。

### 3. 让类型信息随包分发

包发布出去之后，下游用 mypy 检查代码时能不能看到你的类型注解？默认**不能**——需要显式声明。

```text
src/myops/
├── __init__.py
├── py.typed              ← 空文件，声明本包提供类型信息
├── core.py
└── _C.pyi                ← C 扩展的类型存根
```

```toml
[tool.setuptools.package-data]
myops = ["py.typed", "*.pyi"]
```

这一步很容易漏——`py.typed` 在源码目录里存在，但如果没配 `package-data`，构建 wheel 时不会被打进去，下游依然看不到类型。

> `py.typed` 背后的 PEP 561 机制、`.pyi` 存根的写法、typeshed 与 `types-*` 存根包的关系、以及 inline types 与 stub 的选择，见[篇二](/python-type-system-and-data-contract-design.html)的"类型载体与分发"一节。这里只强调打包时别漏掉这一步。

对带 CUDA 扩展的包，`.pyi` 存根几乎是必需的——类型检查器无法分析 `.so` 里的内容，只能靠存根知道 `myops._C.fused_attention` 的签名。

### 4. 版本号与发布

**版本号方案**。Python 的版本规范是 PEP 440，和 SemVer 大体兼容但有自己的预发布记法：

```text
1.0.0           正式版
1.0.0a1         alpha
1.0.0b2         beta
1.0.0rc1        release candidate
1.0.0.post1     发布后修订（只改包装，不改代码）
1.0.0.dev3      开发版
1.0.0+cu121     本地版本标识（第四章）
```

注意 PEP 440 的写法是 `1.0.0a1` 而不是 SemVer 的 `1.0.0-alpha.1`。

**单一版本源**。版本号写在两个地方就会不一致，让构建后端从代码里读：

```toml
[project]
name = "myops"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = { attr = "myops.__version__" }
```

```python
# src/myops/__init__.py
__version__ = "0.1.0"
```

或者从 git tag 推导（`setuptools-scm` / `hatch-vcs`），这样打 tag 就等于定版本，不会忘记改代码：

```toml
[build-system]
requires = ["setuptools>=75", "setuptools-scm>=8"]

[tool.setuptools_scm]
```

**发布流程**：

```bash
python -m build                          # 构建
pip install twine
twine check dist/*                       # 检查元数据和 README 渲染
twine upload --repository testpypi dist/* # 先发到 TestPyPI 验证
twine upload dist/*                      # 正式发布
```

两条纪律：

- **先发 TestPyPI**。PyPI 上的版本号**不能重用**——一旦上传了 `1.0.0`，即使删除也不能再传同名版本，只能发 `1.0.1`。
- **用 Trusted Publishing 而不是 API token**。GitHub Actions 可以通过 OIDC 直接向 PyPI 认证，不需要在仓库里存长期 token，避免密钥泄露风险。

对应 Java：`twine upload` 相当于 `mvn deploy`，TestPyPI 相当于 snapshot 仓库。但有个重要区别——Maven 的 SNAPSHOT 版本可以反复覆盖，PyPI 的正式版本**永久不可变**，这个约束比 Maven 严格得多。


## 七、容器化：Python 服务的交付形态

Java 的交付物是一个 `jar`——自包含、平台无关，`java -jar` 就能跑。Python 没有这种东西：一个 wheel 不含解释器，也不含系统库。所以 **Python 服务的实际交付单位是容器镜像**。

本章只讨论与 Python 直接相关的部分：镜像分层、依赖安装策略、进程模型。Kubernetes 层面的编排不在范围内。

### 1. 分层与缓存：依赖和代码要分开装

先看一个**反面例子**：

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .                                  # ← 问题在这里
RUN pip install -e .
CMD ["uvicorn", "inference_service.main:app"]
```

问题是 `COPY . .` 把代码和依赖声明一起拷进来了。Docker 的层缓存以"COPY 的内容是否变化"为准，**改任何一行业务代码都会让这一层失效**，于是 `pip install` 整个重跑。如果依赖里有 torch，每次改代码都要重新下载几 GB。

正确做法是**先拷依赖声明，装完依赖，再拷代码**：

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# 第一层：只拷依赖声明。只有依赖变化时这一层才失效
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-install-project --no-dev

# 第二层：拷代码。改代码只失效这一层
COPY src/ ./src/
RUN uv sync --frozen --no-dev

CMD [".venv/bin/uvicorn", "inference_service.main:app", "--host", "0.0.0.0"]
```

关键点：

- `--no-install-project` 让第一步只装依赖、不装项目本身，这样项目代码还没拷进来也能跑；
- `--frozen` 严格按锁文件装，锁文件与 `pyproject.toml` 不一致就失败（对应第三章）；
- `--no-dev` 排除测试和 lint 工具，它们不该进生产镜像；
- `--no-cache-dir`（pip）避免把下载缓存留在镜像层里。

如果用 BuildKit，还可以让 pip/uv 的缓存跨构建复用，同时不进镜像：

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
```

### 2. torch 装在哪一层：镜像 8GB 的来源

一个装了 torch 的镜像轻松到 8–12GB。构成大致是：

```text
python:3.11-slim 基础镜像            ~150 MB
torch + nvidia-* CUDA 库            ~5-7 GB      ← 主要来源
其他 Python 依赖                     ~200 MB
业务代码                             ~1 MB
模型权重（如果打进镜像）              ~几 GB       ← 应该避免
```

三条针对性建议：

**其一，用官方 CUDA/PyTorch 基础镜像，别自己 pip 装 torch。**

```dockerfile
# 推荐：torch 和 CUDA 已在基础镜像里，且被多个镜像共享缓存
FROM nvcr.io/nvidia/pytorch:24.10-py3

# 或者用 CUDA 运行时镜像自己装
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

这样做的好处不只是省事：**同一个基础镜像在同一台机器上只存一份**。十个服务都基于 `nvcr.io/nvidia/pytorch:24.10-py3`，那 7GB 的 torch 层只占一份磁盘、只拉一次。而如果每个服务各自 `pip install torch`，即使版本相同，层的哈希也不同，会重复存十份。

这与第四章"把 torch 从项目依赖里摘出去"是同一个决策的两面。

**其二，选对 CUDA 镜像变体。** `nvidia/cuda` 有三个变体：

| 变体 | 内容 | 用途 |
|---|---|---|
| `base` | 最小 CUDA 运行时 | 很少直接用 |
| `runtime` | + cuBLAS、cuDNN 等库 | **生产推理服务用这个** |
| `devel` | + nvcc 编译器、头文件 | 需要现场编译扩展时用 |

`devel` 比 `runtime` 大好几 GB。如果需要编译自定义算子，用多阶段构建：`devel` 阶段编译，`runtime` 阶段只拷产物。

**其三，模型权重不要打进镜像。** 权重应该在启动时从对象存储或挂载卷加载。理由：镜像不可变但模型要频繁换版本；几 GB 的权重让镜像推拉极慢；同一镜像应该能服务不同的模型。

**其四，用 `.dockerignore`。** 否则 `.venv/`（几 GB）、`.git/`、模型文件、`__pycache__` 都会进构建上下文：

```text
.venv/
.git/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.pt
*.safetensors
models/
data/
```

### 3. 为什么容器里仍然建议用虚拟环境

这是个常见疑问：容器本身就是隔离的，为什么还要 venv？

理由是**多阶段构建时便于整体拷贝**。虚拟环境是一个自包含目录，可以从构建阶段整体拷到运行阶段：

```dockerfile
# ---------- 构建阶段 ----------
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---------- 运行阶段 ----------
FROM python:3.11-slim AS runtime
WORKDIR /app

# 只拷虚拟环境和代码，构建工具（uv、编译器、缓存）都留在 builder 里
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

# 不要用 root 跑服务
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

如果装在系统 Python 里，就得挑挑拣拣地拷 `site-packages` 和 `bin/`，容易漏。用 venv 只需一条 `COPY --from`。

注意 `ENV PATH="/app/.venv/bin:$PATH"` ——这就是第二章说的"激活只是改 PATH"，在 Dockerfile 里直接设 `PATH` 比 `source activate` 可靠，因为每条 `RUN` 都是新 shell，`source` 的效果不会保留。

### 4. ASGI 部署：uvicorn 与 worker 模型

Java 的 servlet 容器用线程池处理并发请求。Python 这里的模型不同，而且**直接受 GIL 约束**。

```bash
# 单进程，适合开发和调试
uvicorn inference_service.main:app --host 0.0.0.0 --port 8000

# 多进程
uvicorn inference_service.main:app --workers 4

# 或用 gunicorn 管理 uvicorn worker（生产更常见）
gunicorn inference_service.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000
```

理解这套模型的要点：

**每个 worker 是独立进程，各有自己的 GIL 和事件循环。** 单个 worker 内部靠 asyncio 并发处理大量 I/O 等待中的请求；多个 worker 之间靠多进程绕过 GIL 实现真正的 CPU 并行。

**worker 数怎么定，取决于瓶颈在哪：**

| 场景 | 建议 | 原因 |
|---|---|---|
| 纯 I/O 转发（调用远端模型服务） | worker 数 ≈ 2–4，靠 asyncio 撑并发 | I/O 等待不占 GIL，单 worker 就能扛很高并发 |
| Python 层有 CPU 工作（tokenize、后处理） | worker 数 ≈ CPU 核数 | 需要多进程绕过 GIL |
| **本地 GPU 推理** | **worker 数 = 1**（或按 GPU 数） | 见下 |

**GPU 推理服务的 worker 数是个坑**。每个 worker 是独立进程，会**各自加载一份模型到显存**。4 个 worker × 一个 14GB 的模型 = 56GB 显存，直接 OOM。而且多进程抢同一块 GPU 会导致上下文切换开销和显存碎片。

所以 GPU 服务的常规做法是：**每个 GPU 一个进程**，进程内用单事件循环 + 批处理（continuous batching）来提升吞吐，而不是靠多 worker。vLLM、TGI 都是这个架构。要横向扩容就起多个容器，每个容器绑一块 GPU，前面挂负载均衡。

> 单 worker 内部如何用 asyncio 组织并发、如何避免阻塞事件循环、如何做批处理与背压——这些是[篇三](/python-concurrency-asynchrony-and-task-collaboration.html)的主题。本章只关注进程层面的部署形态。

**健康检查**要区分两种探针：

```python
@app.get("/healthz")          # liveness：进程还活着吗
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")           # readiness：模型加载完了吗，能接流量吗
async def readyz():
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ready"}
```

这个区分对 AI 服务尤其重要——加载几十 GB 权重可能要几分钟，这段时间进程是活的（liveness 通过）但不能服务（readiness 不通过）。如果只有一个探针，编排系统会在模型还没加载完时就把流量打进来，或者误判进程挂了反复重启。

### 5. 为什么 AI-Infra 服务层收敛到 FastAPI + uvicorn

Python Web 生态里 Django、Flask、FastAPI 三者并存，但 AI-Infra 的模型服务层几乎清一色是 FastAPI + uvicorn（vLLM、TGI、Ray Serve、多数自研推理服务都是）。原因是**部署形态和工作负载的匹配度**，不是框架功能强弱：

**其一，模型服务的负载是"少量重请求 + 长时间等待"。** 一个生成请求可能要几秒到几十秒，期间 Python 主要在等 GPU。这正是 asyncio 擅长的场景——单进程就能维持成千上万个等待中的连接。Django/Flask 的传统 WSGI 同步模型下，每个请求占一个线程，扛不住这种并发。

**其二，流式响应是刚需。** token 逐个返回（SSE 或 chunked），需要框架原生支持异步生成器：

```python
from fastapi.responses import StreamingResponse

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    async def token_stream():
        async for token in engine.generate(request.prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(token_stream(), media_type="text/event-stream")
```

**其三，Django 的核心价值用不上。** ORM、模板、admin、用户认证、数据迁移——模型服务通常没有关系数据库，不渲染页面，认证在网关层做。带上整套 Django 只是负担。

**其四，与类型系统的协同。** FastAPI 直接消费 Pydantic 模型做校验和 OpenAPI 生成，这正是[篇二](/python-type-system-and-data-contract-design.html)第三章讲的数据契约在服务边界上的落地。

需要说明的是，这个结论**只针对模型服务层**。如果你要做的是带管理后台、用户体系、复杂数据模型的平台类系统（比如训练任务管理平台），Django 依然是合理选择——它的 admin 和 ORM 能省掉大量工作。选型取决于负载特征，不存在普遍更优的框架。


## 八、串起来：一个可复现的项目骨架

把前面七章的决策合到一起，得到一个可以直接用的骨架。

### 1. 目录结构

```text
inference-service/
├── pyproject.toml              # 元数据、依赖、工具配置（第一章）
├── uv.lock                     # 锁文件，提交进 git（第三章）
├── .python-version             # 解释器版本，uv 会读它（第二章）
├── .pre-commit-config.yaml     # 提交前检查（第五章）
├── .dockerignore               # 排除 .venv、模型、缓存（第七章）
├── Dockerfile                  # 多阶段构建（第七章）
├── Makefile                    # 常用命令入口
├── README.md
├── src/
│   └── inference_service/      # src 布局（见篇一）
│       ├── __init__.py
│       ├── py.typed            # 类型信息随包分发（第六章）
│       ├── config.py           # Pydantic BaseSettings（见篇二）
│       ├── main.py             # FastAPI app
│       ├── cli.py              # [project.scripts] 入口（第一章）
│       ├── engine.py
│       └── backends/
│           ├── __init__.py
│           └── torch_backend.py
├── tests/
│   ├── conftest.py             # 见篇六
│   ├── test_engine.py
│   └── test_api.py
└── scripts/
    └── benchmark.py
```

### 2. pyproject.toml

```toml
[project]
name = "inference-service"
version = "0.1.0"
description = "Model inference service"
readme = "README.md"
requires-python = ">=3.11"

# 抽象依赖：不含 torch，torch 由基础镜像提供（第四章）
dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "uvicorn[standard]>=0.32",
    "httpx>=0.27",
]

[project.optional-dependencies]
# 本地开发时装：uv sync --extra torch-cu121
torch-cu121 = ["torch==2.4.0"]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.8",
    "mypy>=1.13",
    "pre-commit>=4.0",
]

[project.scripts]
inference-server = "inference_service.cli:main"

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
inference_service = ["py.typed"]

# ---- torch 走 PyTorch 官方索引（第四章）----
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }

# ---- 质量工具（第五章）----
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "ASYNC", "RUF", "PGH"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]
"__init__.py" = ["F401"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-q --cov=inference_service --cov-report=term-missing"
```

### 3. Dockerfile

```dockerfile
# syntax=docker/dockerfile:1

# ---------- 构建阶段 ----------
# 基础镜像 tag 必须固定，可复现性由它保证（第四章）
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS builder

WORKDIR /app
RUN pip install --no-cache-dir uv

# 先装依赖：只有依赖变化才重跑这一层（第七章）
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 再拷代码
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- 运行阶段 ----------
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
# GPU 服务单 worker，靠 asyncio + 批处理提升吞吐（第七章）
CMD ["uvicorn", "inference_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

两个环境变量值得说明：`PYTHONUNBUFFERED=1` 让日志立即输出而不是缓冲（容器里日志靠 stdout 采集，缓冲会导致日志延迟甚至丢失）；`PYTHONDONTWRITEBYTECODE=1` 不生成 `.pyc`（镜像里的代码不会变，写 `.pyc` 只是增加层体积）。

### 4. Makefile

把常用命令固化下来，新人不用记一堆参数：

```makefile
.PHONY: install lint format typecheck test check build docker

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/

test:
	uv run pytest

# CI 入口：一条命令跑完所有门禁
check: lint typecheck test
	uv run ruff format --check .

build:
	uv build

docker:
	docker build -t inference-service:$(shell git rev-parse --short HEAD) .
```

### 5. CI

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      # --frozen：锁文件与 pyproject.toml 不一致就失败（第三章）
      - run: uv sync --frozen --no-extra torch-cu121

      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src/
      - run: uv run pytest
```

注意 `--no-extra torch-cu121`：CI 机器没有 GPU，不需要装 torch，省掉几分钟和几 GB 流量。

### 6. 这套骨架回答了什么

回到本文开头列的那几个问题：

| 问题 | 由什么解决 |
|---|---|
| 依赖版本三个月后就变了 | `uv.lock` + `uv sync --frozen` |
| 装到的 torch 与 GPU driver 不兼容 | 固定 tag 的基础镜像 + `[tool.uv.index]` |
| 本地能跑、装到别的机器 `ImportError` | src 布局 + editable 安装（篇一） |
| 改一行代码就重装几 GB 依赖 | Dockerfile 分层 + BuildKit 缓存 |
| 镜像从 2GB 涨到 12GB | 基础镜像共享 torch 层 + `.dockerignore` + 权重外置 |
| 类型错误上线才发现 | `mypy src/` 进 CI 门禁 |
| 代码风格各写各的 | Ruff + pre-commit + `make check` |


## 附：Java 与 Python 工程化工具链对照

| 关注点 | Java | Python | 关键差异 |
|---|---|---|---|
| 项目描述文件 | `pom.xml` / `build.gradle` | `pyproject.toml` | Python 的文件本身不含依赖解析器 |
| 依赖解析与安装 | Maven / Gradle 内建 | pip / uv / Poetry（外部工具） | 需自行选型 |
| 版本锁定 | 依赖树可确定性推导 | `uv.lock` / `poetry.lock` | Python **必须**额外维护锁文件 |
| 包命名空间 | `groupId:artifactId` | **扁平包名** | PyPI 有抢注与混淆风险 |
| 环境隔离 | JVM classpath 天然隔离 | venv / conda（显式创建） | Python 同解释器下一个包只能一个版本 |
| 解释器/JDK 版本管理 | SDKMAN / jenv | uv / pyenv / conda | — |
| 多模块 | parent pom | 无内建机制（uv workspace） | — |
| 环境切换 | Maven profile | 环境变量 / 多份配置 | Python 无内建 profile |
| 代码风格 | Checkstyle | Ruff（`E`/`W`） | — |
| 缺陷模式 | SpotBugs / Error Prone | Ruff（`B`/`SIM`/`ASYNC`） | — |
| 格式化 | spotless / google-java-format | Ruff format / Black | — |
| **类型检查** | **编译器强制** | **mypy / pyright（可选外挂）** | **最根本的差异** |
| 单元测试 | JUnit | pytest | 见篇六 |
| 日志 | SLF4J + Logback | `logging` | 见篇六 |
| 提交前检查 | git hook + spotless | pre-commit | — |
| 构建产物 | jar / war（自包含） | wheel / sdist（**不含解释器**） | Python 产物无法独立运行 |
| 平台相关产物 | 罕见（JNI） | 常见（`cp311-linux_x86_64`） | AI 库几乎都是平台相关的 |
| 制品仓库 | Maven Central（可覆盖 SNAPSHOT） | PyPI（**版本永久不可变**） | PyPI 约束更严 |
| 实际交付单位 | jar 或容器镜像 | **容器镜像** | Python 基本只能靠容器 |
| 应用框架 | Spring（统摄一切） | **无等价物** | 规范需自行建立 |

两条贯穿全表的结论：

1. **Java 的工程化由框架和编译器强制，Python 的工程化靠团队纪律。** Maven 强制你声明依赖，编译器强制你类型正确，Spring 强制你按它的方式组织代码。Python 每一项都是可选的——这带来灵活性，也意味着一个没立规矩的 Python 项目会迅速腐化。

2. **Python 的交付物不自包含。** jar 里有字节码，随便哪台装了 JVM 的机器都能跑。wheel 里没有解释器、没有系统库、可能还绑定了特定平台和 CUDA 版本。所以 Python 服务的交付必然落到容器上。


## 结语

这一篇和前六篇关注的东西不同。前面讲的是**代码怎么写对**——语言机制、类型与契约、并发、元编程、内存、测试；这一篇讲的是**代码怎么交付出去**。

对 Java 开发者，这一篇的核心认知转换是：**Python 把 Java 里由框架和编译器强制的事情，交还给了你。**

- 没有 Maven 内建的依赖解析，所以要自己选工具、自己维护锁文件；
- 没有 classpath 隔离，所以要自己创建虚拟环境；
- 没有编译器把关类型，所以要自己把 mypy 装进 CI；
- 没有 Spring 定义项目结构，所以要自己立目录约定和代码规范；
- 没有自包含的 jar，所以要自己处理镜像分层和平台差异。

这份自由度是 Python 能在 AI 领域胜出的原因之一——它让 Python 可以做薄薄的一层壳，把重活交给 CUDA、C++ 和专用引擎，而不被框架的假设绑住。但同样的自由度也意味着，**一个没有工程纪律的 Python 项目会比同等的 Java 项目腐化得更快**。

如果这一篇只留三条，我会选：

1. **锁文件必须有，且必须进 CI 强制。** `uv sync --frozen` 是可复现的底线。抽象依赖表达兼容范围，锁文件保证这次装的和上次一样，两者不能互相替代。
2. **把 torch 和 CUDA 交给固定 tag 的基础镜像。** 这一条同时解决了依赖体积、构建缓存、跨平台锁文件三个问题。代价是可复现性的责任从锁文件转移到镜像 tag——它没消失，只是换了地方，所以 tag 绝不能用 `latest`。
3. **静态检查是 Python 的编译器替代品，不是可选项。** `ruff check` + `mypy src/` 进 CI 门禁。Java 里编译不过就交付不了，Python 里这道门得你自己装。

### 关于这个系列

七篇写完，回头看是这样一条链：

```text
篇一  核心机制与工程基础       代码是怎么被加载和执行的
篇二  类型系统与数据契约设计    怎么把意图写清楚，并在边界上强制它
篇三  并发、异步与任务协作      怎么组织任务，让资源不闲着也不打架
篇四  反射、元编程与插件化      怎么让系统可扩展而不失控
篇五  内存管理与优化           内存如何分配回收，开销与泄漏如何定位
篇六  单元测试、问题定位与调试   怎么确认它真的对，出问题怎么查
篇七  从依赖管理到生产交付      怎么把它可复现地交付出去
```

前六篇解决"写对"，第七篇解决"交付"。AI-Infra 工程里这两件事的权重是相当的——一个跑得再好但只能在作者机器上复现的服务，工程价值接近于零。

最后提醒一点：**这一篇是全系列最容易过期的。** uv 仍在快速演进，PyTorch 的 CUDA 索引和版本矩阵每个大版本都在变，PEP 735 这类标准也还在落地过程中。文中的版本号和命令请以官方文档为准；但**分层的思路、抽象依赖与锁定依赖的分工、把平台相关的重依赖交给基础镜像**这些判断，应该会比具体工具活得更久。
