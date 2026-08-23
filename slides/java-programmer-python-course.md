---
layout: slides
title: "Java程序员的Python课"
subtitle: "用 Java 思维快速理解 Python 的关键差异"
permalink: /slides/java-programmer-python-course.html
date: 2026-08-23
author: arganzheng
theme: white
transition: slide
---

## 这门课给谁？

- 写 Java 3 年以上，准备转 Python 的工程师
- 想在 AI / 数据 / 自动化场景里更快落地
- 不追求“语法大全”，先抓住思维差异

---

## 先统一心智模型

- Java：强调类型显式、编译期约束、工程边界
- Python：强调表达速度、运行期灵活、少样板代码
- 迁移关键：保留工程习惯，拥抱语言弹性

---

## 变量与类型：动态但不是“无类型”

- Python 是动态类型：变量没有固定类型，值有类型
- Duck Typing：关注“能不能做”而不是“是不是某接口”
- 需要靠测试、类型注解和代码评审守住质量

```python
x = 1
x = "now string"   # 合法，x 绑定了新对象
```

---

## 空值语义：`null` vs `None`

- Java 常见 `NullPointerException`
- Python 用 `None` 表示“无值”，但同样会触发属性访问错误
- 推荐写法：显式判断 `is None` / `is not None`

```python
if user is None:
    return "guest"
```

---

## 集合表达力：推导式很强

- Java 8+ Stream 已经很强，但写法更“管道化”
- Python 列表/字典/集合推导式非常紧凑

```java
List<Integer> evens = nums.stream()
    .filter(n -> n % 2 == 0)
    .collect(Collectors.toList());
```

```python
evens = [n for n in nums if n % 2 == 0]
```

---

## 函数是一等公民

- 函数可以赋值、传参、返回
- 闭包和装饰器在框架里大量出现（Flask/FastAPI）

```python
def timed(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper
```

---

## OOP 不同点：继承不是唯一主角

- Python 同样支持类、继承、多态
- 但更常见“组合 + 函数式工具 + 协议化接口”
- 少量类 + 清晰模块边界，往往比全类层级更 Pythonic

---

## 并发模型：线程不是唯一答案

- Java：线程池 + CompletableFuture 很常用
- Python：I/O 场景优先 `asyncio`，CPU 场景考虑多进程
- 认识 GIL：多线程不等于 CPU 并行

```python
import asyncio

async def fetch_all(urls):
    tasks = [fetch(url) for url in urls]
    return await asyncio.gather(*tasks)
```

---

## 异常处理：更轻但要更自律

- Java 强调受检异常（checked exception）
- Python 全是运行时异常风格，易写但也易漏
- 实践：只捕获你知道能处理的异常类型

---

## 工程化：不是“脚本语言就随便写”

- 包管理：`uv` / `pip` / `poetry`
- 质量工具：`ruff`、`black`、`mypy`、`pytest`
- 建议最小基线：格式化 + lint + 单测 + 类型检查

---

## 数据类：`dataclass` 很像轻量 POJO

```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
```

- 自动生成 `__init__` / `__repr__` / 比较方法
- 明确数据结构，又不写大量模板代码

---

## 迁移策略（实战）

1. 先用 Python 写边缘工具（报表、自动化、ETL）
2. 再写服务侧非核心链路（异步任务、AI 编排）
3. 最后再考虑核心服务迁移与性能基线重建

---

## Java 程序员常见坑

- 把所有逻辑都“类化”，导致 Python 写得很重
- 过度追求一开始就抽象完美
- 忽略虚拟环境、依赖锁定、CI 质量门禁

---

## 一页总结

- 保留 Java 的工程纪律（分层、测试、可观测）
- 学会 Python 的表达方式（简洁、可读、快速迭代）
- 先跑起来，再持续收敛为“可维护系统”

---

## Q&A

谢谢，欢迎讨论你正在迁移的真实业务场景。
