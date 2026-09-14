---
layout: post
series: python-for-ai-infra
title: Python 在 AI-Infra（01）：语言机制与运行时原理
subtitle: Python Language Mechanisms and Runtime Internals
tags: [Python]
catalog: true
updated: 2026-09-10
---

Python 常被认为是一门"简单易学"的语言。但在深度学习框架、推理服务和分布式训练系统里，真正需要掌握的不是语法，而是语法背后的运行时模型。下面这些在 AI-Infra 代码里随处可见的写法，每一行都依赖一个可以被替换、被拦截、被扩展的机制：

```text
import torch                       触发 .so 扩展加载、算子注册、后端初始化
model(x)                           走的是 __call__，中间可能插入 hooks
for batch in loader                迭代协议 + 生成器的暂停与恢复
with torch.inference_mode()        上下文管理协议：进入时改状态、退出时恢复
@register("cuda")                  装饰器在模块导入时执行，注册表能否填上取决于谁导入了它
self.linear = nn.Linear(4, 4)      __setattr__ 拦截赋值，把子模块登记到 _modules
except Exception: log; raise       异常是否重抛，决定 Worker 是否退出
```

这七行分别在第四、六、六 / 八、九、七、五、十章展开，第十二章会逐行给出答案。不理解这些机制，读 PyTorch 或 vLLM 的源码就只能逐行翻译语法；理解之后，才能看出一个框架"为什么这样设计"，也才能解释那些经典故障——"本地能跑、换台机器就 `ImportError`"、"明明写了注册装饰器、运行时却找不到后端"、"生成器持有的文件一直没关"。

本文不试图覆盖 Python 的全部特性，只回答一个问题：

> **一段 AI-Infra 代码从被加载、创建对象、执行任务到释放资源，Python 运行时究竟做了什么？**


## 一、总览

### 1. 一个贯穿全文的例子

为了让后面的机制讨论有一个共同的落点，先给出一个极简的推理组件。它没有任何真实的模型逻辑，但用到了本文要讲的每一种机制：

```python
# runner.py
from contextlib import nullcontext

REGISTRY = {}


def registered(name):                        # 带参数的装饰器：注册表
    def decorator(cls):
        if name in REGISTRY:
            raise ValueError(f"duplicate registration: {name}")
        REGISTRY[name] = cls
        return cls

    return decorator


class InferenceContext:                      # 上下文管理器：进入/退出推理模式
    def __enter__(self):
        print("enter inference mode")
        return self

    def __exit__(self, exc_type, exc, tb):
        print("exit inference mode")
        return False


@registered("runner")
class Runner:
    def __init__(self, model, inference=True):
        self.model = model
        self.inference = inference

    def __call__(self, batch):               # 可调用协议
        context = InferenceContext() if self.inference else nullcontext()
        with context:
            return self.model(batch)

    def stream(self, batch):                 # 生成器：流式输出
        yield from self.model.generate(batch)
```

这段代码从写下到跑完，Python 运行时做了这些事：

1. 有人 `import runner`，导入系统找到文件、编译成字节码、执行模块顶层代码——`REGISTRY` 被创建，`@registered("runner")` 在这一刻把 `Runner` 写进注册表；
2. `Runner(model)` 创建实例，经过 `__new__` 和 `__init__`；
3. `runner(batch)` 触发 `__call__`，其中 `with context:` 进入并退出上下文；
4. `for out in runner.stream(batch)` 创建生成器，逐步暂停与恢复；
5. 任何一步抛出异常，异常沿执行帧向上传播，途经 `__exit__` 和调用方的 `try/except`。

本文的每一章解释其中一个阶段，第十一章再把它们串起来完整追踪一遍。

### 2. 全文的主线：一段代码的生命周期

上面五个步骤给出了本文的组织顺序。这不是 Python 特性的清单，而是一条从"代码文本"到"运行结束"的时间线：

```text
源码 ──编译──► code object ──封装──► 函数/类对象 ──调用──► 执行帧
                                                          │
  ▲ 第二章 执行模型                                        │ 第三章 作用域与闭包：帧里的名称怎么解析
  │                                                       ▼
import 语句 ──finder/loader──► 模块对象 ──执行顶层代码──► 注册表、类定义生效
  ▲ 第四章 导入系统
  │
类语句 ──type──► 类对象 ──__call__──► 实例 ──属性查找/描述符──► 方法、property
  ▲ 第五章 类与对象模型      ▲ 第六章 对象协议：__call__、迭代、__getitem__ ...
  │
装饰器改写调用路径（第七章）→ 生成器暂停帧（第八章）→ 上下文管理器管资源（第九章）→ 异常传播（第十章）
```

各章之间的依赖是单向的：第三章的闭包依赖第二章的函数对象与帧；第五章的方法绑定、`property` 依赖描述符协议，所以描述符放在第五章内部讲，而不是拖到对象协议之后；第七章装饰器同时依赖闭包（第三章）和描述符（第五章），因此排在两者之后。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 执行模型：源码如何变成正在运行的代码 | code object、字节码、函数对象、执行帧、名称绑定 |
| 三 | 作用域与闭包：名称在哪里被解析 | LEGB 的编译期本质、cell、`nonlocal`、延迟绑定 |
| 四 | 模块与导入系统：代码如何被加载 | `import` 的执行过程、`meta_path` / `PathFinder` / loader 三层机制、`sys.path` 与启动方式、项目布局与 editable 安装、`sys.modules` 与注册前提、循环导入、动态导入、导入语法 |
| 五 | 类与对象模型：对象如何被创建和查找 | `type`、属性查找算法、描述符、方法绑定与三种内建描述符、`__getattr__`、`__new__` / `__init__`、MRO 与 `super`、Mixin 与组合 |
| 六 | 对象协议：语法背后的特殊方法 | `__call__`、迭代协议、`__getitem__`、真值、`__eq__` / `__hash__` |
| 七 | 装饰器：用闭包和描述符改写调用路径 | 基本机制、`wraps`、参数化、类装饰器与注册、与描述符的叠放顺序、异步 |
| 八 | 生成器与惰性执行 | 帧的挂起、惰性的成本、`yield from`、关闭与清理 |
| 九 | 上下文管理器 | `with` 的展开、`contextlib`、AI-Infra 中的用法、异步版本 |
| 十 | 异常处理与失败传播 | 层级、错误边界、异常链、记录并重抛 |
| 十一 | 一个推理组件的完整运行时追踪 | 按导入、创建、调用、流式、异常五个阶段追踪 `Runner`，并归纳工程建议 |
| 十二 | 本文小结 | 开头七行代码的答案 |

示例输出基于 CPython 3.12，PyTorch 源码以 2.9.0 为准。


## 二、执行模型：源码如何变成正在运行的代码

一切机制都建立在一个事实上：Python 源码在运行前会被编译成一种中间表示，运行时操作的是这种表示，而不是文本。这一章先把这条链路建立起来，后面讨论导入、闭包、生成器时都会回到它。

### 1. Python 不是"逐行解释"

"解释器读一行、执行一行"是入门时的近似，但它解释不了为什么一个函数里的语法错误会让整个模块都无法导入，也解释不了为什么闭包能记住外层变量。实际过程是：

```text
源代码(.py)  ──parser──►  AST  ──compiler──►  code object（含字节码）  ──解释器──►  在执行帧中运行
```

编译以**整个模块**为单位发生在导入时（或 `python script.py` 启动时），每个函数体、类体、生成器体各自编译成一个嵌套的 code object。用 `dis` 可以看到函数的字节码：

```python
import dis


def add(x, y):
    return x + y


dis.dis(add)
```

CPython 3.12 的输出：

```text
  4           0 RESUME                   0

  5           2 LOAD_FAST                0 (x)
              4 LOAD_FAST                1 (y)
              6 BINARY_OP                0 (+)
             10 RETURN_VALUE
```

四条指令：把两个局部变量压栈、做二元运算、返回。指令集会随 CPython 版本变化（3.11 起大量指令被重写以支持自适应特化），但读法是稳定的：左列是源码行号，中间是字节偏移，右边是指令名和参数。

code object 本身是一个普通对象，函数只是它的一个引用：

```python
c = add.__code__
print(type(c))                                    # <class 'code'>
print(c.co_varnames, c.co_argcount, c.co_consts)  # ('x', 'y') 2 (None,)
```

这里需要分清几个概念，后面每一章都会用到：

| 概念 | 是什么 | 什么时候产生 |
| :--- | :--- | :--- |
| 源代码 | `.py` 文本 | 开发者编写 |
| code object | 编译产物：字节码、常量表、变量名表、行号表 | 编译时，一个作用域一个 |
| 函数对象 | code object + 默认参数 + 全局命名空间 + 闭包 + 注解 | 执行到 `def` 语句时 |
| 执行帧（frame） | 一次调用的运行状态：局部变量、当前指令位置、指向调用者的链 | 每次调用时创建 |
| 解释器 | 执行字节码，维护调用栈、异常状态 | 进程级 |

字节码格式和执行循环属于 CPython 实现细节，不属于语言规范。但 code object、函数对象、帧这三个概念在所有主流实现中都存在，理解它们不会被版本差异推翻。

### 2. 函数对象：code object 的运行时封装

执行到 `def` 语句时，解释器用编译好的 code object 创建一个函数对象，并把名称绑定到它。函数对象携带了运行这段代码所需的全部上下文：

```python
def predict(x, scale=2):
    """Scale the input."""
    return x * scale


print(type(predict))            # <class 'function'>
print(predict.__code__)         # <code object predict at 0x..., file "...", line 1>
print(predict.__defaults__)     # (2,)
print(predict.__doc__)          # Scale the input.
print(predict.__qualname__)     # predict
print(predict.__globals__ is globals())   # True
```

`__globals__` 是函数定义所在模块的命名空间字典——这决定了函数里的全局名称在**定义它的模块**中查找，而不是在调用它的模块中。第三章的作用域规则、第四章"模块的顶层代码执行后名称才存在"，都以这个字段为基础。

`def` 是语句，每执行一次就创建一个新的函数对象。两个函数对象可以共享同一个 code object：

```python
def make():
    def f():
        return 1
    return f


a, b = make(), make()
print(a is b, a.__code__ is b.__code__)   # False True
```

这就是为什么函数在 Python 里是"一等对象"：它可以被赋值、传参、返回、放进容器、动态添加属性——本质上它就是一个带有 `__call__` 的普通对象。第七章的装饰器把这件事用到极致：`@log_call` 只是把名称 `predict` 重新绑定到另一个函数对象。

### 3. 执行帧与调用栈

每次调用函数，解释器创建一个执行帧，包含局部变量、当前执行到的指令位置、指向调用者帧的引用。帧对象可以被观察：

```python
import inspect


def show_frame():
    frame = inspect.currentframe()
    print(frame.f_code.co_name, list(frame.f_locals), frame.f_back.f_code.co_name)


show_frame()    # show_frame ['frame'] <module>
```

traceback 显示的调用链，就是异常发生时沿 `f_back` 串起来的帧：

```python
def inner():
    raise ValueError("boom")


def outer():
    inner()


try:
    outer()
except ValueError as exc:
    tb = exc.__traceback__
    while tb:
        print(tb.tb_frame.f_code.co_name, tb.tb_lineno)
        tb = tb.tb_next
```

```text
<module> 9
outer 6
inner 2
```

帧这个概念解释了后面几件事：

- 第三章：局部变量、闭包变量、全局变量的查找方式不同，因为它们存放在帧的不同位置；
- 第七章：装饰器会在调用栈里多插入一层 `wrapper` 帧，所以 traceback 里会出现它；
- 第八章：生成器的本质是"一个可以被挂起、之后再恢复的帧"；
- 第十章：异常传播就是沿帧链逐层向上寻找 `except`。

生产代码不应依赖 `inspect.currentframe()` 或修改帧对象，但调试器、profiler、`logging` 的调用位置记录都建立在帧之上。

### 4. 名称绑定：名称不是变量盒子

Python 的赋值不是"把值放进一个盒子"，而是"让一个名称指向一个对象"。

```python
x = []
y = x
y.append(1)
print(x, y, x is y)     # [1] [1] True
```

这里只有一个列表对象，`x` 和 `y` 是它的两个名称。因此要分开四件事：**名称**、**对象**、**对象身份**（`id()`、`is`）、**对象可变性**。

`==` 比较值，`is` 比较身份。对值做 `is` 判断结果不可预期：

```python
a = 1000
b = 1000
print(a == b, a is b)                         # True True   —— 同一编译单元里的常量被合并
p = int("1000")
q = int("1000")
print(p == q, p is q)                         # True False  —— 运行时各自创建
```

第一组为 `True` 只是因为编译器把同一 code object 里相同的常量合并进了常量表；第二组是运行时分别创建的两个对象。所以工程代码中 `is` 只用于单例：`is None`、`is True`、哨兵对象。

在 AI-Infra 中，名称绑定和别名会直接影响：

- 配置对象被传入多个组件后，某个组件的原地修改会被其他组件看到；
- 一个 Batch 被 DataLoader、预处理器和模型共享时，谁做了原地操作；
- 缓存里放的是可变对象时，缓存内容会随外部修改而"漂移"；
- 多进程 Worker 各自持有的是**副本**（`fork` 后各自的地址空间），进程内的全局注册表并不共享——这一点在第四章 §6 会再遇到。

### 5. 与 Java 的对照

Java 程序员对上面大部分内容并不陌生：Java 的对象变量同样是引用，`==` 比较引用、`equals()` 比较值，`.class` 文件里的字节码也需要 JVM 解释或 JIT 执行。真正的差异在两处：

| | Python | Java |
| :--- | :--- | :--- |
| 名称携带类型吗 | 不。名称只是绑定，任何对象都能绑上去 | 变量有静态类型，编译期检查 |
| 编译发生在什么时候 | 导入/启动时按模块编译，编译产物（`.pyc`）只是缓存 | 构建期生成 `.class`，运行期由类加载器加载 |
| 函数是对象吗 | 是，`def` 是运行时语句 | 方法不是对象；lambda 是函数式接口的实例 |
| 帧能被程序访问吗 | 能，`inspect.currentframe()`、traceback 对象 | 只能通过 `StackTraceElement` 等有限视图 |

最重要的一条是第一行：Python 名称没有类型，类型信息全在对象上。这也是为什么本系列第二篇要单独讨论类型系统——Python 把"提供类型信息"和"消费类型信息"拆成了两层。


## 三、作用域与闭包：名称在哪里被解析

第二章说函数对象携带 `__globals__`，帧里存放局部变量。那么函数体里写下一个名称时，Python 在哪里找它？答案是**编译期就决定了**——这是理解闭包、`nonlocal` 和"循环里的 lambda 全都返回同一个值"的钥匙。

### 1. LEGB 是编译期规则

名称查找的四层通常叫 LEGB：Local（当前函数）→ Enclosing（外层函数）→ Global（模块）→ Builtins。

```python
value = "global"


def outer():
    value = "enclosing"

    def inner():
        return value

    return inner


f = outer()
print(f())      # enclosing
```

关键在于：编译器在编译 `inner` 时，扫描整个函数体，把每个名称分类——有赋值的是局部变量，没有赋值但在外层函数中有定义的是**自由变量**（free variable），其余都当作全局/内置。分类结果直接体现在字节码里：

```python
import dis
dis.dis(f)
```

```text
              0 COPY_FREE_VARS           1

  7           2 RESUME                   0

  8           4 LOAD_DEREF               0 (value)
              6 RETURN_VALUE
```

`LOAD_DEREF` 表示"从 cell 中取值"，而不是 `LOAD_FAST`（局部）或 `LOAD_GLOBAL`（全局）。查找**位置**是编译期确定的，运行时只是按位置取值。这也是为什么在函数里对一个名称赋值，会让**整个函数**中该名称都变成局部变量，哪怕赋值语句在后面——下面 §3 的 `UnboundLocalError` 就是这个规则的直接后果。

### 2. 闭包就是"函数对象 + cell"

`outer()` 返回之后，它的帧已经销毁，`inner` 为什么还能读到 `value`？因为编译器发现 `value` 被内层函数引用，就把它存放在一个 **cell** 对象里而不是普通局部槽位；`inner` 的函数对象通过 `__closure__` 持有这些 cell：

```python
print(outer.__code__.co_cellvars)        # ('value',)   outer 中被内层引用的变量
print(f.__code__.co_freevars)            # ('value',)   inner 中来自外层的变量
print(f.__closure__)                     # (<cell at 0x...: str object at 0x...>,)
print(f.__closure__[0].cell_contents)    # enclosing
```

所以闭包不是什么魔法："外层帧销毁了，但 cell 对象被内层函数对象引用着，所以还活着。" 这就是**词法作用域**：名称的解析依据函数**定义时**的代码结构，与调用位置无关。

在 AI-Infra 中闭包常用来携带少量配置进入回调：后端名、重试次数、指标标签、模型引用。但它也是隐式依赖——读者无法从函数签名看出它依赖了什么。

### 3. `nonlocal` 与 `UnboundLocalError`

读外层变量不需要声明，**重新绑定**外层变量需要 `nonlocal`：

```python
def make_counter(start=0):
    count = start

    def increment():
        nonlocal count
        count += 1
        return count

    return increment


counter = make_counter()
print(counter(), counter())                 # 1 2
print(counter.__closure__[0].cell_contents)  # 2   —— 修改的是同一个 cell
```

去掉 `nonlocal` 会得到：

```text
UnboundLocalError: cannot access local variable 'count' where it is not associated with a value
```

按 §1 的规则，`count += 1` 是赋值，编译器因此把 `count` 判定为 `increment` 的**局部变量**；执行时先读局部槽位，而它还没有值。这个错误经常出现在把一个模块级计数器、开关或缓存直接在函数里 `+=` 的时候——那时需要的是 `global`。

### 4. 延迟绑定：cell 是共享的

闭包最常见的陷阱：

```python
def make_fns():
    fns = []
    for i in range(3):
        fns.append(lambda: i)
    return fns


print([fn() for fn in make_fns()])          # [2, 2, 2]
```

先看清 `lambda: i` 这个写法本身。`lambda 参数列表: 表达式` 定义一个匿名函数，冒号前是参数、冒号后是返回值；`lambda: i` 的参数列表是**空的**，所以 `i` 不是参数，而是一个要到外层去找的自由变量——每次调用它，都去 `make_fns` 的作用域里取 `i` **当前**的值。

用 §2 的知识解释结果：`i` 是 `make_fns` 的 cell 变量，三个 lambda 的 `__closure__` 指向**同一个** cell（可以打印 `id(fn.__closure__[0])` 验证，三者相同）。lambda 体里的 `i` 在**调用时**才 `LOAD_DEREF`，此时循环早已结束，cell 里是最后一个值。

修法是在创建时就把值固定下来——用默认参数（默认值在 `def`/`lambda` 求值时计算一次）：

```python
fns.append(lambda i=i: i)                    # [0, 1, 2]
```

`i=i` 里等号左边是 lambda 自己的参数 `i`，右边是循环变量 `i` 此刻的值，被当作默认值存进函数对象（`fn.__defaults__`）；调用 `fn()` 时不传参，就用这个默认值，于是 lambda 体里的 `i` 变成了局部变量，不再去外层找。注意不能写成 `lambda i: i`——那是一个**必须**接收一个参数的恒等函数，`fn()` 会直接报 `TypeError: <lambda>() missing 1 required positional argument: 'i'`。

或者用 `functools.partial(handler, i)`，语义更明确。构造一批 Worker 回调、为每个 GPU 生成一个任务函数、在循环里注册 hook，都会遇到这个问题。

### 5. 闭包还是类

闭包和类都能"把状态和行为绑在一起"。取舍标准是状态的数量和生命周期：

- 一两个只读配置、一个简单计数：闭包即可，代码短；
- 状态多于两三个、需要被外部检查或重置、有清理逻辑：用类。类的状态在 `__dict__` 里可见可调试，闭包的状态藏在 `cell_contents` 里。

第七章会看到，装饰器既可以用闭包写，也可以用类写，取舍标准相同。

### 6. 与 Java 的对照

Java 的 lambda 和匿名内部类也能捕获外层变量，但捕获的变量必须是 **effectively final**——编译器直接禁止 §4 那种"捕获一个会变的循环变量"的写法，也就不存在延迟绑定问题。注意 final 约束的是**引用**不能重新赋值，引用指向的对象本身照样可以修改：lambda 里不能写 `count++`，但可以 `counter.incrementAndGet()` 或 `holder[0]++`。所以 Java 无法像 §3 那样用 `nonlocal` 直接重绑一个外层局部变量，而是要把可变状态装进一个对象——数组、`AtomicInteger` 或对象字段——再捕获这个对象的引用。

两种设计的根本差异：Java 捕获的是**值**（拷贝进 lambda 对象），Python 捕获的是**变量**（共享 cell）。读 Python 代码时要时刻记得这点：闭包看到的是变量的当前值，不是创建时的快照。


## 四、模块与导入系统：代码如何被加载

前两章讨论的是一个函数内部发生的事。但函数要先存在才能被调用，而让它存在的动作是 `import`：找到文件、编译、执行顶层代码、把结果放进一个模块对象。第一章的 `Runner` 之所以能出现在 `REGISTRY` 里，唯一的原因是 `runner.py` 被导入并执行过。这一章把导入系统拆开：它由哪几层组成、`sys.path` 从哪里来、为什么同一份代码换一种启动方式就 `ImportError`。

### 1. 模块是对象，包是带 `__path__` 的模块

一个 `.py` 文件导入后就是一个模块对象，它的属性字典就是那份文件的全局命名空间：

```python
import json, types

print(type(json), isinstance(json, types.ModuleType))   # <class 'module'> True
print(list(json.__dict__)[:6])
# ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__path__']
```

前几个名字是导入系统写入的元数据。`json.dumps` 这样的属性访问，就是在这个字典里查键。第二章说函数对象的 `__globals__` 指向定义它的模块命名空间——指的正是这个 `__dict__`。

包是**多了一个 `__path__` 属性**的模块。`__path__` 是一个目录列表，告诉导入系统"这个包的子模块去哪里找"：

```python
print(json.__path__)      # ['.../lib/python3.12/json']
print(json.__package__)   # json
import json.decoder
print(json.decoder.__package__)   # json
```

`mypackage/__init__.py` 是包被导入时执行的模块体。它有两个合理用途：把内部模块的公共对象提升到包级（`from .model import Model`），以及用 `__all__` 声明 `from mypackage import *` 的导出集合。`__all__` 只影响 `import *`、文档和静态分析工具，不是访问控制。

`__init__.py` 里不应放重逻辑。导入包就会执行它，一个 `import mypackage.utils` 会先跑完 `mypackage/__init__.py`——如果那里面 `import torch`、检查 CUDA、建立连接，所有导入方都要付这个代价，并且更容易形成循环导入（§7）。没有 `__init__.py` 的目录是**命名空间包**（PEP 420），能被导入，但缺少一个明确的初始化入口；后面 §4 会看到 PyTorch 自己的一个报错正是命名空间包造成的。

### 2. `import` 语句的执行过程

`import mypackage.model` 在解释器内部大致经历以下步骤（对应 CPython 标准库 `importlib/_bootstrap.py` 中的 `_find_and_load` → `_find_spec` → `_load_unlocked`）：

```text
1. 查 sys.modules["mypackage.model"]，命中则直接返回                   —— 缓存
2. 未命中：先确保父包 "mypackage" 已导入（递归走同一流程）
3. 依次询问 sys.meta_path 里的每个 finder：find_spec(name, path)
   第一个返回非 None 的 ModuleSpec 胜出；全部 None → ModuleNotFoundError    —— 查找
4. spec.loader.create_module(spec) 创建空模块对象，写入 __name__/__spec__/__file__ 等元数据
5. sys.modules[name] = module                                             —— 注意：在执行之前
6. spec.loader.exec_module(module)：编译（或读 .pyc）并在 module.__dict__ 中执行顶层代码   —— 执行
7. 把结果绑定到当前命名空间：import a.b 绑定 a；from a.b import c 绑定 c
```

有三点决定了后面几节的全部内容：

- **第 1 步**说明同一个模块名在一个进程里只执行一次。注册表、单例、模块级缓存都依赖这一点（§6）。
- **第 5 步在第 6 步之前**。模块在顶层代码执行完之前就已经在 `sys.modules` 里了，此时它是一个"部分初始化"的模块。循环导入之所以表现为 `AttributeError`/`ImportError: cannot import name`，就是有人在这个窗口期拿到了它（§7）。
- **第 3 步是可插拔的**。`sys.meta_path` 是一个普通列表，谁都可以往里插 finder。下一节展开。

### 3. 三层查找机制：`sys.meta_path`、`PathFinder` 与 loader

Python 的导入系统不是"在目录里找 `.py` 文件"，而是一个三层协议。先看一个模块导入之后留下的痕迹：

```python
import json
print(json.__spec__)
```

```text
ModuleSpec(name='json',
           loader=<_frozen_importlib_external.SourceFileLoader object at 0x...>,
           origin='/.../lib/python3.12/json/__init__.py',
           submodule_search_locations=['/.../lib/python3.12/json'])
```

`ModuleSpec` 是 finder 交给 loader 的"工单"：模块叫什么、从哪来（`origin`）、由谁加载（`loader`）、如果是包则子模块去哪找（`submodule_search_locations`，也就是 `__path__` 的来源）。

**第一层：`sys.meta_path` 里的 finder**

```python
import sys
for finder in sys.meta_path:
    print(finder)
```

在一个装了 setuptools 的 venv 里：

```text
<_distutils_hack.DistutilsMetaFinder object at 0x...>
<class '_frozen_importlib.BuiltinImporter'>
<class '_frozen_importlib.FrozenImporter'>
<class '_frozen_importlib_external.PathFinder'>
```

三个内建 finder 各管一类来源：`BuiltinImporter` 负责编译进解释器的 C 模块（`math.__spec__` 的 `origin` 是 `'built-in'`，没有 `__file__`），`FrozenImporter` 负责冻结进解释器的启动模块，`PathFinder` 负责文件系统。第一个不是内建的——它是 setuptools 通过 `site-packages/distutils-precedence.pth` 装进来的，用来把 `import distutils` 重定向到 setuptools 自带的副本。这是一个真实生产环境中"第三方代码接管导入过程"的例子，后面 §5 的 editable 安装也用同样的手法。

**第二层：`PathFinder` 沿 `sys.path` 找目录 finder**

`PathFinder.find_spec` 遍历 `sys.path`（子模块则遍历父包的 `__path__`），对每个目录调用 `sys.path_hooks` 中的钩子，得到一个负责该目录的 finder，并缓存在 `sys.path_importer_cache`：

```python
for hook in sys.path_hooks:
    print(hook)
```

```text
<class 'zipimport.zipimporter'>
<function FileFinder.path_hook.<locals>.path_hook_for_FileFinder at 0x...>
```

所以 zip 文件和普通目录都能出现在 `sys.path` 上：前者由 `zipimporter` 接管，后者由 `FileFinder` 接管。

**第三层：`FileFinder` 按后缀选 loader**

`FileFinder` 在自己负责的目录里查找 `name/__init__.py`、`name.<后缀>`，后缀与 loader 的对应表来自 `importlib.machinery`：

```python
import importlib.machinery as m
print(m.SOURCE_SUFFIXES)       # ['.py']                                     → SourceFileLoader
print(m.BYTECODE_SUFFIXES)     # ['.pyc']                                    → SourcelessFileLoader
print(m.EXTENSION_SUFFIXES)    # ['.cpython-312-darwin.so', '.abi3.so', '.so'] → ExtensionFileLoader
```

`ExtensionFileLoader` 就是 C 扩展进入 Python 的入口：它 `dlopen` 这个 `.so`，调用其中的 `PyInit_<name>` 函数拿到模块对象。看一个真实的扩展模块：

```python
import numpy, sys
m = sys.modules["numpy._core._multiarray_umath"]
print(m.__spec__.loader)   # <_frozen_importlib_external.ExtensionFileLoader object at 0x...>
print(m.__spec__.origin)   # .../site-packages/numpy/_core/_multiarray_umath.cpython-312-darwin.so
```

`import torch` 时加载的 `torch._C` 走的是完全相同的路径：`torch/__init__.py` 里的 `from torch._C import *` 让 `PathFinder` 在 `torch/` 目录下找到 `_C.cpython-312-x86_64-linux-gnu.so`，由 `ExtensionFileLoader` 载入——数百 MB 的 libtorch 就这样进入进程。

**为什么这套机制是插件系统的基础**

上面三层的每一层都是可编程的，这直接支撑了三类 AI-Infra 中的常见做法：

① **不导入就能探测**。`importlib.util.find_spec` 只走查找阶段，不执行模块：

```python
import importlib.util
print(importlib.util.find_spec("triton"))    # None —— 没装，且没有触发任何导入副作用
```

"如果装了 Triton/flash-attn 就用它"这类可选依赖判断，用 `find_spec` 比 `try: import` 更轻，因为后者会真的加载扩展库。

② **往 `sys.meta_path` 插 finder 就能拦截导入**。下面这个 finder 让测试代码可以模拟"某个依赖没安装"：

```python
import importlib.abc, sys


class Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, names):
        self.names = names

    def find_spec(self, fullname, path, target=None):
        if fullname in self.names:
            raise ModuleNotFoundError(f"{fullname} blocked for testing", name=fullname)
        return None


sys.meta_path.insert(0, Blocker({"numpy"}))
import numpy    # ModuleNotFoundError: numpy blocked for testing
```

`find_spec` 返回 `None` 表示"不归我管，问下一个"；返回 spec 或抛异常则终结查找。把 `Blocker` 换成一个记录 `fullname` 的 finder，就能看到 `import xml.dom.minidom` 实际请求了 `xml`、`xml.dom`、`xml.dom.domreg`、`xml.dom.minidom`、`copy`……——这是排查"启动时到底导入了什么"的最直接手段。

③ **手工构造 spec 就能加载任意位置的文件**。PyTorch 的 JIT 扩展编译（`torch.utils.cpp_extension.load`）编译出 `.so` 之后，通过 `_import_module_from_library` 把它变成模块（`torch/utils/cpp_extension.py`）：

```python
spec = importlib.util.spec_from_file_location(module_name, filepath)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

`spec_from_file_location` 根据文件后缀自动选出 `ExtensionFileLoader`，剩下的两步就是 §2 中的第 4 和第 6 步。这里没有任何 `sys.path` 参与——所以刚编译出来的扩展可以放在临时目录里。

至此可以回答本节开头的问题：finder 决定"从哪找"、loader 决定"怎么变成模块"、`ModuleSpec` 是二者之间的契约。三层都可替换，所以 zip 包、命名空间包、editable 安装、被 mock 掉的依赖、JIT 编译出来的 `.so`，都能被同一条 `import` 语句处理。

### 4. `sys.path` 的构成，以及 `python script.py` 与 `python -m` 的差别

`PathFinder` 沿 `sys.path` 找，那么 `sys.path` 里有什么？启动时它按顺序由这几部分拼成：

```text
sys.path[0]         取决于启动方式（见下）
PYTHONPATH          环境变量中的目录
标准库               .../lib/python3.12、python312.zip、lib-dynload
site-packages       当前解释器或虚拟环境的第三方包目录
.pth 文件里的行      site 模块处理 site-packages 下的 *.pth：每行一个目录追加到 sys.path，
                    以 import 开头的行会被直接执行（§3 的 DistutilsMetaFinder 就是这么装进来的）
```

真正制造差异的是 `sys.path[0]`。规则是：

| 启动方式 | `sys.path[0]` | `__name__` | `__package__` |
| :--- | :--- | :--- | :--- |
| `python path/to/script.py` | **脚本所在目录** | `__main__` | `None` |
| `python -m pkg.mod` | **当前工作目录** | `__main__` | `pkg` |
| `python -c` / 交互式 | `''`（当前目录） | `__main__` | `None` |

用一个 src 布局的项目验证。`src/myproject/cli.py` 打印自己的身份，然后做一次相对导入：

```python
# src/myproject/cli.py
import sys
print("__name__    =", __name__)
print("__package__ =", __package__)
print("sys.path[0] =", repr(sys.path[0]))
from .core import register
print("relative import ok")
```

第一种启动方式：

```bash
$ python src/myproject/cli.py
__name__    = __main__
__package__ = None
sys.path[0] = '/.../proj/src/myproject'
ImportError: attempted relative import with no known parent package
```

第二种启动方式（在仓库根目录，包已 editable 安装，安装方式见 §5）：

```bash
$ python -m myproject.cli
__name__    = __main__
__package__ = myproject
sys.path[0] = '/.../proj'
relative import ok
```

差别的机制：

- **相对导入依赖 `__package__`**。`from .core import x` 被解析成 `__package__ + ".core"`。作为脚本运行时 `__package__` 是 `None`，解释器不知道"当前包"是谁，于是报错。`python -m` 则是先把 `myproject.cli` 当作一个**模块**找到（走 §3 的全部流程，因此需要 `myproject` 可导入），再把它当作 `__main__` 执行，`__package__` 由此得到正确的值。
- **`sys.path[0]` 是脚本目录，会制造"意外可导入"**。在 `cli.py` 里写 `import core`（绝对导入一个叫 `core` 的顶层模块）用第一种方式能成功——因为 `src/myproject/` 恰好在 `sys.path[0]`。但这个模块的名字是 `core` 而不是 `myproject.core`；一旦别处以 `myproject.core` 导入同一个文件，它会被**第二次执行**，得到两个互不相认的模块。

第二点值得单独演示，因为它是"注册表里出现重复项"和"`isinstance` 莫名其妙为 False"的常见根源。一个脚本 `app.py` 通过装饰器注册 `Runner`，同时它又被别的模块按模块名导入：

```python
# app.py
from registry import register

@register
class Runner: ...

if __name__ == "__main__":
    import app                    # 模拟别的模块（如 worker）按模块名导入 app
    print(Runner is app.Runner, isinstance(app.Runner(), Runner))
```

```bash
$ python app.py
duplicate registration: Runner False
False False
```

同一个文件以 `__main__` 和 `app` 两个名字各执行了一遍，`sys.modules` 里有两个条目，两个 `Runner` 类对象，注册表里第二个覆盖了第一个。规避方法是让入口文件尽量薄：`__main__` 里只 `from myproject.cli import main; main()`，业务代码不放在会被当作脚本运行的文件里。

PyTorch 源码里有一段专门针对这个陷阱的报错文案。`torch/__init__.py` 在 `from torch._C import _initExtension` 失败后检查 `torch._C.__file__ is None`，如果成立就提示：

> It appears that PyTorch has loaded the `torch/_C` folder of the PyTorch repository rather than the C extensions ... This error can generally be solved ... by running Python from a different directory.

原因正是上面的规则：在 PyTorch 仓库根目录执行 `python -c "import torch"`，`sys.path[0]` 是 `''`（当前目录），`PathFinder` 先在这里找到源码树中的 `torch/` 目录，而不是 `site-packages` 里安装好的那个；源码树里 `torch/_C/` 只是一个放 `.pyi` 存根的目录、没有 `__init__.py`，于是被当作命名空间包导入，`__file__` 为 `None`，所有 C 函数都不存在。

### 5. 项目布局、测试与 editable 安装

§4 的规则解释了 flat 布局与 src 布局之争的全部内容。两种布局：

```text
flat 布局                        src 布局
myproject/                       myproject/
├── pyproject.toml               ├── pyproject.toml
├── myproject/                   ├── src/
│   ├── __init__.py              │   └── myproject/
│   └── core.py                  │       ├── __init__.py
└── tests/                       │       └── core.py
    └── test_core.py             └── tests/
                                     └── test_core.py
```

用一个只做 `import myproject.core` 的测试，在**没有安装**包的情况下分别运行：

| 布局 | 命令 | 结果 | 原因 |
| :--- | :--- | :--- | :--- |
| flat | `python -m pytest tests/` | 通过 | `-m` 把当前目录（仓库根）放进 `sys.path`，源码树直接可导入 |
| flat | `pytest tests/` | `ModuleNotFoundError` | `pytest` 可执行文件不加当前目录；pytest 默认的 prepend 模式只把 `tests/` 插入 `sys.path` |
| flat | `pytest tests/`，且 `tests/__init__.py` 存在 | 通过 | prepend 模式向上找到第一个不含 `__init__.py` 的目录（仓库根）插入 `sys.path` |
| src | 任何方式 | `ModuleNotFoundError` | `src/` 不在任何路径上 |

flat 布局的三行结果说明了什么叫"测试环境可以导入、安装后不能"：测试通过与否取决于用 `pytest` 还是 `python -m pytest`、`tests/` 下有没有 `__init__.py`，而这些都与包装得对不对无关。`pyproject.toml` 里漏掉一个子包、忘了带上数据文件，flat 布局的测试照样绿——因为测的根本不是安装产物。

src 布局把这条路堵死：源码树不在任何默认路径上，测试**只能**导入安装后的包。开发时用 editable 安装让"安装后的包"指回源码：

```bash
$ pip install -e .
$ ls site-packages | grep myproject
__editable__.myproject-0.1.0.pth
myproject-0.1.0.dist-info
$ cat site-packages/__editable__.myproject-0.1.0.pth
/.../proj/src
```

这一行 `.pth` 就是 §4 开头那份 `sys.path` 构成清单里的最后一项——`site` 模块启动时把 `src/` 追加到 `sys.path`，`PathFinder` 由此能找到 `src/myproject/`。这是 setuptools 对 src 布局这类"简单布局"的做法；当包结构无法用一条路径表达时（多个源码根、包名与目录名不一致、`--config-settings editable_mode=strict`），setuptools 64 及以后（PEP 660）会改为生成一个 `__editable___<name>_finder.py`，并通过 `.pth` 把其中的 `MetaPathFinder` 插进 `sys.meta_path`——也就是 §3 第一层的自定义 finder。两种方式殊途同归：**editable 安装不是复制文件，而是修改导入系统的查找路径**。

由此可以推出它的边界：修改 `.py` 文件立即生效（每次导入都重新走查找与编译）；但新增顶层包、修改 `[project.scripts]` 入口、变更依赖，都需要重新 `pip install -e .`，因为这些信息记录在 `dist-info` 和 `.pth` 里，不会自动更新。

`pyproject.toml` 的完整写法、依赖锁定与打包分发，见本系列第七篇[《工程化与生产交付》](/python-engineering-and-production-delivery.html)。

### 6. `sys.modules` 缓存、导入副作用与注册机制

回到 §2 的第 1 步：同一个模块名在同一进程里只执行一次。

```python
import sys, json
first = sys.modules["json"]
import json as j2
print(j2 is json, sys.modules["json"] is first)     # True True
```

这条规则是**所有基于导入的注册机制的前提**。第一章的 `@registered("runner")` 在模块顶层执行，意味着：

- 模块被导入 → 注册发生一次；
- 模块没被任何人导入 → 注册表是空的。"写了装饰器却找不到后端"，原因永远是这一条；
- 模块被以两个名字导入（§4 的 `__main__`/`app`）→ 注册两次，后者覆盖前者；
- `importlib.reload(module)` → 顶层代码重新执行，注册表被再写一遍；
- 多进程 → 每个进程有自己的 `sys.modules`，注册表在 `fork` 之后是各自的副本，主进程后来的注册子进程看不到。

因此导入不是"声明依赖"，而是"执行代码"。AI-Infra 中导入触发的典型副作用有：算子/后端注册、插件发现、CUDA 扩展加载、环境变量读取、设备初始化、日志配置、全局缓存创建。这些副作用本身不是坏事——PyTorch 的算子注册就依赖它——但它们的**时机**由导入顺序决定，而导入顺序通常不是显式控制的。工程上的两条原则：

- 顶层只做"声明式"的事（定义类、注册名字），把创建连接、加载模型、初始化 CUDA 放进显式调用的函数；
- 需要保证注册发生时，显式导入插件模块，不要依赖"总会有人导入它"。

### 7. 循环导入与依赖方向

```python
# a.py
from b import B          # ← 执行到这里时，a 已在 sys.modules，但 A 还没定义
class A: ...

# b.py
from a import A          # ← 拿到的是部分初始化的 a，A 不存在 → ImportError: cannot import name 'A'
class B: ...
```

按 §2 的步骤：导入 `a` → `a` 进入 `sys.modules` → 执行 `a` 顶层 → 遇到 `from b import B` → 导入 `b` → 执行 `b` 顶层 → 遇到 `from a import A` → `a` 已在 `sys.modules`，直接取 → 但 `a` 的顶层代码还停在第一行，`A` 尚未定义 → 失败。

`import a` 而不是 `from a import A` 有时能"绕过去"，因为它只绑定模块对象，属性访问推迟到运行时；`TYPE_CHECKING` 块和函数内的局部导入同理。但这些只是改变了**导入时机**，没有消除模块之间的环。首选解法是调整依赖方向：

```text
原结构：   runner ──► backend ──► runner
调整后：   runner ──► protocols ◄── backend        protocols 只放接口与数据结构，不依赖实现
```

如果一个项目需要靠局部导入来"修"循环导入，通常说明模块边界需要重新划分。

### 8. 动态导入：`importlib.import_module`

模块名在运行时才确定时用 `importlib.import_module`：

```python
import importlib


def load_backend(name):
    module = importlib.import_module(f"mypackage.backends.{name}")
    return module.create_backend()
```

它走的是 §2 的同一条流程，只是名字来自字符串。代价是依赖关系对静态工具不可见、拼写错误延迟到运行时、字符串来自不可信输入时会变成任意代码执行。按配置加载后端、`pkgutil.iter_modules` 扫描插件包、`importlib.metadata.entry_points` 发现第三方插件——这三种插件发现方式的对比与选型，在本系列第四篇[《反射、元编程与插件架构》](/python-reflection-metaprogramming-and-plugin-architecture.html)中展开。

### 9. 导入语法，以及与 Java `import` 的对照

`import math`、`import numpy as np`、`from math import sqrt` 三种写法的区别只有一点：**§2 第 7 步绑定什么名称**。`import` 绑定模块名；`as` 只是在当前命名空间起一个别名，不会创建第二个模块对象（`import numpy as np; import numpy; np is numpy` 为 `True`）；`from ... import` 直接绑定成员，简洁但来源不明显、容易被同名覆盖。

工程惯例：模块级依赖用 `import package.submodule`，生态约定的缩写用 `as`（`np`、`pd`），少量稳定的公共对象用 `from ... import`（`Path`、`contextmanager`），不用 `from module import *`。

Java 程序员最需要注意的差异不在语法，而在语义：

| | Python `import` | Java `import` |
| :--- | :--- | :--- |
| 本质 | **运行时语句**：查找、加载、执行模块顶层代码、绑定名称 | **编译期声明**：简化类名书写，不产生任何运行时动作 |
| 执行代码吗 | 会。模块顶层代码在首次导入时执行 | 不会。类初始化推迟到首次主动使用（`new`、静态成员访问） |
| 缓存 | `sys.modules`，按模块名 | 类加载器，按（加载器, 全限定名） |
| 别名 | `as` | 无 |
| 导入成员 | `from m import f` | `import static C.f` |
| 可以放在函数里吗 | 可以，作为延迟导入 | 不可以 |

Java 中"加载类"这件事由类加载器在**使用时**惰性完成，代码不会因为写了一行 `import` 就去执行什么；Python 的 `import` 则是一个会产生副作用的动作，且顺序由代码书写顺序决定。§6 的注册机制、§7 的循环导入，在 Java 里几乎没有对应的问题——反过来，Java 的类加载器隔离、`ClassNotFoundException` vs `NoClassDefFoundError` 那些问题，在 Python 里对应的是 `sys.path`、`sys.modules` 和 §4 的启动方式差异。


## 五、类与对象模型：对象如何被创建和查找

模块被导入、顶层代码执行时，`class Runner:` 语句创建了一个类对象。接下来 `Runner(model)` 创建实例，`runner.model`、`runner.stream` 读取属性。这一章回答的核心问题只有一个：**`obj.attr` 到底做了什么**。方法绑定、`property`、`classmethod`、`__getattr__`、`nn.Module` 把子模块藏在 `_modules` 里却能用 `self.linear` 访问——全部是这一个算法的不同分支。

### 1. 类也是对象：`type` 创建类

`class` 是运行时语句。执行它时，解释器：

1. 新建一个命名空间字典，在其中执行类体（所以类体里的 `def` 创建的是普通函数对象，并被放进这个字典）；
2. 调用元类（默认是 `type`）：`type(name, bases, namespace)`，得到类对象；
3. 把名称绑定到类对象。

```python
class Runner:
    def run(self, batch):
        return batch


print(type(Runner))                       # <class 'type'>
print(Runner.__mro__)                     # (<class 'Runner'>, <class 'object'>)
print(type(Runner.__dict__))              # <class 'mappingproxy'>   类属性字典的只读视图
print(type(Runner.__dict__["run"]))       # <class 'function'>       类里存的是普通函数
```

第 2 步可以手工做，效果完全相同：

```python
Runner = type("Runner", (object,), {"run": lambda self, batch: batch})
```

两个事实值得记住：**类的 `__dict__` 里存的方法就是普通函数对象**（后面 §5 解释它如何变成"方法"）；**类由 `type` 创建**，所以替换 `type` 为自定义元类就能干预类的创建——`nn.Module` 没有用元类，但许多 ORM、配置框架和 `abc.ABC` 用了。元类的用法在本系列第四篇展开，这里只需知道它是 `class` 语句的第 2 步。

### 2. 实例属性与类属性

实例有自己的 `__dict__`，类有类的 `__dict__`：

```python
class Demo:
    value = 10


obj = Demo()
print(obj.__dict__)                     # {}            实例字典是空的，value 在类上
obj.value = 20
print(obj.__dict__)                     # {'value': 20}
print(Demo.__dict__["value"], Demo.value, obj.value)   # 10 10 20
```

赋值 `obj.value = 20` 写的是实例字典，类属性不受影响。反过来，可变的类属性会被所有实例共享——`class Bad: items = []` 之后 `a.items.append(1)` 会让 `b.items` 也变成 `[1]`，因为没有任何赋值发生在实例上，`a.items` 读到的是类字典里的同一个列表。需要每实例独立的可变状态，在 `__init__` 里创建。

### 3. 属性查找算法：`object.__getattribute__`

现在可以给出核心算法。`obj.attr` 默认调用 `type(obj).__getattribute__(obj, "attr")`，对普通对象它的逻辑是：

```python
def __getattribute__(obj, name):                 # object.__getattribute__ 的等价伪代码
    cls = type(obj)
    cls_attr = lookup_in_mro(cls, name)          # ① 沿 cls.__mro__ 逐个查 __dict__，找到第一个就停

    if cls_attr is not None and is_data_descriptor(cls_attr):    # ② 数据描述符（有 __set__ 或 __delete__）优先
        return cls_attr.__get__(obj, cls)

    if name in obj.__dict__:                     # ③ 实例字典
        return obj.__dict__[name]

    if cls_attr is not None:
        if hasattr(cls_attr, "__get__"):         # ④ 非数据描述符（只有 __get__）：函数、classmethod、staticmethod
            return cls_attr.__get__(obj, cls)
        return cls_attr                          # ⑤ 普通类属性

    raise AttributeError(name)                   # ⑥ 都没有 → 抛出；若类定义了 __getattr__，解释器接着调它
```

用三个实验验证这个顺序。首先，数据描述符（`property` 有 `__set__`）赢过实例字典：

```python
class D:
    @property
    def value(self):
        return "from property"


d = D()
d.__dict__["value"] = "from instance dict"
print(d.value)          # from property      —— ② 先于 ③
```

其次，非数据描述符（普通函数）输给实例字典：

```python
class Runner2:
    def run(self):
        return "method"


r = Runner2()
r.__dict__["run"] = lambda: "instance attr"
print(r.run())          # instance attr      —— ③ 先于 ④
```

最后，§2 的 `obj.value = 20` 之所以能覆盖类属性 `10`，是因为 `int` 不是描述符，走的是 ③ 先于 ⑤。

这个顺序设计的意图：`property` 这类需要**控制读写**的东西必须不能被实例字典绕过，所以放最前；方法这类**只提供默认行为**的东西允许被实例覆盖（monkey patch 单个对象的方法就靠这一点），所以放实例字典之后。

### 4. 描述符协议：属性查找的可编程点

算法里的 ② 和 ④ 都在调用一个东西的 `__get__`。凡是定义了 `__get__`、`__set__` 或 `__delete__` 之一的对象，放在**类属性**位置上，就是描述符：

- 只有 `__get__`：**非数据描述符**，可被实例字典覆盖；
- 有 `__set__` 或 `__delete__`：**数据描述符**，优先于实例字典。

一个做类型校验的描述符：

```python
class Typed:
    def __init__(self, expected_type):
        self.expected_type = expected_type

    def __set_name__(self, owner, name):      # 类创建时被调用，拿到自己在类里的名字
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:                       # 通过类访问（Config.batch_size）时 obj 为 None
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} expects {self.expected_type.__name__}, got {type(value).__name__}")
        obj.__dict__[self.name] = value


class Config:
    batch_size = Typed(int)
    device = Typed(str)

    def __init__(self, batch_size, device):
        self.batch_size = batch_size          # 触发 Typed.__set__
        self.device = device


config = Config(32, "cuda")
config.batch_size = "big"                    # TypeError: batch_size expects int, got str
```

描述符必须放在类上才生效——算法第 ① 步只查类的 MRO，实例字典里的对象即便有 `__get__` 也不会被调用。

`__slots__` 是描述符的一个内建应用：`class S: __slots__ = ("a",)` 会让 `type` 为每个槽位在类上创建一个 `member_descriptor`（数据描述符），直接读写实例的固定内存偏移，实例因此不再需要 `__dict__`。第五篇讨论内存时会回到它。

### 5. 方法绑定、`classmethod`、`staticmethod`、`property`：描述符的四种形态

有了 §3 的算法和 §4 的协议，"方法"就不再需要单独解释：**函数对象实现了 `__get__`**，所以它是非数据描述符。

```python
class Runner3:
    def run(self, batch):
        return batch


r3 = Runner3()
print(Runner3.run)          # <function Runner3.run at 0x...>          通过类访问：__get__(None, cls) 返回函数本身
print(r3.run)               # <bound method Runner3.run of <Runner3 ...>>  通过实例访问：__get__(r3, cls) 返回绑定方法
print(r3.run.__self__ is r3, r3.run.__func__ is Runner3.run)          # True True
print(Runner3.__dict__["run"].__get__(r3, Runner3))                    # 手工调用 __get__，得到同样的绑定方法
```

绑定方法是一个小对象，持有实例（`__self__`）和原函数（`__func__`），调用时把 `__self__` 塞到第一个参数位置。`self` 不是关键字，只是这个协议约定的第一个参数。

`classmethod`、`staticmethod`、`property` 是三个内建的描述符类型，区别只在 `__get__` 返回什么，以及是否有 `__set__`：

```python
class K:
    @classmethod
    def c(cls): return cls
    @staticmethod
    def s(): return "s"
    @property
    def p(self): return "p"


print(type(K.__dict__["c"]), type(K.__dict__["s"]), type(K.__dict__["p"]))
# <class 'classmethod'> <class 'staticmethod'> <class 'property'>
print(hasattr(classmethod, "__set__"), hasattr(staticmethod, "__set__"), hasattr(property, "__set__"))
# False False True
```

| 类属性上的对象 | `__get__(obj, cls)` 返回 | 数据描述符？ | 效果 |
| :--- | :--- | :--- | :--- |
| 函数 | `obj` 为 `None` 时返回函数；否则返回绑定 `obj` 的方法 | 否 | 实例方法 |
| `classmethod` | 无论怎么访问，都返回绑定 **`cls`** 的方法 | 否 | 第一个参数是类 |
| `staticmethod` | 原函数本身，不绑定任何东西 | 否 | 普通函数挂在类命名空间 |
| `property` | 调用 `fget(obj)` 的结果 | **是**（有 `__set__`，未提供 setter 时抛 `AttributeError`） | 计算属性，且不可被实例字典覆盖 |

从这张表能直接推出三者的用法：

- **`classmethod` 的价值在于 `cls` 是"实际被调用的那个类"**。`Sub.c()` 和 `Sub().c()` 都返回 `Sub`，不是 `K`。这让它成为替代构造函数的标准写法：

  ```python
  class Runner:
      @classmethod
      def from_config(cls, config):
          return cls(model=config["model"], device=config.get("device", "cpu"))


  class GPURunner(Runner): ...


  GPURunner.from_config(cfg)     # 创建的是 GPURunner 实例
  ```

  对象尚未存在，所以不能是实例方法；需要构造"当前类或子类"，所以不该写死 `Runner(...)`——`cls(...)` 自动跟随继承。

- **`staticmethod` 只是命名空间归属**。既不需要实例也不需要类的函数，放在类里是为了表达"逻辑上属于它"；很多时候放模块级更简单。
- **`property` 是数据描述符**，所以 §3 的第一个实验成立。它适合"看起来像字段、实际需要计算或校验"的场景；不适合隐藏昂贵操作——调用方看到 `runner.device` 不会预期它去查询硬件。

**与 Java 的对照**

| Python | Java 中最接近的形式 | 差异 |
| :--- | :--- | :--- |
| 实例方法 | 实例方法 | 相同；Python 的 `self` 显式出现在签名里 |
| `staticmethod` | `static` 方法 | 相同 |
| `property` | getter/setter 约定，或 record 的访问器 | Java 没有语法级支持，`obj.getX()` 不能写成 `obj.x` |
| `classmethod` | 静态工厂方法 | 见下 |

`classmethod` 与 Java 静态工厂的差异需要说准确。Java 的静态方法**不按接收者分派**：`GpuRunner.createDefault()` 在编译期就被解析为 `Runner.createDefault()`，方法体内没有任何途径知道调用方写的是 `GpuRunner`。要让工厂创建子类，必须由调用方**显式传入**类型信息——`Class<T>` 令牌加反射（`cls.getDeclaredConstructor().newInstance()`），或一个 `Supplier<T>`。这并不难写，但它是显式的；Python 的 `classmethod` 则由描述符协议在 `__get__` 时**自动**把实际的类绑定进去。两者能达到同样的目的，区别在"谁负责提供类对象"：Java 靠调用方传参，Python 靠属性查找机制注入。

### 6. `__getattr__` 与 `__getattribute__`：查找失败后的钩子

§3 算法的第 ⑥ 步：`__getattribute__` 抛出 `AttributeError` 之后，如果类定义了 `__getattr__`，解释器会调用它作为最后手段。

```python
class Cfg:
    def __init__(self):
        self.a = 1

    def __getattr__(self, name):
        print("  __getattr__ called for", name)
        return None


c = Cfg()
print(c.a)      # 1                                —— 正常查找命中，__getattr__ 不参与
print(c.b)      # __getattr__ called for b / None  —— 只在失败后调用
```

`__getattribute__` 则拦截**所有**属性访问，包括 `self.__dict__`。重写它时必须用 `object.__getattribute__(self, name)` 去取真实值，否则无限递归。业务代码几乎不需要它。

`__getattr__` 有一个隐蔽的陷阱：**它会掩盖 `property` 内部的 `AttributeError`**。

```python
class P:
    @property
    def v(self):
        raise AttributeError("bug inside property")   # 比如 property 里访问了一个拼错的字段

    def __getattr__(self, name):
        return f"fallback({name})"


print(P().v)    # fallback(v)
```

`property` 的 getter 抛出 `AttributeError`，`__getattribute__` 把它当作"没找到 `v`"，转而调用 `__getattr__("v")`——真正的 bug 被吞掉，换成一个看似合理的返回值。有 `__getattr__` 的类里，`property` 中的异常应转换成其他类型再抛出。

一个真实的例子是 `torch.nn.Module`。它把参数、缓冲区和子模块分别存在 `_parameters`、`_buffers`、`_modules` 三个字典里，**不放在实例 `__dict__`**。这依靠两个钩子配合（PyTorch 2.9.0，`torch/nn/modules/module.py`）：

```python
# Module.__setattr__ 的骨架
def __setattr__(self, name, value):
    if isinstance(value, Parameter):
        remove_from(self.__dict__, self._buffers, self._modules, ...)   # 保证同名只存在于一处
        self.register_parameter(name, value)
    elif isinstance(value, Module):
        remove_from(self.__dict__, self._parameters, self._buffers, ...)
        self._modules[name] = value
    else:
        super().__setattr__(name, value)                                  # 普通属性才进 __dict__


# Module.__getattr__ 的骨架
def __getattr__(self, name):
    if name in self.__dict__["_parameters"]: return self.__dict__["_parameters"][name]
    if name in self.__dict__["_buffers"]:    return self.__dict__["_buffers"][name]
    if name in self.__dict__["_modules"]:    return self.__dict__["_modules"][name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

于是 `self.linear = nn.Linear(4, 4)` 被 `__setattr__` 截获，写进 `_modules`；之后 `self.linear` 走 §3 的算法在类和实例字典都找不到，落到 `__getattr__`，从 `_modules` 取回。这就是 `parameters()`、`state_dict()`、`.to(device)` 能够递归遍历所有子模块的原因——它们只需遍历三个字典，而不需要扫描 `__dict__` 猜哪些属性是模块。代价是每次访问子模块都多走一次失败查找加一次 Python 级函数调用，所以在热路径上 PyTorch 自己也会先把 `self.linear` 取到局部变量。

模块级也有同样的钩子（PEP 562）：在模块顶层定义 `__getattr__(name)`，访问不存在的模块属性时会被调用。`torch/__init__.py` 用它做两件事——对已废弃的名字发出 `DeprecationWarning` 后返回替代品，以及对少数子模块做惰性导入（第一次访问时才 `importlib.import_module`）。

### 7. 对象创建：`__new__`、`__init__` 与 `type.__call__`

`Runner(model)` 是对**类对象**的调用，按第六章的可调用协议，执行的是 `type(Runner).__call__`，即 `type.__call__`。它做两件事：

```python
def __call__(cls, *args, **kwargs):          # type.__call__ 的等价伪代码
    obj = cls.__new__(cls, *args, **kwargs)  # 分配并返回实例
    if isinstance(obj, cls):
        obj.__init__(*args, **kwargs)        # 初始化实例；返回值必须是 None
    return obj
```

`__new__` 是静态方法（唯一一个不需要写 `@staticmethod` 的特例），负责"造出对象"；`__init__` 负责"填充对象"。重写 `__new__` 的场景很少：不可变类型（`int`、`tuple`、`str`）的子类化必须在 `__new__` 里改值，因为 `__init__` 拿到的已经是造好的不可变对象；单例、实例缓存、根据参数返回不同子类的工厂类也在这里做。其他情况只写 `__init__`。

如果 `__new__` 返回的不是 `cls` 的实例，`__init__` 不会被调用——这是"工厂类"模式的机制基础。

### 8. 继承、MRO 与协作式 `super()`

属性查找算法第 ① 步"沿 `cls.__mro__` 查找"，MRO（Method Resolution Order）就是这个顺序。单继承时它就是从子类到 `object` 的链；多继承时由 C3 线性化算法计算，保证子类先于父类、多个父类之间保持声明顺序：

```python
class A:
    def run(self): return ["A"]
class B:
    def run(self): return ["B"]
class C(A, B): pass


print([k.__name__ for k in C.__mro__], C().run())    # ['C', 'A', 'B', 'object'] ['A']
```

`C().run()` 找到 `A.run` 就停，**不会**自动接着调 `B.run`。想让链上的每一层都执行，每一层必须自己调 `super()`：

```python
class Base:
    def run(self): return ["base"]
class Logging:
    def run(self):
        result = super().run(); result.append("logging"); return result
class Metrics:
    def run(self):
        result = super().run(); result.append("metrics"); return result
class Run(Logging, Metrics, Base): pass


print([k.__name__ for k in Run.__mro__])   # ['Run', 'Logging', 'Metrics', 'Base', 'object']
print(Run().run())                         # ['base', 'metrics', 'logging']
```

`Logging.run` 里的 `super().run()` 调到的是 `Metrics.run`，不是 `Base.run`——`super()` 的含义是"**在实例的 MRO 中，当前类之后的下一个**"，它取决于实例的类型，而不是写代码时看到的父类。`Logging` 类的作者并不知道 `Metrics` 的存在，链条却能接上，这就是协作式多继承。

零参数 `super()` 之所以知道"当前类"是谁，是因为编译器在使用了 `super()` 的方法里注入了一个名为 `__class__` 的闭包变量（可以用 `Child.run.__code__.co_freevars` 看到 `('__class__',)`）——第三章的 cell 机制在这里又出现了一次。

协作式多继承的约定：每一层都调 `super()`（包括 `__init__`）；同名方法签名兼容，通常用 `*args, **kwargs` 透传；不假设 `super()` 指向某个具体类。任何一层漏掉 `super()`，它之后的所有层都会被跳过。声明顺序不一致时 C3 会直接拒绝：`class X(A, C)` 抛出 `TypeError: Cannot create a consistent method resolution order (MRO) for bases A, C`，因为 `C` 已经要求 `A` 在自己之后。

### 9. Mixin 与组合

Mixin 是协作式多继承的一种受限用法：一个只提供横向能力（日志、指标、序列化）、不代表完整实体、不持有业务依赖的类，被放在基类列表前部：

```python
class LoggingMixin:
    def log(self, message):
        print(f"[LOG] {message}")


class Runner(LoggingMixin, BaseRunner):
    def run(self):
        self.log("start")
```

Mixin 没有绕开 MRO 的复杂性，只是把它限制在一个容易验证的范围内。约束是：职责单一；对宿主类的要求明确；`__init__` 遵守协作式 `super()`；用 `__mro__` 和测试固定实际顺序。

而 `model`、`scheduler`、`tokenizer` 这类**业务依赖**，应该用组合——持有而不是继承：

```python
class InferenceRunner:
    def __init__(self, model, scheduler, tokenizer):
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
```

组合的依赖是显式的（构造参数），可以注入替身，不依赖 MRO 和 `super()` 协议。选择规则：

| 关系 | 方式 |
| :--- | :--- |
| 稳定的"是一种"关系、遵守框架协议（如 `nn.Module`） | 继承 |
| 复用独立的横向能力 | Mixin，谨慎 |
| 给单个函数加日志、重试、tracing | 装饰器（第七章） |
| 管理业务依赖 | 组合 |

**与 Java 的对照**：Java 单继承加接口，接口的 default 方法能提供一部分 Mixin 的效果，但没有 MRO——两个接口的同名 default 方法冲突时必须在实现类里显式选择，不存在"沿链自动接力"的 `super()`。Java 的 `super.method()` 永远指向直接父类，是静态的；Python 的 `super()` 是动态的，取决于实例的 MRO。这一点是 Java 程序员读 Python 多继承代码时最容易误判的地方。


## 六、对象协议：语法背后的特殊方法

第五章解释了 `obj.attr`。但 `runner(batch)`、`for x in loader`、`batch[0]`、`if tensor:`、`with ctx:` 这些**不是**属性访问的语法，Python 也是交给对象自己决定的——通过一组以双下划线命名的特殊方法。这一章讨论其中最常见的几个；`with` 和 `yield` 分别留给第八、九章。

### 1. 语法到特殊方法的映射，且在类型上查找

| 表达式 | 特殊方法 |
| :--- | :--- |
| `x(...)` | `__call__` |
| `for item in x` | `__iter__`，然后对迭代器反复 `__next__` |
| `x[key]`、`x[key] = v`、`del x[key]` | `__getitem__`、`__setitem__`、`__delitem__` |
| `len(x)` | `__len__` |
| `if x`、`bool(x)` | `__bool__`，缺失时退化为 `__len__` |
| `x == y`、`hash(x)` | `__eq__`、`__hash__` |
| `x + y` | `__add__`，失败时尝试 `y.__radd__` |
| `with x:` | `__enter__`、`__exit__` |
| `obj.attr` | `__getattribute__`、`__getattr__`（第五章） |

有一个规则与第五章的属性查找不同：**特殊方法由解释器直接在类型上查找，跳过实例字典**。

```python
class C:
    def __call__(self):
        return "type-level"


c = C()
c.__call__ = lambda: "instance-level"
print(c())              # type-level        —— 解释器查的是 type(c).__call__
print(c.__call__())     # instance-level    —— 普通属性访问才看实例字典
```

这是性能考虑的结果：`a + b` 如果每次都走完整的属性查找算法太慢，CPython 在类型对象上为每个特殊方法维护了一个槽位（slot），运算时直接取槽位。后果是"给单个对象打补丁换掉 `__call__`"不起作用——必须换类，或者在类的 `__call__` 里做转发。

### 2. 可调用对象：`__call__`

实现 `__call__` 的实例可以像函数一样被调用。第一章的 `Runner.__call__` 就是这样让 `runner(batch)` 成立的。相比函数，可调用对象能同时携带配置、缓存、依赖和生命周期状态，因此模型、推理器、预处理器、Hook、Loss 大多是可调用对象。

`nn.Module` 是最典型的例子。`model(x)` 不直接调 `forward`，而是 `Module.__call__`——在 PyTorch 2.9.0 中它被赋值为 `_wrapped_call_impl`（`torch/nn/modules/module.py`）：先检查有没有 `torch.compile` 生成的编译版本，再进入 `_call_impl`；`_call_impl` 在没有任何 hook 注册时直接调用 `forward`，否则依次执行 forward pre-hooks、`forward`、forward hooks，并处理 backward hooks 的挂载。所以直接调 `model.forward(x)` 会绕过全部 hook——这就是文档要求"调用模块而不是 forward"的机制原因。

### 3. 迭代协议：可迭代对象与迭代器

`for item in x` 展开为：

```python
it = iter(x)               # 调 x.__iter__()，得到迭代器
while True:
    try:
        item = next(it)    # 调 it.__next__()
    except StopIteration:
        break
    ...
```

**可迭代对象**（iterable）实现 `__iter__` 并返回一个迭代器；**迭代器**（iterator）实现 `__next__`，并且自己的 `__iter__` 返回自己。两者常被混为一谈，但差别决定了能否重复遍历：

```python
lst = [1, 2]
print(iter(lst) is lst)                 # False   列表每次 iter() 返回一个新迭代器，可反复遍历


class CountDown:                        # 既是可迭代对象又是迭代器
    def __init__(self, start): self.current = start
    def __iter__(self): return self
    def __next__(self):
        if self.current <= 0: raise StopIteration
        self.current -= 1
        return self.current + 1


c = CountDown(2)
print(list(c), list(c))                 # [2, 1] []    —— 第二次遍历是空的：迭代器耗尽后不会重置
```

`torch.utils.data.DataLoader` 是可迭代对象——每个 epoch 的 `for batch in loader` 都调用 `loader.__iter__()` 创建一个**新的**迭代器（在多 worker 模式下这意味着重新启动 worker 进程）。而 `loader.__iter__()` 返回的那个对象是迭代器，用完即弃。理解这一层，就能解释为什么 `iter(loader)` 拿出来手工 `next()` 的对象不能跨 epoch 复用。

还有一条历史兼容规则：没有 `__iter__` 但有 `__getitem__` 的对象也可迭代——解释器从 0 开始调 `__getitem__` 直到 `IndexError`。`Dataset` 子类只实现 `__getitem__` 和 `__len__` 就能被 `for` 遍历，靠的就是它。

### 4. `__getitem__` 与容器协议

```python
class Batch:
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self.values[index]


batch = Batch([1, 2, 3])
print(batch[0], batch[1:])              # 1 [2, 3]     —— 切片对象 slice(1, None, None) 原样传给 __getitem__
```

`__getitem__` 收到的 `index` 可以是整数、切片、元组（`x[i, j]` 传入 `(i, j)`）、甚至任意对象——Tensor 的高级索引 `t[mask]`、`t[..., 0]` 全部走这一个方法，由实现自己解释参数。

### 5. 真值判断

`if x:` 先找 `__bool__`，没有则用 `__len__` 是否为零，两者都没有则一律为真。这带来一个 AI-Infra 特有的陷阱：多元素 Tensor 的真值没有明确定义，PyTorch 会抛 `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。所以 `if tensor:`、`tensor and other`、`x if mask else y` 在 Tensor 上都不安全，要用 `.any()`/`.all()`/`torch.where` 明确意图。同理，`if batch:` 判断"是否有数据"时，要确认 `Batch` 类实现了 `__len__` 或 `__bool__`，否则永远为真。

### 6. `__eq__` 与 `__hash__`

`==` 调 `__eq__`；作为字典键或集合元素则需要 `__hash__`。两者有一条契约：**相等的对象必须有相同的哈希值**。为了防止违反它，只定义 `__eq__` 会让 Python 把 `__hash__` 设为 `None`：

```python
class User:
    def __init__(self, uid): self.uid = uid
    def __eq__(self, other): return isinstance(other, User) and self.uid == other.uid


print(User.__hash__)     # None
{User(1)}                # TypeError: unhashable type: 'User'
```

需要哈希就显式实现 `__hash__`，并且只使用参与 `__eq__` 且在对象生命周期内不变的字段。`@dataclass(frozen=True)` 会自动生成一致的两者；可变的 dataclass 默认也是不可哈希的，原因相同。

### 7. 与 Java 的对照

Java 的做法是**接口**：想被 `for-each` 遍历就实现 `Iterable<T>`，想能 `try-with-resources` 就实现 `AutoCloseable`，`equals`/`hashCode` 契约与 Python 完全一致（`HashMap` 同样依赖它）。Python 的特殊方法是**协议**：不需要声明实现了什么，只要方法存在、签名对得上，语法就生效。

差异带来两点后果：Python 里"这个对象能不能 `for`"无法从类型声明得知，只能看有没有 `__iter__`（第二篇的 `Protocol` 就是为了给这类协议补上静态描述）；Java 没有对应 `__call__` 的机制，"可调用对象"要么是函数式接口的实例、要么是显式的 `.apply()`/`.call()`，`model(x)` 这种写法不存在，`nn.Module` 那种"在调用路径上插 hook"只能靠动态代理或 AOP 实现。


## 七、装饰器：用闭包和描述符改写调用路径

第一章的 `@registered("runner")` 让 `Runner` 在被定义的同时进入注册表。装饰器的全部机制在前面已经准备好了：函数是对象（第二章），闭包能记住参数（第三章），类属性上的函数是描述符（第五章）。这一章把它们组装起来。

### 1. 基本机制

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print("calling", func.__name__)
        return func(*args, **kwargs)
    return wrapper


@log_call
def predict(x):
    return x * 2
```

`@log_call` 是语法糖，等价于在 `def` 执行完之后立刻做一次重新绑定：

```python
def predict(x): ...
predict = log_call(predict)
```

所以：装饰器在**定义时**执行一次（模块导入时、类体执行时），返回的对象在**每次调用**时执行。此后名称 `predict` 指向 `wrapper`，原函数只存在于 `wrapper` 的闭包 cell 里。多个装饰器从下往上应用：

```python
@outer
@inner
def run(): ...
# run = outer(inner(run))     调用时先进 outer 的 wrapper，再进 inner 的
```

### 2. 装饰器与闭包的关系

`wrapper` 能在 `log_call` 返回之后仍然找到 `func`，靠的是第三章的 cell：`wrapper.__closure__[0].cell_contents is predict_original`。闭包是实现装饰器最常见的方式，但不是必需的——任何"接收可调用对象、返回可调用对象"的东西都是装饰器，包括类：

```python
class LogCall:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("calling", self.func.__name__)
        return self.func(*args, **kwargs)


@LogCall
def predict(x): ...
```

状态少用闭包，状态多、需要暴露方法（如 `cache.clear()`）用类——与第三章 §5 的取舍一致。注意类实现的装饰器用在**方法**上时有一个坑，见 §6。

### 3. `functools.wraps` 与 `__wrapped__`

`wrapper` 是一个新函数对象，它的 `__name__`、`__doc__`、`__module__`、`__qualname__`、`__annotations__` 都是自己的。不处理的话，`predict.__name__` 变成 `'wrapper'`，日志、文档、`pickle`、测试框架的参数化 ID 全部受影响。`functools.wraps` 把这些元数据从原函数复制过来，并额外设置 `wrapper.__wrapped__ = func`：

```python
from functools import wraps


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

`__wrapped__` 让 `inspect.signature` 能穿透装饰器看到原始签名，也让调试时可以用 `predict.__wrapped__` 拿回原函数。`wraps` 复制的只是元数据，不改变行为：包装器仍然可能改变参数校验、返回值、异常类型和执行时机。

### 4. 带参数的装饰器

`@retry(max_attempts=3)` 多一层：先用参数调用 `retry`，得到真正的装饰器，再用它装饰函数。

```python
def retry(max_attempts, exceptions=(TimeoutError,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator
```

```text
retry(max_attempts)  →  decorator(func)  →  wrapper(*args, **kwargs)
   配置层                 装饰层                运行层
```

`max_attempts` 和 `exceptions` 经由两层闭包到达 `wrapper`。重试装饰器应只捕获明确可重试的异常，让其余异常原样传播——`except Exception` 会把断言失败、类型错误和 CUDA OOM 一起重试。

### 5. 装饰类：注册表

装饰器的参数可以是类，返回原类不变、只做登记，就是第一章的 `registered`：

```python
def registered(name):
    def decorator(cls):
        if name in REGISTRY:
            raise ValueError(f"duplicate registration: {name}")
        REGISTRY[name] = cls
        return cls
    return decorator
```

由 §1 的时机规则和第四章 §6 的导入规则可推出它的全部行为：注册发生在类定义执行时，也就是模块被导入时；模块没被导入则注册表为空；模块以两个名字导入则第二次触发 `duplicate registration`——这正是第四章 §4 那个 `__main__`/`app` 双重导入实验里看到的报错。重复注册抛异常而不是静默覆盖，是刻意的：它把"同一个文件被导入了两次"这个隐蔽问题变成一个显眼的启动失败。

### 6. 装饰方法时的叠放顺序：与描述符的交互

装饰器应用在**类体执行时**，此时 `def` 产生的还是普通函数（第五章 §1），描述符协议要等到属性访问时才起作用。这决定了几件事：

```python
class Runner:
    @classmethod
    @log_call                  # 先应用：包装的是普通函数，wrapper 还是普通函数
    def create(cls): ...       # 再应用 classmethod：把 wrapper 包成 classmethod 对象
```

- `log_call` 必须在 `classmethod`/`staticmethod`/`property` **下面**。反过来写，`log_call` 收到的是一个 `classmethod` 对象，它不可调用，`wrapper` 里 `func(*args)` 会报 `TypeError: 'classmethod' object is not callable`。
- `wrapper(*args, **kwargs)` 的 `args[0]` 是 `self`/`cls`——因为 `wrapper` 作为普通函数放在类上，本身就是非数据描述符，绑定后 `self` 会正常传入。这是函数式装饰器能"透明"用于方法的原因。
- §2 那种用类实现的装饰器**不能**透明用于方法：`LogCall` 实例没有 `__get__`，不是描述符，`obj.method` 得到的是 `LogCall` 实例本身，调用时不会传 `self`（报 `TypeError: S.m() missing 1 required positional argument: 'self'`）。要修复就给 `LogCall` 加一个 `__get__`，返回 `functools.partial(self, obj)` 或 `types.MethodType(self, obj)`。

第五章的知识在这里直接决定了代码能不能跑。

### 7. 同步与异步装饰器

`async def` 函数被调用时只返回协程对象，不执行函数体。同步 `wrapper` 包裹异步函数时，`func(*args)` 立即返回一个协程，`wrapper` 的"结束"日志在函数体真正运行之前就打出来了，异常也捕获不到。异步函数需要异步包装器：

```python
def async_log_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        print("start")
        result = await func(*args, **kwargs)
        print("end")
        return result
    return wrapper
```

需要同时支持两种函数的装饰器，用 `inspect.iscoroutinefunction(func)` 分派。并发与异步在本系列第三篇展开。

### 8. 与 Java 的对照

Java 的注解（`@Transactional`、`@Retryable`）只是**元数据**，本身不改变任何行为；行为来自框架在运行时扫描注解、生成代理对象（Spring AOP 的 JDK 动态代理或 CGLIB 子类）并拦截调用。Python 的装饰器则是**普通函数调用**，在定义处直接替换掉目标对象，没有框架也能工作。

后果是双向的：Python 装饰器透明得多——`grep` 就能找到它做了什么，调试时能直接进 `wrapper`；但它也没有 Java 代理的"边界"——Spring 的自调用不经过代理是常见坑，Python 里则是 §6 的描述符顺序和 §7 的同步/异步区分。两边共同的原则是：被包装的调用路径必须能被开发者看到。


## 八、生成器与惰性执行

第二章说函数调用创建一个帧，返回时销毁它。生成器打破了这个规则：`yield` 让帧**挂起**而不销毁，下次 `next()` 时从原地恢复。第一章的 `Runner.stream` 用它实现流式输出。

### 1. `yield`：帧被挂起而不是销毁

```python
import inspect


def stream():
    print("  start")
    yield 1
    print("  resume")
    yield 2
    print("  end")


g = stream()
print(type(g), inspect.getgeneratorstate(g))   # <class 'generator'> GEN_CREATED   —— 函数体一行都没执行
print(next(g))                                  #   start / 1
print(inspect.getgeneratorstate(g))            # GEN_SUSPENDED
print(next(g))                                  #   resume / 2
next(g)                                         #   end → StopIteration；状态变为 GEN_CLOSED
```

调用生成器函数不执行函数体，只创建一个生成器对象，它内部持有一个帧（`g.gi_frame`）；第一次 `next()` 才开始执行；到 `yield` 处暂停，帧里的局部变量、指令位置全部保留（`g.gi_frame.f_locals` 可以直接看到）；函数体结束时抛出 `StopIteration`。生成器同时实现了 `__iter__`（返回自己）和 `__next__`，所以它是第六章意义上的迭代器，可以直接放进 `for`。

### 2. 惰性的价值与成本

生成器让"生产一个、消费一个"成为默认模式，不需要把全部结果放进内存：

```python
def read_lines(path):
    with open(path) as file:
        for line in file:
            yield line
```

流式 Token 输出、大数据集分片读取、Batch 生成、日志消费都是这个模式。但惰性有两个经常被忽视的成本：

- **生成器持有资源的时间等于它存活的时间**。上面 `read_lines` 的文件句柄在第一次 `next()` 时打开，要到生成器结束或被关闭时才关——如果消费方中途 `break` 并把生成器丢在一边，文件就一直开着（§4）。GPU 张量、数据库游标、进程间队列同理。
- **错误延迟出现**。生成器函数里的 bug 在 `next()` 时才暴露，traceback 的调用方是消费者而不是创建者。

### 3. `yield from`

`yield from sub()` 把 `sub` 产生的每个值直接转发给外层的消费者，同时把 `send()`、`throw()`、`close()` 也透传给子生成器，并且能拿到子生成器的返回值：

```python
def sub():
    yield 1
    return "sub-result"


def outer():
    result = yield from sub()
    print("sub returned:", result)      # sub returned: sub-result
```

它不只是嵌套 `for` 的简写。`Runner.stream` 用 `yield from self.model.generate(batch)` 把模型的生成器**原样**暴露给调用方，包括调用方 `close()` 时能一路传到模型层面做清理。`async`/`await` 在语法上就是从 `yield from` 演化来的。

### 4. 关闭与清理：`GeneratorExit`、`finally` 与 `break`

`g.close()` 在生成器挂起的 `yield` 处抛入 `GeneratorExit`，让 `try/finally` 和 `with` 的清理逻辑有机会执行：

```python
def reader():
    print("  open")
    try:
        yield 1
        yield 2
    finally:
        print("  close file")


r = reader()
next(r)          #   open
r.close()        #   close file
```

`for` 循环正常跑完会耗尽生成器，`finally` 自然执行；但 `break` 只是**退出循环**，不会关闭生成器：

```python
r = reader()
for v in r:
    break
print("after break")      # 此时 "close file" 还没有出现
del r                     #   close file   —— 引用计数归零时，生成器的析构才调用 close()
```

在 CPython 里，生成器对象被回收时会自动 `close()`，所以上面的例子在 `del` 后立刻清理。但依赖这一点是脆弱的：生成器被别处引用（放进列表、被 traceback 持有、被闭包捕获）就不会及时回收；在其他实现或存在循环引用时，回收时机不确定。**需要确定性清理的生成器，要么由消费方在 `finally` 里显式 `close()`，要么用 `contextlib.closing(gen)` 包起来放进 `with`。** 对象回收的细节在本系列第五篇。

### 5. 与 Java 的对照

Java 的 `Iterator`/`Stream` 也是惰性的，但它们是**对象**，"暂停"靠的是对象字段记录状态，每一步的状态机要自己写；Python 的生成器把状态机交给帧，一个 `yield` 就完成了。Java 没有对应 `yield` 的语言机制（Loom 的 `Continuation` 是内部 API），流式响应通常靠回调或 `Flow.Publisher`。另一方面，Java 的 `Stream` 有明确的"终止操作"边界，而 Python 的生成器随时可能被半途丢弃，所以 §4 的清理问题在 Java 中要少得多。


## 九、上下文管理器：把资源生命周期交给协议

第八章的结论是"资源的释放时机不能靠猜"。`with` 语句就是为此设计的：把"进入/退出"两个动作绑定到一个代码块上，无论块内是正常结束、`return`、还是抛异常，退出动作都执行。第一章的 `InferenceContext` 和 `torch.inference_mode()` 都是这个协议。

### 1. `with` 的展开

```python
with resource() as value:
    use(value)
```

等价于：

```python
manager = resource()
value = manager.__enter__()
try:
    use(value)
except BaseException as exc:
    if not manager.__exit__(type(exc), exc, exc.__traceback__):
        raise                                   # __exit__ 返回假值 → 异常继续传播
else:
    manager.__exit__(None, None, None)
```

由展开可以读出几条规则，每一条都在实验中验证过：

- `__enter__` 抛异常时，`__exit__` **不会**被调用——资源还没获取，无需释放；
- `__exit__` 收到异常三元组，返回 `True` 表示**抑制**该异常（块外看不到），返回 `False`/`None` 则异常照常传播；`contextlib.suppress` 就是一个返回 `True` 的 `__exit__`；
- 块内 `return`、`break` 同样触发 `__exit__`，因为它们也走 `try` 的退出路径。

### 2. 类实现与 `contextlib.contextmanager`

类实现直接写两个方法：

```python
class Resource:
    def __enter__(self):
        print("acquire"); return self
    def __exit__(self, exc_type, exc, tb):
        print("release"); return False
```

生成器实现更短——`@contextmanager` 把一个只 `yield` 一次的生成器包装成上下文管理器：

```python
from contextlib import contextmanager


@contextmanager
def resource():
    handle = acquire()
    try:
        yield handle              # yield 之前是 __enter__，之后是 __exit__
    finally:
        release(handle)
```

它的机制是第八章的内容：`__enter__` 调一次 `next()` 跑到 `yield`；`__exit__` 在没有异常时再调一次 `next()` 让它跑完，有异常时用 `gen.throw(exc)` 把异常**在 `yield` 那一行抛出**。所以 `try/finally` 不是可选的——没有它，块内的异常会让生成器在 `yield` 处直接终止，`release` 永远不会执行。同理，想在 `@contextmanager` 里抑制异常，要在 `except` 里捕获并不再抛出。

### 3. AI-Infra 中的上下文管理器

`torch.inference_mode()`、`torch.autocast()`、`torch.profiler.profile()`、`torch.cuda.stream(s)`、`torch.no_grad()`、分布式通信组的生命周期，大多数是"**进入时切换一个全局或线程局部状态，退出时恢复原值**"，而不是获取/释放一个句柄。读这类代码时要问：

1. 进入时改了什么状态？是线程局部还是进程全局？（`no_grad` 是线程局部的；多线程数据加载时要注意）
2. 退出时是否恢复到**进入前**的值，还是恢复到某个默认值？嵌套使用时二者行为不同；
3. 异常路径是否也恢复？（用 `try/finally` 或类实现的 `__exit__` 才能保证）
4. 作为装饰器使用时（`@torch.no_grad()`）语义是否一致？

第一章的 `Runner.__call__` 用 `nullcontext()` 处理"不需要上下文"的分支，避免了 `if` 两个分支重复写 `self.model(batch)`——这是 `contextlib` 里最常用的小工具之一。

### 4. 异步上下文管理器

`async with` 对应 `__aenter__`/`__aexit__`，两者都是协程，允许在进入和退出时 `await`（建立连接、优雅关闭）。`contextlib.asynccontextmanager` 是对应的生成器版本。

### 5. 与 Java 的对照

Java 7 的 try-with-resources 是同一个思路：实现 `AutoCloseable`，`close()` 在块结束时自动调用。差异有三点：Java 只有"退出"钩子，没有 `__enter__` 的返回值和 `__exit__` 的异常参数，因此**不能抑制异常**，也不能根据异常类型做不同清理；Java 的资源必须是一个对象，Python 的 `@contextmanager` 让任何一段"前置/后置"逻辑都能变成上下文管理器；Java 生态里几乎没有 §3 那种"临时切换全局状态"的惯用法，这是 Python 科学计算生态特有的模式。


## 十、异常处理与失败传播

前面每一章都有一个"异常发生时会怎样"的分支：`__exit__` 收到异常、`finally` 在生成器关闭时执行、导入失败让注册表为空。这一章讨论异常本身：它如何沿帧链传播，以及在 AI-Infra 中什么样的处理方式是安全的。

### 1. `try` 结构与传播路径

```python
try:
    result = run()
except ValueError as exc:
    handle(exc)
else:
    process(result)          # 只在 try 块没有异常时执行
finally:
    cleanup()                # 总是执行，包括 return 和未被捕获的异常
```

异常发生时，解释器沿当前帧的 `f_back` 链向上寻找匹配的 `except`；每经过一帧，就把这一帧记进 `__traceback__`。找不到就终止线程（主线程则终止进程）。这就是第二章 §3 中看到的 traceback 结构。

### 2. `BaseException` 层级

```text
BaseException
├── BaseExceptionGroup        3.11+
├── Exception                 ← 业务代码应捕获的根
├── GeneratorExit             生成器/协程被关闭（第八章）
├── KeyboardInterrupt         Ctrl+C
└── SystemExit                sys.exit()
```

`except BaseException:` 或裸 `except:` 会吞掉 `KeyboardInterrupt` 和 `SystemExit`——Worker 收到终止信号却继续跑；也会吞掉 `GeneratorExit`，让生成器无法关闭。业务代码捕获 `Exception` 即可。

### 3. 自定义异常与错误边界

异常类型是给**调用方**用的，它应该回答"发生了什么"以及"能怎么办"：

```python
class BackendError(Exception): ...
class BackendUnavailableError(BackendError): ...     # 可以换后端
class BackendExecutionError(BackendError): ...       # 可以重试


try:
    run_backend()
except BackendUnavailableError:
    fallback_to_cpu()
except BackendExecutionError:
    retry()
```

按业务边界建一个异常基类，让调用方可以选择粗粒度（`except BackendError`）或细粒度处理。错误边界就是"在哪一层把底层异常翻译成调用方能决策的类型"——通常在模块或服务的公共接口处。

### 4. 异常链：`from exc` 与 `from None`

在处理一个异常时抛出另一个，Python 自动记录原异常：

```python
try:
    load_config()
except OSError as exc:
    raise RuntimeError("failed to load configuration") from exc
```

`from exc` 设置 `__cause__`，traceback 显示 "The above exception was the direct cause of the following exception"；不写 `from` 则隐式记录到 `__context__`，显示 "During handling of the above exception, another exception occurred"。两者都保留根因，区别只是措辞和意图的明确程度。`from None` 抑制链条——只在底层异常确实对调用方无意义时使用，否则排障时会丢掉关键信息。

### 5. 记录并重抛：`raise` 与 `raise exc` 的区别

```python
try:
    worker.run()
except Exception:
    logger.exception("worker failed")    # 记录完整 traceback
    raise                                # 原样重抛
```

裸 `raise` 重新抛出当前异常，traceback 不变；`raise exc` 则会把**当前这一行**追加进 traceback，实验中同一个异常的帧列表从 `['<module>', 'w']` 变成 `['<module>', '<module>', 'w']`，多出一层噪音。

只记录不重抛是最常见的错误处理反模式：

```python
except Exception as exc:
    logger.error(str(exc))     # 丢了 traceback，也丢了异常本身
```

上层以为任务成功，Worker 带着不一致的状态继续跑，分布式训练中表现为某个 rank 悄悄掉队、其他 rank 在集合通信上无限等待。原则是：**能处理就处理并明确恢复到一致状态，不能处理就重抛；日志不是处理**。

### 6. 与 Java 的对照

Python 没有受检异常，所有异常都是 Java 意义上的 `RuntimeException`；方法签名不声明会抛什么，这是文档和类型注解（第二篇）的职责。`raise ... from exc` 对应 Java 的 `new RuntimeException(msg, cause)`，`__cause__` 对应 `getCause()`。裸 `raise` 对应 `throw;`——Java 里 `throw e;` 不会改变 `e` 的栈信息（栈在构造时固定），而 Python 的 traceback 是在传播过程中逐帧累加的，所以 §5 的差别在 Java 里不存在。`finally` 语义一致；`else` 子句是 Python 独有的。


## 十一、一个推理组件的完整运行时追踪

回到第一章的 `runner.py`。现在可以按时间顺序，用前面九章的机制精确描述它的每一步。

### 1. 导入阶段

某个模块执行 `import runner`（或 `from runner import Runner`）：

1. `sys.modules` 中没有 `"runner"`，进入查找（第四章 §2）；
2. `sys.meta_path` 上的 `PathFinder` 沿 `sys.path` 找到 `runner.py`，`FileFinder` 按 `.py` 后缀选出 `SourceFileLoader`，生成 `ModuleSpec`（第四章 §3）；
3. 创建空模块对象，写入 `sys.modules["runner"]`——此刻它还是空的（第四章 §7 循环导入的窗口期）；
4. 编译整个文件为 code object（第二章 §1），在模块 `__dict__` 中执行顶层代码：
   - `from contextlib import nullcontext`：`contextlib` 已在 `sys.modules`，直接绑定名称；
   - `REGISTRY = {}`：创建字典；
   - `def registered(name)`：创建函数对象，`__globals__` 指向本模块的 `__dict__`（第二章 §2）；
   - `class InferenceContext:`：执行类体、调用 `type` 创建类对象（第五章 §1）；
   - `@registered("runner") class Runner:`：先执行类体得到类对象，然后调用 `registered("runner")` 得到 `decorator`（闭包持有 `name`，第三章 §2），再调用 `decorator(Runner)`——写入 `REGISTRY`，返回原类（第七章 §5）；名称 `Runner` 绑定到它。
5. 导入方拿到模块对象或 `Runner` 名称。

如果没有任何模块导入 `runner`，第 4 步不会发生，`REGISTRY` 里不会有 `"runner"`。如果 `runner.py` 同时被当作脚本运行又被别的模块导入，第 4 步会执行两次，第二次抛出 `duplicate registration`（第四章 §4）。

### 2. 创建对象阶段

`runner = Runner(model)`：

1. `Runner` 是类对象，调用它执行 `type.__call__(Runner, model)`（第五章 §7）；
2. `Runner.__new__(Runner)` 分配实例（未重写，走 `object.__new__`）；
3. `Runner.__init__(instance, model)` 执行：`self.model = model`、`self.inference = True`，两次赋值走默认 `__setattr__`，写入实例 `__dict__`（第五章 §2）；
4. 返回实例，名称 `runner` 绑定到它。

### 3. 调用阶段

`output = runner(batch)`：

1. 解释器在 `type(runner)` 上查找 `__call__` 槽位（第六章 §1），找到 `Runner.__call__`；
2. 函数作为非数据描述符被绑定，`self = runner`（第五章 §5），创建新的执行帧（第二章 §3）；
3. `self.inference`：类 MRO 上没有同名描述符，实例 `__dict__` 中命中（第五章 §3 的 ③）；
4. `InferenceContext()` 创建上下文管理器实例；
5. `with context:` 调用 `__enter__`，打印 `enter inference mode`（第九章 §1）；
6. `self.model(batch)`：`model` 是什么类型就走什么类型的 `__call__`——如果是 `nn.Module`，进入 `_wrapped_call_impl` → hooks → `forward`（第六章 §2）；
7. `return` 触发 `with` 的退出路径，`__exit__(None, None, None)` 打印 `exit inference mode`，返回 `False`；
8. 帧销毁，返回值绑定到 `output`。

### 4. 流式执行阶段

`for out in runner.stream(batch):`：

1. `runner.stream` 绑定为方法，调用它**不执行函数体**，返回一个 `GEN_CREATED` 状态的生成器对象（第八章 §1）；
2. `for` 调用 `iter()`，生成器返回自身（第六章 §3）；
3. 第一次 `next()`：进入函数体，`self.model.generate(batch)` 被调用，返回模型的生成器；`yield from` 开始转发（第八章 §3）；
4. 每产生一个 token，两个生成器的帧都挂起，控制权回到 `for` 循环体；
5. 模型生成器结束，`yield from` 收到 `StopIteration`，`stream` 的帧也结束，`for` 退出。

如果调用方在第 4 步 `break`，两个生成器都停在挂起状态；它们持有的资源要到生成器对象被回收（或显式 `close()`）时才释放（第八章 §4）。

### 5. 异常阶段

若第 3 阶段第 6 步 `self.model(batch)` 抛出 `RuntimeError`：

1. 异常在 `model` 的帧中产生，沿 `f_back` 回到 `Runner.__call__` 的帧（第十章 §1）；
2. 经过 `with` 块：`InferenceContext.__exit__(RuntimeError, exc, tb)` 被调用，打印 `exit inference mode`，返回 `False`，异常继续传播（第九章 §1）；
3. `__call__` 的帧被加入 traceback 后销毁；
4. 到达调用方的 `try`：

   ```python
   try:
       output = runner(batch)
   except Exception:
       logger.exception("inference failed")
       raise
   ```

   记录完整 traceback 后原样重抛（第十章 §5）；再向上由 Worker 主循环决定重试、降级还是退出。

五个阶段对应的机制：

| 阶段 | 机制 | 章节 |
| :--- | :--- | :--- |
| 导入 | finder/loader、`sys.modules`、顶层代码执行、装饰器注册 | 四、七 |
| 创建 | `type.__call__`、`__new__`/`__init__`、`__setattr__` | 五 |
| 调用 | `__call__` 槽位、方法绑定、属性查找、`with` | 五、六、九 |
| 流式 | 生成器帧挂起、`yield from`、迭代协议 | 六、八 |
| 异常 | 沿帧传播、`__exit__` 参与、异常链与重抛 | 九、十 |

### 6. 从追踪到工程建议

上面的追踪也解释了为什么下面这些常见建议是对的：

- **减少导入副作用**。顶层代码在 `import` 时执行，时机由导入顺序决定。把"定义类、注册名字"留在顶层，把"加载模型、初始化 CUDA、建立连接"放进显式调用的函数。
- **入口文件要薄**。被当作脚本运行的文件不要包含会被别处导入的定义，否则会出现 `__main__`/模块名双重导入。
- **依赖方向单向**。`基础 → 核心 → 服务`，循环依赖用提取协议模块解决，不用局部导入掩盖。
- **隐式机制留给框架层**。`__getattr__`、元类、自定义 finder、复杂装饰器在业务代码里应该很少见；它们让"这行代码到底调了什么"不再能靠阅读得知。
- **资源清理走协议**。文件、锁、连接、CUDA 上下文、通信组都用 `with` 或 `try/finally`；生成器持有资源时要有确定性的关闭路径。
- **异常要么处理要么重抛**。`except Exception: log` 不是处理；捕获 `BaseException` 会吞掉终止信号。
- **给重要对象一个好的 `__repr__`**。日志里 `Request(id='r-1', batch_size=32, device='cuda:0')` 比五个字段各打一行更容易排查。
- **不确定就做实验**。`dis`、`inspect`、`sys.modules`、`__mro__`、`__closure__`、`gi_frame` 都是现成的观察工具。本文的所有输出都是这样得到的，而不是凭记忆写的。


## 十二、本文小结

本文沿着"一段代码的生命周期"讨论了 Python 运行时的九组机制。它们不是并列的特性清单，而是层层依赖的：

```text
执行模型      源码编译成 code object，函数对象封装它，调用时创建帧
    │
作用域与闭包   名称的归属在编译期决定；闭包 = 函数对象 + cell；cell 是共享的
    │
导入系统      import 是运行时动作：sys.modules → meta_path finder → loader → 执行顶层代码
              sys.path[0] 由启动方式决定；editable 安装修改的是查找路径
    │
类与对象模型   类由 type 创建；obj.attr 是一个固定算法：数据描述符 → 实例字典 → 非数据描述符 → __getattr__
              方法、classmethod、staticmethod、property 都只是描述符的 __get__ 返回值不同
    │
对象协议      语法交给类型上的特殊方法；特殊方法跳过实例字典
    │
装饰器        定义时执行一次的函数调用；靠闭包记参数，靠描述符协议对方法透明
    │
生成器        帧被挂起而非销毁；资源随生成器存活；break 不等于 close
    │
上下文管理器   __enter__/__exit__ 绑定到代码块；@contextmanager 把异常 throw 进 yield
    │
异常          沿帧链传播，经过每个 __exit__ 和 finally；只记录不重抛是反模式
```

回到开头那七行代码，现在每一行都有了答案：

| 开头的写法 | 背后的机制 | 在哪一节 |
|---|---|---|
| `import torch` 触发 `.so` 加载、算子注册 | `import` 是运行时动作：`PathFinder` 找到 `torch/_C.*.so`，`ExtensionFileLoader` `dlopen` 它并调 `PyInit__C`；顶层代码顺带执行注册 | 四 §3、§6 |
| `model(x)` 走 `__call__`，中间插入 hooks | 调用语法查的是**类型**上的 `__call__`；`nn.Module.__call__` = `_wrapped_call_impl` → pre-hooks → `forward` → hooks，所以 `model.forward(x)` 绕过全部 hook | 六 §1、§2 |
| `for batch in loader` | 迭代协议：`for` 先调 `loader.__iter__()` 拿一个新迭代器（每个 epoch 一个），再反复调它的 `__next__`；用生成器写迭代器时，"暂停 / 恢复"就是帧在 `yield` 处挂起、下一次 `next` 恢复 | 六 §3、八 §1 |
| `with torch.inference_mode()` | `with` 展开为 `__enter__` / `__exit__`；这类上下文管理器进入时切换一个线程局部状态、退出时恢复，异常也照样恢复 | 九 §1、§3 |
| `@register("cuda")` 注册表能否填上取决于谁导入了它 | 装饰器在**定义时**执行一次；定义所在的模块没被导入、或被以两个名字导入，注册就不会发生或发生两次 | 七 §5、四 §6 |
| `self.linear = nn.Linear(4, 4)` 登记到 `_modules` | `Module.__setattr__` 拦截赋值写进 `_modules`（不进 `__dict__`），读取时属性查找算法在类和实例字典都找不到，落到 `__getattr__` 从 `_modules` 取回 | 五 §3、§6 |
| `except Exception: log; raise` | 异常沿帧链传播，途经每个 `__exit__` 和 `finally`；只记录不重抛，Worker 会带着错误状态继续跑；裸 `raise` 保留原始 traceback | 十 §1、§5 |

再往外一层，读 AI-Infra 代码时常见的这些疑问，也都落在同一组机制上：

- 一个后端为什么"写了注册装饰器却找不到"——它所在的模块没被导入，或被以两个名字导入了；
- 一份代码为什么"`python -m` 能跑、`python script.py` 不能"——`sys.path[0]` 和 `__package__` 不同；
- 测试为什么"本地过、装完挂"——flat 布局让测试导入的是源码树而不是安装产物；
- `import torch` 为什么能加载几百 MB 的 C++ 库——`ExtensionFileLoader` 是导入系统的一个普通 loader；
- `nn.Module` 为什么能用 `self.linear` 访问一个不在 `__dict__` 里的子模块——`__setattr__` 拦截写、`__getattr__` 兜底读；
- `model(x)` 和 `model.forward(x)` 为什么不等价——`__call__` 在类型上查找，hooks 挂在那里；
- `super().__init__()` 为什么不一定调到"父类"——它调的是实例 MRO 中的下一个；
- 一个生成器持有的文件为什么一直没关——消费方 `break` 了但没 `close()`；
- 一个 Worker 为什么带着错误状态继续跑——异常被记录后没有重抛。

这是从"会写 Python"到"能读懂 Python 工程和 AI-Infra 框架"的第一步。后续六篇分别讨论类型系统、并发与异步、动态机制与插件架构、内存管理、测试与调试、工程化与交付，每一篇都会用到本文的某一组机制。

配套代码：本文验证各个结论用的小脚本（`dis` 与 code object、import 系统、描述符优先级、生成器与上下文管理器、自定义 `MetaPathFinder`、装饰器顺序）在 [ai-learning-labs/python-for-ai-infra/01-language-mechanisms](https://github.com/arganzheng/ai-learning-labs/tree/main/python-for-ai-infra/01-language-mechanisms)，只依赖标准库。


## 下一篇

[类型系统与数据契约设计](/python-type-system-and-data-contract-design.html)
