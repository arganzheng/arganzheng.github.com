---
layout: post
series: python-for-ai-infra
title: "Python 在 AI-Infra（01 下）：对象如何工作——对象模型、协议、装饰器与生成器"
subtitle: "How Objects Work — Object Model, Protocols, Decorators and Generators"
tags: [Python]
catalog: true
date: 2026-01-24 20:00:00 +0800
updated: 2026-11-16
---

[上篇](/python-execution-model-scopes-imports-and-exceptions.html)讲的是代码怎么跑：源码编译成 code object，调用创建帧，名称在帧里按 LEGB 解析，`import` 找到并执行模块，异常沿帧链传播。本篇讲**对象怎么工作**——上篇开头那七行里剩下的五行：

- **model(x)**：走的是 __call__，中间可能插入 hooks
- **for batch in loader**：迭代协议 + 生成器的暂停与恢复
- **with torch.inference_mode()**：上下文管理协议：进入时改状态、退出时恢复
- **@register("cuda")**：装饰器在模块导入时执行，注册表能否填上取决于谁导入了它
- **self.linear = nn.Linear(4, 4)**：__setattr__ 拦截赋值，把子模块登记到 _modules

这五行分别在第三、三 / 五、六、四、二章展开，第七章会把上下两篇的机制串在一起，逐步追踪开头那个 `Runner` 从导入到异常的完整生命周期。

本篇只回答一个问题：

> **`Runner(model)`、`runner(batch)`、`runner.stream(batch)` 这几行背后，对象是怎么被创建的、属性和方法是怎么被找到的、语法是怎么交给对象自己决定的？[^q0]**

答案分四层：类由 `type` 创建，`obj.attr` 是一个固定算法（描述符协议是它的可编程点）；`runner(batch)`、`for`、`with`、`[]` 这些语法查的是类型上的特殊方法；装饰器用上篇的闭包加本篇的描述符改写调用路径；生成器让上篇的帧挂起而不销毁。

## 一、总览

### 1. 例子与依赖关系

例子仍是上篇的 `runner.py`——一个用了装饰器注册表、上下文管理器、`__call__` 与生成器的极简推理组件：

```python title='runner.py：装饰器注册表、上下文管理器、__call__ 与生成器'
# runner.py
from contextlib import nullcontext

REGISTRY = {}


def registered(name):                        # 带参数的装饰器：registered("runner") 先被调用，返回下面的 decorator
    def decorator(cls):                      # 真正的装饰器：拿到被装饰的类 Runner，登记后原样返回
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

上篇解释了它被 `import` 时发生了什么（第 1 步）以及异常怎么传播（第 5 步）；本篇解释中间三步：`Runner(model)` 创建实例，`runner(batch)` 触发 `__call__` 并进入 `with`，`runner.stream(batch)` 创建生成器。

```mermaid
%% 图：Runner 从定义到调用：哪些发生在导入时，哪些发生在使用时
flowchart TB
    subgraph DEFINE["定义阶段：首次导入执行模块代码时"]
        BODY["<b>执行 class Runner 的类体</b><br/>def 创建函数对象，写进类命名空间"]
        CLS["<b>元类创建类对象</b><br/>默认由 type 处理方法与描述符"]
        REG["<b>应用 registered 装饰器</b><br/>登记类对象，再把结果绑定到 Runner"]
        BODY --> CLS --> REG
    end
    subgraph USE["使用阶段：每次创建或调用时"]
        OBJ["<b>Runner(model)</b><br/>__new__ 创建实例，__init__ 写入状态"]
        CALL["<b>runner(batch)</b><br/>类型上的 __call__ → with → model(batch)"]
        GEN["<b>runner.stream(batch)</b><br/>绑定方法，调用后只创建生成器"]
        NEXT["<b>next(g) / for</b><br/>恢复帧，运行到 yield，再次挂起"]
        OBJ --> CALL
        OBJ --> GEN --> NEXT
    end
    REG --> OBJ
```

图从**模块代码已经开始执行**的地方接上篇。源码和函数体的编译发生在此前；执行 `class`、应用装饰器、调用 `Runner(model)` 都不是重新编译。装饰器表达式 `registered("runner")` 在类体执行前求值，得到的装饰器在类对象创建后应用，第四章会把这两个时机分开。

先区分链路上的对象，再看它们如何协作。下表说的是运行中的对象；除了 code object 可写入 `.pyc`，它们不是磁盘上的另一份代码文件。

| 对象 | 什么时候产生 | 保存在哪里 | Java 对照 |
| :--- | :--- | :--- | :--- |
| 类对象 `Runner` | 执行类体后，由元类创建 | 进程内存；模块字典里的 `Runner`、注册表等引用它；类的 `__dict__` 提供属性映射的只读视图 | `Class<?>` 可反映类的运行时信息，但普通 Java 类不是执行 `class` 语句时创建的 |
| 实例 `runner` | 调用 `Runner(model)` 时 | 进程内存；普通实例的状态在 `__dict__`，使用 `__slots__` 的类也可放在槽位中 | `new Runner(model)` 创建的堆对象 |
| 函数 `Runner.__dict__["stream"]` | 类体执行到 `def stream` 时 | 类属性映射引用它；`__code__` 指向此前已编译的 code object | 类的方法有字节码，但不是 Python 这种可直接取出、替换的函数对象 |
| 绑定方法 `runner.stream` | 通过实例读取类上的函数属性时 | 临时的方法对象，持有 `__self__` 和 `__func__`；赋给变量后可继续存活 | `runner::stream` 的绑定方法引用可类比，但需要函数式接口作为目标类型 |
| 描述符，如 `property`、`Typed(int)` | 构造该对象并放入类属性时 | 描述符对象由类引用；它管理的值可存在实例字典、槽位或其他存储中 | getter/setter 只能类比用途；Java 没有同样的属性访问协议 |
| 生成器 `g` | 调用含 `yield` 的函数时 | 生成器对象保留暂停执行所需的帧状态，包括局部变量和指令位置 | 手写 `Iterator` 把进度存进字段；Java 没有对应的生成器函数语法 |

Table: 类、实例、函数、绑定方法、描述符与生成器的产生时机和保存位置

各章之间的依赖是单向的：第二章的方法绑定、`property` 依赖描述符协议，所以描述符放在第二章内部讲，而不是拖到对象协议之后；第四章装饰器同时依赖闭包（[上篇第三章](/python-execution-model-scopes-imports-and-exceptions.html#三作用域与闭包名称在哪里被解析)）和描述符（本篇第二章），因此排在两者之后；第五章生成器建立在[上篇第二章](/python-execution-model-scopes-imports-and-exceptions.html#二执行模型源码如何变成正在运行的代码)的帧之上；第六章 `contextlib` 的实现又建立在第五章的生成器之上。

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 类与对象模型：对象如何被创建和查找 | `type`、属性查找算法、描述符、方法绑定与三种内建描述符、`__getattr__`、`__new__` / `__init__`、MRO 与 `super`、Mixin 与组合 |
| 三 | 对象协议：语法背后的特殊方法 | `__call__`、迭代协议、`__getitem__`、真值、`__eq__` / `__hash__` |
| 四 | 装饰器：用闭包和描述符改写调用路径 | 基本机制、`wraps`、参数化、类装饰器与注册、与描述符的叠放顺序、异步 |
| 五 | 生成器与惰性执行 | 帧的挂起、惰性的成本、`yield from`、关闭与清理 |
| 六 | 上下文管理器 | `with` 的展开、`contextlib`、AI-Infra 中的用法、异步版本 |
| 七 | 一个推理组件的完整运行时追踪 | 按导入、创建、调用、流式、异常五个阶段追踪 `Runner`，并归纳工程建议 |
| 八 | 本文小结 |  |
| 九 | 自测 | 5 道题 |

Table: 本文的章节安排

## 二、类与对象模型：对象如何被创建和查找

模块被导入、顶层代码执行时，`class Runner:` 语句创建了一个类对象。接下来 `Runner(model)` 创建实例，`runner.model`、`runner.stream` 读取属性。这一章回答的核心问题只有一个：**`obj.attr` 到底做了什么**。方法绑定、`property`、`classmethod`、`__getattr__`、`nn.Module` 把子模块藏在 `_modules` 里却能用 `self.linear` 访问——全部是这一个算法的不同分支。

### 1. 类也是对象：`type` 创建类

`class` 是运行时语句。执行它时，解释器：

1. 新建一个命名空间字典，在其中执行类体（所以类体里的 `def` 创建的是普通函数对象，并被放进这个字典）；
2. 调用元类（默认是 `type`）：`type(name, bases, namespace)`，得到类对象；
3. 把名称绑定到类对象。

```python title='类也是对象：type(Runner)、__mro__ 与只读的 __dict__'
class Runner:
    def run(self, batch):
        return batch


print(type(Runner))                       # <class 'type'>
print(Runner.__mro__)                     # (<class 'Runner'>, <class 'object'>)
print(type(Runner.__dict__))              # <class 'mappingproxy'>   类属性字典的只读视图
print(type(Runner.__dict__["run"]))       # <class 'function'>       类里存的是普通函数
```

不涉及自定义元类等扩展时，可以手工完成第 2 步，得到具有相同方法的类：

```python title='手工调用 type(name, bases, namespace) 造出同样的类'
Runner = type("Runner", (object,), {"run": lambda self, batch: batch})
```

两个事实值得记住：**类的 `__dict__` 里存的方法就是普通函数对象**（后面 §5 解释它如何变成"方法"）；**类由 `type` 创建**，所以替换 `type` 为自定义元类就能干预类的创建——`nn.Module` 没有用元类，但许多 ORM、配置框架和 `abc.ABC` 用了。元类的用法在本系列第四篇展开，这里只需知道它是 `class` 语句的第 2 步。

> [!NOTE] 与 Java 对照：类对象与实例
> Java 的 `Runner.class` 得到 `Class<Runner>`，`new Runner(...)` 创建实例；Python 的 `Runner` 本身就是可调用的类对象，默认由 `type` 创建。两边都有类的运行时信息，但 Python 的类体是会执行的代码：里面的赋值、`def` 和装饰器会在执行类定义时产生对象，而不是每次创建实例时重做。

### 2. 实例属性与类属性

实例有自己的 `__dict__`，类有类的 `__dict__`：

```python title='实例 __dict__ 与类 __dict__ 各是各的'
class Demo:
    value = 10


obj = Demo()
print(obj.__dict__)                     # {}            实例字典是空的，value 在类上
obj.value = 20
print(obj.__dict__)                     # {'value': 20}
print(Demo.__dict__["value"], Demo.value, obj.value)   # 10 10 20
```

这里的 `value` 是普通类属性，所以赋值 `obj.value = 20` 写的是实例字典，类属性不受影响；遇到 `property` 等数据描述符或自定义 `__setattr__` 时，写入会被接管。反过来，可变的类属性会被所有实例共享——`class Bad: items = []` 之后 `a.items.append(1)` 会让 `b.items` 也变成 `[1]`，因为没有任何赋值发生在实例上，`a.items` 读到的是类字典里的同一个列表。需要每实例独立的可变状态，在 `__init__` 里创建。

把前一个例子里的引用画出来，`obj.value = 20` 并不是把类里的 `10` 改成了 `20`：

```mermaid
%% 图：普通同名属性分别存放在实例字典和类字典
flowchart TB
    O["<b>obj 实例</b>"] -- "__dict__" --> I["<b>实例字典</b><br/>value → 20"]
    O -- "type(obj)" --> C["<b>Demo 类对象</b>"]
    C -- "__dict__ 只读视图" --> D["<b>类属性映射</b><br/>value → 10"]
    R["<b>读取 obj.value</b><br/>普通类属性没有优先权"] -- "先命中实例字典" --> I
    R -. "实例里没有 value 时才取" .-> D
```

> [!NOTE] 与 Java 对照：类属性不等于 static 字段
> Java 用 `static` 区分类字段与实例字段；Python 按数据描述符、实例字典、其余类属性的优先级读取。`Demo.value` 改类属性，`obj.value = 20` 在这个例子里却会新建同名实例属性，并不修改 `Demo.value`。可变的类属性又会被实例共享，不能仅凭写法 `obj.value` 判断数据属于谁。

### 3. 属性查找算法：`object.__getattribute__`

先补一个马上要用到的词：**MRO（Method Resolution Order，方法解析顺序）就是查找类属性时依次检查哪些类**，不只用于找方法。单继承时很好理解：

```python title='先看单继承的 MRO：当前类、父类、object'
class Base:
    value = 10


class Child(Base):
    pass


print([cls.__name__ for cls in Child.__mro__])  # ['Child', 'Base', 'object']
print(Child().value)                           # 10：Child 没有，再找 Base
```

这里先按 `Child → Base → object` 的顺序逐个查类的 `__dict__` 就够了；多继承时如何排出这个顺序，以及 `super()` 怎么沿它继续找，放在本章 §8 展开。**MRO 决定类之间的先后，下面的算法决定类属性和实例属性谁优先。**

现在可以给出核心算法。`obj.attr` 默认调用 `type(obj).__getattribute__(obj, "attr")`，对普通对象它的逻辑是：

```python title='object.__getattribute__ 的等价伪代码：六步查找顺序'
MISSING = object()                              # 与合法的类属性值 None 区分


def __getattribute__(obj, name):                 # 默认查找算法的简化伪代码
    cls = type(obj)
    cls_attr = lookup_in_mro(cls, name, MISSING) # ① 沿 cls.__mro__ 查 __dict__，未找到才返回 MISSING
    descr_get = getattr(type(cls_attr), "__get__", None)

    if descr_get is not None and is_data_descriptor(cls_attr):   # ② 有读取方法的数据描述符优先
        return descr_get(cls_attr, obj, cls)

    instance_dict = get_instance_dict(obj)      # ③ 不经过 obj.__getattribute__；无字典时视为空
    if name in instance_dict:
        return instance_dict[name]

    if cls_attr is not MISSING:
        if descr_get is not None:              # ④ 非数据描述符：函数、classmethod、staticmethod
            return descr_get(cls_attr, obj, cls)
        return cls_attr                          # ⑤ 普通类属性

    raise AttributeError(name)                   # ⑥ 都没有 → 抛出；若类定义了 __getattr__，解释器接着调它
```

伪代码里的 ① 只是"先在类上找一遍"，真正决定结果的是后面几次判断。描述符方法要查**描述符对象的类型**，而不是在它自己的实例字典里找。下面流程假设描述符正常返回：类上找到的东西要**先看它有没有 `__get__` 且是数据描述符**，再决定实例字典有没有机会；查找抛出 `AttributeError` 后，点号访问才会尝试 `__getattr__`：

```mermaid
%% 图：obj.attr 的属性查找判断流程
flowchart TB
    S["obj.attr"] --> M["① 沿 type(obj).__mro__ 查各类的 __dict__<br/>得到 cls_attr（可能没有）"]
    M --> D{"有 __get__，且是数据描述符？<br/>类型上有 __set__ 或 __delete__"}
    D -- "是" --> R2["② 返回 cls_attr.__get__(obj, cls)<br/>例：property"]
    D -- "否" --> I{"name 在 obj.__dict__ 里？"}
    I -- "是" --> R3["③ 返回实例字典里的值"]
    I -- "否" --> F{"类上找到了 cls_attr？"}
    F -- "否" --> E["⑥ AttributeError"]
    F -- "是" --> G{"cls_attr 有 __get__？"}
    G -- "是" --> R4["④ 返回 cls_attr.__get__(obj, cls)<br/>例：函数 → 绑定方法"]
    G -- "否" --> R5["⑤ 返回类属性本身"]
    E --> GA{"类定义了 __getattr__？"}
    GA -- "是" --> R6["调用 __getattr__(obj, name)"]
    GA -- "否" --> R7["AttributeError 抛给调用方"]
```

用一个实验同时放进三种类属性，再给实例字典塞入同名值，看哪一个生效：

```python title='一个实验比较 property、函数和普通类属性的查找优先级'
class LookupDemo:
    count = 10

    @property
    def value(self):
        return "from property"

    def run(self):
        return "from class method"


obj = LookupDemo()
obj.__dict__.update(
    value="from instance dict",
    run=lambda: "from instance function",
    count=20,
)
print(obj.value)             # from property：数据描述符优先，② 先于 ③
print(obj.run())             # from instance function：实例优先，③ 先于 ④
print(obj.count)             # 20：实例优先于普通类属性，③ 先于 ⑤
print(LookupDemo.count)      # 10：类属性没有改变

del obj.__dict__["run"]
del obj.__dict__["count"]
print(obj.run())             # from class method：实例没有，才绑定类上的函数
print(obj.count)             # 10：实例没有，才返回普通类属性
```

这里刻意直接写实例字典，让三种同名属性同时存在：`property` 即使没有 setter，类型上仍有 `__set__`，所以读时优先；函数是非数据描述符；`count` 对应的整数则根本不是描述符。后两种都允许实例里的值遮住类上的值，删除实例值后才重新露出类属性。

这个顺序设计的意图：`property` 这类需要**控制读写**的东西必须不能被实例字典绕过，所以放最前；方法这类**只提供默认行为**的东西允许被实例覆盖（monkey patch 单个对象的方法就靠这一点），所以放实例字典之后。

### 4. 描述符协议：属性查找的可编程点

算法里的 ② 和 ④ 都在调用一个东西的 `__get__`。凡是定义了 `__get__`、`__set__` 或 `__delete__` 之一的对象，放在**类属性**位置上，就是描述符：

- 只有 `__get__`：**非数据描述符**，可被实例字典覆盖；
- 有 `__set__` 或 `__delete__`：**数据描述符**，优先于实例字典。

一个做类型校验的描述符：

```python title='一个做类型校验的描述符 Typed：__set_name__、__get__、__set__'
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

`__slots__` 是描述符的一个内建应用：`class S: __slots__ = ("a",)` 会让 `type` 为每个槽位在类上创建一个 `member_descriptor`（数据描述符），直接读写实例的固定内存偏移。若基类没有提供实例字典，且槽位里没有 `"__dict__"`，这种实例就没有 `__dict__`；子类不声明 `__slots__` 时又会获得字典。第五篇讨论内存时会回到它。

> [!NOTE] 与 Java 对照：描述符接管的是访问语法
> Java 的字段访问和 getter/setter 调用是两种写法；Python 的描述符能让 `config.batch_size`、`config.batch_size = 32` 分别执行 `__get__`、`__set__`。类上保存的是管理访问的 `Typed` 对象，本例中的整数值仍在各个实例字典里，不是所有实例共用一个值。

### 5. 方法绑定、`classmethod`、`staticmethod`、`property`：描述符的四种形态

有了 §3 的算法和 §4 的协议，"方法"就不再需要单独解释：**函数对象实现了 `__get__`**，所以它是非数据描述符。

```python title='函数是非数据描述符：通过类访问得函数，通过实例访问得绑定方法'
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

```mermaid
%% 图：同一函数通过两个实例读取时，绑定方法分别持有谁
flowchart TB
    F["<b>类字典中的 run 函数</b><br/>Runner3.__dict__#91;'run'#93;，只有一份"]
    A["<b>a = Runner3()</b><br/>读取 a.run，触发函数的 __get__"]
    B["<b>b = Runner3()</b><br/>读取 b.run，触发函数的 __get__"]
    MA["<b>绑定方法 a.run</b><br/>__self__ → a，__func__ → run"]
    MB["<b>绑定方法 b.run</b><br/>__self__ → b，__func__ → run"]
    A --> MA
    B --> MB
    MA -- "__func__" --> F
    MB -- "__func__" --> F
```

不是把函数复制进每个实例，而是在读取属性时临时配上接收者：`a.run(batch)` 相当于调用 `Runner3.run(a, batch)`；`b.run(batch)` 则传入 `b`。这也解释了为什么直接把函数写进**实例字典**不会自动绑定：它不经过类上描述符的分支。

> [!NOTE] 与 Java 对照：实例方法与绑定方法
> 两边的实例方法调用都会携带接收者；Python 把它显式写成第一个参数 `self`，并把“函数 + 接收者”暴露成可读取的绑定方法对象。Java 的 `obj::method` 也能记住接收者，但它是在函数式接口的目标类型下创建的方法引用，不是每次读取 `obj.method` 的结果。

`classmethod`、`staticmethod`、`property` 是三个内建的描述符类型，区别只在 `__get__` 返回什么，以及是否有 `__set__`：

```python title='classmethod、staticmethod、property 三个内建描述符的 __get__ 返回值'
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

| 类属性上的对象 | `__get__(obj, cls)` 返回 | 数据描述符？ | 效果 | Java 对照 |
| :--- | :--- | :--- | :--- | :--- |
| 函数 | `obj` 为 `None` 时返回函数；否则返回绑定 `obj` 的方法 | 否 | 实例方法 | 实例方法；Python 显式声明 `self` |
| `classmethod` | 无论通过类还是实例访问，都绑定到访问所用的类 | 否 | 第一个参数是类 | 静态工厂可类比用途，但不会自动得到调用处的子类 |
| `staticmethod` | 原函数本身，不绑定任何东西 | 否 | 普通函数挂在类命名空间 | `static` 方法可类比用途；Python 仍按运行时属性查找规则取值 |
| `property` | 实例访问返回 `fget(obj)`；类访问返回 property 本身 | **是**（有 `__set__`，未提供 setter 时抛 `AttributeError`） | 计算属性，且不可被实例字典覆盖 | getter/setter 或 record 访问器；Java 仍要调用方法 |

Table: 函数、classmethod、staticmethod、property 的 __get__ 行为

从这张表能直接推出三者的用法：

- **`classmethod` 的价值在于 `cls` 是"实际被调用的那个类"**。`Sub.c()` 和 `Sub().c()` 都返回 `Sub`，不是 `K`。这让它成为替代构造函数的标准写法：

  ```python title='classmethod 作替代构造函数：cls 是实际被调用的那个类'
  class Runner:
      def __init__(self, model, device="cpu"):
          self.model = model
          self.device = device

      @classmethod
      def from_config(cls, config):
          return cls(model=config["model"], device=config.get("device", "cpu"))


  class GPURunner(Runner): ...


  cfg = {"model": "demo", "device": "cuda"}
  print(type(GPURunner.from_config(cfg)).__name__)  # GPURunner
  ```

  对象尚未存在，所以不能是实例方法；需要构造"当前类或子类"，所以不该写死 `Runner(...)`——`cls(...)` 自动跟随继承。

- **`staticmethod` 只是命名空间归属**。既不需要实例也不需要类的函数，放在类里是为了表达"逻辑上属于它"；很多时候放模块级更简单。
- **`property` 是数据描述符**，所以 §3 的实验中它赢过实例字典。它适合"看起来像字段、实际需要计算或校验"的场景；不适合隐藏昂贵操作——调用方看到 `runner.device` 不会预期它去查询硬件。

> [!NOTE] 与 Java 对照：classmethod 与静态工厂
> Java 的 `GPURunner.fromConfig(cfg)` 若继承的是 `Runner` 的静态工厂，方法体里的 `new Runner(...)` 不会自动变成 `new GPURunner(...)`。Python 的 `GPURunner.from_config(cfg)` 则自动把 `GPURunner` 绑定给 `cls`，所以 `cls(...)` 创建子类实例。Java 要实现类似工厂，可以显式传构造器 `GPURunner::new`；区别在于由调用方传入，而不是由属性查找注入。

### 6. `__getattr__` 与 `__getattribute__`：查找失败后的钩子

§3 算法的第 ⑥ 步：`__getattribute__` 抛出 `AttributeError` 之后，如果类定义了 `__getattr__`，解释器会调用它作为最后手段。

```python title='__getattr__ 只在查找失败后被调用'
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

```python title='__getattr__ 掩盖 property 内部的 AttributeError'
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

```python title='torch.nn.Module 的 __setattr__ 骨架：参数与子模块分开存放'
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

注意，`linear` 不直接在实例字典里，不等于三个注册字典也不在：`self.__dict__["_modules"]` 持有的就是子模块注册表。

```mermaid
%% 图：nn.Module 子模块属性的写入、存储与读取路径
flowchart TB
    W["<b>self.linear = nn.Linear(4, 4)</b>"] --> SET["__setattr__ 识别 Module"]
    SET -- "登记 linear" --> STORE["<b>self.__dict__['_modules']</b><br/>linear → Linear 子模块"]
    R["<b>读取 self.linear</b>"] --> LOOK["默认属性查找<br/>类和实例字典没有 linear"]
    LOOK -- "AttributeError 后兜底" --> GET["__getattr__('linear')"]
    GET -- "查注册表" --> STORE
    STORE --> RESULT["返回同一个 Linear 子模块"]
```

模块级也有同样的钩子（PEP 562）：在模块顶层定义 `__getattr__(name)`，访问不存在的模块属性时会被调用。`torch/__init__.py` 用它做两件事——对已废弃的名字发出 `DeprecationWarning` 后返回替代品，以及对少数子模块做惰性导入（第一次访问时才 `importlib.import_module`）。

### 7. 对象创建：`__new__`、`__init__` 与 `type.__call__`

`Runner(model)` 是对**类对象**的调用；这里的元类是默认的 `type`，因此由 `type.__call__` 组织创建过程。省略底层槽位调用与返回值校验后，可以概括为：

```python title='type.__call__ 的简化伪代码：__new__ 再 __init__'
def __call__(cls, *args, **kwargs):          # 说明创建顺序，不是底层实现
    obj = cls.__new__(cls, *args, **kwargs)  # 分配并返回实例
    if isinstance(obj, cls):               # __new__ 可返回 cls 的子类实例
        type(obj).__init__(obj, *args, **kwargs)  # 按返回对象的类型初始化；必须返回 None
    return obj
```

`__new__` 被特殊处理为静态方法，不需要显式写 `@staticmethod`，负责“造出对象”；`__init__` 负责“填充对象”。重写 `__new__` 的场景很少：不可变类型（`int`、`tuple`、`str`）的子类化必须在 `__new__` 里改值，因为 `__init__` 拿到的已经是造好的不可变对象；单例、实例缓存、根据参数返回不同子类的工厂类也在这里做。其他情况只写 `__init__`。

如果 `__new__` 返回的不是 `cls` 的实例，`__init__` 不会被调用——这是"工厂类"模式的机制基础。

```mermaid
%% 图：类调用与实例调用的两条不同路径
flowchart TB
    C["<b>Runner(model)</b><br/>调用类对象"] --> T["元类 type.__call__"]
    T --> N["Runner.__new__<br/>分配并返回对象 obj"]
    N --> CHECK{"obj 是 Runner 的实例？"}
    CHECK -- "是" --> INIT["按 obj 的实际类型调用 __init__<br/>初始化后返回 obj"]
    CHECK -- "否" --> RET["直接返回 obj<br/>不调用 __init__"]
    O["<b>runner(batch)</b><br/>调用已有实例"] --> CALL["类型上的 Runner.__call__<br/>执行业务逻辑，不是创建 Runner"]
```

两条路径里都有 `__call__`，但接收者不同：前者由元类负责造实例，后者由实例的类型负责执行模型调用。

> [!NOTE] 与 Java 对照：分配与初始化
> Java 的 `new Runner(...)` 把分配对象和执行构造器连在一起；Python 在协议上分成 `__new__` 返回对象、`__init__` 初始化对象。因此 `__init__` 不是分配内存的构造步骤，不能用 `return another_object` 换掉实例；这类替换发生在 `__new__` 或工厂函数里。

### 8. 继承、MRO 与协作式 `super()`

属性查找算法第 ① 步"沿 `cls.__mro__` 查找"，MRO（Method Resolution Order）就是这个顺序。单继承时它就是从子类到 `object` 的链；多继承时由 C3 线性化算法计算，保证子类先于父类、多个父类之间保持声明顺序：

```python title='多继承的 MRO：C(A, B) 找到 A.run 就停'
class A:
    def run(self): return ["A"]
class B:
    def run(self): return ["B"]
class C(A, B): pass


print([k.__name__ for k in C.__mro__], C().run())    # ['C', 'A', 'B', 'object'] ['A']
```

`C().run()` 找到 `A.run` 就停，**不会**自动接着调 `B.run`。想让链上的每一层都执行，每一层必须自己调 `super()`：

```python title='协作式 super()：链上每一层都自己调 super().run()'
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

把这次调用按时间画出来：调用沿 MRO 往下走到 `Base`，返回值再沿原路往回传，每一层在**拿到下一层的结果之后**才 `append`，所以最先追加的是离 `Base` 最近的 `Metrics`：

```mermaid
%% 图：协作式 super() 的调用链：沿 MRO 向下调用，结果向上返回
flowchart TB
    U["调用方：Run().run()"] -- "Run 没有 run，按 MRO 找到" --> L["<b>Logging.run</b><br/>super().run()：MRO 里 Logging 之后是 Metrics"]
    L -- "调用" --> M["<b>Metrics.run</b><br/>super().run()：MRO 里 Metrics 之后是 Base"]
    M -- "调用" --> B["<b>Base.run</b><br/>返回 #91;'base'#93;"]
    B -. "返回 #91;'base'#93;" .-> M2["Metrics.run 追加 'metrics'"]
    M2 -. "返回 #91;'base', 'metrics'#93;" .-> L2["Logging.run 追加 'logging'"]
    L2 -. "返回 #91;'base', 'metrics', 'logging'#93;" .-> U2["调用方拿到结果"]
```

`Logging.run` 里的 `super().run()` 调到的是 `Metrics.run`，不是 `Base.run`——`super()` 的含义是"**在实例的 MRO 中，当前类之后的下一个**"，它取决于实例的类型，而不是写代码时看到的父类。`Logging` 类的作者并不知道 `Metrics` 的存在，链条却能接上，这就是协作式多继承。

零参数 `super()` 之所以知道"当前类"是谁，是因为编译器在使用了 `super()` 的方法里注入了一个名为 `__class__` 的闭包变量（可以用 `Child.run.__code__.co_freevars` 看到 `('__class__',)`）——[上篇第三章](/python-execution-model-scopes-imports-and-exceptions.html#三作用域与闭包名称在哪里被解析)的 cell 机制在这里又出现了一次。

协作式多继承的约定：每一层都调 `super()`（包括 `__init__`）；同名方法签名兼容，通常用 `*args, **kwargs` 透传；不假设 `super()` 指向某个具体类。任何一层漏掉 `super()`，它之后的所有层都会被跳过。声明顺序不一致时 C3 会直接拒绝：`class X(A, C)` 抛出 `TypeError: Cannot create a consistent method resolution order (MRO) for bases A, C`，因为 `C` 已经要求 `A` 在自己之后。

> [!NOTE] 与 Java 对照：super 不是固定的父类名称
> Java 单继承加接口；`super.method()` 从当前类的直接父类一侧查找，不会沿多个类组成的协作链继续分派。两个无继承关系接口的同名 default 方法冲突时，需要实现类显式选择。Python 的 `super()` 则按实际实例的 MRO，从当前类之后继续查找：上面的 `Logging` 没有继承 `Metrics`，却能经由 `super().run()` 调到它。

### 9. Mixin 与组合

Mixin 是协作式多继承的一种受限用法：一个只提供横向能力（日志、指标、序列化）、不代表完整实体、不持有业务依赖的类，被放在基类列表前部：

```python title='LoggingMixin 放在基类列表前部'
class LoggingMixin:
    def log(self, message):
        print(f"[LOG] {message}")


class Runner(LoggingMixin, BaseRunner):
    def run(self):
        self.log("start")
```

Mixin 没有绕开 MRO 的复杂性，只是把它限制在一个容易验证的范围内。约束是：职责单一；对宿主类的要求明确；`__init__` 遵守协作式 `super()`；用 `__mro__` 和测试固定实际顺序。

而 `model`、`scheduler`、`tokenizer` 这类**业务依赖**，应该用组合——持有而不是继承：

```python title='业务依赖用组合：InferenceRunner 持有 model、scheduler、tokenizer'
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
| 给单个函数加日志、重试、tracing | 装饰器（第四章） |
| 管理业务依赖 | 组合 |

Table: 继承与组合的选择规则

## 三、对象协议：语法背后的特殊方法

第二章解释了 `obj.attr`。但 `runner(batch)`、`for x in loader`、`batch[0]`、`if tensor:`、`with ctx:` 这些**不是**属性访问的语法，Python 也是交给对象自己决定的——通过一组以双下划线命名的特殊方法。这一章讨论其中最常见的几个；`with` 和 `yield` 分别留给第六、五章。

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
| `obj.attr` | `__getattribute__`、`__getattr__`（第二章） |

Table: 语法到特殊方法的映射

有一个规则与第二章的属性查找不同：**特殊方法由解释器直接在类型上查找，跳过实例字典**。

```python title='特殊方法在类型上查找：实例上的 __call__ 被跳过'
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

> [!NOTE] 与 Java 对照：可调用对象
> Java 可以用函数式接口实例表达行为，但仍通过 `.apply()`、`.call()` 等方法调用，没有让任意实例支持 `model(x)` 的运算符协议。在调用前后加 hook 也不一定要 AOP：显式包装一层方法就能做到；Python 的 `__call__` 让这层包装仍可使用函数调用语法。

### 3. 迭代协议：可迭代对象与迭代器

`for item in x` 展开为：

```python title='for 循环的展开：iter() 与 next() 直到 StopIteration'
it = iter(x)               # 调 x.__iter__()，得到迭代器
while True:
    try:
        item = next(it)    # 调 it.__next__()
    except StopIteration:
        break
    ...
```

**可迭代对象**（iterable）实现 `__iter__` 并返回一个迭代器；**迭代器**（iterator）实现 `__next__`，并且自己的 `__iter__` 返回自己。两者常被混为一谈，但差别决定了能否重复遍历：

```python title='可迭代对象与迭代器的差别：列表可反复遍历，CountDown 只能一次'
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

`torch.utils.data.DataLoader` 是可迭代对象——每个 epoch 的 `for batch in loader` 都重新调用 `iter(loader)`，开始一轮迭代。默认情况下会创建新迭代器，多 worker 模式也会重建 worker；但 `persistent_workers=True` 且 `num_workers > 0` 时，会复用迭代器及 worker，并重置本轮迭代状态。不能因此笼统地说“每次都创建新对象”；应在每个 epoch 重新向 loader 请求迭代，而不是继续对上一轮已经耗尽的迭代器调用 `next()`。

> [!NOTE] 与 Java 对照：Iterable 与 Iterator
> Java 的 `Iterable<T>.iterator()`、`Iterator<T>.hasNext()/next()` 与这里的两层角色相近；Python 用 `iter()`、`next()` 和 `StopIteration`，没有单独的 `hasNext()`。Java 要声明实现接口，Python 的运行时协议不要求继承特定基类。能否重复遍历，取决于 `iter(x)` 如何提供或重置迭代状态，不是只看有没有 `__iter__`。

还有一条历史兼容规则：没有 `__iter__` 但有 `__getitem__` 的对象也可迭代——解释器从 0 开始调 `__getitem__` 直到 `IndexError`。`Dataset` 子类只实现 `__getitem__` 和 `__len__` 就能被 `for` 遍历，靠的就是它。

### 4. `__getitem__` 与容器协议

```python title='只实现 __getitem__ 的 Batch：索引与切片都原样传入'
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

```python title='只定义 __eq__ 会让 __hash__ 变成 None'
class User:
    def __init__(self, uid): self.uid = uid
    def __eq__(self, other): return isinstance(other, User) and self.uid == other.uid


print(User.__hash__)     # None
{User(1)}                # TypeError: unhashable type: 'User'
```

需要哈希就显式实现 `__hash__`，并且只使用参与 `__eq__` 且在对象生命周期内不变的字段。`@dataclass(frozen=True)` 会自动生成一致的两者；可变的 dataclass 默认也是不可哈希的，原因相同。

> [!NOTE] 与 Java 对照：equals / hashCode 契约
> Java 的 `equals` / `hashCode` 也要求相等对象具有相同哈希值，`HashMap`、`HashSet` 都依赖它。不同的是 Python 在类只重写 `__eq__` 时会自动禁用继承来的默认哈希；Java 不会替开发者禁用 `hashCode()`，需要自己保证两者一致。Python 的 `==` 对应值比较，`is` 对应身份比较，不要把它们与 Java 引用上的 `==` 混淆。

## 四、装饰器：用闭包和描述符改写调用路径

第一章的 `@registered("runner")` 让 `Runner` 在被定义的同时进入注册表。装饰器的全部机制在前面已经准备好了：函数是对象（[上篇第二章](/python-execution-model-scopes-imports-and-exceptions.html#二执行模型源码如何变成正在运行的代码)），闭包能记住参数（[上篇第三章](/python-execution-model-scopes-imports-and-exceptions.html#三作用域与闭包名称在哪里被解析)），类属性上的函数是描述符（第二章）。这一章把它们组装起来。

### 1. 基本机制

```python title='最简单的装饰器 log_call：wrapper 包住原函数'
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

```python title='@log_call 等价于定义后立刻重新绑定'
def predict(x): ...
predict = log_call(predict)
```

所以：每次执行到这条 `def` 时，装饰器都会被调用；如果定义在模块顶层，通常就是首次导入执行模块代码时，而不是每次调用 `predict` 时。此后名称 `predict` 指向 `wrapper`，原函数由 `wrapper` 的闭包 cell 保留；若没有其他引用，就不再有独立名称指向它。多个装饰器从下往上应用：

```python title='多个装饰器从下往上应用'
@outer
@inner
def run(): ...
# run = outer(inner(run))     调用时先进 outer 的 wrapper，再进 inner 的
```

> [!NOTE] 与 Java 对照：装饰器不是注解
> Java 的 `@Transactional`、`@Retryable` 是元数据，调用拦截来自读取它们的框架，例如 Spring 创建的代理；Python 的 `@log_call` 本身就是一次调用和名称重新绑定，不依赖扫描或代理框架。读代码时分别追踪“谁读取注解”和“装饰器返回了谁”，不能只看两个语言里都有 `@`。

### 2. 装饰器与闭包的关系

`wrapper` 能在 `log_call` 返回之后仍然找到 `func`，靠的是[上篇第三章](/python-execution-model-scopes-imports-and-exceptions.html#三作用域与闭包名称在哪里被解析)的 cell：`wrapper.__closure__[0].cell_contents is predict_original`。闭包是实现装饰器最常见的方式，但不是必需的——任何"接收可调用对象、返回可调用对象"的东西都是装饰器，包括类：

```python title='用类实现装饰器：__init__ 收函数、__call__ 转发'
class LogCall:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("calling", self.func.__name__)
        return self.func(*args, **kwargs)


@LogCall
def predict(x): ...
```

状态少用闭包，状态多、需要暴露方法（如 `cache.clear()`）用类——与[上篇第三章](/python-execution-model-scopes-imports-and-exceptions.html#三作用域与闭包名称在哪里被解析) §5 的取舍一致。注意类实现的装饰器用在**方法**上时有一个坑，见 §6。

### 3. `functools.wraps` 与 `__wrapped__`

`wrapper` 是一个新函数对象，它的 `__name__`、`__doc__`、`__module__`、`__qualname__`、`__annotations__` 都是自己的。不处理的话，`predict.__name__` 变成 `'wrapper'`，日志、文档、`pickle`、测试框架的参数化 ID 全部受影响。`functools.wraps` 把这些元数据从原函数复制过来，并额外设置 `wrapper.__wrapped__ = func`：

```python title='functools.wraps 复制元数据并设置 __wrapped__'
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

```python title='带参数的装饰器 retry：三层函数'
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

| 层 | 函数 | 执行时机 | 返回 / 保存在哪里 |
|---|---|---|---|
| 配置层 | `retry(max_attempts)` | 执行被装饰的定义时，先求值装饰器表达式 | 返回 `decorator`；参数经闭包 cell 保留 |
| 装饰层 | `decorator(func)` | 函数对象创建后，名称最终绑定前 | 返回 `wrapper`；闭包保留 `func` 和重试参数 |
| 运行层 | `wrapper(*args, **kwargs)` | 每次调用装饰后的函数时 | 新建本次调用帧，返回 `func` 的结果 |

Table: retry 的三层：配置层、装饰层、运行层

`max_attempts` 和 `exceptions` 经由两层闭包到达 `wrapper`。重试装饰器应只捕获明确可重试的异常，让其余异常原样传播——`except Exception` 会把断言失败、类型错误和 CUDA OOM 一起重试。

```mermaid
%% 图：参数化装饰器的定义阶段与调用阶段
flowchart TB
    subgraph DEF["定义阶段：执行这条 def 时"]
        EXPR["<b>求值 retry(3)</b><br/>返回 decorator，记住 max_attempts = 3"]
        FUNC["<b>创建原函数对象 f</b><br/>函数体此时不运行"]
        WRAP["<b>应用 decorator(f)</b><br/>返回 wrapper，把名称 predict 绑定到它"]
        EXPR --> FUNC --> WRAP
    end
    subgraph RUN["调用阶段：执行 predict(5) 时"]
        WC["<b>wrapper 的执行帧</b><br/>从闭包取 func、max_attempts、exceptions"]
        FC["<b>原函数 f 的执行帧</b><br/>计算结果；超时则由 wrapper 决定重试"]
        WC --> FC
    end
    WRAP -- "每次调用" --> WC
```

定义阶段产出可复用的包装函数和闭包；调用阶段才创建本次的 `args`、`kwargs`、`attempt` 等局部状态。若定义写在另一个函数内部，每次运行到该定义都会重新装饰，并不是整个进程只装饰一次。

### 5. 装饰类：注册表

先说清装饰器**接收什么**：紧贴 `def` 的装饰器收到函数对象，紧贴 `class` 的收到类对象；`async def` 也产生函数对象。多层叠放时，外层收到的是**内层装饰器的返回值**，不一定仍是函数（§6 的 `property` / `staticmethod` 就会改变它）。装饰器语法只能写在函数或类定义前，但其**返回值**可以是任何对象：`@property` 返回描述符；`@functools.lru_cache` 返回带 `cache_info()` 的可调用对象；默认的 `@dataclass` 原地改造并返回原类，`slots=True` 则返回新类。

装饰器的参数可以是类，返回原类不变、只做登记，就是第一章的 `registered`：

```python title='装饰类做注册：registered 返回原类只做登记'
def registered(name):
    def decorator(cls):
        if name in REGISTRY:
            raise ValueError(f"duplicate registration: {name}")
        REGISTRY[name] = cls
        return cls
    return decorator
```

由 §1 的时机规则和[上篇第四章](/python-execution-model-scopes-imports-and-exceptions.html#四模块与导入系统代码如何被加载) §6 的导入规则可推出它的行为：注册发生在类定义执行时，模块顶层的类通常就是在导入时注册；模块没被导入则注册不会发生。若同一文件以两个模块名执行，可能向**同一个注册表**重复登记并报错，也可能各自创建一份注册表，导致查找的一方看不到另一方的注册。重复注册抛异常而不是静默覆盖，能让共享注册表里的重复定义尽早暴露，但不能替代对模块身份的检查。

### 6. 装饰方法时的叠放顺序：与描述符的交互

装饰器应用在**类体执行时**，此时 `def` 产生的还是普通函数（第二章 §1），描述符协议要等到属性访问时才起作用。这决定了几件事：

```python title='@classmethod 与 @log_call 的叠放顺序'
class Runner:
    @classmethod
    @log_call                  # 先应用：包装的是普通函数，wrapper 还是普通函数
    def create(cls): ...       # 再应用 classmethod：把 wrapper 包成 classmethod 对象
```

- 对这种只会调用 `func(*args)` 的通用包装器，`log_call` 放在 `classmethod`/`staticmethod`/`property` **下面**最稳妥，让它先拿到普通函数。反过来包 `classmethod` 或 `property`，包装器拿到不可直接调用的描述符，会报错。`staticmethod` 从 Python 3.10 起可直接调用，所以反序在某些类访问中能跑，但包装结果变成普通函数后，实例访问又会多绑定一个 `self`，不再保留静态方法的语义。
- `wrapper(*args, **kwargs)` 的 `args[0]` 是 `self`/`cls`——因为 `wrapper` 作为普通函数放在类上，本身就是非数据描述符，绑定后 `self` 会正常传入。这是函数式装饰器能"透明"用于方法的原因。
- §2 那种用类实现的装饰器**不能**透明用于方法：`LogCall` 实例没有 `__get__`，不是描述符，`obj.method` 得到的是 `LogCall` 实例本身，调用时不会传 `self`（报 `TypeError: S.m() missing 1 required positional argument: 'self'`）。要修复就给 `LogCall` 加一个 `__get__`，返回 `functools.partial(self, obj)` 或 `types.MethodType(self, obj)`。

第二章的知识在这里直接决定了代码能不能跑。

### 7. 同步与异步装饰器

`async def` 函数被调用时只返回协程对象，不执行函数体。同步 `wrapper` 包裹异步函数时，`func(*args)` 立即返回一个协程，`wrapper` 的"结束"日志在函数体真正运行之前就打出来了，异常也捕获不到。异步函数需要异步包装器：

```python title='异步函数需要 async 的包装器'
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

## 五、生成器与惰性执行

[上篇第二章](/python-execution-model-scopes-imports-and-exceptions.html#二执行模型源码如何变成正在运行的代码)说普通函数调用创建一次执行，返回时退出活动调用栈。生成器则可以在 `yield` 处**挂起**，下次 `next()` 时从原地恢复。

先把两个词说白：**帧**就是一次执行的“活动记录”——这次执行的局部变量、以及执行到了哪一条指令；**挂起**是指函数走到 `yield` 处停下来，把值交给调用方，但这份活动记录保留，下次 `next()` 从暂停点继续，局部变量还是刚才的值。普通帧退出调用栈后，如果仍被 traceback 或调试器引用，也不一定立即释放；生成器的不同在于它保留了**可继续执行**的状态。第一章的 `Runner.stream` 用它实现流式输出。

> [!NOTE] 与 Java 对照：yield 与惰性迭代
> Java 的 `Iterator`、`Stream` 也能惰性地产生值，但没有 Python 这种生成器函数语法；手写迭代器通常把进度保存在对象字段，Python 则替开发者保留暂停的帧。Java `switch` 表达式中的 `yield` 是提交分支结果，不是暂停函数。两边的惰性数据源都可能持有文件等资源，不能把消费结束当成自动关闭的保证。

### 1. `yield`：帧被挂起而不是销毁

```python title='用 inspect 看生成器帧的挂起与恢复'
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

把这段执行画成状态变化，关键不在于“函数被调用了几次”，而在于**同一份执行状态被恢复了几次**：

```mermaid
%% 图：stream 生成器的创建、两次挂起与耗尽
flowchart TB
    C["<b>g = stream()</b><br/>GEN_CREATED：函数体尚未运行"]
    C -- "第一次 next(g)" --> R1["GEN_RUNNING<br/>打印 start，执行到 yield 1"]
    R1 -- "向调用方交出 1" --> S1["<b>GEN_SUSPENDED</b><br/>保留帧状态，停在第一个 yield"]
    S1 -- "第二次 next(g)" --> R2["GEN_RUNNING<br/>从暂停点继续，打印 resume"]
    R2 -- "yield 2，交出 2" --> S2["<b>GEN_SUSPENDED</b><br/>同一份帧状态，停在第二个 yield"]
    S2 -- "第三次 next(g)" --> END["打印 end，函数体结束<br/>向调用方抛 StopIteration"]
    END --> CLOSED["<b>GEN_CLOSED</b><br/>g.gi_frame 为 None"]
```

调用生成器函数不执行函数体，只创建一个生成器对象，它内部保留帧状态（可通过 `g.gi_frame` 观察）；第一次 `next()` 才开始执行；到 `yield` 处暂停，帧里的局部变量、指令位置全部保留（`g.gi_frame.f_locals` 可以直接看到）；函数体结束时通过 `StopIteration` 告知消费者。生成器的帧在两次 `next()` 之间仍然可恢复，这是惰性执行的基础。生成器同时实现了 `__iter__`（返回自己）和 `__next__`，所以它是第三章意义上的迭代器，可以直接放进 `for`。

再用一个局部变量看“保留”的到底是什么：

```python title='生成器保留的是同一次执行：局部变量 n 从 2 变成 1'
def countdown(n):
    while n > 0:
        yield n
        n -= 1


g = countdown(2)
print(inspect.getgeneratorstate(g), g.gi_frame.f_locals)  # GEN_CREATED {'n': 2}
print(next(g), g.gi_frame.f_locals)                       # 2 {'n': 2}
print(next(g), g.gi_frame.f_locals)                       # 1 {'n': 1}
print(list(g), g.gi_frame)                               # [] None
```

第一次 `yield 2` 时还没执行 `n -= 1`，所以帧里仍是 `n = 2`；第二次 `next()` 才继续减到 1。正常耗尽后，生成器不再持有执行帧，`gi_frame` 变成 `None`；若此前有代码另存了帧引用，那个帧对象仍可存活。

### 2. 惰性的价值与成本

生成器让"生产一个、消费一个"成为默认模式，不需要把全部结果放进内存：

```python title='逐行读文件的生成器 read_lines'
def read_lines(path):
    with open(path) as file:
        for line in file:
            yield line
```

流式 Token 输出、大数据集分片读取、Batch 生成、日志消费都是这个模式。但惰性有两个经常被忽视的成本：

- **暂停也可能延长资源占用**。上面 `read_lines` 的文件句柄在第一次 `next()` 时打开，要到生成器结束或被关闭时才关——如果消费方中途 `break` 并保留着未关闭的生成器，文件就一直开着（§4）。GPU 张量、数据库游标、进程间队列同理；正常耗尽或显式关闭后，即使生成器对象还在，清理逻辑也已经执行。
- **错误延迟出现**。生成器函数里的 bug 在 `next()` 时才暴露，traceback 的调用方是消费者而不是创建者。

### 3. `yield from`

`yield from sub()` 把 `sub` 产生的每个值直接转发给外层的消费者，同时把 `send()`、`throw()`、`close()` 也透传给子生成器，并且能拿到子生成器的返回值：

```python title='yield from 转发子生成器并拿到它的返回值'
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

```python title='带 try/finally 的生成器 reader'
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

```python title='break 不关闭生成器：finally 要等到 del 才执行'
r = reader()
for v in r:
    break
print("after break")      # 此时 "close file" 还没有出现
del r                     #   close file   —— 引用计数归零时，生成器的析构才调用 close()
```

在 CPython 里，生成器对象被回收时会自动 `close()`，所以上面的例子在 `del` 后立刻清理。但依赖这一点是脆弱的：生成器被别处引用（放进列表、被 traceback 持有、被闭包捕获）就不会及时回收；在其他实现或存在循环引用时，回收时机不确定。**需要确定性清理的生成器，要么由消费方在 `finally` 里显式 `close()`，要么用 `contextlib.closing(gen)` 包起来放进 `with`。** 对象回收的细节在本系列第五篇。

同一个 `reader()`，消费方式决定资源什么时候释放：

```mermaid
%% 图：生成器正常耗尽、提前 break 与显式关闭的资源路径
flowchart TB
    OPEN["<b>next(r) 后进入 reader</b><br/>资源已获取，停在 yield"]
    OPEN -- "继续消费直到结束" --> FIN["执行 finally<br/>释放资源"]
    OPEN -- "break，且 r 仍被引用" --> HOLD["<b>只退出消费循环</b><br/>生成器仍挂起，资源仍被持有"]
    HOLD -- "r.close()" --> EXIT["在暂停点注入 GeneratorExit"]
    EXIT --> FIN
    FIN --> CLOSED["生成器关闭"]
    WITH["<b>with closing(reader()) as r</b>"] -- "退出 with 时调用 r.close()" --> EXIT
```

`closing` 不会把数据提前读完，只保证离开 `with` 时调用 `close()`。即使循环只取一个值就 `break`，资源也有明确的退出路径。

> [!NOTE] 与 Java 对照：终止操作不等于 close
> Java `Stream` 的终止操作（如 `findFirst()`）不会自动调用 `close()`；`Files.lines(path)` 这类持有资源的流仍要放进 try-with-resources。对应到 Python，`break` 不会关闭生成器，持有资源时应使用 `with closing(g)` 或显式 `close()`，不能依赖两种语言各自的垃圾回收时机。

## 六、上下文管理器：把资源生命周期交给协议

第五章的结论是"资源的释放时机不能靠猜"。`with` 语句就是为此设计的：把"进入/退出"两个动作绑定到一个代码块上，无论块内是正常结束、`return`、还是抛异常，退出动作都执行。第一章的 `InferenceContext` 和 `torch.inference_mode()` 都是这个协议。

### 1. `with` 的展开

```python title='一个 with 语句'
with resource() as value:
    use(value)
```

等价于：

```python title='with 的等价展开：__enter__、__exit__ 与异常传递'
manager = resource()
value = manager.__enter__()
normal_exit = True
try:
    use(value)
except BaseException as exc:
    normal_exit = False
    if not manager.__exit__(type(exc), exc, exc.__traceback__):
        raise                                   # __exit__ 返回假值 → 异常继续传播
finally:
    if normal_exit:
        manager.__exit__(None, None, None)      # 正常结束、return、break、continue 都走这里
```

这里展示控制流，省略了特殊方法查找细节：真实的 `with` 在管理器的**类型**上取 `__enter__`、`__exit__`，并在进入块体前保存退出方法，而不是走实例字典里的同名属性。PEP 343 的展开采用 try / except / finally；若把 `finally` 错换成 `else`，块内 `return` 或 `break` 时就会漏掉退出动作。

把这段展开按"块怎么结束"分成几条路径，`__exit__` 被调用几次、收到什么参数、异常最后去了哪里就一目了然：

```mermaid
%% 图：with 语句的执行路径：__enter__、块的四种结束方式与 __exit__ 的返回值
flowchart TB
    S["manager = resource()"] --> EN{"manager.__enter__() 抛异常？"}
    EN -- "是" --> P0["异常直接向外传播<br/>__exit__ 不被调用"]
    EN -- "否" --> BODY["执行块体 use(value)"]
    BODY -- "正常结束、return、break、continue" --> X0["__exit__(None, None, None)"]
    X0 --> C0["按原来的方式继续：<br/>执行下一句或完成 return / break"]
    BODY -- "块内抛异常 exc" --> X1["__exit__(type(exc), exc, tb)"]
    X1 --> T{"__exit__ 返回真值？"}
    T -- "是" --> SUP["异常被抑制<br/>块后的代码照常执行"]
    T -- "否（False / None）" --> RE["异常继续向外传播"]
```

由展开可以读出几条规则，每一条都在实验中验证过：

- `__enter__` 抛异常时，`__exit__` **不会**被调用；如果进入过程中已经获取了一部分资源，必须由 `__enter__` 自己清理，不能假设“进入失败就没资源”；
- `__exit__` 收到异常三元组，返回 `True` 表示**抑制**该异常（块外看不到），返回 `False`/`None` 则异常照常传播；`contextlib.suppress` 就是一个返回 `True` 的 `__exit__`；
- 块内 `return`、`break` 同样触发 `__exit__`，因为它们也走 `try` 的退出路径。

> [!NOTE] 与 Java 对照：with 与 try-with-resources
> Java 的 try-with-resources 在退出块时调用 `AutoCloseable.close()`；Python 的 `with` 还多了进入钩子及其返回值，退出钩子也能接收异常三元组。`__exit__` 返回真值可抑制块内异常，Java 的 `close()` 返回 `void`，没有这个协议；若块体和清理同时失败，Java 有 suppressed exceptions，Python 则按异常上下文链保留关系。

### 2. 类实现与 `contextlib.contextmanager`

类实现直接写两个方法：

```python title='用类实现上下文管理器：__enter__ 与 __exit__'
class Resource:
    def __enter__(self):
        print("acquire"); return self
    def __exit__(self, exc_type, exc, tb):
        print("release"); return False
```

生成器实现更短——`@contextmanager` 把一个只 `yield` 一次的生成器包装成上下文管理器：

```python title='用 @contextmanager 把只 yield 一次的生成器变成上下文管理器'
from contextlib import contextmanager


@contextmanager
def resource():
    handle = acquire()
    try:
        yield handle              # yield 之前是 __enter__，之后是 __exit__
    finally:
        release(handle)
```

它的机制是第五章的内容：`__enter__` 调一次 `next()` 跑到 `yield`；`__exit__` 在没有异常时再调一次 `next()` 让它跑完，有异常时用 `gen.throw(exc)` 把异常**在 `yield` 那一行抛出**。所以 `try/finally` 不是可选的——没有它，块内的异常会让生成器在 `yield` 处直接终止，`release` 永远不会执行。同理，想在 `@contextmanager` 里抑制异常，要在 `except` 里捕获并不再抛出。

### 3. AI-Infra 中的上下文管理器

`torch.inference_mode()`、`torch.autocast()`、`torch.profiler.profile()`、`torch.cuda.stream(s)`、`torch.no_grad()`、分布式通信组的生命周期，大多数是"**进入时切换一个全局或线程局部状态，退出时恢复原值**"，而不是获取/释放一个句柄。读这类代码时要问：

1. 进入时改了什么状态？是线程局部还是进程全局？（`no_grad` 是线程局部的；多线程数据加载时要注意）
2. 退出时是否恢复到**进入前**的值，还是恢复到某个默认值？嵌套使用时二者行为不同；
3. 异常路径是否也恢复？（用 `try/finally` 或类实现的 `__exit__` 才能保证）
4. 作为装饰器使用时（`@torch.no_grad()`）语义是否一致？

第一章的 `Runner.__call__` 用 `nullcontext()` 处理"不需要上下文"的分支，避免了 `if` 两个分支重复写 `self.model(batch)`——这是 `contextlib` 里最常用的小工具之一。

### 4. 异步上下文管理器

`async with` 对应 `__aenter__`/`__aexit__`，两者都是协程，允许在进入和退出时 `await`（建立连接、优雅关闭）。`contextlib.asynccontextmanager` 是对应的生成器版本。

## 七、一个推理组件的完整运行时追踪

回到第一章的 `runner.py`，为了不用来回翻，把它原样贴在这里（行号旁的蓝色数字可点，下面各阶段的描述会链接回具体的行）：

```python title='runner.py 全文：带 !ref 锚点的完整运行时追踪'
# runner.py
from contextlib import nullcontext

REGISTRY = {}


# !ref registered +7
def registered(name):                        # 带参数的装饰器：registered("runner") 先被调用，返回下面的 decorator
    def decorator(cls):                      # 真正的装饰器：拿到被装饰的类 Runner，登记后原样返回
        if name in REGISTRY:
            raise ValueError(f"duplicate registration: {name}")
        REGISTRY[name] = cls
        return cls

    return decorator


# !ref ctx +7
class InferenceContext:                      # 上下文管理器：进入/退出推理模式
    def __enter__(self):
        print("enter inference mode")
        return self

    def __exit__(self, exc_type, exc, tb):
        print("exit inference mode")
        return False


# !ref apply
@registered("runner")
class Runner:
    # !ref init +2
    def __init__(self, model, inference=True):
        self.model = model
        self.inference = inference

    # !ref call +3
    def __call__(self, batch):               # 可调用协议
        context = InferenceContext() if self.inference else nullcontext()
        with context:
            return self.model(batch)

    # !ref stream +1
    def stream(self, batch):                 # 生成器：流式输出
        yield from self.model.generate(batch)
```

现在可以按时间顺序，用上下两篇的机制精确描述它的每一步——上篇的导入、帧与异常传播，本篇的对象创建、属性查找、协议、生成器与上下文管理器。

### 1. 导入阶段

某个模块执行 `import runner`（或 `from runner import Runner`）：

1. `sys.modules` 中没有 `"runner"`，进入查找（[上篇第四章](/python-execution-model-scopes-imports-and-exceptions.html#四模块与导入系统代码如何被加载) §2）；
2. `sys.meta_path` 上的 `PathFinder` 沿 `sys.path` 找到 `runner.py`，`FileFinder` 按 `.py` 后缀选出 `SourceFileLoader`，生成 `ModuleSpec`（[上篇第四章](/python-execution-model-scopes-imports-and-exceptions.html#四模块与导入系统代码如何被加载) §3）；
3. 创建空模块对象，写入 `sys.modules["runner"]`——此刻它还是空的（[上篇第四章](/python-execution-model-scopes-imports-and-exceptions.html#四模块与导入系统代码如何被加载) §7 循环导入的窗口期）；
4. 取得模块的 code object：没有有效缓存时编译整个文件，否则从 `.pyc` 读回（[上篇第二章](/python-execution-model-scopes-imports-and-exceptions.html#二执行模型源码如何变成正在运行的代码) §1）；随后在模块 `__dict__` 中执行顶层代码：
   - `from contextlib import nullcontext`：若 `contextlib` 已在 `sys.modules`，复用该模块并绑定名称；否则先导入它；
   - `REGISTRY = {}`：创建字典；
   - [`def registered(name)`](#registered)：创建函数对象，`__globals__` 指向本模块的 `__dict__`（[上篇第二章](/python-execution-model-scopes-imports-and-exceptions.html#二执行模型源码如何变成正在运行的代码) §2）；
   - `class InferenceContext:`：执行类体、调用 `type` 创建类对象（第二章 §1）；
   - [`@registered("runner") class Runner:`](#apply)：先求值 `registered("runner")` 得到 `decorator`（闭包持有 `name`，[上篇第三章](/python-execution-model-scopes-imports-and-exceptions.html#三作用域与闭包名称在哪里被解析) §2），再执行类体并由元类创建类对象，随后调用 `decorator(类对象)`——写入 `REGISTRY`，返回原类（第四章 §5）；最后名称 `Runner` 绑定到它。
5. 导入方拿到模块对象或 `Runner` 名称。

如果没有执行 `runner` 模块的顶层代码，注册就不会发生。如果 `runner.py` 同时以 `__main__` 和 `runner` 两个名字执行，本例会连 `REGISTRY = {}` 也执行两次，得到两份注册表，**不会仅因双重导入就报重复注册**；只有装饰器写入同一份共享注册表时，重名才会触发 `duplicate registration`。这两种失败分别是注册状态分裂和重复登记，排查时要区分。

### 2. 创建对象阶段

`runner = Runner(model)`：

1. `Runner` 是类对象，调用它执行 `type.__call__(Runner, model)`（第二章 §7）；
2. `Runner.__new__(Runner)` 分配实例（未重写，走 `object.__new__`）；
3. [`Runner.__init__(instance, model)`](#init) 执行：`self.model = model`、`self.inference = True`，两次赋值走默认 `__setattr__`，写入实例 `__dict__`（第二章 §2）；
4. 返回实例，名称 `runner` 绑定到它。

### 3. 调用阶段

`output = runner(batch)`：

1. 解释器在 `type(runner)` 上查找 `__call__` 槽位（第三章 §1），找到 [`Runner.__call__`](#call)；
2. 函数作为非数据描述符被绑定，`self = runner`（第二章 §5），创建新的执行帧（[上篇第二章](/python-execution-model-scopes-imports-and-exceptions.html#二执行模型源码如何变成正在运行的代码) §3）；
3. `self.inference`：类 MRO 上没有同名描述符，实例 `__dict__` 中命中（第二章 §3 的 ③）；
4. `InferenceContext()` 创建上下文管理器实例；
5. `with context:` 调用 [`__enter__`](#ctx)，打印 `enter inference mode`（第六章 §1）；
6. `self.model(batch)`：`model` 是什么类型就走什么类型的 `__call__`——如果是 `nn.Module`，进入 `_wrapped_call_impl` → hooks → `forward`（第三章 §2）；
7. `return` 触发 `with` 的退出路径，`__exit__(None, None, None)` 打印 `exit inference mode`，返回 `False`；
8. 帧退出活动调用栈，返回值绑定到 `output`；没有其他引用时，帧状态可被释放。

### 4. 流式执行阶段

`for out in runner.stream(batch):`：

1. [`runner.stream`](#stream) 绑定为方法，调用它**不执行函数体**，返回一个 `GEN_CREATED` 状态的生成器对象（第五章 §1）；
2. `for` 调用 `iter()`，生成器返回自身（第三章 §3）；
3. 第一次 `next()`：进入函数体，`self.model.generate(batch)` 被调用，返回模型的生成器；`yield from` 开始转发（第五章 §3）；
4. 每产生一个 token，两个生成器的帧都挂起，控制权回到 `for` 循环体；
5. 模型生成器结束，`yield from` 收到 `StopIteration`，`stream` 的帧也结束，`for` 退出。

如果调用方在第 4 步 `break`，两个生成器都停在挂起状态；它们持有的资源要到生成器对象被回收（或显式 `close()`）时才释放（第五章 §4）。

### 5. 异常阶段

若第 3 阶段第 6 步 `self.model(batch)` 抛出 `RuntimeError`：

1. 异常在 `model` 的帧中产生，沿 `f_back` 回到 `Runner.__call__` 的帧（[上篇第五章](/python-execution-model-scopes-imports-and-exceptions.html#五异常处理与失败传播) §1）；
2. 经过 `with` 块：`InferenceContext.__exit__(RuntimeError, exc, tb)` 被调用，打印 `exit inference mode`，返回 `False`，异常继续传播（第六章 §1）；
3. `__call__` 的帧退出活动调用栈，但仍被 traceback 引用，不能当作已经销毁；
4. 到达调用方的 `try`：

   ```python title='调用方的 try：记录 traceback 后重抛'
   try:
       output = runner(batch)
   except Exception:
       logger.exception("inference failed")
       raise
   ```

   记录完整 traceback 后原样重抛（[上篇第五章](/python-execution-model-scopes-imports-and-exceptions.html#五异常处理与失败传播) §5）；再向上由 Worker 主循环决定重试、降级还是退出。

五个阶段对应的机制：

| 阶段 | 机制 | 章节 |
| :--- | :--- | :--- |
| 导入 | finder/loader、`sys.modules`、顶层代码执行、装饰器注册 | 上篇四、四 |
| 创建 | `type.__call__`、`__new__`/`__init__`、`__setattr__` | 二 |
| 调用 | `__call__` 槽位、方法绑定、属性查找、`with` | 二、三、六 |
| 流式 | 生成器帧挂起、`yield from`、迭代协议 | 三、五 |
| 异常 | 沿帧传播、`__exit__` 参与、异常链与重抛 | 六、上篇五 |

Table: 五个阶段对应的机制

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

## 八、本文小结

本篇讲的五种机制，都围绕**对象**：

| 机制 | 要点 |
|---|---|
| <span style="white-space:nowrap">类与对象模型</span> | 默认由 `type` 创建类；属性读取优先级：有 `__get__` 的数据描述符 → 实例字典 → 非数据描述符或普通类属性；抛出 `AttributeError` 后才尝试 `__getattr__`<br/>方法、`classmethod`、`staticmethod`、`property` 通过描述符决定绑定或取值 |
| 对象协议 | 语法交给类型上的特殊方法；特殊方法跳过实例字典 |
| 装饰器 | 定义时执行一次的函数调用；靠闭包记参数，靠描述符协议对方法透明 |
| 生成器 | `yield` 保留可恢复的帧；暂停可能延长资源占用；`break` 不等于 `close` |
| <span style="white-space:nowrap">上下文管理器</span> | `__enter__` / `__exit__` 绑定到代码块；`@contextmanager` 把异常 `throw` 进 `yield` |

Table: 本篇五种机制围绕对象的小结

回到开头那五行代码，现在每一行都有了答案：

| 开头的写法 | 背后的机制 | 在哪一节 |
|---|---|---|
| `model(x)` 走 `__call__`，中间插入 hooks | 调用语法查的是**类型**上的 `__call__`；`nn.Module.__call__` = `_wrapped_call_impl` → pre-hooks → `forward` → hooks，所以 `model.forward(x)` 绕过全部 hook | 三 §1、§2 |
| `for batch in loader` | `for` 先调用 `iter(loader)` 开始本轮迭代，再反复 `next()`；DataLoader 可新建迭代器，也可在 persistent worker 模式下复用并重置；生成器则在 `yield` 处挂起、下一次 `next` 恢复 | 三 §3、五 §1 |
| `with torch.inference_mode()` | `with` 展开为 `__enter__` / `__exit__`；这类上下文管理器进入时切换一个线程局部状态、退出时恢复，异常也照样恢复 | 六 §1、§3 |
| `@register("cuda")` 注册表能否填上取决于谁导入了它 | 装饰器在**定义时**执行一次；定义所在的模块没被导入、或被以两个名字导入，注册就不会发生或发生两次 | 四 §5、上篇四 §6 |
| `self.linear = nn.Linear(4, 4)` 登记到 `_modules` | `Module.__setattr__` 拦截赋值写进 `_modules`（不进 `__dict__`），读取时属性查找算法在类和实例字典都找不到，落到 `__getattr__` 从 `_modules` 取回 | 二 §3、§6 |

Table: 开头五行代码背后的机制

读 AI-Infra 代码时的这些疑问，也都落在本篇的机制上：

- `nn.Module` 为什么能用 `self.linear` 访问一个不在 `__dict__` 里的子模块——`__setattr__` 拦截写、`__getattr__` 兜底读；
- `model(x)` 和 `model.forward(x)` 为什么不等价——`__call__` 在类型上查找，hooks 挂在那里；
- `super().__init__()` 为什么不一定调到"父类"——它调的是实例 MRO 中的下一个；
- 一个用类实现的装饰器为什么用在方法上就报 `missing self`——它不是描述符，绑定不会发生；
- 一个生成器持有的文件为什么一直没关——消费方 `break` 了但没 `close()`。

这是从"会写 Python"到"能读懂 Python 工程和 AI-Infra 框架"的第一步。后续六篇分别讨论类型系统、并发与异步、动态机制与插件架构、内存管理、测试与调试、工程化与交付，每一篇都会用到上下两篇的某一组机制。

配套代码：本文验证各个结论用的小脚本（描述符优先级、生成器与上下文管理器、装饰器顺序）在 [ai-learning-labs/python-for-ai-infra/01-language-mechanisms](https://github.com/arganzheng/ai-learning-labs/tree/main/python-for-ai-infra/01-language-mechanisms)，只依赖标准库。

## 九、自测

1. `model(x)` 与 `model.forward(x)` 为什么不等价？

   <details markdown="1"><summary>答案</summary>

   调用语法查的是**类型**上的 `__call__`：`nn.Module.__call__` → `_wrapped_call_impl` → pre-hooks → `forward` → hooks。直接调 `forward` 绕过全部 hook（包括 `register_forward_hook` 注册的），行为可能不同。

   </details>

2. 写了 `@register("cuda")` 装饰器的后端在注册表里找不到，两个最常见的原因是什么？

   <details markdown="1"><summary>答案</summary>

   装饰器在**定义时**执行一次：（1）它所在的模块从未被导入，注册没发生；（2）同一模块被以两个名字导入（如 `pkg.mod` 与 `mod`，`sys.path[0]` 差异导致），注册发生了两次或落到另一份模块对象里。

   </details>

3. `self.linear = nn.Linear(4, 4)` 之后 `self.linear` 为什么不在 `self.__dict__` 里却能访问？

   <details markdown="1"><summary>答案</summary>

   `Module.__setattr__` 拦截赋值，把子模块写进 `_modules` 字典而不是实例 `__dict__`；读取时属性查找在类与实例字典都找不到，落到 `__getattr__`，它从 `_modules` / `_parameters` / `_buffers` 取回。

   </details>

4. `for batch in loader` 里 `for` 对 `loader` 做了什么？为什么每个 epoch 能重新迭代？

   <details markdown="1"><summary>答案</summary>

   `for` 先调用 `iter(loader)` 开始本轮迭代，再反复 `next()` 直到 `StopIteration`。默认创建新迭代器，多 worker 时也重建 worker；`persistent_workers=True` 且 `num_workers > 0` 时则复用迭代器和 worker，并重置迭代状态。下一个 epoch 要重新执行 `iter(loader)`，不是继续消费上一轮已耗尽的状态。

   </details>

5. 用类实现的装饰器（`__init__` 存函数、`__call__` 转发）用在普通函数上没问题，用在方法上却报 `missing 1 required positional argument: 'self'`，为什么？

   <details markdown="1"><summary>答案</summary>

   方法能拿到 `self` 靠的是函数对象实现了 `__get__`（非数据描述符），`obj.method` 时被绑定；装饰后类属性上放的是装饰器类的实例，它没有 `__get__`，`obj.method` 得到的就是这个实例本身，调用时不会传 `self`。给装饰器类补一个 `__get__`（返回 `functools.partial(self, obj)` 或 `MethodType`），或改用闭包实现，都能修好。

   </details>

[^q0]: 四层：（1）类由 `type` 创建——`Runner(model)` 先调元类 `type.__call__`，依次 `__new__` 分配、`__init__` 初始化；（2）`obj.attr` 是一个固定算法——先查类型 MRO 上的数据描述符，再查实例 `__dict__`，再查非数据描述符与类属性，最后 `__getattr__`；方法就是函数这个非数据描述符绑定实例的结果；（3）语法交给对象自己决定靠**特殊方法**——`runner(batch)` 查类型上的 `__call__`，`for` 查 `__iter__` / `__next__`，`with` 查 `__enter__` / `__exit__`，`[]` 查 `__getitem__`；（4）装饰器用闭包与描述符改写调用路径，生成器让帧挂起而不销毁，`runner.stream(batch)` 返回的就是一个挂起的生成器。
