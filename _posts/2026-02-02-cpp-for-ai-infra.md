---
layout: post
title: C++ 在 AI-Infra：从对象模型到算子扩展（总纲）
subtitle: C++ for AI-Infra, from the Object Model to Operator Extensions
tags: [C++, AI, AI-Infra]
catalog: true
---


## 内容简介

《C++ 在 AI-Infra：从对象模型到算子扩展》是一组共八篇的系列文章，面向有 Java 或其他托管语言背景、准备阅读和修改 PyTorch、vLLM 等 AI-Infra 项目 C++ 层的工程师。

它不是一本 C++ 教材。C++ 有几十个特性、几千页标准，而 AI-Infra 项目真正大量使用的只是一个子集。这个系列只讲这个子集，并且每一个特性都从一个具体问题引出：

> **PyTorch 和 vLLM 的 C++ 源码里，这段代码为什么这样写？**

一个典型例子。PyTorch 的 `at::Tensor` 在 Python 侧看起来是一个普通对象，但它的 C++ 定义大致是：

```cpp
class TensorBase {
 protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
};
```

要读懂这一行，需要同时理解：

- 为什么 `Tensor` 是一个"句柄"而不是数据本身；
- `intrusive_ptr` 是什么，它和 `shared_ptr` 有什么区别，为什么 PyTorch 不用标准库的那个；
- 模板参数 `UndefinedTensorImpl` 在做什么；
- 拷贝一个 `Tensor` 时到底发生了什么，为什么它几乎不花时间；
- 引用计数归零时谁负责释放 GPU 显存。

这些问题分别对应 C++ 的值语义、所有权、RAII、模板和析构时序。Java 工程师习惯了"一切皆引用、GC 负责回收"的世界，这些概念是进入 AI-Infra C++ 层的第一道门。

系列的目标是让读者在读完后，能够：

- 打开 `c10/`、`aten/src/ATen/`、`torch/csrc/`、vLLM `csrc/` 里的任意一个文件，知道每一行在做什么；
- 写出一个符合这些项目风格、能通过 review 的 C++ 算子或运行时组件；
- 在 C++ 层出错时（编译错误、链接错误、段错误、ABI 不匹配），知道去哪里找原因。


## 为什么写这个系列？

### AI-Infra 的核心项目都是 C++ 内核、Python 外壳

Python 是 AI-Infra 的控制平面，但性能关键的执行平面几乎全是 C++：

```text
项目            Python 层                      C++ 层
PyTorch         torch/                         c10/ · aten/ · torch/csrc/
vLLM            vllm/                          csrc/（attention、量化、MoE、cache）
FlashAttention  flash_attn/                    csrc/（CUTLASS 之上的 kernel）
Triton          python/triton/                 lib/ · include/（MLIR 编译器）
NCCL            —                              src/（全部 C++/CUDA）
TensorRT-LLM    tensorrt_llm/                  cpp/
```

在这些项目中做有价值的贡献——修一个算子的 bug、加一个 kernel、优化一条 dispatch 路径、适配一个新硬件——迟早会碰到 C++。只在 Python 层活动，能做的事情有上限。

### Java 工程师面前的门槛不是 PyTorch，而是 C++

PyTorch 的 C++ 扩展文档和大多数教程，会直接给出这样的代码：

```cpp
at::Tensor scale_shift_cpu(const at::Tensor& x, double alpha, double beta) {
  TORCH_CHECK(x.is_floating_point(), "expected floating point tensor");
  auto x_c = x.contiguous();
  auto out = at::empty_like(x_c);
  AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "scale_shift_cpu", [&] {
    const scalar_t* in = x_c.data_ptr<scalar_t>();
    scalar_t* o = out.data_ptr<scalar_t>();
    at::parallel_for(0, x_c.numel(), 4096, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i)
        o[i] = static_cast<scalar_t>(alpha) * in[i] + static_cast<scalar_t>(beta);
    });
  });
  return out;
}
```

教程假设读者的障碍在 PyTorch 的 API，但一个 Java 工程师读这段代码时，卡住的地方往往与 PyTorch 无关：

- `const at::Tensor&` 里的 `const` 和 `&` 各是什么意思，为什么不直接传 `at::Tensor`？
- `AT_DISPATCH_FLOATING_TYPES` 是函数吗？`scalar_t` 是从哪里冒出来的？
- `[&] { ... }` 是什么？`&` 捕获了什么，会有生命周期问题吗？
- `data_ptr<scalar_t>()` 的 `<>` 是泛型吗？为什么 Java 的泛型不能这样用？
- 这个函数返回 `at::Tensor` 是按值返回，会拷贝整个数据吗？
- `at::parallel_for` 的线程从哪里来？谁在管理它们？

讲 PyTorch 机制的材料对这些问题只能一句话带过，否则就变成了 C++ 教程。这个系列就是把那些"一句话带过"的地方展开。

### 现有材料的断层

学 C++ 的材料很多，但对这个目标读者都不合适：

- **经典教材**（《C++ Primer》《Effective C++》）面向通用 C++ 开发者，深度均匀，读完两千页后还是不知道 `intrusive_ptr` 和 `KernelFunction` 是什么。
- **现代 C++ 特性介绍**（C++11/14/17/20 新特性）按标准版本组织，而不是按问题组织。
- **PyTorch 的官方文档**假设读者已经会 C++，只讲 PyTorch 自己的 API。
- **"给 Java 程序员的 C++"**类材料通常止步于语法对照，不涉及大型工程里真正的难点：模板元编程、静态注册、ABI、构建系统。

这个系列的取法是：**以 PyTorch 和 vLLM 的真实源码为教材，以 Java 为参照系，以一个逐步生长的 mini-c10 项目为练手**。

### Java 是很好的参照系，但要明确它的边界

Java 背景在很多地方是助力：

- 类、继承、接口、虚函数——概念相通，只是 C++ 默认不虚；
- 泛型 → 模板——都是参数化类型，但实现机制完全不同；
- `try-with-resources` → RAII——同一目标，C++ 的做法更彻底；
- `ServiceLoader` → 静态注册——同一目标，C++ 用静态初始化实现；
- JNI → pybind11——同一位置，都是两个运行时的边界；
- JMM 的 `volatile`/happens-before → C++ 内存模型——概念直接对应。

但有几处 Java 的直觉会成为障碍，本系列会反复点出：

- Java 里"变量就是引用"，C++ 里变量默认是值，引用和指针要显式写出来；
- Java 里对象什么时候死由 GC 决定，C++ 里由作用域和所有权决定，而且是确定的；
- Java 泛型在编译后被擦除，C++ 模板在编译期为每组参数生成一份代码；
- Java 有唯一的字节码和 JVM，C++ 没有统一的 ABI，编译器版本、标准库版本、编译选项都会影响二进制兼容；
- Java 的构建产物就是 `.class`/`.jar`，C++ 有翻译单元、目标文件、静态库、动态库、符号可见性一整套概念。


## 适合哪些读者？

### 从 Java/Go 后端转向 AI-Infra 的工程师

这是本系列的主要读者。你已经有多年工程经验，理解类型系统、并发、构建、部署，但没有系统写过 C++，或者只在学校写过 C++98 风格的代码。

你准备做的事情包括：

- 阅读 PyTorch、vLLM、FlashAttention、NCCL 等项目的 C++ 源码；
- 给这些项目修 bug、加算子、做硬件适配；
- 用 C++/CUDA 写自定义算子并接入 PyTorch；
- 排查 C++ 层的崩溃、内存错误和性能问题。

### 会写 C++ 但没接触过 PyTorch 内部的开发者

如果你有 C++ 经验，可以快速翻过每篇的语言部分，重点看"这个特性在 PyTorch/vLLM 里是怎么用的"：`intrusive_ptr` 的设计取舍、`KernelFunction` 的 boxed/unboxed 双路径、`TORCH_LIBRARY` 的宏展开、TLS 守卫的分层、pybind11 与 Python C API 的混用。

### 想读懂 PyTorch Dispatcher 和扩展机制的读者

PyTorch 的算子分发（Dispatcher、`OperatorEntry`、`KernelFunction`）和扩展机制（`TORCH_LIBRARY`、`torch.utils.cpp_extension`）只能在 C++ 源码里看清。如果读这些源码时感到 C++ 本身是障碍，本系列的第二、三、四、七篇正是为此准备的。

### 准备写 CUDA kernel 的读者

CUDA 是 C++ 的方言。写 kernel 之前，需要先能自然地写出 host 侧的 C++：模板参数化 tile 大小、RAII 管理设备内存和 stream、宏做多架构分发。本系列不讲 CUDA，但讲清这些 CUDA 代码所依赖的 C++。


## 系列的整体主线

八篇文章按"读懂一个大型 C++ 项目需要的知识"的依赖顺序展开：

```text
第一篇：C++ 程序是怎么变成二进制的
        ↓  翻译单元 · 头文件 · 链接 · 符号 · 库
第二篇：对象在哪里、活多久、谁负责释放
        ↓  值语义 · 引用 · 移动 · RAII · 智能指针
第三篇：一份代码如何服务多种类型
        ↓  模板 · 推导 · 特化 · constexpr
第四篇：运行时如何选择实现
        ↓  虚函数 · 函数对象 · 类型擦除 · variant
第五篇：代码如何在启动时自己登记进系统
        ↓  宏 · 静态初始化 · 可见性 · 代码生成
第六篇：多线程下如何正确且快
        ↓  内存模型 · 原子 · TLS · 守卫 · 并行
第七篇：C++ 如何与 Python 对话
        ↓  pybind11 · Python C API · GIL · ABI
第八篇：怎么构建、调试、测试
           CMake · 编译选项 · gdb · sanitizer · gtest
```

前四篇是语言核心，后四篇是工程实践。每一篇都有同样的三段结构：

```text
问题        从 PyTorch/vLLM 源码里挑一段，指出读不懂的地方
机制        讲清背后的 C++ 特性，用 Java 做对照，划清类比的边界
回到源码    带着机制重读那段代码，再扩展读几个同类位置
mini-c10    在练手项目里用这个机制实现一小块
```

这条主线也可以归纳为三条线索：

```text
语言线：编译模型 → 对象模型 → 泛型 → 多态 → 元编程 → 并发
工程线：构建 → 链接 → ABI → 调试 → 测试
源码线：c10::intrusive_ptr → TensorImpl → AT_DISPATCH → KernelFunction
        → TORCH_LIBRARY → DeviceGuard → THPVariable → CMakeLists.txt
```


## 章节结构与分章导读

### 1. 从源码到二进制：编译模型与项目布局

Java 工程师第一次面对 C++ 项目时，最陌生的不是语法，而是**它是怎么编译出来的**。为什么有 `.h` 和 `.cpp` 两种文件？为什么改一个头文件要重编半个项目？什么是"未定义的引用"？为什么同一个函数在两个文件里定义会报错？

这一篇会讨论：

- 预处理、编译、汇编、链接四个阶段，每个阶段的输入和输出；
- 翻译单元；声明与定义；头文件的职责与 `#pragma once`；
- One Definition Rule，以及 `inline`、`static`、匿名命名空间如何影响它；
- 目标文件、静态库、动态库；符号、符号表、`nm` 与 `objdump`；
- 动态链接与加载时的符号解析，`LD_LIBRARY_PATH` 与 `RPATH`；
- 命名空间与 `c10::`、`at::`、`torch::` 的分工；
- PyTorch 的源码布局为什么是 `c10/` → `aten/` → `torch/csrc/`，每层编成什么库（`libc10.so`、`libtorch_cpu.so`、`libtorch_cuda.so`、`libtorch_python.so`）。

核心问题是：

> **`import torch` 时加载了哪些 `.so`？它们之间是什么依赖关系？我写的扩展链接到哪一个？**

Java 对照：类加载器在运行时按需加载 `.class`，C++ 在链接时就把符号解析完（动态库是部分例外）。理解这个差别，就理解了为什么 C++ 的"找不到符号"错误出现在编译期和加载期，而不是运行期。

实践：写第一个 libtorch C++ 程序，手工用 `g++` 编译并链接到 PyTorch 的 `.so`，用 `ldd` 和 `nm` 观察结果。mini-c10 项目在这一篇建立目录结构。

### 2. 值、引用与所有权：对象模型与 RAII

这是全系列最重要的一篇。Java 里所有对象都在堆上、变量都是引用、生命周期由 GC 决定；C++ 三者都不同，而且是理解一切后续内容的前提。

这一篇会覆盖：

- 栈对象与堆对象；值语义；拷贝发生在哪里；
- 引用 `T&`、常量引用 `const T&`、指针 `T*`，各自用在什么场景；
- `const` 的位置和含义；`const` 成员函数；
- 构造、析构、拷贝构造、拷贝赋值、移动构造、移动赋值——"六大特殊成员函数"；
- 右值引用与 `std::move`；移动语义为什么让"按值返回"没有代价；
- RAII：把资源的生命周期绑定到对象的生命周期；
- `std::unique_ptr`、`std::shared_ptr`、`std::weak_ptr`；
- `c10::intrusive_ptr`：为什么 PyTorch 自己实现一套引用计数，它比 `shared_ptr` 省了什么；
- `c10::intrusive_ptr_target`、`TensorImpl` 与 `StorageImpl` 的引用关系；
- `at::Tensor` 作为句柄：拷贝一个 Tensor 拷贝了什么，什么时候释放显存。

核心问题是：

> **`at::Tensor y = x;` 之后 `y` 和 `x` 是什么关系？什么时候数据真正被释放？**

这一篇会从 C++ 所有权的角度讲清 PyTorch Tensor 的 `Tensor → TensorImpl → StorageImpl → 数据` 三层结构：`Tensor` 按值持有一个 `intrusive_ptr<TensorImpl>`，`TensorImpl` 按值持有一个 `Storage`，`Storage` 按值持有一个 `intrusive_ptr<StorageImpl>`，`StorageImpl` 持有一个带自定义删除器的 `DataPtr`。整条链上没有一处需要手工 `delete`，这就是 RAII。

Java 对照：`try-with-resources` 只能管理块作用域内的资源；RAII 让资源可以作为成员、被移动、放进容器，生命周期跟着所有者走。GC 解决了"什么时候释放内存"，没有解决"什么时候释放显存、文件句柄、锁"。

实践：mini-c10 实现 `intrusive_ptr` 和一个最简 `TensorImpl`/`StorageImpl`，让第一个 Tensor 对象跑起来，并用析构函数打印证明释放时序。

### 3. 模板与泛型编程

Java 泛型和 C++ 模板都写成 `<T>`，但实现机制相反：Java 在编译后擦除类型，运行时只有一份代码；C++ 为每组模板参数生成一份代码，运行时没有类型信息也不需要。这个差别决定了模板能做什么、编译错误为什么那么长、以及 `AT_DISPATCH` 为什么存在。

这一篇会覆盖：

- 函数模板与类模板；实例化；`typename` 与 `template` 关键字的消歧义作用；
- 模板参数推导；`auto`；`decltype`；返回类型推导；
- 非类型模板参数：`std::array<T, N>`、CUDA kernel 的 `BLOCK_SIZE` 参数化；
- 全特化与偏特化；
- 变参模板与参数包展开；
- `constexpr` 与 `if constexpr`：把分支移到编译期；
- SFINAE 与 `std::enable_if`，以及 C++20 concepts 如何替代它们；
- 编译期分派与运行期分派：为什么 `dtype` 是运行期值，而 kernel 需要编译期类型；
- 读懂 `AT_DISPATCH_FLOATING_TYPES` 的宏展开：一个 `switch` 把运行期 `ScalarType` 映射到编译期 `scalar_t`；
- `c10::ArrayRef`、`IntArrayRef`、`std::optional`（PyTorch 2.4 之后逐步用它替代 `c10::optional`）、`c10::SmallVector`：轻量视图和容器类型；
- lambda：捕获列表、`[&]` 与 `[=]` 的区别、泛型 lambda、lambda 作为模板参数。

核心问题是：

> **`AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "name", [&] { ... scalar_t ... })` 里的 `scalar_t` 从哪里来？这个 lambda 被编译了几次？**

Java 对照：`List<Integer>` 和 `List<String>` 在 JVM 里是同一个类；`std::vector<int>` 和 `std::vector<std::string>` 在 C++ 里是两个完全不相干的类型，各有各的机器码。这也是为什么模板错误信息里会出现几百字符长的类型名。

实践：mini-c10 用模板加上 `dtype` 支持，实现自己的 `DISPATCH_FLOATING_TYPES` 宏，并观察编译产物里确实有多份 kernel 代码。

### 4. 多态与类型擦除：运行时如何选择实现

Java 的多态只有一种：接口与虚方法。C++ 有好几种，各有代价，AI-Infra 项目会按场景混用。理解这一篇，才能读懂 PyTorch 的 Dispatcher 是用什么 C++ 机制把 `add` 分发到 CPU 或 CUDA 实现的。

这一篇会覆盖：

- 虚函数、vtable、纯虚函数与抽象类；`override` 与 `final`；虚析构；
- 为什么 `TensorImpl` 有虚函数而 `Tensor` 没有；
- 函数指针；`std::function` 及其代价；
- 函数对象与 lambda 作为零开销的策略参数；
- CRTP：编译期多态的典型模式；
- 类型擦除：把任意可调用对象装进统一接口——`c10::KernelFunction` 的 boxed 与 unboxed 两条调用路径；
- `c10::IValue`：一个能装下任何算子参数的"盒子"，以及它和 Java `Object` 的异同；
- `std::variant` 与 `std::visit`；`enum class`；
- 异常：`TORCH_CHECK` 抛出的是什么，异常如何跨越 C++/Python 边界。

核心问题是：

> **Dispatcher 拿到一个 `OperatorHandle` 和一组参数后，用什么 C++ 机制调到 CPU kernel？为什么既有 boxed 又有 unboxed？**

这一篇会读 `aten/src/ATen/core/boxing/KernelFunction.h` 和 `aten/src/ATen/core/dispatch/OperatorEntry.h`，把"Operator Table 里按 DispatchKey 索引的槽位"这个概念具体化成 C++ 类型。

Java 对照：Java 的 `Object` 是所有类的基类，任何值都能装进去；C++ 没有统一基类，`IValue` 是手工实现的带类型标签的联合体。Java 的接口调用永远是虚调用；C++ 用模板可以在编译期就把"策略"内联进去，零运行时开销。

实践：mini-c10 实现一个最小 Dispatcher：`OperatorEntry` 保存按 `DispatchKey` 索引的类型擦除 kernel，支持 CPU 和一个假的 "Meta" 后端。

### 5. 宏、静态注册与代码生成

打开 PyTorch 的任何一个算子文件，都会看到大量宏：`TORCH_LIBRARY`、`TORCH_CHECK`、`AT_DISPATCH_*`、`REGISTER_DISPATCH`、`C10_API`、`TORCH_API`、`C10_LIKELY`。Java 里没有预处理器，这些东西看起来像另一种语言。这一篇解释宏在大型 C++ 项目里承担的三种职责，以及它们各自的替代方案。

这一篇会覆盖：

- 预处理器：`#define`、`#include`、`#ifdef`、字符串化 `#` 与拼接 `##`、`__VA_ARGS__`；
- 宏的三种用途：条件编译、生成重复代码、在调用点捕获信息（`__FILE__`、`__LINE__`）；
- `TORCH_CHECK` 与 `TORCH_INTERNAL_ASSERT`：为什么错误检查用宏而不是函数；
- 静态初始化与静态注册模式：一个 `static` 对象的构造函数在 `main` 之前运行，用来把自己登记到全局表里；
- `TORCH_LIBRARY(myops, m)` 展开成什么；`TORCH_LIBRARY_IMPL` 如何按 DispatchKey 注册；
- 静态初始化顺序问题（"static initialization order fiasco"）与 PyTorch 的规避方式；
- 符号可见性：`-fvisibility=hidden`、`__attribute__((visibility("default")))`、`C10_API`/`TORCH_API` 宏；为什么一个静态注册的算子在某些链接方式下"消失"了；
- 代码生成：`torchgen` 从 `native_functions.yaml` 生成 C++ 代码，生成的文件长什么样，在哪里，为什么要生成而不是手写；
- 平台与编译器宏：`__CUDACC__`、`__CUDA_ARCH__`、`_WIN32`、`__GNUC__`。

核心问题是：

> **一个 `.so` 被 `import` 后，里面的算子怎么就出现在 `torch.ops.myops` 下了？没有任何函数被显式调用。**

Java 对照：`ServiceLoader` 靠 `META-INF/services` 文件在运行时发现实现；Spring 靠反射扫描注解。C++ 没有反射，用"静态对象的构造函数在加载时运行"实现同一目标——代价是链接方式会影响它是否生效。

实践：mini-c10 加上 `MINI_LIBRARY` 宏和静态注册，算子实现文件不再需要被任何地方显式引用。

### 6. 并发、内存模型、TLS 与守卫

PyTorch 的 CPU kernel 用 OpenMP 多线程，Autograd 引擎有自己的线程池，`InferenceMode`、`no_grad`、当前设备、当前 stream 全都是线程局部状态。这一篇讲 C++ 并发的基础，以及 PyTorch 用它们搭出的几种模式。

这一篇会覆盖：

- `std::thread`、`std::mutex`、`std::lock_guard`、`std::unique_lock`、`std::condition_variable`；
- C++ 内存模型：`std::atomic`、六种 memory order、happens-before；与 Java `volatile`、JMM 的对照；
- 为什么 `intrusive_ptr` 的引用计数用 `relaxed` 增、`acq_rel` 减；
- `thread_local`；PyTorch 的 TLS 状态：`c10::impl::LocalDispatchKeySet`、`InferenceMode`、`GradMode`、当前 CUDA 设备与 stream；
- 守卫模式：`c10::DeviceGuard`、`c10::cuda::CUDAStreamGuard`、`at::AutoDispatchBelowADInplaceOrView`、`torch::NoGradGuard`——RAII 管理的不只是资源，还有"上下文"；
- `at::parallel_for` 与 OpenMP；grain size；线程数从哪里来；
- 为什么 CUDA kernel 的 launch 不用锁：stream 顺序语义与 host 线程模型的关系；
- SIMD 简介：`at::vec::Vectorized<T>` 与 CPU kernel 的向量化。

核心问题是：

> **`with torch.no_grad():` 在 C++ 层做了什么？为什么它对其他线程不生效？**

Java 对照：`ThreadLocal` 与 `thread_local` 概念相同，但 C++ 的守卫对象让"进入上下文—退出上下文"的配对由析构函数保证，不会忘记 `remove()`。Java 的 `synchronized` 和 `java.util.concurrent` 对应 C++ 的 `mutex` 和 `<atomic>`，但 C++ 的内存序是显式的、可以比 `volatile` 更弱。

实践：mini-c10 加 `parallel_for` 和一个 TLS 守卫 `NoGradGuard`，用两个线程演示 TLS 隔离。

### 7. 与 Python 之间：pybind11、Python C API 与 ABI

C++ 层写好之后要暴露给 Python。这一篇讲两个运行时的边界上发生了什么，以及为什么"在我机器上能跑"在这条边界上经常不成立。

这一篇会覆盖：

- Python C API 基础：`PyObject`、引用计数、GIL、`PyObject_Call`；
- pybind11 的工作方式：模板生成类型转换代码、`py::object` 的 RAII 引用管理、异常翻译；
- `PYBIND11_MODULE`；类绑定、函数绑定、默认参数、关键字参数；
- `py::gil_scoped_release`：什么时候必须释放 GIL，释放后不能碰什么；
- PyTorch 如何绑定 `Tensor`：`THPVariable` 直接用 Python C API 而不是 pybind11，为什么；
- `torch/csrc/utils/pybind.h` 里对 `at::Tensor`、`c10::Device` 等类型的 caster；
- `TORCH_LIBRARY` 与 pybind11 的选择：vLLM 的 `csrc/torch_bindings.cpp` 为什么用前者；
- ABI：name mangling、`_GLIBCXX_USE_CXX11_ABI`、libstdc++ 版本、`manylinux` 标准；PyTorch 2.6 起 Linux wheel 切换到 CXX11 ABI（版本敏感）；
- 扩展与 PyTorch 之间的 ABI 契约：为什么扩展要用编译 PyTorch 的同一编译器大版本；`torch.utils.cpp_extension` 做了哪些检查；
- Python 对象生命周期与 C++ 对象生命周期的交叉：谁持有谁，循环引用怎么断。

核心问题是：

> **一个 `at::Tensor` 从 Python 传到 C++ 又返回 Python，经过了几次类型转换、几次引用计数变化？中间 GIL 状态是什么？**

Java 对照：JNI 是同一位置的技术——`jobject` 对应 `PyObject*`，局部引用/全局引用对应 borrowed/owned reference，`JNIEnv` 对应 GIL 持有状态。区别是 Python 的 GIL 让"释放锁去跑 C++"成为一个显式、常见的动作。

实践：mini-c10 用 pybind11 暴露给 Python，实现 `Tensor` 的 caster，演示一次 ABI 不匹配的真实报错以及如何定位。

### 8. 构建、调试与测试工具链

最后一篇讲工程闭环：怎么把前七篇写的代码可靠地编出来、出问题时怎么看进去、怎么防止它再坏。

这一篇会覆盖：

- CMake 的目标模型：`add_library`、`target_link_libraries`、`target_include_directories`、`PUBLIC`/`PRIVATE`/`INTERFACE`；`find_package(Torch)` 找到了什么；
- Ninja 与 ccache；为什么 PyTorch 全量构建要一小时，增量怎么控制在分钟级；
- 编译选项：`-O0/-O2/-O3`、`-g`、`-fno-omit-frame-pointer`、`-Wall -Werror`、`-march`；Debug 与 Release 构建的区别；
- `compile_commands.json` 与 clangd：让 IDE 真正理解一个百万行的 C++ 项目；
- gdb/lldb：从一个 Python 进程 attach 到 C++、在 kernel launch 前打断点、看 `at::Tensor` 的内容（PyTorch 自带的 gdb 辅助脚本）；
- 段错误、栈溢出、use-after-free 的排查路径；
- Sanitizers：ASan、UBSan、TSan；它们能抓什么、抓不到什么、在 PyTorch 的 CI 里怎么用；
- gtest：`c10/test/`、`aten/src/ATen/test/` 的组织方式；C++ 测试与 Python 测试的分工；
- clang-format、clang-tidy，以及 PyTorch 的 lint 规则；
- 工具链版本矩阵：gcc/clang 版本、CUDA 版本、C++ 标准（PyTorch 2.x 用 C++17）之间的兼容约束。

核心问题是：

> **一个 C++ 改动，从写完到确认正确、没有内存错误、不会在别的编译器上炸，需要跑哪些东西？**

Java 对照：Maven/Gradle 管理依赖、编译和测试一体化，C++ 里这三件事由 CMake、编译器和测试框架分别负责，而且依赖管理没有标准答案。JVM 的调试器无需关心优化级别，C++ 在 `-O2` 下变量可能被优化掉、栈帧可能被内联，Debug 构建是必需的。

实践：mini-c10 补齐 CMake 工程、gtest 测试、ASan 配置，用 gdb 从 Python 端一路断到 C++ kernel。本篇最后给出全系列总结，把八篇讲过的机制映射回开篇那段 `scale_shift_cpu` 代码，逐行作答。


## 贯穿全系列的实践线：mini-c10

本系列的练手项目取名 mini-c10，因为它模仿的正是 PyTorch 最底层那个库——`c10/`——的核心结构，外加 ATen 的 Dispatcher 骨架：

```text
第一篇    目录结构、CMake 骨架、第一个可链接的库
第二篇    intrusive_ptr · TensorImpl · StorageImpl · Tensor 句柄
第三篇    ScalarType · dtype 分发宏 · 第一个模板化 kernel
第四篇    DispatchKey · OperatorEntry · 类型擦除的 KernelFunction · 最小 Dispatcher
第五篇    MINI_LIBRARY 静态注册宏 · 算子文件自注册 · 符号可见性
第六篇    parallel_for · TLS 守卫 · 线程安全的引用计数
第七篇    pybind11 模块 · Tensor caster · 一次 ABI 事故复现
第八篇    gtest · ASan · gdb 会话 · clang-format
```

它最终大约一两千行 C++，实现 `add` 和 `mul` 两个算子的 CPU 实现和 Meta 实现，能从 Python 调用，能被 gdb 调试，有测试。它不追求性能，也不追求功能覆盖，只有一个目标：**读者写完之后，再打开真实的 `c10/core/TensorImpl.h` 和 `aten/src/ATen/core/dispatch/Dispatcher.h`，看到的是熟悉的结构。**

与它平行的是源码阅读线。每篇的"回到源码"部分带读者读真实项目的一到三处代码：

```text
第一篇    c10/CMakeLists.txt · torch/csrc/stub.c · setup.py 的库依赖
第二篇    c10/util/intrusive_ptr.h · c10/core/TensorImpl.h · c10/core/StorageImpl.h · c10/core/Allocator.h
第三篇    ATen/Dispatch.h · c10/util/ArrayRef.h · c10/util/SmallVector.h
第四篇    ATen/core/boxing/KernelFunction.h · ATen/core/dispatch/OperatorEntry.h · ATen/core/ivalue.h
第五篇    torch/library.h · c10/macros/Macros.h · c10/macros/Export.h · 一个 torchgen 生成文件
第六篇    c10/core/impl/LocalDispatchKeySet.h · c10/core/DeviceGuard.h · ATen/Parallel.h · ATen/cpu/vec/
第七篇    torch/csrc/autograd/python_variable.cpp · torch/csrc/utils/pybind.h · vLLM csrc/torch_bindings.cpp
第八篇    c10/test/util/intrusive_ptr_test.cpp · tools/gdb/pytorch-gdb.py · .clang-tidy
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

前四篇是语言核心，建议按顺序读；后四篇相对独立，可以按需调整。

### 只想读懂源码，暂时不写

```text
1 → 2 → 3 → 4 → 5
```

到第五篇为止，能读懂 `c10/`、`aten/` 里绝大多数文件的结构。

### 要马上写一个 C++ 扩展

```text
2 → 3 → 7 → 8
```

第二篇的所有权和第三篇的模板是写扩展的最低要求，第七篇解决暴露给 Python 的问题，第八篇解决构建。

### 准备进入 CUDA

```text
1 → 2 → 3 → 6 → 8
```

CUDA kernel 大量依赖模板参数化和 RAII 管理设备资源；第六篇的 stream 与线程模型、第八篇的 nvcc 工具链是直接前置。

### 有 C++ 经验，只看 PyTorch 特有的部分

每篇的"回到源码"与 mini-c10 部分，重点是第二篇的 `intrusive_ptr`、第四篇的 `KernelFunction`、第五篇的静态注册、第六篇的 TLS 守卫、第七篇的 `THPVariable`。


## 本系列的边界

本系列只讲 C++ 这门语言在 AI-Infra 项目中的用法，以及读懂这些项目 C++ 层所需的工程知识。以下内容不在范围内，虽然它们与本系列的源码阅读对象紧密相关：

- **PyTorch 的机制本身**：Autograd 如何构图、Dispatcher 的分发规则、`torch.compile` 的工作方式。本系列只借用它们的 C++ 代码作为语言特性的例子，读者不需要事先理解这些机制，也不会在本系列里学到它们的完整原理。
- **CUDA 编程**：kernel 的写法、GPU 内存层次、性能优化。本系列停在 host 侧，只覆盖 CUDA 代码所依赖的 C++ 特性。
- **Python 语言本身**：第七篇讨论 C++ 与 Python 的边界，假设读者已经了解 CPython 的引用计数和 GIL 是什么。
- **通用 C++ 知识的完整覆盖**：STL 算法库、iostream、正则、文件系统、协程等在这些项目里很少出现的部分。


## 前置要求与说明

### 前置要求

- 一门静态类型语言的多年工程经验（Java、Go、C# 均可）；
- 理解类、接口、泛型、异常、线程这些概念本身；
- 基本的 Linux 命令行；
- 能读简单 Python（源码阅读会涉及 Python 端的对应部分）。

不要求：

- 写过 C++；
- 了解 PyTorch 内部；
- 会 CUDA。

### 语言标准与版本基线

正文以 **C++17** 为基线：这是 PyTorch 2.x 与大量第三方扩展使用的标准，也是读懂这些项目源码所需的最小集；本系列引用的 PyTorch v2.10.0 树顶层 `CMakeLists.txt` 的 `CMAKE_CXX_STANDARD` 为 17，vLLM v0.15.0 同样是 17，mini-c10 全部用 `-std=c++17` 编译。concepts、`std::span`、`consteval` 等 C++20 特性会在相关位置作为语言知识提及，并标注"源码树中尚未使用"。

源码引用以 **PyTorch 2.x（2.4 及之后，正文片段取自 v2.10.0 源码树）** 和 **vLLM 0.x 主线（v0.15.0）**为准。C++ 层的接口比 Python 层稳定，但以下几处变化较快，正文会随文标注版本：

- `c10::optional` → `std::optional` 的迁移；
- `AT_DISPATCH` 宏族向 `AT_DISPATCH_V2` 的演进；
- Linux wheel 的 C++ ABI 设置；
- CUDA/编译器的兼容矩阵。

### 关于工具链

示例在 Linux + gcc 11 以上 / clang 14 以上验证。macOS 上除 CUDA 相关内容外都可以运行。Windows 不作为目标平台，但会在 ABI 和可见性的讨论中提及 MSVC 的差异。


## 章节目录

1. [从源码到二进制：编译模型与项目布局](/cpp-compilation-model-and-project-layout.html)
2. [值、引用与所有权：对象模型与 RAII](/cpp-value-semantics-ownership-and-raii.html)
3. [模板与泛型编程](/cpp-templates-and-generic-programming.html)
4. [多态与类型擦除：运行时如何选择实现](/cpp-polymorphism-and-type-erasure.html)
5. [宏、静态注册与代码生成](/cpp-macros-static-registration-and-codegen.html)
6. [并发、内存模型、TLS 与守卫](/cpp-concurrency-memory-model-tls-and-guards.html)
7. [与 Python 之间：pybind11、Python C API 与 ABI](/cpp-pybind11-python-c-api-and-abi.html)
8. [构建、调试与测试工具链](/cpp-build-debug-and-test-toolchain.html)


## 最终目标

读完这套系列之后，读者应该能够回到开篇的那段代码：

```cpp
at::Tensor scale_shift_cpu(const at::Tensor& x, double alpha, double beta) {
  TORCH_CHECK(x.is_floating_point(), "expected floating point tensor");
  auto x_c = x.contiguous();
  auto out = at::empty_like(x_c);
  AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "scale_shift_cpu", [&] {
    ...
  });
  return out;
}
```

并逐行回答：

```text
const at::Tensor& 为什么这样传？            → 第二篇：值语义与常量引用
TORCH_CHECK 为什么是宏？                    → 第五篇：宏在调用点捕获信息
x.contiguous() 返回的对象要拷贝数据吗？      → 第二篇：句柄与移动语义
AT_DISPATCH 如何把运行期 dtype 变成编译期类型？ → 第三篇：模板与编译期分派
[&] 捕获了什么，安全吗？                     → 第三篇：lambda 与生命周期
parallel_for 的线程从哪里来？               → 第六篇：OpenMP 与线程模型
这个函数怎么变成 torch.ops 下的算子？        → 第五篇：静态注册
Python 调用它时经过了什么？                  → 第七篇：pybind11 与 GIL
它编译成哪个 .so，链接到哪些库？             → 第一篇、第八篇：链接与 CMake
```

最终目标不是"会写 C++"，而是三种能力：

1. **阅读能力**：打开 PyTorch、vLLM 的任意 C++ 文件，能够识别其中的模式并理解意图；
2. **修改能力**：写出符合项目风格、通过 review、不引入内存错误和 ABI 问题的 C++ 代码；
3. **排障能力**：面对编译错误、链接错误、段错误和 ABI 不匹配，知道用什么工具、看哪里。

这是从 Python 层走向 AI-Infra 执行平面时，无法绕开的一段路。
