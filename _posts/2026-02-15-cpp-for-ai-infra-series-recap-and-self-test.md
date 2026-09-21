---
layout: post
series: cpp-for-ai-infra
title: "C++ 在 AI-Infra（09）：系列总结与通关自测"
subtitle: "C++ for AI-Infra: Series Recap and Final Self-Test"
tags: [C++, AI, AI-Infra]
catalog: true
date: 2026-02-15 20:00:00
---

八篇正文回答了一个问题：**PyTorch 和 vLLM 的 C++ 源码里，这段代码为什么这样写**。第一篇讲一个 `.cpp` 怎么变成 `.so`、`import torch` 拉起哪几个库；第二篇讲 `at::Tensor` 为什么是一个 8 字节的句柄、数据什么时候释放；第三篇讲 `AT_DISPATCH` 里的 `scalar_t` 从哪里来；第四篇讲 Dispatcher 用什么机制调到 kernel；第五篇讲算子怎么在没有任何显式调用的情况下登记进 `torch.ops`；第六篇讲 `no_grad` 为什么对其他线程不生效；第七篇讲一个 Tensor 跨过 Python/C++ 边界经过了什么；第八篇讲一个改动从写完到能提 PR 要跑哪些东西。八篇合起来，是[总纲](/cpp-for-ai-infra.html)开篇那段 `scale_shift_cpu` 的逐行注解。

本文不讲新内容，做三件事：把八篇压成一张表与八段回顾，把贯穿八篇的几条线拎出来，然后给一套三段式的通关自测——判断与计算、跨篇综合、面试题。各篇末尾的自测检验的是"这一篇读懂了没有"，这里检验的是"八篇能不能连起来用"。

> **读完这八篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**C++ 把 Java 交给运行时的决定——对象放在哪里、活多久、类型是什么、调哪个实现、线程状态怎么传、怎么与另一个运行时对话、编成什么——全部前移到了编译期和链接期，由程序员显式做出。** 这带来了性能和确定性，也带来了八篇讨论的全部复杂性。八篇按"读懂一个大型 C++ 项目需要的知识"的依赖顺序展开：前四篇是语言核心（编译模型、对象模型、模板、多态），后四篇是工程实践（宏与注册、并发、Python 边界、工具链），每篇都从 PyTorch v2.10.0 或 vLLM v0.15.0 的一段真实源码出发，以 Java 为参照系，在 mini-c10 里把机制再实现一遍。

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 判据 |
|---|---|---|---|
| 第一篇：编译模型与项目布局（[上](/cpp-compilation-model-from-cpp-to-shared-object.html) · [下](/cpp-project-layout-namespaces-libraries-and-cmake.html)） | `import torch` 加载了哪些 `.so`？依赖关系是什么？扩展链接到哪一个？ | 编译器只看得见一个翻译单元，符号由链接器与动态加载器在两个时刻解析；每类"找不到"只出现在一个阶段 | 四阶段 + 加载第五步；20 行源文件预处理后 40222 行；四种链接属性；`ld.so` 搜索顺序 RPATH → `LD_LIBRARY_PATH` → RUNPATH；`libtorch_global_deps.so`（`RTLD_GLOBAL`）→ `_C` → `torch_python` → `torch` → `torch_cpu` → `c10` |
| [第二篇：对象模型与 RAII](/cpp-value-semantics-ownership-and-raii.html) | `at::Tensor y = x;` 之后 `y` 与 `x` 是什么关系？数据什么时候释放？ | `Tensor` 是 8 字节值句柄，`=` 拷贝句柄、计数 +1；数据在最后一个 `TensorImpl` 与所有 view 的 `StorageImpl` 计数归零那一刻同步释放 | `Tensor` 8 字节 vs `shared_ptr` 16 字节 + 控制块；400M 个活 tensor 每加一个字多 3.2 GB；`y = x` / view / clone 三种关系；`del` 后显存未降的四步排查；`const` 五个位置且 `Tensor` 的 `const` 是浅的 |
| [第三篇：模板与泛型编程](/cpp-templates-and-generic-programming.html) | `AT_DISPATCH` 里的 `scalar_t` 从哪里来？lambda 被编译了几次？ | 模板为每组参数生成一份代码；`AT_DISPATCH` 是一个 `switch`，每个 `case` 里 `using scalar_t = ...` 再粘贴 lambda，N 个 dtype 就编 N 份 | 浮点两份、`ALL_TYPES_AND_HALF` 十几份；不从返回值推导 → `data_ptr<T>()` 必须显式；`DimVector` 内联 5 维；vLLM 3 dtype × 2 width = 6 份 kernel；`[&]` 同步不逃逸才安全 |
| [第四篇：多态与类型擦除](/cpp-polymorphism-and-type-erasure.html) | Dispatcher 用什么机制调到 CPU kernel？为什么既有 boxed 又有 unboxed？ | 不是虚函数，是函数指针 + 模板生成的适配器；unboxed 为快，boxed（`Stack*` 上的 `IValue`）为通用层写一次 | `KernelFunction` = `intrusive_ptr<OperatorKernel>` + boxed 指针 + `void*` unboxed；`std::function` 32 / `function_ref` 16 / 函数指针 8 字节；`IValue` 16 字节（4 tag + 8 payload）；`lookup` 一次数组下标、全程无虚调用 |
| [第五篇：宏、静态注册与代码生成](/cpp-macros-static-registration-and-codegen.html) | 一个 `.so` 被 `import` 后，算子怎么出现在 `torch.ops.myops` 下？ | `TORCH_LIBRARY` 展开成静态对象，`dlopen` 执行 `.init_array` 时其构造函数向 `Dispatcher` 登记；调用者是加载器，不是用户代码 | 宏三种用途；`TORCH_CHECK` 必须是宏的两个理由；四种链接方式里只有"静态库直接链"注册表为空；`native_functions.yaml` 2666 条目、一条生成十几处；跨库类型要 `C10_API` 导出 typeinfo |
| [第六篇：并发、内存模型、TLS 与守卫](/cpp-concurrency-memory-model-tls-and-guards.html) | `with torch.no_grad():` 在 C++ 层做了什么？为什么对其他线程不生效？ | 改的是一个 `thread_local` 变量；`thread_local` 每线程一份、新线程不继承，要跨线程必须用 `ThreadLocalState` 显式传播 | 六种 memory order；计数 relaxed 增、acq_rel 减；64 位合并计数（低 32 强、高 31 弱、第 63 位 PyObject）；守卫三步骨架；线程数优先级 `set_num_threads` > `OMP_NUM_THREADS` > `MKL_NUM_THREADS` > 核数；`GRAIN_SIZE` 32768 |
| [第七篇：pybind11、Python C API 与 ABI](/cpp-pybind11-python-c-api-and-abi.html) | 一个 Tensor 从 Python 到 C++ 再回来，几次转换、几次计数变化、GIL 状态如何？ | 输入 3 次、输出 2 次类型转换，无一步拷贝数据；C++ 计数 1 → 2 → 1 联动一次 `Py_INCREF`/`Py_DECREF`；GIL 转参数时持有、跑 kernel 时释放 | new / borrowed / stolen 三种引用；1 → 2 `Py_INCREF`、2 → 1 `Py_DECREF`；ABI 三层（CPython、libstdc++、PyTorch）；2.7 起 Linux wheel 全部 CXX11 ABI，v2.10.0 开关已删；`cpp_extension` 检查 GCC ≥ 5、不检查 PyTorch 版本 |
| [第八篇：构建、调试与测试工具链](/cpp-build-debug-and-test-toolchain.html) | 一个 C++ 改动，从写完到确认正确、无内存错误、不在别的编译器上炸，要跑哪些东西？ | 八步清单：clangd → clang-format → Debug 构建 → gtest/pytest → clang-tidy → ASan+UBSan → TSan → CI 矩阵；"不靠推理，靠矩阵" | `-O0` 比 `-O2` 慢 3–10 倍、`-g` 不影响速度；ASan 约 2× 时间、TSan 5–15×、二者互斥；`detect_leaks=0`；三张栈；GCC ≥ 9.3、CUDA ≥ 12.0、C++17；`intrusive_ptr_test.cpp` 325 个测试 |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿八篇的五条线：引用计数、RAII 从资源到上下文、编译期与运行期分派、符号与 ABI、错误从抛出到 Python |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |

## 二、逐篇回顾

### 1. 第一篇：从源码到二进制——编译模型与项目布局

（拆成[上：编译模型](/cpp-compilation-model-from-cpp-to-shared-object.html)与[下：工程布局](/cpp-project-layout-namespaces-libraries-and-cmake.html)两篇。）

**核心问题**：`import torch` 时加载了哪些 `.so`？它们之间是什么依赖关系？我写的扩展链接到哪一个？

**结论**：一个 `.cpp` 经预处理、编译、汇编、链接四个阶段变成二进制，加载是运行时的第五步；编译器只看得见一个翻译单元（一个 20 行的例子预处理后是 40222 行），符号的地址由链接器（链接期）与 `ld.so`（加载期）两次解析，所以每类错误只在一个阶段出现：`not declared` 是编译期、`undefined reference` 是链接期、`undefined symbol` 是加载期。ODR 规定非 inline 实体全程序一个定义，`inline`、模板、类可多份但必须逐字相同，`static` 与匿名命名空间是真正没有外部名字的内部链接。`-fvisibility=hidden` 加逐库的 `C10_API`/`TORCH_API` 决定 `.so` 导出什么；`.a` 在链接期裁剪拷贝、`.so` 只记一条 `DT_NEEDED` 到加载期递归解析——不管用不用，找不到任何一个就整个失败。`import torch` 先以 `RTLD_GLOBAL` 加载空壳 `libtorch_global_deps.so`，再加载 15 行 stub 编成的 `torch/_C.cpython-*.so`，后者沿 `$ORIGIN/lib` 的 RUNPATH 拉起 `libtorch_python.so` → `libtorch.so`（空壳）→ `libtorch_cpu.so`（+ `libtorch_cuda.so`）→ `libc10.so`（+ `libc10_cuda.so`）。命名空间 `c10::`/`at::`/`torch::` 大致对应前三层，但 `torch::` 横跨两个库——命名空间、目录、库是三个独立维度。

**必记**：

- 四阶段 + 加载第五步；三类"找不到"分别落在编译期、链接期、加载期。
- 四种链接属性：普通函数 = 外部链接（多份 multiple definition）；`inline`/模板/类内成员/`constexpr` = 允许重复但必须相同；`static`/匿名命名空间 = 内部链接；`extern` 声明不算定义。
- `ld.so` 搜索顺序：`DT_RPATH`（无 RUNPATH 时）→ `LD_LIBRARY_PATH` → `DT_RUNPATH` → `ld.so.cache` → `/lib`、`/usr/lib`；现代链接器默认写 RUNPATH，`$ORIGIN` 是部署的正解。
- 扩展链接：用 `c10::` 链 `-lc10`；用 `at::Tensor`、算子、`TORCH_LIBRARY` 链 `-ltorch_cpu`（或 `-ltorch`）；CUDA 加 `-lc10_cuda -ltorch_cuda`；pybind11 的 Tensor 转换加 `-ltorch_python`。GNU ld 默认 `--no-copy-dt-needed-entries`，`libc10.so` 里的符号必须显式 `-lc10`。
- 定位技巧：`c10::` → `c10/` → `libc10.so`；`at::` → `aten/` → `libtorch_cpu.so`；`build_variables.bzl` 是"哪个 `.cpp` 进哪个库"的权威清单。

**常见误解**："编译器能看到整个项目"——它只能看到一个翻译单元，头文件是手写的接口；"找不到类是运行时异常"——在 C++ 里它是构建失败或进程起不来，三个阶段都在业务代码运行之前。

### 2. 第二篇：值、引用与所有权——对象模型与 RAII

**核心问题**：`at::Tensor y = x;` 之后 `y` 和 `x` 是什么关系？什么时候数据真正被释放？

**结论**：C++ 变量默认就是对象，`=` 与按值传参、返回默认是拷贝；`const T&` 是只读输入的零开销传法，`T*` 表达非拥有或可空，引用不延长寿命。编译器默认逐成员生成六大特殊成员函数，直接持有裸资源的类逐成员拷贝会 double free，要么 Rule of Five 六个都想一遍，要么 Rule of Zero 一个都不写；`= default` 的是值类型（`Tensor`、`Storage`），`= delete` 的是实体（`TensorImpl`、`StorageImpl`）。右值是马上要死的临时对象，`std::move` 只是类型转换，NRVO 让 `return out;` 零成本。RAII 把资源生命周期绑到对象上，析构确定、同步、逆序、异常时也运行，比 `try-with-resources` 强在可以作为成员、被移动、放进容器。PyTorch 自己造 `intrusive_ptr` 是因为 `Tensor` 是最高频被拷贝的对象：计数放进对象，8 字节、无控制块、可与裸指针互转（`release`/`reclaim` 是穿越 C/Python 边界的桥），`make_intrusive` 直接写计数不用原子加。持有链是 `Tensor`（值）→ `intrusive_ptr<TensorImpl>` → `TensorImpl`（实体）→ `Storage`（值）→ `intrusive_ptr<StorageImpl>` → `StorageImpl` → `DataPtr` → 删除器 → `free`/`cudaFree`/缓存池，整条链没有一处手工 `delete`。`y = x` 让两个 8 字节句柄指向同一个 `TensorImpl`（计数 1 → 2，共享一切包括形状与 autograd 状态）；view 新建 `TensorImpl` 只共享 `StorageImpl`；clone 什么都不共享。

**必记**：

- `sizeof`：`Tensor`/`unique_ptr`/裸指针 8 字节，`shared_ptr` 16 字节 + 单独分配的控制块，带函数指针删除器的 `unique_ptr` 16 字节；`TensorImpl.h` 注释：400M 个活 tensor，每加一个 64 位字多 3.2 GB。
- 传参三条：只读 `Tensor` 用 `const at::Tensor&`（按值多两次原子操作，`at::Tensor&` 拒绝临时对象）；`double`、`Device`、`ScalarType` 等小值按值；要存下来的按值接收再 `std::move`。
- 数据释放当且仅当：所有指向该 `TensorImpl` 的句柄析构 → 没有别的 view 持有同一 `StorageImpl` → `DataPtr` 析构调删除器 → CPU `free_cpu`、CUDA 默认还给缓存池。
- "`del` 了显存没降"四步排查：别的 Python 引用 → autograd `SavedVariable` → view 还活着 → 在缓存池里。
- `const Tensor&` 保护的是句柄不是数据：in-place 算子的输出参数写成 `const Tensor&` 照样能写数据。
- 移动构造/移动赋值/析构标 `noexcept`，否则 `std::vector` 扩容退回拷贝；不要在构造函数里从 `this` 创建 `intrusive_ptr`（计数还是 0）。

**常见误解**："`=` 是起别名"——它是拷贝，要别名用引用或指针；"拷贝一个大对象一定贵"——`Tensor` 看起来大拷起来只是 8 字节，`std::vector<int64_t>` 看起来小拷起来要分配，要看它是句柄还是值。

### 3. 第三篇：模板与泛型编程

**核心问题**：`AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "name", [&] { ... scalar_t ... })` 里的 `scalar_t` 从哪里来？这个 lambda 被编译了几次？

**结论**：模板是生成代码的配方，一个模板 + N 组参数 = N 份独立机器码，各有各的符号；Java 泛型擦除类型、只有一份字节码，C++ 恰恰相反，所以能用基本类型、能按类型换实现、能把值当参数，代价是编译慢、二进制大、错误信息长。函数模板从实参推 `T`、不能从返回值推，所以 `data_ptr<scalar_t>()` 必须显式写。非类型模板参数（`std::array<T, N>`、`SmallVector<T, N>`、CUDA kernel 的 `BLOCK_SIZE`）必须是编译期常量，运行期的值要经过一组分支才能进入模板——这是 `AT_DISPATCH` 与 vLLM if 链的共同本质。`AT_DISPATCH_FLOATING_TYPES` 是三层宏：外层生成 `case` 列表，`AT_DISPATCH_SWITCH` 是一个 IIFE 里的 `switch`（`default` 抛 `NotImplementedError`），每个 `case` 里 `using scalar_t = ScalarTypeToCPPTypeT<enum_type>;` 然后 `return lambda();`。`scalar_t` 是 `case` 块作用域里的类型别名，名字由宏硬编码、类型由 `ScalarTypeToCPPType` 全特化表查出；lambda 的文本被复制到每个 `case`，N 个 dtype 就是 N 个闭包类型各编译一次。`ArrayRef` 是指针 + 长度的不拥有视图，只做参数和返回值；`DimVector` 是 5 维内联的 shape 容器。lambda 的闭包类型唯一、可内联，`[&]` 不延长寿命，同步且不逃逸时安全，要存起来或异步执行时按值或初始化捕获。

**必记**：

- `AT_DISPATCH_FLOATING_TYPES` 两个 `case`（`Double`、`Float`），lambda 文本复制两份各编译一次；传 `Half`/`BFloat16` 走 `default` 抛错，要支持它们用 `_AND2(kHalf, kBFloat16, ...)`——`TORCH_CHECK(x.is_floating_point())` 挡不住它们。
- 外层两个 `case` 意味着内层 `parallel_for` 也有两份实例（`F` 分别是 `double` 版与 `float` 版闭包类型）。
- vLLM `VLLM_DISPATCH_FLOATING_TYPES` 是 Float/Half/BFloat16 三种，乘 `width` 两种 = 6 份 kernel，运行期两层分支选一份。
- `C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE` 是 5，`DimVector = SmallVector<int64_t, 5>`，5 维以内不分配。
- 一个 256 位寄存器装 8 个 `float` 或 4 个 `double`，`Vectorized<float>` 与 `Vectorized<double>` 是不同的类——dtype 判断不能放进最内层循环。
- 接受可调用对象：热路径、要内联 → 模板参数 `const F&`；要跨 `.so`、要存起来 → `std::function`。

**常见误解**："C++ 模板就是 Java 泛型"——`vector<int>` 与 `vector<string>` 没有共同基类、不能向上转型，`instanceof T` 不存在，按类型分支用 `if constexpr`；"`[&]` 总是安全"——只在 lambda 于被捕获变量的生命周期内被调用完时安全，扔进线程池后函数返回就是悬垂。

### 4. 第四篇：多态与类型擦除——运行时如何选择实现

**核心问题**：Dispatcher 拿到一个 `OperatorHandle` 和一组参数后，用什么 C++ 机制调到 CPU kernel？为什么既有 boxed 又有 unboxed？

**结论**：C++ 方法默认不虚，`virtual`/`override`/`final`/`= 0` 各有职责；每个有虚函数的对象多一个 vptr，每次虚调用一次不可内联的间接跳转；通过基类指针删除必须有虚析构，这是 `intrusive_ptr_target` 与 `OperatorKernel`（只有一个虚析构的空结构体）存在的理由。`TensorImpl` 用"policy 字节快路径 + `*_custom` 虚定制点"兼顾速度与扩展，`Tensor` 作为值句柄没有任何虚函数。运行期选实现有五种以上做法——虚函数、函数指针、`std::function`/`function_ref`、模板参数、CRTP、类型擦除、`std::variant`——虚函数只是其中开销最大、最灵活的一种。`KernelFunction` 是手工类型擦除：`intrusive_ptr<OperatorKernel> functor_` + `boxed_kernel_func_`（永远有效）+ `void* unboxed_kernel_func_`（可能为空）；注册时模板把具体签名包成统一形态，调用时 `call<Return, Args...>` 把 `void*` cast 回精确签名做一次间接调用，全程无虚调用，签名一致性由 `CppSignature` 在注册与 `typed<>()` 时双重校验。unboxed 保留 C++ 签名、零装箱、可内联，是热路径；boxed 用 `Stack*`（`IValue` 向量）作统一约定，让 autograd、tracing、fallback、Python 这些"不知道具体签名"的通用层写一次即可，两者可互转。`IValue` 是手工的 tagged union，不是基类、类型集合封闭。`TORCH_CHECK` 抛 `c10::Error : std::exception`，子类决定跨到 Python 后的异常类型。

**必记**：

- 三种可调用抽象：`std::function` 拥有、32 字节、可能堆分配；`c10::function_ref` 借用、16 字节、无分配、不能存；函数指针 8 字节、不能带状态；模板参数零开销可向量化。
- `IValue` = 4 字节 tag + 8 字节 payload = 16 字节值类型，无堆分配；`Tensor` 是 union 里唯一的非平凡成员；`toTensor()` 三个 ref-qualifier 重载区分"借"与"搬"。
- `Dispatcher::call` 算 `DispatchKeySet` → `OperatorEntry::lookup` 一次数组下标 → `KernelFunction::call` 一次间接调用 → 内联的包装链直达 kernel。
- 把函数指针存成 `void*` 后 cast 回来的签名必须一字不差，否则是 UB 不是编译错误；注册优先 `TORCH_FN(fn)`（编译期函数指针可内联）。
- 用 `TORCH_CHECK` 系列不手写 `throw`；`TORCH_CHECK_INDEX`/`_VALUE`/`_TYPE`/`_NOT_IMPLEMENTED` 对应 Python 的 `IndexError`/`ValueError`/`TypeError`/`NotImplementedError`。

**常见误解**："运行期选实现只能用接口 + 虚方法"——PyTorch 的 Dispatcher 全程没有虚调用；"`IValue` 就是 Java 的 `Object`"——它不是所有类型的基类，类型集合封闭，`toX()` 不做转换（`IValue(1).toDouble()` 会抛）。

### 5. 第五篇：宏、静态注册与代码生成

**核心问题**：一个 `.so` 被 `import` 后，里面的算子怎么就出现在 `torch.ops.myops` 下了？没有任何函数被显式调用。

**结论**：预处理器是编译前的文本变换，`#` 字符串化、`##` 拼接、`__VA_ARGS__` 变参；参数紧挨 `#`/`##` 时不先展开，所以要两层间接（`C10_STRINGIZE`、vLLM 的 `TORCH_LIBRARY_EXPAND`）。宏承担三种职责：条件编译、生成重复代码（X-macro `AT_FORALL_SCALAR_TYPES`）、在调用点捕获信息。`TORCH_CHECK` 必须是宏，因为函数拿不到调用点的 `__FILE__`/`__LINE__` 与条件表达式文本，且 `msg` 要惰性求值。静态注册的机制是：`TORCH_LIBRARY(myops, m) { ... }` 展开成一个函数定义加一个静态存储期对象 `TorchLibraryInit`，`import` 是 `dlopen`，`dlopen` 返回前执行 `.init_array`，这个对象的构造函数向 `Dispatcher::singleton()` 登记命名空间并调用函数体，`m.def`/`m.impl` 落到 `registerDef`/`registerImpl`；`torch.ops.myops.foo` 由 `_OpNamespace.__getattr__` 用字符串 `"myops::foo"` 惰性查表。让它可靠的三个配套：注册表容忍任意顺序（`impl` 可先于 `def`，`OperatorEntry::schema_` 是 `optional`）、单例用函数内静态规避初始化顺序问题、静态库要 `--whole-archive` 否则没被引用的注册 `.o` 被裁掉——这是 Java 里完全不存在的失效条件。宏做不了跨文件、读外部数据的生成，PyTorch 用 `torchgen` 从 `native_functions.yaml` 生成 C++。

**必记**：

- `TORCH_CHECK` 是宏的两个理由：调用点捕获 `__FILE__`/`__LINE__`/`__func__`；`msg` 只在 `cond` 为假时才拼字符串。
- 四种链接方式：直接链 `.o` 正常；打成静态库直接链——**无编译错误、无链接错误、注册表为空**；静态库 + `--whole-archive`（macOS `-force_load`）正常；动态库正常（加载时全部 `.init_array` 执行）。
- 函数内静态单例 `static Registry& instance() { static Registry r; return r; }`——第一次调用时构造、线程安全，`Dispatcher::realSingleton()` 就是这样。
- 跨库使用的类型必须导出：不只函数，还有异常类（`class C10_API Error`）、多态基类的 vtable/typeinfo；漏掉一个，Linux 可能链接错误、macOS 可能 `catch` 失效。
- `native_functions.yaml` 16000 多行、2666 个 `- func:` 条目（v2.10.0）；一个 `bincount` 条目至少产出 `Functions.h`、`RegisterCPU.cpp`、`TensorBody.h` 等十几处代码；PyTorch 2600 多个算子，vLLM 百来个所以胶水手写。
- 注册器构造函数只做登记：不初始化 CUDA、不读配置、不打日志。

**常见误解**："宏只是文本替换，能不用就不用"——`TORCH_CHECK` 与 `TORCH_LIBRARY` 做的事函数与模板做不到；"登记了就一定在"——静态库会把没被引用的目标文件整个扔掉，算子"消失"时要看链接命令而不是代码。

### 6. 第六篇：并发、内存模型、TLS 与守卫

**核心问题**：`with torch.no_grad():` 在 C++ 层做了什么？为什么它对其他线程不生效？

**结论**：`__enter__` 调 `torch._C._set_grad_enabled(False)` → `GradMode::set_enabled(false)`，改的是 `thread_local AutogradState autograd_state_tls` 里的 grad-mode 位；每个 autograd kernel 建图前读 `GradMode::is_enabled()`；`__exit__` 写回旧值。Python 的 `no_grad` 与 C++ 的 `NoGradGuard` 是同一个东西：保存旧值、设新值、退出时恢复。`thread_local` 是语言级存储类别，每线程一份、访问通常一条指令、新线程不继承创建者的值，所以主线程 `no_grad` 里启动的 worker 读到的是默认值 `True`；要跨线程生效必须显式传递——`at::ThreadLocalState` 打包 grad mode、dispatch key 集合、当前 stream 等，`ThreadLocalStateGuard` 在工作线程里恢复，`parallel_for`、autograd 引擎、`torch.jit.fork` 都这样做。C++ 内存模型与 JMM 同源但把 `volatile` 一档拆成六档；`intrusive_ptr` 计数 +1 用 relaxed（对象已被持有，不需要发布或获取任何数据），−1 用 acq_rel（每次减都可能是最后一次，要同时扮演 release 与 acquire 才能安全析构）。守卫是 RAII 从"管资源"推广到"管上下文"：三步骨架，删掉拷贝与移动，可作为成员组合；`DeviceGuard` 用"虚接口 + 内联模板"两层在不依赖 CUDA 的 `libc10` 里切换 CUDA 设备。CUDA launch 不用锁，因为 runtime 线程安全、同一 stream 按入队顺序执行、"当前设备"与"当前 stream"都是 TLS。

**必记**：

- 六种 memory order；`relaxed` 只保证原子性，release 写与 acquire 读配对建立 happens-before；不确定就用默认 `seq_cst`。
- v2.10.0 把强、弱计数合并成一个 64 位 `combined_refcount_`：低 32 位强引用、高 31 位弱引用、第 63 位 `kHasPyObject`；一条原子指令同时操作两个计数。
- 守卫三步骨架：构造保存旧值并设新值、析构恢复；`= delete` 拷贝与移动；守卫必须是有名字的局部变量（`c10::InferenceMode();` 立刻析构等于没生效）。
- `parallel_for` 并行的三个条件：元素数超过 `grain_size`、不在并行区域内、线程数大于 1；不能嵌套；多数 ATen 算子用 `GRAIN_SIZE` 32768；线程数优先级 `torch.set_num_threads()` > `OMP_NUM_THREADS` > `MKL_NUM_THREADS` > 核数。
- `std::mutex` 不可重入（重入是 UB）；`condition_variable::wait` 一律带谓词、配 `unique_lock`；不在持锁时调用户回调（`ThreadPool::main_loop` 先 `unlock` 再执行任务）。
- CUDA host 代码用 `c10::cuda::CUDAGuard`（模板实例化、去虚化）而不是 `c10::DeviceGuard`（多一次虚调用和注册表查找）；launch 后紧跟 `C10_CUDA_KERNEL_LAUNCH_CHECK()`。

**常见误解**："C++ `volatile` 像 Java 一样提供可见性与顺序"——它与线程无关，同步用 `std::atomic`；"守卫对象可以传给别的线程去析构"——它恢复的是构造线程的状态，PyTorch 通过删除移动构造从根本上杜绝这条路。

### 7. 第七篇：与 Python 之间——pybind11、Python C API 与 ABI

**核心问题**：一个 `at::Tensor` 从 Python 传到 C++ 又返回 Python，经过了几次类型转换、几次引用计数变化？中间 GIL 状态是什么？

**结论**：`PyObject_HEAD` 是每个 Python 对象开头的引用计数与类型指针，`THPVariable` 以它为前缀、后面跟一个 `at::Tensor cdata`——"嵌"就是内存布局上的前缀兼容。引用分 new/borrowed/stolen 三种约定，`THPVariable_Wrap` 返回新引用，`THPVariable_Unpack` 返回借用。GIL 保护的是解释器状态而非 C++ 内存：`gil_scoped_release` 是 `PyEval_SaveThread/RestoreThread` 的 RAII，进 kernel 前释放、参数转换必须在此之前，释放后不能碰任何 `PyObject`（包括让 `py::object` 析构），但可以随便用已取出的 C++ 值。PyTorch 两条绑定路线并存：普通类型走 pybind11，`Tensor` 用手写 C API，因为需要透明子类化、双向持有与同一性（`pyobj_slot`）、GC 集成、`PythonArgParser` 的性能；双向持有靠 `kHasPyObject` 位加"C++ 计数 1 → 2 时 `Py_INCREF`、2 → 1 时 `Py_DECREF`"，环靠 `tp_traverse` 报告给 CPython GC。以 `torch.ops.myops.op(t)` 为例：输入 3 次转换（`PyObject*` → `at::Tensor` → `IValue` → `const at::Tensor&`），输出 2 次（`Tensor` → `IValue` → `PyObject*`），全是句柄级操作；C++ 计数 1 → 2 → 1，Python 计数经钩子 +1/−1；GIL 在参数与结果转换时持有、kernel 执行时释放、出栈时 `decref_pyobject` 在无 GIL 区间短暂拿回。进计算图的算子必须走 `TORCH_LIBRARY`（可 trace、可注册 Meta/Autograd、不依赖 `libtorch_python`），vLLM 全部如此。ABI 有三层：CPython、C++ 标准库、PyTorch 自身，每一层都能让 `import` 失败。

**必记**：

- 一次往返：输入 3 次、输出 2 次类型转换；C++ 计数 1 → 2 → 1；一次 `Py_INCREF` 一次 `Py_DECREF`；没有一步拷贝数据；原生 `torch.add` 不经 `IValue`，少两次转换。
- 三条 GIL 规则：进 kernel 前 `gil_scoped_release`；作用域内不声明 `py::object`、不调 C API；C++ 线程回调 Python 先 `Py_IsInitialized()` 再 `gil_scoped_acquire`。
- `_GLIBCXX_USE_CXX11_ABI`：2.6 部分 wheel 切到 `=1`（同时切 manylinux_2_28）、2.7 起所有 Linux wheel 与 libtorch 都是 `=1`、v2.10.0 开关已不存在（`torch._C._GLIBCXX_USE_CXX11_ABI` 恒 `True`，`cpp_extension.py` 不再传 `-D`）——在 v2.10.0 上 `__cxx11` 类的 undefined symbol 只在你自己显式加了 `=0` 时出现。
- `cpp_extension.py` 自动处理头文件、链接库、`-std=c++17`、`TORCH_EXTENSION_NAME`，检查编译器种类、GCC ≥ 5、CUDA 与 host 编译器区间；不检查也检查不了"PyTorch 版本一致"；`TORCH_DONT_CHECK_COMPILER_ABI=1` 是关警告不是修问题。
- `undefined symbol` 三步：`c++filt`，看符号在不在目标库（`nm -D`）、参数里有没有 `__cxx11`、函数在这个 PyTorch 版本里是否存在；`GLIBCXX_3.4.x not found` 是运行时 `libstdc++.so.6` 太旧。
- 需要 `abi3` 就不能用 `libtorch_python.so`，也就不能用 pybind11 里的 PyTorch caster，只能走 `TORCH_LIBRARY`。

**常见误解**："放掉 GIL 就不能碰任何东西"——不能碰的是 `PyObject`，`self` 那个 `at::Tensor` 是 C++ 对象，只要有人持有它的计数就活着；"PyTorch 全用 pybind11"——`Tensor` 用手写 C API，两者在同一个文件里并存。

### 8. 第八篇：构建、调试与测试工具链

**核心问题**：一个 C++ 改动，从写完到确认正确、没有内存错误、不会在别的编译器上炸，需要跑哪些东西？

**结论**："确认正确"靠 Debug 构建加 gtest/pytest——Debug 构建让 gdb/lldb 看到真实变量，gtest 测语言层面的契约，pytest 测能从 Python 观察到的行为；"没有内存错误"靠 ASan+UBSan 构建再跑一遍测试——越界与 use-after-free 默认什么都不发生，只有插桩构建能把它们变成带三张栈（访问、释放、分配）的确定报告，改了并发代码再加 TSan；"不会在别的编译器上炸"靠 CI 矩阵——gcc 与 clang 各编一遍、x86 与 aarch64 各跑一遍，不靠推理靠矩阵。CMake 描述目标与属性，`PUBLIC`/`PRIVATE`/`INTERFACE` 决定传递，`find_package(Torch)` 给的 `torch` 目标一行带来头文件、库与 `-std=c++17`；PyTorch 全量构建慢在翻译单元多且每个展开几十万行、`.cu` 编译约为 `.cpp` 十倍，日常靠 Ninja、ccache、`USE_CUDA=0`、`USE_CUSTOM_DEBINFO` 把增量控制在分钟级。`-O2` 下变量被优化掉、帧被内联，`<optimized out>` 不是调试器的错。CI 在 ASan 构建下先用 `_crash_if_csrc_asan(3)` 这类函数故意让进程崩、确认崩了再跑测试，是对检测工具本身的自检。版本矩阵三个轴——C++ 标准、主机编译器、CUDA——加上第七篇的标准库 ABI 第四轴，没有 `--release` 这样的开关能屏蔽差异。

**必记**：

- 八步清单：clangd 无红线 → clang-format → Debug 构建 `-Wall -Wextra` 无新警告 → 相关 gtest/pytest → clang-tidy → 改了内存代码跑 ASan+UBSan → 改了并发代码跑 TSan → CI（gcc 11 + clang 12/18 + aarch64 + CUDA）。1–4 每次做，5 提交前，6、7 按改动性质，8 由 CI 承担。
- `-O0` 比 `-O2` 慢 3–10 倍是常态；`-g` 对运行速度无影响。
- Sanitizer 开销：ASan 约 2× 时间、2–3× 内存；UBSan 很小、可与 ASan 同开；TSan 5–15× 时间、5–10× 内存、与 ASan 互斥；只检查插了桩的代码；嵌入 Python 一律 `detect_leaks=0`。
- 32 核机器 CPU-only Debug 干净构建二三十分钟；改一个 `.cpp` 到 `import torch` 一两分钟（瓶颈是链接）；改 `TensorImpl.h` 回到十几分钟；ccache 对头文件改动命中率接近 0。
- 版本矩阵：C++17、GCC ≥ 9.3、CUDA ≥ 12.0；`CUDA_GCC_VERSIONS` 里 CUDA 11.7 支持 gcc 6 到 gcc 11（上界 12 不含），表只维护到 11.7，12.x 只警告放行。
- `c10/test/util/intrusive_ptr_test.cpp` 3500 多行、325 个 `TEST`；`TORCH_SHOW_CPP_STACKTRACES=1` 是最便宜的诊断；头文件路径用 `PUBLIC`、编译选项用 `PRIVATE`。

**常见误解**："越界写会崩"——它是未定义行为，多数情况下落在填充或邻居对象上程序继续跑，只有 ASan 的红区与影子内存能把它变成确定报告；"Release 也能调试"——在 `-O2` 构建里调试是在读汇编，先 `DEBUG=1` 或 `USE_CUSTOM_DEBINFO` 再打断点。

## 三、贯穿全系列的几条线

### 1. 引用计数：一个数字穿过五篇

第二篇建立起 `intrusive_ptr`：计数放进对象、8 字节句柄、`release`/`reclaim` 可与裸指针互转；`at::Tensor y = x;` 就是这个计数从 1 变 2。第四篇的 `KernelFunction` 里 `functor_` 是一个 `intrusive_ptr<OperatorKernel>`，`OperatorKernel` 继承 `intrusive_ptr_target` 并只有一个虚析构——同一套所有权语义被用来持有类型擦除后的 kernel。

第六篇给这个计数加上并发语义：+1 用 relaxed、−1 用 acq_rel，v2.10.0 把强、弱计数合并进一个 64 位字段，低 32 位强、高 31 位弱、第 63 位标记是否有 Python 包装对象。第七篇解释这一位的用途：C++ 计数在 1 与 2 之间变化时联动一次 `Py_INCREF`/`Py_DECREF`，让 Python 与 C++ 的两套计数互相托住；一次 `torch.ops` 调用里 C++ 计数 1 → 2 → 1 的每一步都对应第二篇的一次拷贝或析构。

第八篇给它验证与观察的手段：`intrusive_ptr_test.cpp` 的 325 个测试全部是"计数何时变、对象何时死"这类语言层面的契约；ASan 报告的三张栈里，释放栈就是第二篇的析构链 `~TensorImpl` → `~StorageImpl` → `free` 一层不差地展开；TSan 在这个计数上的报告通常是 relaxed 原子操作的假阳性。

### 2. RAII：从资源到上下文到 GIL

第二篇的 RAII 讲的是资源——`unique_ptr`、`DataPtr` 的删除器、显存在最后一个句柄析构那条语句里同步释放；比 `try-with-resources` 强在可以作为成员、被移动、放进容器。第四篇借它替代 Java 的 `finally`：异常没有 checked/unchecked 之分、不能穿 C 边界，清理全靠析构。

第六篇把 RAII 推广到"上下文"：`NoGradGuard`、`InferenceMode`、`DeviceGuard`、`CUDAStreamGuard`、`ThreadLocalStateGuard`、`lock_guard` 全是同一个三步骨架——保存旧值、设新值、析构恢复——并且删掉拷贝与移动、要求有名字的局部变量，因为守卫必须与作用域一一对应、不能跨线程。第七篇在另一个运行时上再用一次：`py::object` 是 `Py_INCREF`/`Py_DECREF` 的 RAII 化，`gil_scoped_release`/`gil_scoped_acquire` 是 `PyEval_SaveThread`/`RestoreThread` 的守卫。

第三篇的 lambda 捕获与 RAII 是一体两面：`[&]` 引用捕获不延长寿命，所以只有在 lambda 被同步调用完（`AT_DISPATCH` 的 IIFE、`parallel_for` 返回前 join）时才安全；要异步就按值捕获一个 8 字节的 `Tensor` 句柄，让所有权跟着闭包走。

### 3. 编译期与运行期的分派：dtype、device、注册

第三篇解决"按 dtype 选 kernel"：`dtype` 是运行期值，kernel 需要编译期类型，`AT_DISPATCH` 用一个 `switch` 把有限个运行期值映射到有限个编译期实例——每个 `case` 一份闭包、一份 `parallel_for`、一份向量化循环。第四篇解决"按 device 选 kernel"：这必须是运行期的事，PyTorch 没有用虚函数，而是把任意签名的 kernel 用 `KernelFunction` 擦成两个函数指针加一个基类指针，`OperatorEntry` 按 `DispatchKey` 一次数组下标查到它。

第五篇解释这张表是怎么填的：`TORCH_LIBRARY_IMPL(myops, CPU, m)` 展开成静态对象，`dlopen` 时其构造函数把 `KernelFunction` 塞进 `OperatorEntry` 对应 key 的槽位；`torchgen` 从 2666 个 yaml 条目生成 `RegisterCPU.cpp` 等十几处样板，对原生算子做同样的事。第七篇则是这张表的 Python 入口：`torch.ops.myops.op(t)` 把参数装进 `IValue` 走 boxed 路径，这正是第四篇"通用层写一次"的用途。

三段合起来是一条完整的路：Python 值 → `IValue`（运行期、类型擦除）→ `KernelFunction`（运行期、按 device）→ `AT_DISPATCH`（运行期到编译期的桥）→ `scalar_t` 专用循环（编译期）。每一段的"份数"都是代价：dtype 集合越大二进制越大，`case` 之外用 `data_ptr<T>()` 就要在运行期检查。

### 4. 符号、链接与 ABI：为什么"在我机器上能跑"不成立

第一篇给出全部基本概念：翻译单元、ODR、四种链接属性、name mangling、`-fvisibility=hidden` 与 `C10_API`、`.a` 的链接期裁剪与 `.so` 的 `DT_NEEDED`、RUNPATH 与 `$ORIGIN`。第五篇把它们用在静态注册的失效条件上：静态库丢弃没被引用的 `.o` 让注册消失，`--whole-archive` 是解药；跨 `.so` 的 `catch (const c10::Error&)` 要求 typeinfo 同一份，所以异常类必须 `C10_API` 导出。

第七篇把符号问题推进到 ABI：`__cxx11` inline namespace 让 `std::string` 在两种 ABI 下 mangled 名不同，`GLIBCXX_3.4.x` 符号版本让新编译器编的库在旧运行时上找不到，CPython 的 `PyObject` 布局公开可解引用所以扩展对解释器版本敏感——三层 ABI 每一层都能让 `import` 失败，而 `cpp_extension.py` 只能检查其中一部分。第八篇给它工程化的兜底：`TORCH_USE_RTLD_GLOBAL=1` 让 UBSan 的 vptr 检查不因跨库重复 typeinfo 误报（第一篇的 `RTLD_LOCAL` 在这里露面），`detect_odr_violation=1` 抓同一符号在两个 `.so` 里的不同定义，版本矩阵与 Docker 镜像把编译器、CUDA、标准库 ABI 三个轴钉死。

### 5. 错误从抛出到 Python：一条穿过六篇的路

第五篇讲 `TORCH_CHECK` 为什么必须是宏：在调用点捕获 `__FILE__`/`__LINE__` 与条件文本，`msg` 惰性求值。第四篇讲它抛的是什么：`c10::Error : std::exception`，子类 `IndexError`/`ValueError`/`TypeError`/`NotImplementedError` 决定跨到 Python 后的异常类型；异常不能穿过 C 边界、析构函数、`noexcept` 函数。

第七篇讲边界上怎么翻译：pybind11 的调用桩自动 `catch` 并按注册的翻译器设置 Python 异常；C API 风格的 `THPVariable_xxx` 用 `HANDLE_TH_ERRORS`/`END_HANDLE_TH_ERRORS` 手工 `catch`、`PyErr_SetString`、`return nullptr`。第一篇与第五篇给出它跨库成立的前提：`class C10_API Error` 导出 typeinfo，否则 `-fvisibility=hidden` 下两个 `.so` 各有一份 typeinfo，`catch` 匹配失败直接 `terminate`。第八篇给出观察它的工具：`TORCH_SHOW_CPP_STACKTRACES=1` 让 `c10::Error` 附上 C++ 栈，`.clang-tidy` 的 `hicpp-exception-baseclass` 保证凡 `throw` 出去的都派生自 `std::exception`。第三篇也在这条路上：`AT_DISPATCH` 的 `default` 分支抛的 `NotImplementedError`、`data_ptr<T>()` 类型不匹配抛的 `TORCH_CHECK` 错误，都沿这条路到达 Python。

| 概念 | 出现的篇 | 关系 |
|---|---|---|
| `intrusive_ptr` 引用计数 | 二、四、六、七、八 | 二定义所有权；四用它持有 `OperatorKernel`；六给内存序与 64 位合并字段；七让它与 `Py_INCREF` 联动；八用 325 个测试与 ASan 三张栈观察它 |
| RAII / 守卫 | 二、三、四、六、七 | 二管资源；三的 `[&]` 以"同步调用完"为安全条件；四替代 `finally`；六推广到上下文并删掉拷贝移动；七变成 `py::object` 与 `gil_scoped_release` |
| 编译期 vs 运行期分派 | 三、四、五、七 | 三按 dtype（`switch` 到编译期）；四按 device（类型擦除）；五填表（静态注册、torchgen）；七从 Python 经 boxed 路径进表 |
| 符号可见性与链接 | 一、五、七、八 | 一给概念；五给静态注册的失效条件与 typeinfo 导出；七推进到三层 ABI；八用 `RTLD_GLOBAL`、ODR 检测与版本矩阵兜底 |
| `TORCH_CHECK` 与 `c10::Error` | 三、四、五、七、八 | 五说为什么是宏；四说抛的是什么；七说怎么翻译到 Python；三的 `default`/`data_ptr` 走同一条路；八给 `TORCH_SHOW_CPP_STACKTRACES` 与 lint 规则 |
| `thread_local` 与 TLS 传播 | 三、六、七 | 三说 `parallel_for` 的 lambda 只碰裸指针；六说新线程不继承、`ThreadLocalState` 显式传播；七的 GIL 是"全局一把"与 TLS 的对照 |
| C++17 与 v2.10.0 基线 | 一、二、三、七、八 | 一与二在 `CMakeLists.txt` 确认 `CMAKE_CXX_STANDARD 17`；三说 concepts 源码树未用；七说 `cpp_extension` 传 `-std=c++17`；八说扩展的标准要跟 PyTorch 走 |

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| 找不到符号是运行时异常，像 `NoClassDefFoundError` | 三类"找不到"分别在编译期、链接期、加载期，都在业务代码运行之前 | 按错误信息判断阶段：`not declared` / `undefined reference` / `undefined symbol` | [第一篇](/cpp-compilation-model-from-cpp-to-shared-object.html) |
| `-ltorch_cpu` 已经依赖 `libc10.so`，不用再写 `-lc10` | GNU ld 默认 `--no-copy-dt-needed-entries`，不通过依赖的依赖满足引用 | 引用了 `c10::` 的符号就显式 `-lc10` | [第一篇](/cpp-compilation-model-from-cpp-to-shared-object.html) |
| `Tensor y = x;` 是起别名，和 Java 一样不花钱 | 它是值拷贝，拷的是带引用计数的 8 字节句柄，一次原子加、将来一次原子减 | 只读参数用 `const Tensor&` 省掉这两次原子操作 | [第二篇](/cpp-value-semantics-ownership-and-raii.html) |
| `del x` 之后显存立刻回来 | 数据只在最后一个 `TensorImpl` 与所有 view 的 `StorageImpl` 计数归零时释放，CUDA 默认还给缓存池 | 四步排查：别的 Python 引用 → autograd 保存 → view 活着 → 在缓存池 | [第二篇](/cpp-value-semantics-ownership-and-raii.html) |
| `const Tensor&` 保证函数不改数据 | `const` 只看句柄自己的字节，不追踪指针指向的地方 | `Tensor` 的 `const` 是浅的；in-place 算子的输出参数就是 `const Tensor&` | [第二篇](/cpp-value-semantics-ownership-and-raii.html) |
| `AT_DISPATCH` 里的 lambda 只编译一份，运行时按 dtype 分支 | lambda 文本被复制进每个 `case`，每个 `case` 一个闭包类型、一份机器码 | 浮点两份、`ALL_TYPES_AND_HALF` 十几份；运行期只有一个 `switch` | [第三篇](/cpp-templates-and-generic-programming.html) |
| `TORCH_CHECK(x.is_floating_point())` 之后 `AT_DISPATCH_FLOATING_TYPES` 一定能处理 | `Half`/`BFloat16` 也是浮点，但不在 `FLOATING_TYPES` 的两个 `case` 里 | 要支持它们换 `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` | [第三篇](/cpp-templates-and-generic-programming.html) |
| Dispatcher 是 `Map<DispatchKey, Kernel>` 加接口虚调用 | `KernelFunction` 是函数指针 + 模板适配器的手工类型擦除，`lookup` 是一次数组下标 | 全程无虚调用；签名一致性靠 `CppSignature` 校验 | [第四篇](/cpp-polymorphism-and-type-erasure.html) |
| 算子编译链接都没报错，所以一定注册上了 | 静态库里没被引用的注册 `.o` 被链接器静默丢弃 | 用 `--whole-archive`/`-force_load` 或动态库；`nm -C` 看 `TORCH_LIBRARY` 符号在不在 | [第五篇](/cpp-macros-static-registration-and-codegen.html) |
| 主线程 `no_grad` 里启动的线程也在 `no_grad` 下 | `thread_local` 新线程不继承，worker 读到默认值 `True` | 用 `at::ThreadLocalState` + `ThreadLocalStateGuard` 显式传播 | [第六篇](/cpp-concurrency-memory-model-tls-and-guards.html) |
| 放掉 GIL 后不能再用参数里的 Tensor | GIL 保护的是解释器状态不是 C++ 内存；`at::Tensor` 是 C++ 对象 | 不能碰 `PyObject*`、不能 `Py_INCREF/DECREF`、不能让 `py::object` 析构；C++ 值随便用 | [第七篇](/cpp-pybind11-python-c-api-and-abi.html) |
| 在 PyTorch 2.10 上要给扩展传 `-D_GLIBCXX_USE_CXX11_ABI=1` 才安全 | v2.10.0 已删掉这个开关，只有编译器默认一种 ABI；显式传 `=0` 反而制造 `__cxx11` 类 undefined symbol | 不传；≤ 2.7 才需要与 `torch._C._GLIBCXX_USE_CXX11_ABI` 一致 | [第七篇](/cpp-pybind11-python-c-api-and-abi.html) |
| 越界一个字节程序会崩，测试过了就没有内存错误 | 未定义行为默认什么都不发生，数据悄悄错；只检查插了桩的代码 | ASan 构建跑一遍相关测试，CI 里先跑"故意崩"自检确认 ASan 生效 | [第八篇](/cpp-build-debug-and-test-toolchain.html) |

## 五、通关自测

### A. 判断与计算（10 题）

1. `x` 是一个 1 GB 的 CUDA tensor，`y = x[0]` 之后 `del x`：显存释放了多少？`y = x.clone()[0]` 之后 `del x` 呢？

   <details markdown="1"><summary>答案</summary>

   第一种 0：`x[0]` 是 view，新建 `TensorImpl` 但共享同一个 `StorageImpl`（计数 +1），只要 `y` 活着 1 GB 就活着。第二种：`clone` 新建了 `StorageImpl` 并拷贝数据，`del x` 后原来那 1 GB 的 `StorageImpl` 计数归零、`DataPtr` 析构，CUDA 默认还给缓存池——`nvidia-smi` 上不一定立刻下降，但可供下一个 tensor 复用。

   </details>

2. 一个 struct 有三个 `at::Tensor` 成员和一个 `std::shared_ptr<Foo>` 成员，没有虚函数：`sizeof` 至少多少？若把 `shared_ptr` 换成 `c10::intrusive_ptr<Foo>` 呢？

   <details markdown="1"><summary>答案</summary>

   三个 `Tensor` 各 8 字节（只含一个 `intrusive_ptr<TensorImpl>`），`shared_ptr` 16 字节（两个指针），至少 40 字节；换成 `intrusive_ptr` 是 8 字节，共 32 字节。这正是第二篇里 `Tensor` 不用 `shared_ptr` 的理由之一：一个 cache line 对两个 cache line。

   </details>

3. 按 `TensorImpl.h` 注释里的规模（4 亿个活 tensor），给 `TensorImpl` 加两个 `int64_t` 字段会多占多少内存？

   <details markdown="1"><summary>答案</summary>

   注释说每加一个 64 位字多 3.2 GB，两个字就是 6.4 GB。这是 `AutogradMeta` 做成可为空的 `unique_ptr`（不需要梯度的 tensor 不分配它）而不是内嵌成员的原因。

   </details>

4. `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, x.scalar_type(), "f", [&] { ... at::parallel_for(0, n, 4096, [&](int64_t b, int64_t e) { ... }); })`：外层 lambda 被实例化几次？`parallel_for` 有几份实例？

   <details markdown="1"><summary>答案</summary>

   四个 `case`（`Double`、`Float`、`Half`、`BFloat16`），外层 lambda 的文本被复制四份、四个闭包类型各编译一次；每个外层实例里的内层 lambda 也是一个独立的闭包类型，所以 `parallel_for(const F&)` 有四份实例。第三篇讨论两个 `case` 时是两份，规律是一样的。

   </details>

5. vLLM 某个 kernel 用 `VLLM_DISPATCH_FLOATING_TYPES` 分派 dtype，再用一条 if 链把运行期的 `width` 映射到 1、2、4 三个编译期常量：编译产物里有几份这个 kernel？运行期做几次选择？

   <details markdown="1"><summary>答案</summary>

   3 种 dtype（Float/Half/BFloat16）× 3 种 `width` = 9 份，全部在编译期生成；运行期两层分支（一个 `switch`、一条 if 链）选一份。第三篇里 `width` 两种时是 6 份。每加一个分支就多一份代码与编译时间，所以 if 链要有兜底、不能为每个可能的值都开分支。

   </details>

6. 用 `DimVector` 存一个 7 维 tensor 的 `sizes`，会堆分配吗？5 维呢？

   <details markdown="1"><summary>答案</summary>

   会：`DimVector = SmallVector<int64_t, 5>`，内联缓冲区放 5 个 `int64_t`，第 6 个元素起转到堆上；5 维不分配。`C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE` 取 5 是因为绝大多数 tensor 不超过 5 维。

   </details>

7. 没有调用 `torch.set_num_threads`，环境里 `OMP_NUM_THREADS=16`、`MKL_NUM_THREADS=4`，机器 8 核：`at::parallel_for` 用几个线程？一个 2 万元素的 tensor 用默认 `GRAIN_SIZE` 会并行吗？

   <details markdown="1"><summary>答案</summary>

   16——优先级 `set_num_threads` > `OMP_NUM_THREADS` > `MKL_NUM_THREADS` > 核数，没调 `set_num_threads` 就轮到 `OMP_NUM_THREADS`，核数只是最后的兜底（这也是多进程训练要显式设 `OMP_NUM_THREADS` 的原因——否则每个进程默认开满核数）。不会：默认 `GRAIN_SIZE` 是 32768，2 万元素没超过阈值，直接串行。

   </details>

8. 一个热循环里每次迭代调用一次 `void f(at::Tensor t)`，改成 `void f(const at::Tensor& t)` 后，一百万次迭代少了多少次原子操作？

   <details markdown="1"><summary>答案</summary>

   按值传拷贝一个 `intrusive_ptr`，进入时原子 +1（relaxed）、参数析构时原子 −1（acq_rel），每次两次；`const&` 零次。一百万次迭代少两百万次原子操作。第二篇说这就是 PyTorch 内部默认 `const Tensor&` 的原因。

   </details>

9. 在 v2.10.0 的 `combined_refcount_` 里，一个对象强引用 3、弱引用 1、有 Python 包装对象：这个 64 位字段的哪些位是 1？

   <details markdown="1"><summary>答案</summary>

   低 32 位存强引用 3（第 0、1 位为 1）；弱引用从第 32 位起（`kWeakReferenceCountOne = 1 << 32`），弱引用 1 就是第 32 位为 1；第 63 位是 `kHasPyObject`。所以是第 0、1、32、63 位。`weakcount()` 读取时要先屏蔽掉 `kHasPyObject` 再右移 32。

   </details>

10. `torch.ops.myops.op(t)` 一次往返经过几次类型转换？原生的 `torch.add(t, t)` 对一个输入呢？两者哪一步会拷贝数据？

    <details markdown="1"><summary>答案</summary>

    自定义算子 5 次：输入 3 次（`PyObject*` → `at::Tensor` → `IValue` → `const at::Tensor&`）加输出 2 次（`Tensor` → `IValue` → `PyObject*`）；原生算子不经过 `IValue`，少两次，是 3 次。没有一步拷贝数据——全是句柄级操作，`THPVariable_Unpack` 是前缀 cast 加借用，`THPVariable_Wrap` 是查 `pyobj_slot` 或新建包装对象。

    </details>

### B. 跨篇综合（5 题）

1. 扩展 `.so` 用 `-fvisibility=hidden` 编译，里面 `TORCH_CHECK` 失败抛出的错误在 Python 侧要变成 `RuntimeError`。从抛出到 Python，哪几件事必须成立？各在哪一篇？

   <details markdown="1"><summary>答案</summary>

   第五篇：`TORCH_CHECK` 是宏，在调用点捕获位置并惰性拼消息；第四篇：它抛的是 `c10::Error : std::exception`，子类决定 Python 异常类型；第一篇与第五篇：`c10::Error` 必须 `class C10_API Error` 导出，否则两个 `.so` 各有一份 typeinfo，`catch (const c10::Error&)` 匹配失败直接 `terminate`——`-fvisibility=hidden` 藏的不只是函数；第七篇：边界上由 pybind11 的翻译器或 `HANDLE_TH_ERRORS` 把它 `catch` 住、`PyErr_SetString`、`return nullptr`，C++ 异常绝不能穿过 C API 边界；第八篇：`TORCH_SHOW_CPP_STACKTRACES=1` 让它附上 C++ 栈。

   </details>

2. 在 `with torch.no_grad():` 里调用一个自定义 CPU 算子，它的 `at::parallel_for` 循环体里调了一个 tensor 算子。会发生什么？改法是什么？

   <details markdown="1"><summary>答案</summary>

   第六篇：`parallel_for` 的工作线程来自 OpenMP 线程池，不继承调用线程的 `thread_local`，在 worker 上 `GradMode::is_enabled()` 读到的是默认值 `True`，那个 tensor 算子会建 autograd 图；PyTorch 自己在进入 kernel 前用 `ThreadLocalState` 打包、worker 里用 `ThreadLocalStateGuard` 恢复，但循环体里的算子调用不在这条保护下。第三篇：`parallel_for` 的 lambda 体只应碰裸指针和标量，不调 tensor 算子。第二篇：把 `data_ptr` 取出来在循环体里用——裸指针是借用，`parallel_for` 返回前 `x_c` 还活着所以安全。改法是把算子调用移到 `parallel_for` 外面，循环体只做逐元素运算。

   </details>

3. 一个 Python 扩展把算子实现编成静态库 `libmyops.a`，再链进 `myops.so`；`import` 成功、`torch.ops.myops.foo` 却报 `AttributeError`。用三篇的知识给出原因、验证手段与修法。

   <details markdown="1"><summary>答案</summary>

   第五篇：`TORCH_LIBRARY` 靠静态对象的构造函数在 `dlopen` 时登记，注册所在的 `.o` 没有任何符号被引用，链接器从 `.a` 里取文件时把它丢掉——无编译错误、无链接错误、注册表为空（mini-c10 的四种链接方式里的 B）。第一篇：`.a` 是链接期按需拷贝、`.so` 才是整体加载，这是两种库的本质差别。第八篇与第五篇的验证：`nm -C myops.so` 里搜 `TORCH_LIBRARY`，看注册符号在不在；`torch.ops.loaded_libraries` 看 `.so` 是否真的加载了；`LD_DEBUG=libs` 看加载过程。修法：`-Wl,--whole-archive`（macOS `-force_load`）或直接把算子实现编进动态库。

   </details>

4. 一个 pybind11 绑定的函数接收 `const at::Tensor&`，内部跑 2 秒的 CPU 计算并返回新 tensor。指出这个签名与实现里分别涉及哪几篇的规则，以及为什么进计算图的算子不该这样绑。

   <details markdown="1"><summary>答案</summary>

   第二篇：`const at::Tensor&` 是只读输入的零开销传法，返回 `at::Tensor` 按值靠 NRVO 不拷数据。第七篇：pybind11 的 `type_caster<at::Tensor>::load` 用 `THPVariable_Unpack` 从 `THPVariable::cdata` 拷贝一个 `Tensor` 句柄（C++ 计数 +1，可能触发 1 → 2 钩子的 `Py_INCREF`），参数转换完后第一行 `py::gil_scoped_release`，否则解释器其他线程全停 2 秒；释放期间不能碰任何 `PyObject`，但 `Tensor` 是 C++ 对象随便用；返回时 `THPVariable_Wrap` 生成新引用。第四篇与第五篇：pybind11 直绑的函数不在 Dispatcher 的表里，不能按 `DispatchKey` 分发、不能注册 Meta/Autograd、不能被 `torch.compile` trace；进计算图的算子要用 `TORCH_LIBRARY` 注册，让它经 `KernelFunction` 走 boxed/unboxed 路径。第一篇：用了 pybind11 的 Tensor caster 就要链 `libtorch_python.so`，也就放弃了 `abi3`。

   </details>

5. ASan 报告一段 use-after-free：访问栈在你的 kernel 里对一个 `float*` 读，释放栈是 `~TensorImpl` → `~StorageImpl` → `free_cpu`，分配栈是 `at::empty` → `alloc_cpu`。用三篇的知识说出这个 bug 是什么、为什么普通构建下测试都过了。

   <details markdown="1"><summary>答案</summary>

   第三篇：`data_ptr<float>()` 返回的裸指针是借用，不持有 `Storage`；第二篇：释放栈就是持有链反向析构——最后一个 `Tensor` 句柄析构让 `TensorImpl` 计数归零、`StorageImpl` 归零、`DataPtr` 的删除器 `free_cpu`——说明持有那块数据的最后一个 tensor 在指针使用前已经死了——第八篇 `uaf.cpp` 的形状是"取了 `data_ptr` 之后把句柄重新赋值成 `Tensor()`"，实际代码里更常见的是从 `x.contiguous()` 这样的临时对象上取指针而没有把返回的 `Tensor` 存下来。第八篇：越界与 use-after-free 是未定义行为，普通构建下那块内存多半还没被复用、读出来的值恰好是对的，测试全过；只有 ASan 的红区与影子内存能在第一次访问时给出确定报告，所以改了生命周期相关代码必须跑 ASan。

   </details>

### C. 面试题（7 题）

1. PyTorch 为什么自己实现 `intrusive_ptr` 而不用 `std::shared_ptr`？给出数字与它带来的额外能力。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 大小：`shared_ptr` 16 字节加单独分配的控制块，`intrusive_ptr` 8 字节、计数在对象里、无额外分配——`Tensor` 是最高频被拷贝的对象，4 亿个活 tensor 时每个字 3.2 GB；(2) 缓存局部性：一个 cache line 对两个；(3) 从裸指针恢复：`reclaim`/`release` 让 `TensorImpl*` 能穿越 C/Python 边界，`shared_ptr` 做不到（除非 `enable_shared_from_this` 再多 16 字节）；(4) `make_intrusive` 直接写计数不用原子加；(5) 可定制：`NullType` 让 undefined `Tensor` 不是空指针，`release_resources()` 让弱引用不拖住显存，64 位合并计数把强、弱与 `kHasPyObject` 放进一条原子指令，`pyobj_slot` 实现与 Python 对象的双向持有。
   **追问方向**：+1 为什么 relaxed、−1 为什么 acq_rel；`weak_intrusive_ptr` 在 autograd 图里断什么环；不能在构造函数里从 `this` 创建它的原因（计数为 0）。
   **好答案与一般答案的区别**：一般答案说"省一次分配、少 8 字节"；好答案说出 `reclaim`/`release` 与 Python 双向持有——这才是标准库给不了、PyTorch 必须自己造的部分。

   </details>

2. 从 Python 写 `torch.add(a, b)` 到 CPU kernel 的循环体，中间经过了哪些 C++ 机制？每一跳为什么用那种机制而不是别的？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) Python → C++：`THPVariable_Unpack` 前缀 cast 借出 `at::Tensor`，`PythonArgParser` 解析参数，持有 GIL；(2) torchgen 生成的 `at::add` 入口拿到 `TypedOperatorHandle`，`Dispatcher::call` 从参数算 `DispatchKeySet`；(3) `OperatorEntry::lookup` 一次数组下标拿到 `KernelFunction`——类型擦除而不是虚函数，因为要把任意签名装进一张表且不付虚调用；(4) unboxed 路径 `call<Return, Args...>` 把 `void*` cast 回精确签名，热路径零装箱；autograd 等通用层走 boxed 的 `IValue` 栈；(5) 进 kernel 后 `AT_DISPATCH` 的 `switch` 把运行期 dtype 变成编译期 `scalar_t`，每个 dtype 一份向量化循环；(6) `parallel_for` 按 `grain_size` 切给 OpenMP 线程，TLS 用 `ThreadLocalState` 传播；(7) 返回 `Tensor` 按值、NRVO，`THPVariable_Wrap` 包回 Python。
   **追问方向**：boxed 与 unboxed 怎么互转；`TORCH_LIBRARY_IMPL(aten, CPU, m)` 什么时候把 kernel 填进表；为什么原生 `torch.add` 比 `torch.ops.myops.op` 少两次类型转换（不经过 `IValue`）。
   **好答案与一般答案的区别**：一般答案背"Dispatcher 按 DispatchKey 查表"；好答案在每一跳说出"为什么不是虚函数 / 为什么不是运行期 if"，把编译期分派与运行期分派的边界画清楚。

   </details>

3. Java 泛型与 C++ 模板都写 `<T>`，为什么 PyTorch 需要 `AT_DISPATCH` 这样的宏而 Java 不需要？代价是什么，怎么控制？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) Java 擦除类型，`List<Integer>` 与 `List<String>` 是一份字节码、元素装箱；C++ 为每组参数生成一份代码，`vector<int>` 与 `vector<string>` 是两个类型；(2) kernel 需要编译期类型——`float*` 与 `double*` 的解引用是不同指令，`Vectorized<float>` 装 8 个、`Vectorized<double>` 装 4 个——而 dtype 是运行期值，中间需要一座桥；(3) `AT_DISPATCH` 就是这座桥：一个 `switch`，每个 `case` 里 `using scalar_t = ...` 再粘贴 lambda，N 个 dtype 编 N 份；(4) 代价是编译时间与二进制体积：`.cu` 约为 `.cpp` 十倍，vLLM 3 dtype × 2 width 就是 6 份 kernel；(5) 控制手段：只列真正需要的 dtype（推理 kernel 不要 `Double`）、if 链有兜底、`if constexpr` 在一份函数体内做局部差异、显式实例化把模板定义放 `.cpp`。
   **追问方向**：为什么 `data_ptr<T>()` 必须显式写 `<T>`；`AT_DISPATCH_V2` 的 `AT_WRAP` 解决什么；模板错误信息怎么读（找第一条 `error:`，顺 `note: in instantiation of` 找自己的帧）。
   **好答案与一般答案的区别**：一般答案说"C++ 模板会代码膨胀"；好答案说出运行期值到编译期常量必须经过有限个分支，并给出份数怎么算。

   </details>

4. `import myext` 报 `undefined symbol: _ZN3c105Error...`，符号明明在 `libc10.so` 里。你的排查顺序是什么？每一步用什么工具？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先定位阶段：`undefined symbol` 在 `import` 时出现是加载期，链接期已过，说明链接时看到的头文件与运行时加载的 `.so` 不一致；(2) `c++filt` 还原符号，看参数类型里有没有 `__cxx11`——有就是 libstdc++ ABI 不一致（v2.10.0 上只可能是自己显式传了 `=0`；≤ 2.7 上是没和 `torch._C._GLIBCXX_USE_CXX11_ABI` 对齐）；(3) `nm -DC libc10.so` 看目标库里到底有没有这个签名、是否导出（`C10_API`）、修饰名是否一致——函数可能在这个 PyTorch 版本里不存在或签名变了；(4) `ldd myext.so`、`LD_DEBUG=libs` 看加载的是哪一份 `libc10.so`，是否有第二份 libtorch 或 `LD_LIBRARY_PATH` 把库换掉了；(5) `GLIBCXX_3.4.x not found` 则是运行时 `libstdc++.so.6` 太旧，`strings` 对比；(6) 最后核对版本矩阵：编译扩展的 PyTorch 版本、GCC 版本不低于 wheel 所用版本、CUDA 与 host 编译器区间。
   **追问方向**：`cpp_extension.py` 替你检查了什么、不检查什么；为什么 vLLM 的 wheel 要钉死 PyTorch 版本；`torch/csrc/stable/` 的 C shim 想解决哪一层。
   **好答案与一般答案的区别**：一般答案说"重编一下"；好答案按加载期 → 符号 → ABI 三层 → 版本矩阵的顺序逐层排除，每步给工具与判据。

   </details>

5. 让你在 C++ 里实现一个像 `torch.no_grad()` 那样的开关，你会怎么设计？为什么它必须删掉拷贝与移动、为什么对其他线程不生效？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 状态放 `thread_local`——语言级存储类别、每线程一份、访问一条指令，让一个线程的开关不影响并发跑的 DataLoader worker；(2) 守卫类三步骨架：构造保存旧值并设新值、析构恢复，业务逻辑全在这两处，异常路径也生效；(3) `= delete` 拷贝与移动，因为两份副本说不清谁在析构时恢复、恢复成什么，守卫必须与作用域一一对应，也从根本上杜绝跨线程传递；(4) 守卫必须是有名字的局部变量，临时对象立刻析构等于没生效；(5) 新线程的 TLS 是初始值不是父线程的值，跨线程要 `ThreadLocalState` 打包、`ThreadLocalStateGuard` 恢复；(6) Python 上下文管理器在 C++ 侧只能用非 RAII 的 `set`/`get` API，因为 `__enter__` 与 `__exit__` 之间 C++ 栈已展开。
   **追问方向**：`LocalDispatchKeySet` 为什么零初始化 + XOR；`DeviceGuard` 为什么分虚接口与内联模板两层；`gil_scoped_release` 与它有什么共同之处。
   **好答案与一般答案的区别**：一般答案说"用 RAII 恢复旧值"；好答案解释删掉拷贝移动与"有名字的局部变量"这两条纪律各防什么，并把 TLS 不继承与 `ThreadLocalState` 说成同一件事的两面。

   </details>

6. 解释 `TORCH_LIBRARY(myops, m) { ... }` 在没有任何显式调用的情况下是怎么生效的，以及它有哪些 Java 里不存在的失效条件。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 宏展开成一个函数定义加一个静态存储期对象 `TorchLibraryInit`，构造函数参数就是那个函数；(2) `import` 是 `dlopen`，`dlopen` 返回前执行 `.so` 的 `.init_array`，静态对象此刻构造，向 `Dispatcher::singleton()` 登记命名空间并调用函数体，`m.def`/`m.impl` 落到 `registerDef`/`registerImpl`——调用者是加载器；(3) `torch.ops.myops.foo` 是 `_OpNamespace.__getattr__` 按字符串惰性查表；(4) 失效条件一：静态库丢弃没被引用的 `.o`，注册消失且无任何报错，要 `--whole-archive`；(5) 失效条件二：静态初始化顺序跨翻译单元未定义，注册器不能依赖别处的全局对象，单例用函数内静态；(6) 失效条件三：注册表必须容忍 `impl` 先于 `def`（不同 `.so` 的初始化顺序不受控）；(7) 配套纪律：注册器构造函数只做登记，`import torch` 必须先于 `load_library`。
   **追问方向**：`TORCH_LIBRARY_FRAGMENT` 解决什么；`C10_USED` 与 `--gc-sections` 的关系；`TORCH_EXTENSION_NAME` 为什么要套一层 `TORCH_LIBRARY_EXPAND`。
   **好答案与一般答案的区别**：一般答案说"静态变量的构造函数在 main 之前跑"；好答案说出"链接方式会静默取消登记"——这是 `ServiceLoader` 使用者从未遇到过、也是算子"消失"时该去看链接命令的原因。

   </details>

7. 你改了 `c10` 里一个与引用计数相关的函数，提 PR 之前要跑什么？按成本排序，并说明每一步在抓什么、为什么前一步抓不到。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) clangd 无红线、clang-format 过——零成本；(2) Debug 构建 `-Wall -Wextra` 无新警告，`-O0` 让 gdb/lldb 看得到变量，`-O2` 下会 `<optimized out>`；(3) 相关 gtest：`intrusive_ptr_test.cpp` 有 325 个测试专测计数契约，析构时序用布尔指针断言而非打印；再跑能从 Python 观察的 pytest；(4) clang-tidy（需要 `compile_commands.json`）；(5) 引用计数是生命周期代码，必须单独一个 `build-asan` 目录跑 ASan+UBSan——use-after-free 与 double free 在普通构建下默认什么都不发生，只有插桩能给出三张栈，约 2× 时间；`detect_leaks=0`，CI 先跑"故意崩"自检；(6) 计数是并发代码，再单独一个 `build-tsan` 跑 TSan（与 ASan 互斥，5–15×），并会区分 relaxed 原子操作的假阳性与非原子读写的真竞争；(7) 交给 CI 矩阵：gcc 11 + clang 12/18 + aarch64 + CUDA，警告集与 UB 的表现是编译器相关的。
   **追问方向**：为什么 `-fvisibility=hidden` 在 sanitizer 构建里要关；`TORCH_USE_RTLD_GLOBAL=1` 与 UBSan vptr 误报；改了 `TensorImpl.h` 这类头文件为什么增量构建回到十几分钟。
   **好答案与一般答案的区别**：一般答案说"跑测试、跑 CI"；好答案按成本排序并对每一步说出"上一步为什么抓不到"，尤其是 ASan 不是可选项。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出八篇各讲什么；知道 `intrusive_ptr`、`AT_DISPATCH`、`KernelFunction`、`TORCH_LIBRARY`、`thread_local`、GIL、ASan 这些名词，能把总纲那段 `scale_shift_cpu` 的每一行指到对应的篇 |
| 掌握 | A 组能不翻书算出 8 题以上；B 组能说出每题用了哪几篇的什么；打开 `c10/core/TensorImpl.h` 或 `Dispatcher.h` 能认出每一行的机制；遇到 `undefined symbol`、算子"消失"、`del` 后显存不降能按篇里的步骤排查 |
| 能教人 | C 组每题能给出全部要点并预判追问；能解释八篇里每个反直觉结论（Dispatcher 无虚调用、静态库会取消注册、越界不崩、放掉 GIL 还能用 Tensor、`const Tensor&` 能写数据）为什么成立，并用 Java 的对照说清类比在哪里失效 |

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的"必记"，再回该篇正文的相应章节；mini-c10 的对应文件是最直接的复习材料。

## 六、下一步

八篇讲的是 C++ 这门语言在 AI-Infra 项目里的用法，以及读懂这些项目 C++ 层所需的工程知识。四个方向紧邻但不在范围内：

- **PyTorch 的机制本身**——Autograd 如何构图、Dispatcher 的分发规则、`torch.compile` 的工作方式——本系列只借它们的 C++ 代码作为语言特性的例子，完整原理在[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)。
- **CUDA 编程**——kernel 的写法、GPU 内存层次、性能优化——本系列停在 host 侧，只覆盖 CUDA 代码所依赖的 C++（模板参数化 tile、RAII 管理 stream、宏做多架构分发），kernel 本身在[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)。
- **Python 语言本身**——第七篇假设读者已了解 CPython 的引用计数与 GIL；它们的来历与 Python 侧的工程实践在[《Python 在 AI-Infra：从语言机制到生产交付》](/python-for-ai-infra.html)。
- **通用 C++ 知识的完整覆盖**——STL 算法库、iostream、正则、文件系统、协程等在这些项目里很少出现的部分——不在任何一篇里，本系列只讲 AI-Infra 项目真正大量使用的那个子集。
- 这一层在整张地图上的位置，以及语言层之后往哪走，见[《AI-Infra 工程师学习地图》](/ai-infra-learning-roadmap.html)。

回到总纲：[《C++ 在 AI-Infra：从对象模型到算子扩展》](/cpp-for-ai-infra.html)。

## 七、延伸阅读

本系列只讲 C++ 这门语言在 AI-Infra 项目中的用法，以及读懂这些项目 C++ 层所需的工程知识。以下内容不在范围内，虽然它们与本系列的源码阅读对象紧密相关：

- **PyTorch 的机制本身**：Autograd 如何构图、Dispatcher 的分发规则、`torch.compile` 的工作方式。本系列只借用它们的 C++ 代码作为语言特性的例子，读者不需要事先理解这些机制，也不会在本系列里学到它们的完整原理。
- **CUDA 编程**：kernel 的写法、GPU 内存层次、性能优化。本系列停在 host 侧，只覆盖 CUDA 代码所依赖的 C++ 特性。
- **Python 语言本身**：第七篇讨论 C++ 与 Python 的边界，假设读者已经了解 CPython 的引用计数和 GIL 是什么。
- **通用 C++ 知识的完整覆盖**：STL 算法库、iostream、正则、文件系统、协程等在这些项目里很少出现的部分。


[^q0]: 八个：`import torch` 加载了哪些 `.so`、依赖关系是什么、扩展链接到哪一个（编译模型与库布局）；`at::Tensor y = x;` 之后两者是什么关系、数据什么时候释放（句柄、`intrusive_ptr`、持有链）；`AT_DISPATCH` 里的 `scalar_t` 从哪里来、lambda 编译了几次（模板与编译期分派）；Dispatcher 用什么机制调到 kernel、为什么既有 boxed 又有 unboxed（类型擦除、`IValue`）；一个 `.so` 被 `import` 后算子怎么出现在 `torch.ops` 下（静态注册、链接方式）；`no_grad` 在 C++ 层做了什么、为什么对其他线程不生效（`thread_local`、守卫、`ThreadLocalState`）；一个 Tensor 跨过 Python/C++ 边界经过几次转换与计数变化、GIL 状态如何（C API、pybind11、ABI）；一个改动从写完到能提 PR 要跑什么（构建、调试、sanitizer、矩阵）。详见[第二章](#二逐篇回顾)。
[^q1]: `Tensor` 8 字节、`shared_ptr` 16 字节、每个字 3.2 GB；`y = x` / view / clone 三种关系与 `del` 后的四步排查；`AT_DISPATCH_FLOATING_TYPES` 两个 `case` 编两份、`DimVector` 内联 5 维；`std::function` 32 / `function_ref` 16 / 函数指针 8 字节、`IValue` 16 字节、`lookup` 一次数组下标无虚调用；`TORCH_CHECK` 是宏的两个理由、四种链接方式里静态库直接链注册表为空、`native_functions.yaml` 2666 条目；relaxed 增 acq_rel 减、64 位合并计数低 32 强高 31 弱第 63 位 PyObject、线程数优先级 `set_num_threads` > `OMP_NUM_THREADS` > `MKL_NUM_THREADS` > 核数、`GRAIN_SIZE` 32768；一次往返输入 3 次输出 2 次转换、C++ 计数 1 → 2 → 1、ABI 三层、2.7 起全部 CXX11 ABI 且 v2.10.0 开关已删；`-O0` 慢 3–10 倍、ASan 约 2×、TSan 5–15× 且与 ASan 互斥、GCC ≥ 9.3、CUDA ≥ 12.0、C++17、325 个测试。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。
[^q2]: 用第五章的三段自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出"读过 / 掌握 / 能教人"三级的表现。详见[第五章](#五通关自测)。
