---
layout: post
title: "C++ 在 AI-Infra（03）：模板与泛型编程"
subtitle: "Templates and Generic Programming"
tags: [C++, AI, AI-Infra]
catalog: true
---

打开 `aten/src/ATen/native/cpu/Activation.cpp`，`log_sigmoid` 的 CPU kernel 里有这么一段（`log_sigmoid_cpu_kernel` 的 `else` 分支，删节）：

```cpp
AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&] {
  using Vec = Vectorized<scalar_t>;
  scalar_t* output_data = output.data_ptr<scalar_t>();
  scalar_t* buffer_data = buffer.data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  parallel_for(0, input.numel(), 1, [&] (int64_t begin, int64_t end) {
    // ...
    Vec min_vec = vec::minimum(data_vec, Vec(scalar_t(0)));
    // ...
  });
});
```

一个 Java 工程师读到这里，会在第一行就停住。`AT_DISPATCH_FLOATING_TYPES` 全大写，显然是宏，但它接受一个 lambda，看起来又像函数；`scalar_t` 在整段代码里被当成一个类型来用——声明指针、构造 `Vec`、做 `static_cast`——可是它没有在任何地方被声明，也没有 `#include` 进来。往上翻整个文件，找不到 `typedef` 或 `using scalar_t`。它是从哪里来的？

第二个疑问跟着来：`input.data_ptr<scalar_t>()` 的 `<>`。Java 泛型也用尖括号，但 Java 里没有"按类型参数返回不同的裸指针"这种事；`List<Integer>` 和 `List<String>` 在 JVM 里是同一个类，方法体只有一份字节码。而这里，`data_ptr<float>()` 和 `data_ptr<double>()` 显然要返回不同类型、做不同的事。如果 `scalar_t` 是某种"运行时才知道"的类型，这段代码是怎么编译的？

第三个疑问：`[&]`。它捕获了什么？`output`、`buffer`、`input` 都是外层函数的局部变量，用引用捕获它们安全吗？内层的 `parallel_for` 又用了一次 `[&]`，那是多线程执行的，为什么不用担心？

这些问题的答案在同一个机制上：C++ 模板。Java 泛型和 C++ 模板都写成 `<T>`，但实现机制相反——Java 在编译后擦除类型，运行时只有一份代码；C++ 为每组模板参数在编译期生成一份代码，运行时没有类型信息也不需要。这个差别决定了模板能做什么（可以按类型生成完全不同的机器码）、编译错误为什么那么长（错误发生在实例化链的深处）、以及 `AT_DISPATCH` 为什么必须存在（`dtype` 是运行期的值，kernel 需要编译期的类型，中间要有一座桥）。

本文的核心问题是总纲里的这句：

> **`AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "name", [&] { ... scalar_t ... })` 里的 `scalar_t` 从哪里来？这个 lambda 被编译了几次？**

全文按下面的顺序展开：

1. 模板是生成代码的配方：函数模板、类模板、实例化；模板为什么必须放在头文件；与 Java 类型擦除的根本差别；错误信息为什么巨长；`typename` 与 `template` 的消歧义；
2. 推导：模板参数推导、显式指定、`auto`、`decltype`、返回类型推导、CTAD；
3. 非类型模板参数：`std::array<T, N>`、`SmallVector<T, N>`、以枚举为参数的 `ScalarTypeToCPPType<N>`、vLLM CUDA kernel 的 `BLOCK_Y_SIZE`/`width` 参数化；
4. 特化与变参模板：全特化、偏特化、函数模板为什么不能偏特化；参数包展开与完美转发；
5. 把分支移到编译期：`constexpr`、`if constexpr`、`static_assert`、SFINAE 与 `enable_if`、C++20 concepts；
6. 编译期分派与运行期分派：为什么 `dtype` 是运行期值而 kernel 需要编译期类型；逐层展开 `AT_DISPATCH_FLOATING_TYPES`（`Dispatch.h` → `torch/headeronly/core/Dispatch.h` → `ScalarType.h`），回答核心问题；vLLM 的 `dispatch_utils.h`；`Dispatch_v2.h` 的演进；
7. 轻量视图与容器：`c10::ArrayRef`/`IntArrayRef`、`std::optional`（`c10::optional` 迁移现状）、`c10::SmallVector`；
8. lambda：闭包类型、捕获列表、`[&]` 与 `[=]`、泛型 lambda、lambda 作为模板参数、引用捕获的生命周期陷阱、`at::parallel_for` 的 `[&]` 为什么安全；
9. 回到源码：重读 `scale_shift_cpu` 与 `TensorMethods.cpp` 里 `data_ptr<T>` 的显式实例化；
10. mini-c10：`core/ScalarType.h`、`core/Dispatch.h`、`util/ArrayRef.h`、第一个模板化 CPU kernel `ops/add.cpp`/`ops/mul.cpp`，并用 `nm` 与 `clang++ -S` 观察多份 kernel 代码；
11. 工程实践建议与常见错误；
12. 总结。

正文以 C++17 为基线（PyTorch 2.x 中的变化：本机 v2.13.0 源码树的顶层 `CMakeLists.txt` 将 `CMAKE_CXX_STANDARD` 设为 20，并在检测到环境变量指定其他标准时报错 "PyTorch requires -std=c++20"；vLLM 也是 C++20。但本文涉及的机制在 C++17 中全部存在，源码树里也没有用到 concepts 等 C++20 特有语法，所有 mini-c10 片段用 `clang++ -std=c++17 -Wall -Wextra` 验证）。


## 一、模板是生成代码的配方

### 1.1 函数模板与隐式实例化

先看最小的例子。一个对任意数值类型做逐元素相加的函数，在 C++ 里写成函数模板：

```cpp
template <typename scalar_t>
void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
```

`template <typename scalar_t>` 声明了一个模板参数 `scalar_t`。关键的认知转换是：**这段代码本身不是函数，编译器不会为它生成任何机器码**。它是一份配方，只有当某处用具体类型使用它时——`add_kernel<float>(...)` 或者让编译器从实参推导出 `scalar_t = float`——编译器才会把 `scalar_t` 替换成 `float`，生成一个真正的函数 `add_kernel<float>`，这一步叫**实例化**（instantiation）。再用 `double` 调用一次，就再实例化一个 `add_kernel<double>`。两个实例是两个独立的函数，各有各的符号、各有各的机器码：一个里面是单精度乘法指令，另一个是双精度。

第十节的 mini-c10 会用 `nm` 把这两个符号真的列出来。这里先记住结论：**一个模板 + N 组参数 = N 份代码**。这是 C++ 模板与 Java 泛型的根本区别，后面几乎所有现象都由它推出。

### 1.2 类模板：`vector<int>` 和 `vector<string>` 是两个不相干的类型

类模板同理。`std::vector<int>` 和 `std::vector<std::string>` 不是"同一个类的两种用法"，而是由同一份配方生成的两个完全独立的类，各有各的成员函数机器码、各有各的 `sizeof`、彼此之间没有任何继承或转换关系。你不能把 `vector<int>*` 赋给一个"泛型 vector 指针"，因为不存在这样的东西。

第二篇的 `c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>` 就是类模板的实例：`intrusive_ptr` 是模板，`<TensorImpl, UndefinedTensorImpl>` 是两个类型参数，整个东西才是一个类型。`intrusive_ptr<StorageImpl>` 是另一个类型，两者的 `operator->` 返回不同的指针类型，编译器在编译期就知道 `impl_->sizes()` 调的是 `TensorImpl::sizes`，不需要任何运行期查找。

Java 里 `List<Integer>` 和 `List<String>` 在 JVM 里是同一个 `java.util.List` 类，泛型参数在编译后被擦除，`get()` 返回 `Object` 再由编译器插入 `checkcast`。这带来两个后果：Java 泛型不能用基本类型（`List<int>` 不合法，只能装箱成 `Integer`），也不能按类型参数做不同的事（没有办法为 `List<Integer>` 单独写一个更快的实现）。C++ 模板恰恰相反：`vector<int>` 里就是紧密排列的 `int`，没有装箱；而且可以为特定类型参数写完全不同的实现——这就是第四节的特化。

### 1.3 实例化发生在哪里：模板为什么必须放在头文件

第一篇讲过翻译单元：每个 `.cpp` 独立编译。现在把它和实例化放在一起看，会得到一条重要的工程规则。

编译器要实例化 `add_kernel<float>`，必须在**当前翻译单元**里看得到 `add_kernel` 的完整定义（不只是声明），否则它不知道函数体长什么样，无法替换 `scalar_t`。这意味着模板的定义通常写在头文件里，谁包含谁实例化。这也是为什么 `c10/util/intrusive_ptr.h`、`c10/util/ArrayRef.h`、`c10/util/SmallVector.h` 这些文件几乎全部代码都在 `.h` 里，而不是像普通类那样声明在 `.h`、实现在 `.cpp`。

同一个实例（比如 `intrusive_ptr<TensorImpl>::reset_()`）可能在几百个翻译单元里各实例化一份，链接器会把它们合并成一个——模板实例默认是第一篇讲过的 `inline`/弱符号语义，不违反 ODR。代价是编译时间：每个翻译单元都要重做一遍实例化。大型 C++ 项目编译慢，模板是首要原因之一。

有一个例外值得专门看，因为 PyTorch 用它实现了 `Tensor::data_ptr<T>()`。`aten/src/ATen/core/TensorBase.h` 里只有声明：

```cpp
  // Implemented in aten/src/ATen/templates/TensorMethods.cpp
  template <typename T>
  const T* const_data_ptr() const;

  template <typename T>
  T* mutable_data_ptr() const;
  // ...
  template <typename T>
  T* data_ptr() const;
```

定义在 `aten/src/ATen/templates/TensorMethods.cpp`（这是一个 codegen 模板文件，生成后进入 build 目录，内容如下）：

```cpp
template <typename T>
T* TensorBase::mutable_data_ptr() const {
  check_type(*this, c10::CppTypeToScalarType<T>());
  return this->unsafeGetTensorImpl()->mutable_data_ptr_impl<T>();
}

template <typename T>
T* TensorBase::data_ptr() const {
  return this->mutable_data_ptr<T>();
}

#define DEFINE_CAST(T, name)                                                \
   template TORCH_API const T* TensorBase::const_data_ptr<T>() const;       \
   template TORCH_API const T* TensorBase::const_data_ptr<const T>() const; \
   template TORCH_API T* TensorBase::mutable_data_ptr() const;              \
   template TORCH_API T* TensorBase::data_ptr() const;

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 DEFINE_CAST(uint16_t, UInt16)
 DEFINE_CAST(uint32_t, UInt32)
 DEFINE_CAST(uint64_t, UInt64)
 #undef DEFINE_CAST
```

`template TORCH_API T* TensorBase::data_ptr() const;`（开头只有 `template` 而没有 `template <...>`，也没有函数体；`T` 由返回类型给出）是**显式实例化定义**：命令编译器在这个翻译单元里生成这个实例，并用 `TORCH_API` 导出符号。这样做的理由有两个：一是 `data_ptr<T>` 只对有限的一组 `T` 有意义（PyTorch 支持的 dtype 列表），把实例化收口到一个 `.cpp` 里，头文件就不必暴露 `check_type` 等实现细节；二是所有翻译单元共享 `libtorch_cpu.so` 里这一份实例，不再各自实例化。代价是：如果你在扩展里写 `x.data_ptr<MyStruct>()`，编译能过（声明可见），链接时报"未定义的引用"——因为没有任何地方实例化过它。第八篇排查链接错误时会再遇到这种情况。

### 1.4 Java 对照：类型擦除 vs 单态化，以及错误信息为什么巨长

把两种机制放在一起：

| | Java 泛型 | C++ 模板 |
|---|---|---|
| 类型参数在哪里消失 | 编译成字节码时被擦除 | 编译成机器码时被替换成具体类型 |
| `List<Integer>` / `vector<int>` | 与 `List<String>` 同一个类 | 与 `vector<string>` 两个不相干类型 |
| 代码份数 | 一份 | 每组参数一份 |
| 能否用基本类型 | 不能（需装箱） | 能，且没有额外开销 |
| 能否按类型参数换实现 | 不能 | 能（特化、`if constexpr`、重载） |
| 值可以做参数吗 | 不能 | 能（非类型模板参数，第三节） |
| 类型检查发生在 | 泛型定义处（有 `extends` 约束） | 实例化处（默认没有约束，第五节） |
| 运行期能问 `T` 是什么吗 | 不能（已擦除） | 不需要问，`T` 已经编进代码 |

最后两行解释了 C++ 模板错误信息为什么巨长。Java 在泛型定义处就检查 `T` 满足 `extends Comparable<T>`，错在哪一行就报哪一行。C++ 模板默认对 `T` 没有任何约束，`add_kernel` 里的 `a[i] + b[i]` 是否合法要等到 `scalar_t` 确定之后才知道；如果你传了一个没有 `operator+` 的类型，错误发生在**实例化的深处**，编译器会把整条实例化链打印出来。本机做一个最小实验，用 `std::sort` 排序一个 `std::list`（`list` 的迭代器不支持随机访问）：

```cpp
#include <algorithm>
#include <list>
int main() {
  std::list<int> l{3, 1, 2};
  std::sort(l.begin(), l.end());
}
```

`clang++ -std=c++17 -c` 的输出共 138 行、约 15KB，其中真正的 `error:` 只有 4 条，第一条是：

```text
.../__algorithm/make_heap.h:35:34: error: invalid operands to binary expression
    ('std::__list_iterator<int, void *>' and 'std::__list_iterator<int, void *>')
   35 |   difference_type __n   = __last - __first;
note: in instantiation of function template specialization
    'std::__make_heap<std::_ClassicAlgPolicy, std::__less<void, void> &, std::__list_iterator<int, void *>>' requested here
note: in instantiation of function template specialization
    'std::__partial_sort_impl<...>' requested here
...
```

错误出在标准库内部第五层的 `__last - __first`，而不是你写 `std::sort` 那一行；每一层 `note: in instantiation of ... requested here` 是实例化栈的一帧，类型名带着完整的模板参数，所以一行就有几百字符。读这类错误的方法：先找第一条 `error:`，再顺着 `note: in instantiation ... requested here` 往下找到**自己代码里**的那一帧，问题几乎总在那里。PyTorch 里一条 `AT_DISPATCH` 里的类型错误可以轻松产生上千行输出，读法相同。

### 1.5 `typename` 与 `template` 的消歧义

模板里有两个关键字的用法是纯语法层面的，Java 没有对应物，但读源码时随处可见。

**`typename`**：当一个名字依赖于模板参数、并且它是一个类型时，必须在前面写 `typename`，否则编译器把它当成值。`c10/util/SmallVector.h` 的构造函数：

```cpp
  template <
      typename ItTy,
      typename = std::enable_if_t<std::is_convertible_v<
          typename std::iterator_traits<ItTy>::iterator_category,
          std::input_iterator_tag>>>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
```

`std::iterator_traits<ItTy>::iterator_category` 依赖 `ItTy`，编译器在解析这段代码时还不知道 `ItTy` 是什么，无法确定 `::iterator_category` 是类型还是静态成员，所以要靠 `typename` 告诉它"这是类型"。vLLM `csrc/libtorch_stable/type_convert.cuh` 里的 `_f16Vec`：

```cpp
template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  // ...
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];
```

同样，`Converter::hip_type` 依赖 `scalar_t`，前面必须有 `typename`。注意 `torch/headeronly/core/ScalarType.h` 里的这一行也是：

```cpp
template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;
```

——这就是第六节 `scalar_t` 的最终来源。

**`template`**：当一个依赖名是成员模板、并且后面紧跟 `<`，要写 `template` 告诉编译器 `<` 是模板参数列表而不是小于号。`c10/util/intrusive_ptr.h` 的移动赋值：

```cpp
  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
    return this->template operator= <TTarget, NullType>(std::move(rhs));
  }
```

`this->operator=` 是成员模板，显式给它模板参数时要写 `this->template operator= <...>`。`c10/util/flat_hash_map.h` 里的 `typename std::allocator_traits<A>::template rebind_alloc<...>` 是两个关键字连用的例子。日常写代码很少需要 `template` 消歧义，但读到时不要以为是什么高级用法，它只是给解析器的提示。


## 二、推导：编译器怎么知道 `T` 是什么

### 2.1 从实参推导

函数模板的类型参数通常不用写，编译器从实参推导。`aten/src/ATen/Parallel.h` 里 `parallel_for` 的声明：

```cpp
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f);
```

调用 `parallel_for(0, n, 4096, [&](int64_t b, int64_t e) { ... })` 时，`F` 被推导为那个 lambda 的闭包类型（第八节）。每个 lambda 表达式都有独一无二的类型，所以**每个调用点都会实例化一份 `parallel_for`**，lambda 体可以被完整内联到那一份里。这和 Java 把 lambda 转成 `Function` 接口对象再虚调用的模型完全不同，也是 C++ 里"传 lambda 零开销"的原因。

`aten/src/ATen/native/cpu/Loops.h` 的 `cpu_kernel` 更进一步：

```cpp
template <typename func_t>
void cpu_kernel(TensorIteratorBase& iter, func_t&& op, int64_t grain_size = at::internal::GRAIN_SIZE, bool check_dynamic_casting = true) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // ...
```

`func_t&&` 是**转发引用**（forwarding reference）：实参是左值时 `func_t` 推导为 `T&`，是右值时推导为 `T`，配合 `std::forward` 可以原样转发。推导出 `func_t` 之后，`function_traits<func_t>` 用模板技巧拆出 lambda 的参数个数 `arity` 和返回类型——`cpu_kernel(iter, [](float a, float b) { return a * b; })` 里 lambda 有两个参数，编译器就知道这个 kernel 有两个输入，在编译期而不是运行期。

推导有几条常见规则需要记住：顶层 `const` 和引用会被剥掉（传一个 `const int` 给 `T`，`T` 是 `int`）；数组和函数名退化成指针，除非参数写成引用（`const T (&arr)[N]` 能保留长度 `N`，第三节的 `ArrayRef` 用到）；`{1, 2, 3}` 这样的花括号列表不能推导 `T`（所以 `ArrayRef` 专门写了 `initializer_list` 构造函数）。

### 2.2 显式指定：为什么 `data_ptr<scalar_t>()` 必须写尖括号

推导只能从**实参**推，不能从返回值推。`TensorBase::data_ptr<T>()` 没有参数，`T` 只出现在返回类型里，编译器无从推导，所以调用时必须显式写 `x.data_ptr<float>()`。这是 `AT_DISPATCH` 那段代码里 `<scalar_t>` 无处不在的原因：`data_ptr<scalar_t>()`、`Vectorized<scalar_t>`、`static_cast<scalar_t>(alpha)`，每一处都是在把编译期的 `scalar_t` 显式喂给另一个模板。

Java 也有类似情形：`Collections.<String>emptyList()` 是显式类型见证（type witness），但 Java 通常能从赋值目标推导，C++ 不能——C++ 的推导只看实参。

### 2.3 `auto`、`decltype` 与 `std::declval`

`auto` 让编译器从初始化表达式推导变量类型，规则和模板参数推导相同。总纲开篇的 `auto x_c = x.contiguous();`、`auto out = at::empty_like(x_c);` 都是 `at::Tensor`。它在两种场合几乎是必需的：类型名太长（迭代器、lambda）或者根本写不出来（lambda 的闭包类型没有名字，只能 `auto f = [&] {...};`）。

`decltype(expr)` 给出表达式的类型而不求值。它在泛型代码里用来"问"一个类型能做什么。`c10/util/StringUtil.h` 里检测一个类型能否被 `<<` 到 `ostream` 的写法：

```cpp
template <class T, class = std::ostream&>
struct Streamable : std::false_type {};

template <class T>
struct Streamable<T, decltype(std::declval<std::ostream&>() << T{})>
    : std::true_type {};
```

`std::declval<X>()` 在不构造对象的前提下"假装"有一个 `X` 类型的值，只能出现在不求值的上下文（`decltype`、`sizeof`）里。`decltype(std::declval<std::ostream&>() << T{})` 的意思是"`ostream << T` 这个表达式的类型"；如果 `T` 不支持 `<<`，这个表达式非法，第二个特化被丢弃（SFINAE，第五节），`Streamable<T>` 就落到第一个定义上，值为 `false`。这段代码服务于 `TORCH_CHECK` 的消息拼接：能 `<<` 的直接输出，是枚举但不能 `<<` 的转成底层整数再输出。

`c10/util/ArrayRef.h` 末尾的推导指引也用了 `decltype` + `declval`：

```cpp
// Generic container constructor (anything with .data() and .size())
template <typename Container>
ArrayRef(const Container&) -> ArrayRef<
    std::remove_pointer_t<decltype(std::declval<Container>().data())>>;
```

"任何有 `.data()` 的容器，`ArrayRef` 的元素类型就是 `data()` 返回指针去掉一层指针后的类型"。

### 2.4 返回类型推导与尾置返回类型

C++14 起函数可以写 `auto` 返回类型，由 `return` 语句推导。`c10/util/StringUtil.h` 的 `c10::str`：

```cpp
template <typename... Args>
inline auto str(const Args&... args) {
  return detail::_str_wrapper<
      typename detail::CanonicalizeStrTypes<Args>::type...>::call(args...);
}
```

返回类型取决于 `_str_wrapper<...>::call` 返回什么——多数情况下是 `std::string`，但只有一个 `const char*` 参数时直接返回 `const char*`（第四节的特化），写 `auto` 就不用为每种情况分别声明。

另一种写法是尾置返回类型 `-> Type`，可以引用参数名。`aten/src/ATen/core/TensorBase.h` 里 `register_hook`：

```cpp
template <typename T>
auto TensorBase::register_hook(T&& hook) const -> TensorBase::hook_return_void_t<T> {
  // Return the grad argument in case of a hook with void return type to have an
  // std::function with Tensor return type
  static_assert(std::is_same_v<decltype(hook(TensorBase())), void>,
                "Expected hook to return void");
  return _register_hook([fn=std::forward<T>(hook)](const TensorBase& grad) {
    fn(grad);
    return TensorBase();
  });
}
```

`hook_return_void_t<T>` 是一个 `enable_if_t<...>` 别名（第五节），在 `T` 不满足条件时让这个重载消失。`decltype(hook(TensorBase()))` 问"调用 hook 的返回类型是什么"，`static_assert` 在编译期检查它是 `void`。

### 2.5 CTAD：类模板参数也能推导

C++17 起类模板的参数可以从构造函数实参推导（Class Template Argument Deduction），`std::pair p(1, 2.0)` 得到 `pair<int, double>`。当构造函数是继承来的时，编译器无法自动推导，需要手写**推导指引**（deduction guide）。`c10/util/ArrayRef.h` 因为把大部分构造函数移到了基类 `HeaderOnlyArrayRef` 里（PyTorch 2.x 中的变化：v2.13.0 把 `ArrayRef` 拆成 `torch/headeronly/util/HeaderOnlyArrayRef.h` 的 header-only 基类和 `c10/util/ArrayRef.h` 的派生类，注释说明是为了让不链接 `libtorch.so` 的扩展也能用），所以在类定义后面补了一组指引：

```cpp
/// Deduction guides for ArrayRef to support CTAD with inherited constructors
/// These mirror the constructors inherited from HeaderOnlyArrayRef
// Single element constructor
template <typename T>
ArrayRef(const T&) -> ArrayRef<T>;

// Pointer and length constructor
template <typename T>
ArrayRef(const T*, size_t) -> ArrayRef<T>;
// ...
// std::vector constructor
template <typename T, typename A>
ArrayRef(const std::vector<T, A>&) -> ArrayRef<T>;
```

有了它们，`c10::ArrayRef ref(vec);` 可以不写 `<int64_t>`。日常代码里直接用别名 `IntArrayRef` 更常见，但读到 `X(...) -> X<T>;` 这种形状时要知道它是推导指引，不是函数声明。

## 三、非类型模板参数：值也可以是模板参数

### 3.1 `std::array<T, N>` 与 `SmallVector<T, N>`

Java 泛型的参数只能是类型。C++ 模板的参数还可以是**值**——整数、枚举、指针、`bool`——只要它在编译期是常量。`std::array<float, 4>` 的 `4` 就是非类型模板参数：长度是类型的一部分，`array<float, 4>` 和 `array<float, 8>` 是两个类型，`sizeof` 不同，不能互相赋值，但也因此不需要在对象里存长度、不需要堆分配。

`c10/util/SmallVector.h`（从 LLVM 移植）把这个思路用在"小容量内联、大容量转堆"的容器上：

```cpp
/// Storage for the SmallVector elements.  This is specialized for the N=0 case
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct SmallVectorStorage {
  alignas(T) char InlineElts[N * sizeof(T)];
};
// ...
template <
    typename T,
    unsigned N = CalculateSmallVectorDefaultInlinedElements<T>::value>
class /* LLVM_GSL_OWNER */ SmallVector : public SmallVectorImpl<T>,
                                         SmallVectorStorage<T, N> {
```

`N` 个元素的内联缓冲区直接作为对象的一部分（`char InlineElts[N * sizeof(T)]`），元素数不超过 `N` 时完全不碰堆。`c10/util/DimVector.h` 把它实例化成 tensor 维度专用的容器：

```cpp
constexpr size_t kDimVectorStaticSize = C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;

/// A container for sizes or strides
using DimVector = SmallVector<int64_t, kDimVectorStaticSize>;
```

`C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE` 在 `c10/core/impl/SizesAndStrides.h` 里是 5：绝大多数 tensor 不超过 5 维，`DimVector` 在栈上就放得下，算 shape 时不分配。第七节回来细看。

第二节提到的 C 数组引用 `const T (&arr)[N]` 也是非类型参数在起作用：`torch/headeronly/util/HeaderOnlyArrayRef.h` 里

```cpp
  /// Construct a HeaderOnlyArrayRef from a C array.
  template <size_t N>
  /* implicit */ constexpr HeaderOnlyArrayRef(const T (&Arr)[N])
      : Data(Arr), Length(N) {}
```

编译器从数组类型推导出 `N`，长度不需要调用方再传一遍。

### 3.2 枚举做模板参数：`ScalarTypeToCPPType<N>`

非类型参数可以是枚举值，这是 `AT_DISPATCH` 的基石。`torch/headeronly/core/ScalarType.h`（PyTorch 2.x 中的变化：v2.13.0 把 `ScalarType` 枚举及其映射从 `c10/core/ScalarType.h` 挪进了 `torch/headeronly/`，`c10/core/ScalarType.h` 现在 `#include` 它并补充 `kFloat` 等常量和类型提升表）：

```cpp
namespace impl {

// These are used to map ScalarTypes to C++ types.

template <c10::ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)                \
  template <>                                                                \
  struct ScalarTypeToCPPType<c10::ScalarType::scalar_type> {                 \
    using type = cpp_type;                                                   \
    /* ... */                                                                \
    static type t;                                                           \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

} // namespace impl
```

`ScalarTypeToCPPType` 的模板参数 `N` 是一个 `ScalarType` 枚举值。主模板只有声明没有定义；`AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 这个宏对表里每一对 `(uint8_t, Byte)`、`(float, Float)`、`(double, Double)`…… 生成一个全特化（第四节），每个特化里 `using type = cpp_type;`。于是 `ScalarTypeToCPPTypeT<ScalarType::Float>` 就是 `float`，`ScalarTypeToCPPTypeT<ScalarType::Double>` 就是 `double`——**一个编译期常量枚举值到 C++ 类型的映射表**。反向映射在同一个文件里：

```cpp
// Map from C++ type to ScalarType enum
template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)                  \
  template <>                                                                  \
  struct CppTypeToScalarType<cpp_type>                                         \
      : std::                                                                  \
            integral_constant<c10::ScalarType, c10::ScalarType::scalar_type> { \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CppTypeToScalarType)
```

`CppTypeToScalarType<float>` 继承 `std::integral_constant<ScalarType, ScalarType::Float>`，所以 `CppTypeToScalarType<float>::value` 是 `ScalarType::Float`，`CppTypeToScalarType<float>()` 可以隐式转换成这个枚举值——1.3 节 `TensorMethods.cpp` 里 `check_type(*this, c10::CppTypeToScalarType<T>())` 就是这样用的：编译期类型 `T` 转成运行期枚举，再和 tensor 的 dtype 比较。

注意这里的限制：模板参数必须是**编译期常量**。`ScalarTypeToCPPTypeT<x.scalar_type()>` 是非法的，因为 `x.scalar_type()` 是运行期的值。这个限制正是第六节整个 `switch` 的由来。

### 3.3 CUDA kernel 的 `BLOCK_SIZE`：为什么 GPU 代码特别依赖非类型参数

CUDA kernel 是 C++ 函数（多一个 `__global__`），host 侧的启动代码是普通 C++。vLLM 的 kernel 几乎都把 tile 大小、向量宽度、是否有权重等参数做成非类型模板参数。`csrc/libtorch_stable/cache_kernels.cu`：

```cpp
template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    // ...
```

host 侧根据运行期的 `num_tokens` 选择实例化哪一个：

```cpp
// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                    \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                  \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,                 \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),                \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                                 \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                       \
          /* ... */                                                           \
          quant_block_size);
// ...
  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  // ...
```

这段 if-else 链是本文最重要的模式之一，第六节的 `AT_DISPATCH` 是它的另一个形态：**运行期的值（`num_tokens`）通过一组分支被映射到有限个编译期常量（1、2、4、8、16……），每个分支实例化一份 kernel**。为什么不直接把 `BLOCK_Y_SIZE` 当普通参数传进 kernel？因为在 GPU 上它决定的东西必须在编译期确定：共享内存数组的大小（`__shared__ float buf[BLOCK_SIZE]` 不能用运行期变量）、`#pragma unroll` 能否完全展开、寄存器分配、以及 `dim3(8, BLOCK_Y_SIZE)` 与 kernel 内部索引计算的一致性。编译期常量让编译器把循环边界、地址偏移全部折叠成立即数；换成运行期参数，每个线程都要多做一次乘除和分支。

`csrc/libtorch_stable/layernorm_kernels.cu` 的 `fused_add_rms_norm_kernel` 把三种参数放在一起：

```cpp
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size], null if !HasWeight
    const float epsilon, const int num_tokens, const int hidden_size,
    const int64_t residual_stride) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);
  // ...
```

`scalar_t` 是类型参数（由 dtype 分发决定），`width` 是向量宽度（8 或 0，由指针对齐决定），`HasWeight` 是 `bool`（由 `weight` 是否有值决定）。host 侧的选择逻辑：

```cpp
#define LAUNCH_FUSED_ADD_RMS_NORM(width, has_weight)                       \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                     \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {              \
        if (has_weight) {                                                  \
          vllm::fused_add_rms_norm_kernel<scalar_t, width, true>           \
              <<<grid, block, 0, stream>>>(                                \
                  input.mutable_data_ptr<scalar_t>(), input_stride,        \
                  residual.mutable_data_ptr<scalar_t>(),                   \
                  weight->const_data_ptr<scalar_t>(), epsilon, num_tokens, \
                  hidden_size, residual_stride);                           \
        } else {                                                           \
          vllm::fused_add_rms_norm_kernel<scalar_t, width, false>          \
              <<<grid, block, 0, stream>>>(/* ... */);                     \
        }                                                                  \
      });
// ...
    if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
        !batch_invariant_launch) {
      LAUNCH_FUSED_ADD_RMS_NORM(8, true);
    } else {
      LAUNCH_FUSED_ADD_RMS_NORM(0, true);
    }
```

数一下实例化了多少份 kernel：dtype 三种（`VLLM_STABLE_DISPATCH_FLOATING_TYPES` 是 Float/Half/BFloat16）× `width` 两种 × `HasWeight` 两种 = 12 份，全部在编译期生成，运行期用三层分支选一份。这就是 CUDA 代码 `.cu` 文件编译慢、二进制大的直接原因，也是 host 侧 C++ 必须熟练的部分：你写的每一个 `if` 都在选择一份已经存在的代码，而不是在改变一份代码的行为。

（`HasWeight` 在上面是通过运行期 `if (has_weight)` 选两份不同实例化，而不是在 kernel 里 `if (HasWeight)`——kernel 里也可以写 `if constexpr (HasWeight)`，第五节讨论。）


## 四、特化与变参模板

### 4.1 全特化：为某组参数单独给一份实现

模板是通用配方，但常常需要"对这一组参数用另一套代码"。**全特化**（explicit/full specialization）就是给出一组完整的参数并单独定义：

```cpp
template <>                                    // 空的模板参数列表：这是特化
struct ScalarTypeToCPPType<c10::ScalarType::Float> {
  using type = float;
};
```

3.2 节的 `ScalarTypeToCPPType` 是全特化最纯粹的用法：主模板不定义，只有特化有定义，于是"查表"变成了"选择特化"，查不到的键直接编译失败（`ScalarTypeToCPPType<ScalarType::Undefined>::type` 会报 incomplete type）。

vLLM `csrc/libtorch_stable/type_convert.cuh` 用全特化实现"某类型是否有向量化转换支持"：

```cpp
template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

template <>
struct _typeConvert<float> {
  static constexpr bool exists = true;
  using hip_type = float;
  using packed_hip_type = float2;
  using packed_hip_type4 = float4;  // For 128-bit vectorization

  __device__ static __forceinline__ float convert(hip_type x) { return x; }
  // ...
};
```

主模板给默认值 `exists = false`，`float`、`Half`、`BFloat16` 各有一个特化把它设为 `true` 并提供转换函数。3.3 节 kernel 签名里的 `_typeConvert<scalar_t>::exists` 就是在编译期查这张表。这种"主模板给默认、特化给例外"的写法在 C++ 里叫 traits（类型特征），标准库的 `std::is_same`、`std::is_trivially_copyable` 全是这个模式。Java 没有对应机制——你不能为 `Foo<Integer>` 单独写一个类体。

函数模板也能全特化。`aten/src/ATen/templates/TensorMethods.cpp`：

```cpp
 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
```

`Tensor::item<float>()`、`Tensor::item<double>()`…… 各是一个特化，把无类型参数的 `item()`（返回 `Scalar`）的结果转成具体类型。

`c10/util/StringUtil.h` 的 `_str_wrapper` 展示全特化用于性能：

```cpp
template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
  }
};

// Specializations for already-a-string types.
template <>
struct _str_wrapper<std::string> final {
  // return by reference to avoid the binary size of a string copy
  static const std::string& call(const std::string& str) {
    return str;
  }
};
template <>
struct _str_wrapper<const char*> final {
  static const char* call(const char* str) {
    return str;
  }
};
```

`TORCH_CHECK(cond, "msg")` 只有一个字符串参数时，走 `_str_wrapper<const char*>` 特化，不构造 `ostringstream`，不拷贝字符串。这就是为什么 `TORCH_CHECK` 可以在热路径上随便写——只有失败时才有开销，而且单字符串消息连 `std::string` 都不构造。

### 4.2 偏特化：只固定一部分参数

**偏特化**（partial specialization）固定部分参数或给参数加上某种模式，其余仍是模板。`c10/util/SmallVector.h` 用它区分"平凡可拷贝的 `T`"和"非平凡的 `T`"：

```cpp
template <
    typename T,
    bool = (std::is_trivially_copy_constructible_v<T>) &&
        (std::is_trivially_move_constructible_v<T>) &&
        std::is_trivially_destructible_v<T>>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
  // ...
  static void destroy_range(T* S, T* E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }
// ...
/// SmallVectorTemplateBase<TriviallyCopyable = true> - This is where we put
/// method implementations that are designed to work with trivially copyable
/// T's. This allows using memcpy in place of copy/move construction and
/// skipping destruction.
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
```

主模板的第二个参数有默认值——一个由 `T` 算出来的 `bool`。`SmallVector<int64_t>` 算出 `true`，选到 `<T, true>` 偏特化，用 `memcpy` 搬元素、不调析构；`SmallVector<Tensor>` 算出 `false`，选主模板，逐个调用构造/析构。这是"按类型参数换实现"的典型用法，也是 `DimVector` 高效的原因之一。

同文件还有一个偏特化处理 `N = 0` 的边界：

```cpp
/// We need the storage to be properly aligned even for small-size of 0 so that
/// the pointer math in \a SmallVectorTemplateCommon::getFirstEl() is
/// well-defined.
template <typename T>
struct alignas(T) SmallVectorStorage<T, 0> {};
```

`T` 仍然是模板参数，只固定了 `N = 0`——零长度数组在 C++ 里不合法，所以要单独给一个空结构体。

偏特化的"模式"可以比固定值更复杂。`c10/util/StringUtil.h`：

```cpp
template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

template <size_t N>
// NOLINTNEXTLINE(*c-arrays*)
struct CanonicalizeStrTypes<char[N]> {
  using type = const char*;
};
```

"如果 `T` 是任意长度的 `char` 数组"——`char[N]` 是一个带非类型参数的模式。字符串字面量 `"msg"` 的类型是 `const char[4]`，这个偏特化把它统一成 `const char*`，让 `_str_wrapper<const char*>` 的全特化能命中。

### 4.3 函数模板不能偏特化，用重载

一个常见的坑：**函数模板只能全特化，不能偏特化**。想对"所有指针类型"或"所有 `optional<T>`"给函数模板另一套实现，要用重载。`c10/util/StringUtil.h` 的 `_str`：

```cpp
template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  if constexpr (std::is_enum_v<T> && !Streamable<T>::value) {
    return _str(ss, static_cast<typename std::underlying_type<T>::type>(t));
  } else {
    ss << t;
    return ss;
  }
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const std::optional<T>& t) {
  if (t.has_value()) {
    return _str(ss, t.value());
  }
  ss << "std::nullopt";
  return ss;
}
```

第二个 `_str` 不是特化，是一个更特殊的重载模板；重载决议优先选更特殊的那个，所以 `_str(ss, std::optional<int>{})` 走第二个。效果与偏特化相同，语法不同。读 PyTorch 源码时看到一组同名函数模板，通常就是在做这件事。

### 4.4 变参模板与参数包展开

Java 的可变参数 `Object... args` 是一个数组，所有参数被装箱成 `Object`，类型在运行期才知道。C++ 的变参模板（variadic template）在编译期知道每个参数的**精确类型**和**个数**。语法是三个点：

```cpp
template <class... Args>          // Args 是一个"类型参数包"：零个或多个类型
static intrusive_ptr make(Args&&... args) {   // args 是"函数参数包"
  return intrusive_ptr(new TTarget(std::forward<Args>(args)...));  // 展开
}
```

这是 `c10/util/intrusive_ptr.h` 里 `make_intrusive` 的核心（第二篇照抄过）。`Args&&... args` 接收任意个任意类型的实参；`std::forward<Args>(args)...` 是**包展开**（pack expansion）：模式 `std::forward<Args>(args)` 对包里每个元素重复一次，用逗号连接。`make_intrusive<TensorImpl>(storage, sizes, dtype)` 展开后就是 `new TensorImpl(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2))`——参数原样转给构造函数，左值仍是左值、右值仍是右值，这叫**完美转发**。第二篇里 `TensorImpl` 的 sink 参数之所以能在 `make_intrusive` 那一层被移动而不是拷贝，就靠这个。

`c10::str` 用递归展开处理"把任意个参数拼成字符串"：

```cpp
inline std::ostream& _str(std::ostream& ss) {
  return ss;
}
// ... 上面 4.3 节的单参数版本 ...
template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}
```

`_str(ss, a, b, c)` 匹配 `T = A, Args = {B, C}`，处理完 `a` 后以 `args...` 递归调用 `_str(ss, b, c)`，最终落到零参数版本终止。每一层递归都是不同的实例化，全部在编译期完成，运行期就是三次 `<<`。C++17 的折叠表达式 `(ss << ... << args)` 可以把这类递归写成一行，PyTorch 代码里两种写法都能看到。

`TORCH_CHECK` 的消息就是这样拼出来的，`c10/util/Exception.h`：

```cpp
namespace c10::detail {
template <typename... Args>
auto torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  return ::c10::str(args...);
}
inline C10_API const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline C10_API const char* torchCheckMsgImpl(
    const char* /*msg*/,
    const char* args) {
  return args;
}
} // namespace c10::detail
```

变参模板加两个非模板重载：零个用户参数用默认消息，一个 `const char*` 直接返回，其他情况才走 `c10::str`。这又是 4.3 节"用重载代替偏特化"的应用。

最后区分一下两种"三个点"：`AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)` 里的 `...` 和 `__VA_ARGS__` 是**预处理器**的变参宏，做的是文本替换，与类型无关（第五篇展开）；`template <class... Args>` 是**模板**的参数包，编译器知道每个参数的类型。`AT_DISPATCH` 用的是前者，所以 lambda 里的逗号会把它切碎——这是 6.4 节 `AT_WRAP` 存在的原因。


## 五、把分支移到编译期

### 5.1 `constexpr`：编译期可求值

`constexpr` 声明一个变量或函数**可以**在编译期求值。变量：`torch/headeronly/core/ScalarType.h` 的

```cpp
constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);
```

可以直接用作数组长度或模板参数。`c10/core/ScalarType.h` 用宏批量定义 `constexpr ScalarType kFloat = ScalarType::Float;` 等常量，这就是源码里 `kFloat`、`kHalf`、`kBFloat16` 的来历。

函数：`aten/src/ATen/Dispatch.h` 开头

```cpp
inline constexpr bool should_include_kernel_dtype(
    const char* /*kernel_tag_str*/,
    at::ScalarType /*scalar_type*/
) {
  return true;
}
```

`constexpr` 函数在实参是常量时于编译期求值，否则退化成普通函数。这个函数是移动端"选择性构建"的钩子：默认返回 `true`；在 `TEMPLATE_SELECTIVE_BUILD` 下由代码生成替换成一张查表函数，对没用到的 dtype 返回 `false`，从而让下一小节的 `if constexpr` 在编译期把那些 `case` 整个删掉。

`constexpr` 与 `const` 的区别：`const` 只是"不可修改"，值可以运行期才确定；`constexpr` 要求编译期可知。第六节宏展开里有一行 `constexpr const char* at_dispatch_name = NAME;`——两个关键字连用，`constexpr` 说指针值编译期确定，`const` 说指向的字符不可改。

### 5.2 `if constexpr`：在编译期删掉一个分支

C++17 的 `if constexpr` 要求条件是编译期常量，**没被选中的分支不会被实例化**——里面的代码可以对当前类型完全不合法，也不会报错。这和普通 `if` 有本质区别：普通 `if` 两个分支都要编译通过，只是运行期不执行其中一个。

`c10/core/TensorImpl.h`：

```cpp
  template <typename T>
  ArrayRef<T> generic_sizes() {
    static_assert(
        std::is_same_v<T, int64_t> || std::is_same_v<T, c10::SymInt>,
        "Only supports int64_t and c10::SymInt.");

    if constexpr (std::is_same_v<T, int64_t>) {
      return sizes();
    } else {
      return sym_sizes();
    }
  }
```

`sizes()` 返回 `IntArrayRef`，`sym_sizes()` 返回 `SymIntArrayRef`，返回类型不同。用普通 `if`，`generic_sizes<int64_t>` 的 `else` 分支里 `return sym_sizes();` 会因为类型不匹配编译失败；`if constexpr` 让那个分支在 `T = int64_t` 时根本不存在。

vLLM `csrc/libtorch_stable/type_convert.cuh` 的 `_f16Vec::operator+=`：

```cpp
  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] += other.data[i];
          data[i + 1] += other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp += T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          // ...
```

两层 `if constexpr`：外层看非类型参数 `width` 的奇偶，内层看类型参数 `T2` 是不是 `float2`。生成的每份代码里都没有这些判断，只剩被选中的那条路径。3.3 节提到的 `HasWeight` 如果放进 kernel 体，写法就是 `if constexpr (HasWeight) { ... }`。

`aten/src/ATen/Dispatch.h` 的选择性构建钩子也用了它：

```cpp
#define AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type)   \
  do {                                                \
    if constexpr (!at::should_include_kernel_dtype(   \
                      at_dispatch_name, enum_type)) { \
      TORCH_CHECK(                                    \
          false,                                      \
          "dtype '",                                  \
          toString(enum_type),                        \
          "' not selected for kernel tag ",           \
          at_dispatch_name);                          \
    }                                                 \
  } while (0)
```

默认构建里 `should_include_kernel_dtype` 恒为 `true`，整个 `if constexpr` 体被删掉，零开销。

### 5.3 `static_assert`：编译期断言

`static_assert(cond, "msg")` 在编译期检查常量条件，失败即编译错误。它是给模板加约束最直接的手段，错误信息也是你自己写的、可读的。上面 `generic_sizes` 的 `static_assert` 让 `generic_sizes<float>()` 报 "Only supports int64_t and c10::SymInt." 而不是一堆实例化栈。`HeaderOnlyArrayRef` 用它挡住 `vector<bool>`：

```cpp
  template <typename A>
  /* implicit */ HeaderOnlyArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same_v<T, bool>,
        "HeaderOnlyArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }
```

`vector<bool>` 是位压缩的，没有 `bool*` 形式的 `data()`，一个视图类型无法指向它。3.3 节 kernel 里的两条 `static_assert` 检查 `_f16Vec` 的内存布局能被 `reinterpret_cast` 安全地按 `scalar_t` 数组解释。

### 5.4 SFINAE 与 `std::enable_if`：让一个重载"消失"

SFINAE 是 "Substitution Failure Is Not An Error" 的缩写：在推导模板参数并把它代入函数签名时，如果代入产生非法类型，这个模板**不报错，只是从候选集里被移除**。这是 C++11/14/17 时代给模板加约束的主要手段，读 PyTorch 源码绕不开。

`std::enable_if_t<cond, T>` 在 `cond` 为真时是 `T`，为假时不存在（代入失败）。三种常见摆放位置在源码里都有：

**放在模板参数列表里**（`HeaderOnlyArrayRef.h` 的通用容器构造函数）：

```cpp
  template <
      typename Container,
      typename U = decltype(std::declval<Container>().data()),
      // NOLINTNEXTLINE(modernize-use-constraints)
      typename = std::enable_if_t<
          (std::is_same_v<U, T*> || std::is_same_v<U, T const*>)>>
  /* implicit */ HeaderOnlyArrayRef(const Container& container)
      : Data(container.data()), Length(container.size()) {}
```

"任何容器，只要它的 `data()` 返回 `T*` 或 `const T*`"——`decltype(std::declval<Container>().data())` 若 `Container` 没有 `data()` 就代入失败，`enable_if_t` 若返回类型不对也代入失败，两种情况下这个构造函数都安静地消失，不影响其他构造函数。

**放在返回类型上**（3.3 节 vLLM 的两个 kernel 重载）：

```cpp
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(/* ... */)
// ...
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(/* ... */)
```

两个同名同参数的函数模板，返回类型分别是 `enable_if_t<A>` 和 `enable_if_t<!A>`（省略第二个参数时默认 `void`）。任意一组 `<scalar_t, width, HasWeight>` 只会让其中一个的返回类型合法，另一个被 SFINAE 移除，于是"向量化版本"和"通用版本"互斥共存。这比一个大 `if constexpr` 更适合 CUDA，因为两个版本的 `__shared__` 声明和循环结构完全不同。

**放在一个哑参数上**（`aten/src/ATen/native/cpu/Loops.h`）：

```cpp
template <typename func_t,
    std::enable_if_t<!std::is_void_v<typename function_traits<func_t>::result_type>>* = nullptr>
inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
// ...
template <typename func_t,
    std::enable_if_t<std::is_void_v<typename function_traits<func_t>::result_type>>* = nullptr>
inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
```

`enable_if_t<...>* = nullptr` 是一个有默认值的非类型模板参数，条件不满足时类型不存在、代入失败。这个写法的好处是不占用返回类型，也不需要额外的类型参数名。

2.3 节的 `Streamable` 是 SFINAE 用在**类模板偏特化**上：`decltype(std::declval<std::ostream&>() << T{})` 代入失败时偏特化被丢弃，落回主模板。这个"检测某表达式是否合法"的套路叫 detection idiom，`function_traits`、`needs_dynamic_casting` 等 ATen 内部工具都是这样实现的。

Java 对照：Java 的 `<T extends Comparable<T>>` 是**显式的、在定义处检查的**约束；SFINAE 是**隐式的、在重载决议时起作用的**过滤器。SFINAE 的缺点很明显——意图藏在 `enable_if` 的布尔表达式里，报错时只说"没有匹配的函数"，不说为什么。这正是 C++20 concepts 要解决的问题。

### 5.5 C++20 concepts：约束的正式语法

C++20 把约束变成一等语法。用 concepts 重写 `HeaderOnlyArrayRef` 的容器构造函数，大致是：

```cpp
template <typename Container>
  requires std::is_same_v<decltype(std::declval<Container>().data()), const T*>
HeaderOnlyArrayRef(const Container& container);
```

或者先定义一个具名 concept：

```cpp
template <typename C, typename T>
concept ContiguousContainerOf = requires(const C& c) {
  { c.data() } -> std::convertible_to<const T*>;
  { c.size() } -> std::convertible_to<size_t>;
};
```

不满足时编译器会说"约束 `ContiguousContainerOf<X, int64_t>` 不满足，因为 `c.data()` 不存在"，比 SFINAE 的报错好得多。Java 工程师会觉得这终于像 `extends` 了——但注意 concepts 仍是**鸭子类型**：它检查"能不能这样用"，不要求 `Container` 声明实现了某个接口。

PyTorch v2.13.0 用 C++20 编译，但源码树里没有使用 `requires`/`concept`（在 `c10/` 和 `aten/src/ATen/core/` 下搜不到这两个关键字的语法用法）；`HeaderOnlyArrayRef.h` 里 `enable_if` 上方那行 `// NOLINTNEXTLINE(modernize-use-constraints)` 说明 clang-tidy 已经在建议改成 concepts，只是还没有做。vLLM 同样。所以读这两个项目时，约束都长成 `enable_if` 的样子；写新代码时，如果项目允许 C++20，concepts 是更好的选择。

## 六、编译期分派与运行期分派：逐层展开 `AT_DISPATCH_FLOATING_TYPES`

### 6.1 问题：`dtype` 是运行期的值，kernel 需要编译期的类型

把前五节的结论放在一起，就能看清 `AT_DISPATCH` 为什么必须存在。

一个 `at::Tensor` 的 dtype 存在 `TensorImpl` 里，是一个 `ScalarType` 枚举值——**运行期的值**。用户在 Python 里写 `torch.randn(3, dtype=torch.float64)`，C++ 层只在运行时才知道这个 tensor 是 `Double`。

而一个高效的 kernel 需要**编译期的类型**：`float*` 和 `double*` 的解引用是不同的机器指令，`Vectorized<float>` 和 `Vectorized<double>` 是不同的类（一个 256 位寄存器装 8 个 `float` 或 4 个 `double`），`a[i] + b[i]` 对 `c10::Half` 要先转 `float` 再算。这些差别不能在运行期用一个 `if (dtype == Float)` 包在每次访存外面——那会让最内层循环里全是分支，性能是灾难；正确做法是 1.1 节的模板：每个 dtype 一份 kernel，循环体内没有任何 dtype 判断。

于是问题变成：手里有一个运行期的 `ScalarType`，怎么调到编译期实例化好的 `kernel<float>` 或 `kernel<double>`？3.2 节说过 `ScalarTypeToCPPTypeT<x.scalar_type()>` 是非法的——模板参数必须是常量。**唯一的办法是枚举所有可能的值，每个值写一个分支，每个分支里 dtype 就是常量了**：

```cpp
switch (x.scalar_type()) {
  case ScalarType::Float:  kernel<float>(...);  break;
  case ScalarType::Double: kernel<double>(...); break;
  default: TORCH_CHECK(false, "not implemented for ", toString(x.scalar_type()));
}
```

这就是 3.3 节 vLLM 里 `if (num_tokens < 32) ... else if (num_tokens < 64) ...` 的同一件事：**运行期分派**（一个 `switch` 或 if 链，运行期执行一次）把控制权交给**编译期分派**（模板实例化，每份代码在编译期就固定了类型）。`switch` 只在 kernel 入口跑一次，热循环里没有分支。

手写这个 `switch` 的问题是：PyTorch 有几十个 dtype、几千个 kernel，每个 kernel 都要抄一遍 `case`，而且 kernel 体里到处要写 `float`/`double`。`AT_DISPATCH_*` 宏族就是把这个 `switch` 自动生成出来，并且在每个 `case` 里把当前 dtype 对应的 C++ 类型命名为 `scalar_t`，让 kernel 体只写一遍。

### 6.2 第一层：`AT_DISPATCH_FLOATING_TYPES` = `AT_DISPATCH_SWITCH` + 两个 `CASE`

`aten/src/ATen/Dispatch.h`：

```cpp
#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

`AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)` 三个参数：`TYPE` 是运行期的 `ScalarType` 表达式（`input.scalar_type()`），`NAME` 是一个字符串字面量（报错和 profiling 用），`...` 是 lambda。它展开成一个 `AT_DISPATCH_SWITCH`，第三个参数是两个 `AT_DISPATCH_CASE`——每个都把同一个 lambda（`__VA_ARGS__`）作为参数传进去。注意这里 lambda 的**文本**已经被复制了两份：一份跟着 `Double`，一份跟着 `Float`。

同文件里其他变体只是 `case` 列表不同：`AT_DISPATCH_FLOATING_TYPES_AND_HALF` 多一个 `Half`，`AT_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)` 让调用方再追加两个，`AT_DISPATCH_ALL_TYPES` 加上五种整数。文件开头那段长注释解释了默认集合为什么是 `float`/`double` + 整数而不含 `Half`/`bool`/complex——历史原因加上这些类型"行为不好"。

### 6.3 第二层：`AT_DISPATCH_SWITCH` 是一个立即调用的 lambda

```cpp
#define AT_DISPATCH_SWITCH(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH_TMPL(                 \
      RECORD_KERNEL_FUNCTION_DTYPE,         \
      TORCH_CHECK_NOT_IMPLEMENTED,          \
      TYPE,                                 \
      NAME,                                 \
      __VA_ARGS__)
```

PyTorch 2.x 中的变化：v2.13.0 把 `switch` 的骨架挪到了 `torch/headeronly/core/Dispatch.h`，命名为 `THO_DISPATCH_SWITCH_TMPL`（THO = torch header-only），多出两个"钩子"参数 `PRELUDE` 和 `CHECK_NOT_IMPLEMENTED`，ATen 版传入自己的 profiling 记录宏和 `TORCH_CHECK_NOT_IMPLEMENTED`，header-only 版（供不链接 libtorch 的稳定 ABI 扩展使用）传入空宏和 `STD_TORCH_CHECK`。骨架本身：

```cpp
#define THO_DISPATCH_SWITCH_TMPL(                                           \
    PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)                        \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    constexpr const char* at_dispatch_name = NAME;                          \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    torch::headeronly::ScalarType _st = ::detail::scalar_type(the_type);    \
    PRELUDE(at_dispatch_name, _st);                                         \
    C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")             \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        CHECK_NOT_IMPLEMENTED(                                              \
            false,                                                          \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            torch::headeronly::toString(_st),                               \
            "'");                                                           \
    }                                                                       \
    C10_DIAGNOSTIC_POP()                                                    \
  }()
```

逐行读：

- 整体是 `[&] { ... }()`——定义一个引用捕获一切的 lambda 并**立即调用**它（IIFE，immediately-invoked function expression）。这样做有两个效果：第一，整个宏是一个**表达式**而不是语句，可以写在 `return AT_DISPATCH_...(...)` 或赋值右边；第二，`switch` 里的 `return` 是从这个 lambda 返回，不是从外层函数返回，所以宏可以放在任何函数里而不会意外结束它。
- `const auto& the_type = TYPE;`：`TYPE` 只求值一次（注释说明是为了避免 `TYPE` 是昂贵或有副作用的表达式时被重复求值）。
- `constexpr const char* at_dispatch_name = NAME;`：`NAME` 必须是字面量，5.1 节讲过它被 `if constexpr` 用于选择性构建。
- `::detail::scalar_type(the_type)`：同文件里定义的一个恒等函数（`inline ScalarType scalar_type(ScalarType s) { return s; }`），是早年 `TYPE` 参数还允许传别的类型时留下的转换点，现在只是把 `the_type` 原样取出。
- `switch (_st) { __VA_ARGS__ default: ... }`：`__VA_ARGS__` 是所有 `case`（由第一层的 `AT_DISPATCH_CASE_FLOATING_TYPES` 展开），`default` 抛 `NotImplementedError`——这就是你在 Python 里见到的 `RuntimeError: "log_sigmoid_cpu" not implemented for 'Long'`。
- `-Wswitch-enum` 的 push/pop：`ScalarType` 有几十个枚举值而 `case` 只有两个，编译器默认会警告"枚举值没有全部处理"，这里主动关掉。

### 6.4 第三层：`AT_DISPATCH_CASE` 定义了 `scalar_t`

```cpp
#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(                      \
      AT_PRIVATE_CHECK_SELECTIVE_BUILD, enum_type, HINT, __VA_ARGS__)

#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)
```

`AT_DISPATCH_CASE(enum_type, lambda)` 调用 `AT_PRIVATE_CASE_TYPE_USING_HINT`，把 `HINT` 固定为标识符 `scalar_t`——**这一行就是 `scalar_t` 这个名字的出处**。它再转给 `torch/headeronly/core/Dispatch.h` 里的骨架：

```cpp
#define THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...) \
  case enum_type: {                                                          \
    PRELUDE(enum_type);                                                      \
    using HINT [[maybe_unused]] =                                            \
        torch::headeronly::impl::ScalarTypeToCPPTypeT<enum_type>;            \
    return __VA_ARGS__();                                                    \
  }
```

四行，每一行都是前面几节讲过的机制：

1. `case enum_type: {`——一个 `switch` 分支，`enum_type` 是 `at::ScalarType::Float` 这样的**编译期常量**。
2. `PRELUDE(enum_type);`——ATen 版传入的是 5.2 节的 `AT_PRIVATE_CHECK_SELECTIVE_BUILD`，默认构建下展开为空。
3. `using scalar_t [[maybe_unused]] = ScalarTypeToCPPTypeT<enum_type>;`——**在这个 `case` 的块作用域里定义类型别名 `scalar_t`**。因为 `enum_type` 是常量，可以做 3.2 节那张表的模板参数；`ScalarTypeToCPPTypeT<ScalarType::Float>` 是 `float`。`[[maybe_unused]]` 防止 lambda 体没用到 `scalar_t` 时编译器警告。
4. `return __VA_ARGS__();`——`__VA_ARGS__` 是用户的 lambda，加 `()` 立即调用它；`return` 把它的返回值作为外层 IIFE 的返回值（lambda 返回 `void` 时 `return f();` 也合法）。

`scalar_t` 是一个块作用域里的类型别名，作用域是这个 `case` 的花括号。用户 lambda 的文本恰好被贴在这个作用域里，所以 lambda 体里的 `scalar_t` 按普通的名字查找规则找到它。它不是宏参数，不是全局 typedef，不是魔法——就是一个 `using`。

### 6.5 完整展开：回答核心问题

把三层合起来，开头那段 `log_sigmoid_cpu` 代码（省略 PRELUDE 与诊断宏）预处理后大致是：

```cpp
[&] {
  const auto& the_type = input.scalar_type();
  constexpr const char* at_dispatch_name = "log_sigmoid_cpu";
  torch::headeronly::ScalarType _st = ::detail::scalar_type(the_type);
  switch (_st) {
    case at::ScalarType::Double: {
      using scalar_t [[maybe_unused]] = torch::headeronly::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double>;  // double
      return [&] {
        using Vec = Vectorized<scalar_t>;
        scalar_t* output_data = output.data_ptr<scalar_t>();
        // ... 整个 lambda 体，第一份
      }();
    }
    case at::ScalarType::Float: {
      using scalar_t [[maybe_unused]] = torch::headeronly::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>;   // float
      return [&] {
        using Vec = Vectorized<scalar_t>;
        scalar_t* output_data = output.data_ptr<scalar_t>();
        // ... 整个 lambda 体，第二份
      }();
    }
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(false, '"', at_dispatch_name, "\" not implemented for '", torch::headeronly::toString(_st), "'");
  }
}()
```

现在可以回答核心问题：

**`scalar_t` 从哪里来？** 从 `AT_DISPATCH_CASE` 宏在每个 `case` 块里生成的 `using scalar_t = ScalarTypeToCPPTypeT<enum_type>;`。名字 `scalar_t` 是 `AT_DISPATCH_CASE` 硬编码的 `HINT` 参数；类型由 `torch/headeronly/core/ScalarType.h` 里 `ScalarTypeToCPPType` 的全特化表查出。宏把运行期的枚举值变成了 `case` 标签上的编译期常量，模板再把编译期常量变成类型。

**lambda 被编译了几次？** 源码里只写了一次，但预处理后 lambda 的**文本出现了两次**（每个 `case` 一份），它们是两个**不同的 lambda 表达式**、两个不同的闭包类型，各自编译一次；两个闭包体里的 `scalar_t` 分别绑定到 `double` 和 `float`。所以准确地说：不是"一个 lambda 被实例化两次"，而是"两个长得一样的 lambda 各编译一次"。加上 lambda 体内调用的所有模板（`Vectorized<scalar_t>`、`data_ptr<scalar_t>`、内层 `parallel_for` 的 `F`），每份都独立实例化。`AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(...)` 这样的宏会生成十几份。这是 ATen 的 `.cpp` 编译慢、`libtorch_cpu.so` 巨大的直接原因，也是选择性构建（`should_include_kernel_dtype`）存在的原因。

顺便回答 4.4 节末尾埋的问题：因为 `__VA_ARGS__` 是预处理器的文本变参，lambda 体里如果有**不在括号内的逗号**（比如 `std::pair<int, int> p;` 或 `foo<A, B>()`），预处理器会把它切成多个参数。旧宏靠 `...` 吞掉尾部所有参数再用 `__VA_ARGS__` 原样吐出，所以 lambda 放在**最后一个参数**是安全的；`Dispatch_v2.h` 因为 lambda 后面还有 dtype 列表，就必须用 `AT_WRAP` 保护（下一小节）。

### 6.6 反向：`data_ptr<scalar_t>()` 里的运行期检查

`scalar_t` 是编译期确定的，但 tensor 的 dtype 是运行期的，两者一致靠什么保证？靠 1.3 节 `TensorMethods.cpp` 里的 `check_type`：

```cpp
template <typename T>
T* TensorBase::mutable_data_ptr() const {
  check_type(*this, c10::CppTypeToScalarType<T>());
  return this->unsafeGetTensorImpl()->mutable_data_ptr_impl<T>();
}
```

`CppTypeToScalarType<T>()` 把编译期类型 `T` 反查成运行期枚举，与 `scalar_type()` 比较，不一致抛错。在 `AT_DISPATCH` 的 `case` 里 `T = scalar_t` 恰好等于 `_st` 对应的类型，检查必然通过；但如果你在 `case` 外面写 `x.data_ptr<float>()` 而 `x` 是 `double`，运行时会得到 "expected scalar type Float but found Double"。两张映射表——`ScalarTypeToCPPType`（枚举→类型）和 `CppTypeToScalarType`（类型→枚举）——是一对逆映射，前者用于分派，后者用于校验。

### 6.7 vLLM 的 `dispatch_utils.h`：换一组 dtype，换一个名字

vLLM 没有重新发明这套机制，而是直接复用 ATen 的 `AT_DISPATCH_SWITCH` / `AT_DISPATCH_CASE`，只换 `case` 列表。`csrc/dispatch_utils.h`：

```cpp
/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */
#pragma once

#include <torch/all.h>

// Need a special dispatch case macro since we will nest the FP8 dispatch.
// Instead of the usual 'scalar_t', this names the dispatched type 'fp8_t'.
#define AT_DISPATCH_FP8_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

两点值得注意。第一，vLLM 的"浮点类型"是 Float/Half/BFloat16，**没有 Double**——推理引擎不需要双精度，多一份实例化只会增加编译时间和二进制大小。第二，`AT_DISPATCH_FP8_CASE` 直接调用 `AT_PRIVATE_CASE_TYPE_USING_HINT` 并把 `HINT` 改成 `fp8_t`：当一个 kernel 需要同时按激活 dtype 和 KV cache 的 fp8 dtype 分派时，两层 `AT_DISPATCH` 嵌套，内层用 `fp8_t` 命名，避免与外层的 `scalar_t` 冲突。这正好说明了 6.4 节的结论——`scalar_t` 只是 `HINT` 参数的默认值，不是什么保留字。

vLLM v0.27 还有一套 `csrc/libtorch_stable/dispatch_utils.h`，用的是 header-only 的 `THO_DISPATCH_SWITCH` / `THO_DISPATCH_CASE`（3.3 节的 `VLLM_STABLE_DISPATCH_FLOATING_TYPES`），服务于不依赖 libtorch ABI 的 kernel 文件。

### 6.8 `Dispatch_v2.h`：去掉 `_AND2`/`_AND3` 的算术

旧宏族有一个问题：想在默认集合上追加 N 个 dtype，就要用 `AT_DISPATCH_FLOATING_TYPES_AND2`、`_AND3`、`_AND4`……名字里带着个数，组合爆炸。`aten/src/ATen/Dispatch_v2.h`（PyTorch 2.x 中的变化：V2 在 2.3 引入，与 V1 并存；v2.13.0 中它的骨架同样已挪到 `torch/headeronly/core/Dispatch_v2.h`）用一种新写法解决：

```cpp
//  AT_DISPATCH_V2(
//    self.scalar_type(),
//    "_local_scalar_dense_cpu",
//    AT_WRAP([&] {
//      scalar_t value = *self.data_ptr<scalar_t>();
//      r = Scalar(value);
//    }),
//    AT_EXPAND(AT_ALL_TYPES),
//    AT_EXPAND(AT_COMPLEX_TYPES),
//    kComplexHalf,
//    kHalf,
//  )
```

lambda 移到第三个参数并用 `AT_WRAP` 包住（因为它不再是最后一个参数，6.5 节说的逗号问题必须处理——`#define AT_WRAP(...) __VA_ARGS__` 把带逗号的内容先当成一个参数吃进去再原样吐出），后面跟任意个 dtype，`AT_EXPAND(AT_ALL_TYPES)` 展开成一组预定义集合。实现：

```cpp
#define AT_DISPATCH_V2(TYPE, NAME, BODY, ...) \
  THO_DISPATCH_V2_TMPL(                       \
      AT_DISPATCH_SWITCH,                     \
      AT_DISPATCH_CASE,                       \
      TYPE,                                   \
      NAME,                                   \
      AT_WRAP(BODY),                          \
      __VA_ARGS__)
```

`torch/headeronly/core/Dispatch_v2.h` 里 `THO_DISPATCH_V2_TMPL` 用 `AT_NUM_ARGS(__VA_ARGS__)` 数出 dtype 个数 N（经典的"参数计数"宏技巧：把 `__VA_ARGS__` 后面接一串递减数字，取第 61 个），用 `AT_CONCAT` 拼出 `THO_AP##N`，再由同文件里机器生成的 `THO_AP1`…`THO_AP60` 对每个 dtype 调用一次传入的 `CASE` 宏：

```cpp
#define THO_AP_VAR_TMPL(C, N, T, ...) \
  AT_EXPAND(                          \
      AT_CONCAT(THO_AP, AT_NUM_ARGS(__VA_ARGS__))(C, AT_WRAP(N), __VA_ARGS__))
// ...
#define THO_AP1(C, N, _1) C(_1, N)
#define THO_AP2(C, N, _1, _2) C(_1, N) C(_2, N)
// ... 到 THO_AP60
```

`C` 就是 `AT_DISPATCH_CASE`，`N` 是被 `AT_WRAP` 保护的 lambda。`aten/src/ATen/Dispatch_v2.h` 里还留着一组旧的 `AT_AP1`…`AT_AP60`（注释标明 "Unused helper macros, kept for BC"）和生成它们的 Python 脚本，以及一条 `static_assert(static_cast<int>(c10::ScalarType::NumOptions) < 60);` 防止 dtype 总数超过宏能处理的上限。最终落到的仍然是同一个 `AT_DISPATCH_SWITCH` 和同一个 `AT_DISPATCH_CASE`——`scalar_t` 的来源、lambda 的份数，和 V1 完全一样。`aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` 里的 `_AT_DISPATCH_ALL_TYPES_AND_BOOL` 等本地宏就是用 `AT_DISPATCH_V2` 组合出来的。

读源码时两代宏都会遇到，识别方法：看到 `_AND2`/`_AND3` 后缀是 V1，看到 `AT_WRAP`/`AT_EXPAND` 是 V2。

## 七、轻量视图与容器：`ArrayRef`、`std::optional`、`SmallVector`

这三个类型在 ATen 的函数签名里出现频率极高，它们都是模板，都是为了"在不分配、不拷贝的前提下传递一组值或一个可选值"。理解它们要把本篇的模板知识和第二篇的所有权规则放在一起看。

### 7.1 `c10::ArrayRef<T>` 与 `IntArrayRef`：不拥有的只读视图

`c10/util/ArrayRef.h` 类定义前的注释把设计说得很清楚：

```cpp
/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
```

数据成员只有两个（在基类 `HeaderOnlyArrayRef` 里）：

```cpp
 protected:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_type Length;
```

16 字节，平凡可拷贝，按值传递就是两个寄存器。它像 Java 的 `List<Long>` 接口那样让调用方"不关心底层容器是什么"，但实现方式相反：Java 靠接口和虚调用，`ArrayRef` 靠一组隐式构造函数在编译期把各种容器统一成"指针 + 长度"：

```cpp
  /// Construct a HeaderOnlyArrayRef from a single element.
  constexpr HeaderOnlyArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct a HeaderOnlyArrayRef from a pointer and length.
  constexpr HeaderOnlyArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {}
  // ... 通用容器（5.4 节的 enable_if 版本）、std::vector、std::array<T, N>、C 数组 T[N] ...

  /// Construct a HeaderOnlyArrayRef from a std::initializer_list.
  /* implicit */ constexpr HeaderOnlyArrayRef(
      const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()) {}
```

`/* implicit */` 注释是 PyTorch 的代码约定：标明这个构造函数**故意**不加 `explicit`，允许隐式转换。正因为有 `initializer_list` 的隐式构造，`at::empty({2, 3}, options)` 才能直接把 `{2, 3}` 传给 `IntArrayRef size` 参数；有 `std::vector` 的隐式构造，持有 `std::vector<int64_t>` 的代码也能直接传。

`using IntArrayRef = ArrayRef<int64_t>;` 在文件末尾。`c10/core/TensorImpl.h` 里 `sizes()` 返回的就是它：

```cpp
  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  IntArrayRef sizes() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sizes_custom();
    }
    return sizes_and_strides_.sizes_arrayref();
  }
```

注释里的两个条件就是使用规则：`IntArrayRef s = x.sizes();` 在 `x` 活着且没被 resize 期间有效。第二篇讲过"返回成员的引用有风险"，`ArrayRef` 就是那个风险的具体形态——它是一个不延长生命周期的借用。

派生类 `ArrayRef` 相对基类只多两样东西：一个从 `SmallVector` 构造的模板构造函数（4.2 节），以及把 `front()`/`back()`/`at()`/`slice()` 的检查从 `STD_TORCH_CHECK` 换成 `TORCH_CHECK`（更好的错误信息，但依赖 libtorch）。还有一对刻意删除的赋值运算符：

```cpp
  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  // NOLINTNEXTLINE(modernize-use-constraints)
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;
```

`ref = int64_t{5};` 或 `ref = {int64_t{1}, int64_t{2}};` 会让 `ref` 指向一个语句结束就销毁的临时对象，这两个 `= delete` 的模板把这种写法变成编译错误（`enable_if_t<is_same_v<U, T>>` 保证只拦截元素类型恰好是 `T` 的临时值，`ref = {}` 仍走默认移动赋值）。但它只能拦这一种形态：`IntArrayRef ref = {1, 2, 3};` 作为一条独立语句同样是悬垂的（`initializer_list` 的底层数组在语句结束时销毁），编译器不会报错。规则很简单：**`ArrayRef` 只做参数类型和返回值类型，不做成员，不做跨语句的局部变量**；需要拥有一份时调用 `.vec()` 拷成 `std::vector`。

### 7.2 `std::optional`：`c10::optional` 迁移的现状

`std::optional<T>` 表示"可能没有值的 `T`"，在 ATen 签名里到处都是。`aten/src/ATen/native/TensorFactories.cpp`：

```cpp
Tensor empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
```

这对应 Schema 里的 `ScalarType? dtype=None`。用法：`opt.has_value()`、`*opt` / `opt.value()`、`opt.value_or(default)`、`opt->member`。它是**值类型**：`optional<ScalarType>` 就是一个 `ScalarType` 加一个 `bool`，放在栈上或参数寄存器里，没有堆分配。Java 的 `Optional<T>` 是一个堆对象，包着一个引用，两者语义相近、代价不同；另外 Java 用 `null` 表示缺失的地方远多于 `Optional`，而 C++ 里 `optional<Tensor>` 和"undefined `Tensor`"（第二篇的 `UndefinedTensorImpl`）是两种不同的"没有"，读源码时要分清。

版本演进（PyTorch 2.x 中的变化）：PyTorch 早期用自己实现的 `c10::optional`（`c10/util/Optional.h` 曾是一个完整的自有实现，因为要支持没有 `<optional>` 的旧编译器；2.1 的 `c10/util/Optional.h` 还是这个自有实现），从 2.2 起 `c10::optional` 变成 `std::optional` 的别名，之后几个版本里 PyTorch 自己的代码逐步改写为直接使用 `std::optional`。v2.13.0 源码树里 `c10/util/Optional.h` 只剩下这些：

```cpp
namespace c10 {

#if !defined(FBCODE_CAFFE2) && !defined(C10_NODEPRECATED)
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::bad_optional_access;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::make_optional;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt_t;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::optional;
#endif
// ... 两个标了 [[deprecated]] 的 value_or_else ...
```

`c10::optional` 现在就是 `std::optional` 的一个 `using` 别名，且只在没定义 `C10_NODEPRECATED` 时存在。PyTorch 自己的代码已全部改写为 `std::optional`（在 `aten/`、`c10/`、`torch/csrc/` 下 grep 不到 `c10::optional`）。第三方扩展仍可能写着旧名字——vLLM 的 `csrc/cpu/mamba_cpu.cpp` 就还有 `const c10::optional<at::Tensor>& bias`，能编译只是因为那个 `using`。写新代码一律用 `std::optional`；读到 `c10::optional` 知道它是同一个东西即可。

### 7.3 `c10::SmallVector<T, N>`：小容量不分配

3.1 节和 4.2 节已经看过 `SmallVector` 的两个关键模板技巧（非类型参数 `N` 决定内联缓冲区大小、偏特化按 `T` 是否平凡可拷贝选择 `memcpy` 路径）。这里补齐它的形状和用法。`c10/util/SmallVector.h` 的继承链：

```text
SmallVectorBase<Size_T>              BeginX 指针、Size、Capacity；grow 的非模板部分放在 .cpp 里减少代码膨胀
  └─ SmallVectorTemplateCommon<T>    迭代器、operator[]、isSmall()（BeginX 是否指向内联缓冲区）
       └─ SmallVectorTemplateBase<T, bool>   按 T 是否平凡可拷贝偏特化：拷贝/析构策略
            └─ SmallVectorImpl<T>            push_back/insert/erase 等完整接口；不含 N，可作为"任意 N 的 SmallVector"的公共引用类型
                 └─ SmallVector<T, N>  +  SmallVectorStorage<T, N>   内联缓冲区 alignas(T) char InlineElts[N * sizeof(T)]
```

`SmallVectorImpl<T>` 这一层的设计值得注意：函数参数写 `SmallVectorImpl<T>&` 就能同时接受 `SmallVector<T, 4>` 和 `SmallVector<T, 8>`，避免为每个 `N` 实例化一份调用方代码——这是"用继承擦掉一个模板参数"的常用手法，`ArrayRef` 的那个 `SmallVectorTemplateCommon<T, U>` 构造函数也是同样的考虑。

用法上，`c10::DimVector`（`SmallVector<int64_t, 5>`）是 shape 计算的默认容器，`aten/src/ATen/ExpandUtils.h` 里 `infer_size_dimvector(IntArrayRef a, IntArrayRef b)` 返回它，`aten/src/ATen/native/TensorShape.cpp` 里 `DimVector sizes{0};` 这样的局部变量随处可见。选择规则：结果要**拥有**、长度通常小、在栈上用完就丢——用 `DimVector`；只是**借用**——用 `IntArrayRef`；要长期持有或可能很大——用 `std::vector`。`TensorImpl` 自己存 sizes/strides 用的是 `c10/core/impl/SizesAndStrides.h` 里一个更紧凑的手写结构（5 个元素内联，同样的思路），对外统一以 `IntArrayRef` 暴露。

三个类型放在一起看 ATen 的一条典型签名：

```cpp
Tensor empty_cpu(IntArrayRef size, std::optional<ScalarType> dtype_opt, ...)
```

`IntArrayRef` 借用调用方的 shape，`std::optional` 按值传可选标量，函数内部算出的新 shape 放在 `DimVector` 里，最后写进 `TensorImpl` 的 `SizesAndStrides`。整条链没有一次堆分配（shape 不超过 5 维时），也没有一次不必要的拷贝。


## 八、lambda：捕获、泛型 lambda、作为模板参数与生命周期

### 8.1 lambda 是一个匿名类的对象

`[&](int64_t begin, int64_t end) { ... }` 在编译器眼里等价于：

```cpp
struct __lambda_at_line_74 {
  // 捕获的变量成为成员：[&] 时是引用，[=] 时是拷贝
  Tensor& x_c; Tensor& out; double& alpha; double& beta;
  void operator()(int64_t begin, int64_t end) const { /* 函数体 */ }
};
```

编译器生成一个只有 `operator()` 的类（闭包类型，closure type），lambda 表达式的值就是这个类的一个对象。三个推论：

1. **每个 lambda 表达式的类型都是唯一的**，即使两个 lambda 一模一样。这就是 6.5 节"两个长得一样的 lambda 各编译一次"的语言基础，也是 `parallel_for(const F& f)` 每个调用点实例化一份的原因。
2. lambda 的类型没有名字，只能用 `auto` 接住，或者作为模板参数 `F` 推导出来，或者装进 `std::function`（第四篇讨论它的代价）。
3. 调用 lambda 就是调用 `operator()`，编译器完全知道函数体，可以内联。没有虚调用，没有堆分配（除非装进 `std::function`）。

Java 的 lambda 会被编译成 `invokedynamic` + 一个实现函数式接口的类，调用是接口调用；在 JIT 内联之前，它和 C++ lambda 的性能模型不同。

### 8.2 捕获列表：`[&]`、`[=]`、具名捕获

方括号里的内容决定 lambda 体里能用哪些外层变量，以及**怎么**用：

| 写法 | 含义 | 生成的成员 |
|---|---|---|
| `[]` | 不捕获任何东西 | 无 |
| `[&]` | 用到的外层变量全部按引用捕获 | `T&` 成员 |
| `[=]` | 用到的外层变量全部按值捕获（拷贝） | `T` 成员 |
| `[&x, y]` | `x` 按引用，`y` 按值 | 混合 |
| `[this]` | 捕获 `this` 指针，可以访问成员 | `Cls*` |
| `[fn = std::move(f)]` | 初始化捕获（C++14）：把表达式的结果存成成员 `fn` | 任意类型 |

`[&]` 和 `[=]` 的区别只在两点：**能否修改外层变量**（`[=]` 拷了一份，改的是自己的副本，且默认 `operator()` 是 `const`，需要 `mutable` 才能改），以及**生命周期**（`[&]` 里的引用不延长被引用对象的寿命，8.5 节）。ATen 里 `[&]` 占绝大多数，因为 kernel 代码几乎总是"在当前函数里同步地把活干完"。`[=]` 用于要把 lambda 存起来或传到别处的场合，例如 `aten/src/ATen/native/Linear.cpp` 里 einsum 的辅助函数：

```cpp
  // Convert label in [A-Za-z] to subscript in [0, TOTAL_LABELS)
  auto label_to_subscript = [=](unsigned char label) -> uint8_t {
    return std::isupper(label) ? label - 'A' : label - 'a' + NUM_OF_LETTERS;
  };
```

（`NUM_OF_LETTERS` 是 `constexpr`，实际不需要捕获；`[=]` 在这里是"这个 lambda 不依赖任何外部引用"的声明。）2.4 节 `register_hook` 里的 `[fn=std::forward<T>(hook)]` 是初始化捕获：把用户的 hook **移动**进闭包成为成员，因为这个闭包要被存进 `std::function` 长期持有，按引用捕获会悬垂。

Java 对照：Java lambda 只能捕获 effectively final 的局部变量，而且是按值捕获（对对象来说是拷贝引用）。Java 没有 `[&]`——你不能在 lambda 里给外层局部变量赋值。这条限制的原因正是生命周期：Java lambda 可能在外层方法返回后才执行，按引用捕获栈变量必然悬垂，所以语言直接禁止。C++ 允许 `[&]`，把判断"lambda 会不会活得比外层变量久"的责任交给程序员。

### 8.3 泛型 lambda

C++14 起 lambda 的参数可以写 `auto`，此时 `operator()` 是一个成员函数模板，每种实参类型实例化一次：

```cpp
auto exp_vec = [](const auto& v) { /* ... */ };   // aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
```

`c10/util/Unroll.h` 用泛型 lambda 配合非类型模板参数做编译期循环展开：

```cpp
template <int n>
struct ForcedUnroll {
  template <typename Func, typename... Args>
  C10_ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    ForcedUnroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct ForcedUnroll<1> {
  template <typename Func, typename... Args>
  C10_ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};
```

`ForcedUnroll<4>{}(f)` 递归实例化 `ForcedUnroll<3>`、`<2>`、`<1>`（全特化终止递归），依次调用 `f(integral_constant<int, 0>{})` … `f(integral_constant<int, 3>{})`。传入的 `f` 是泛型 lambda（`aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp`）：

```cpp
  c10::ForcedUnroll<IntegerLog2(kF16RegistersPerIteration)>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (const auto i : c10::irange(offset)) {
      x[i] = x[i] + x[offset + i];
    }
  });
```

`idx` 的类型每次都不同（`integral_constant<int, 0>`、`<int, 1>`……），所以 lambda 体被实例化 N 次，每次 `idx` 都是编译期常量——这是"用类型系统把循环变量变成常量"的技巧，效果等同于 `#pragma unroll` 但可移植。这个例子同时展示了具名捕获 `[&offset, &x]`：只按引用捕获两个变量，其余不可见。

### 8.4 lambda 作为模板参数 vs `std::function`

把 lambda 传给函数有两种方式，性能模型完全不同：

```cpp
template <class F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F& f);   // 模板参数

TORCH_API void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size,
                               const std::function<void(int64_t, int64_t)>& f);          // 类型擦除
```

第一种（`aten/src/ATen/Parallel.h`）：`F` 推导为闭包类型，`f(begin, end)` 是对已知函数体的直接调用，可以内联；每个调用点一份 `parallel_for` 实例。第二种（`aten/src/ATen/ParallelNative.h`）：`std::function` 把任意可调用对象装进统一类型，调用要经过一次间接跳转，构造时可能堆分配；但它是**非模板**，可以放在 `.cpp` 里、导出成 `TORCH_API` 符号、跨动态库边界传递。ATen 的分层正好说明取舍：面向 kernel 作者的 `parallel_for` 是模板（热路径、要内联），面向线程池实现的 `invoke_parallel` 用 `std::function`（要跨 `.so`、一次调用对应一大块工作，间接跳转的开销可忽略）。`aten/src/ATen/Parallel-inl.h` 里 `parallel_for` 把模板 `f` 包进一个 `[&]` lambda 再交给 `invoke_parallel`，转换就发生在那里。第四篇会把 `std::function` 的实现和 `c10::KernelFunction` 一起展开。

2.1 节的 `cpu_kernel(iter, func_t&& op)` 是模板参数方式的极致：不仅内联 `op`，还用 `function_traits<func_t>` 在编译期读出 `op` 的参数个数和类型，据此生成正确步长的循环——这在 `std::function<void(...)>` 上做不到，因为签名信息在类型擦除时已经固定成用户写的那个。

### 8.5 引用捕获的生命周期陷阱

`[&]` 生成的是引用成员。如果闭包对象活得比被引用的变量久，调用时就是悬垂引用——第二篇 2.5 节的问题在 lambda 上的形态。典型错误：

```cpp
std::function<void()> make_task(const Tensor& x) {
  double scale = compute_scale(x);
  return [&] { use(x, scale); };   // 返回后 scale 已销毁，x 也可能已销毁
}
```

判断规则只有一条：**lambda 会不会在当前作用域结束后还被调用？** 会——只能按值捕获或初始化捕获（`[x, scale]`、`[x = std::move(x)]`）；不会——`[&]` 安全且更快。一个很好的信号是参数类型：接收 `std::function<void()>` 的接口（`aten/src/ATen/Parallel.h` 里的 `TORCH_API void launch(std::function<void()> func);` 是异步提交到线程池）通常意味着"我会把它存起来稍后调用"，此时 `[&]` 几乎一定是错的；接收 `const F&` 模板参数并立即调用的接口（`parallel_for`、`AT_DISPATCH`）则可以放心 `[&]`。

另一个隐蔽的陷阱是 `[=]` 捕获 `this`：`[=]` 会隐式捕获 `this` 指针（按值拷贝的是指针本身），对象析构后调用闭包同样悬垂。C++20 已废弃 `[=]` 隐式捕获 `this`，要写 `[=, this]` 或 `[=, *this]` 明示。

### 8.6 为什么 `AT_DISPATCH` 与 `at::parallel_for` 的 `[&]` 安全

回到开头的代码。两层 `[&]`：

外层 `AT_DISPATCH_FLOATING_TYPES(..., [&] { ... })`：6.3 节展开后是 `[&] { switch (...) { case ...: return [&] { 用户体 }(); } }()`——两个 lambda 都在**定义的同一条语句里被立即调用**，调用完成前外层函数的所有局部变量都还活着。这是 IIFE 模式的固有性质：闭包从不逃逸。

内层 `parallel_for(0, n, grain, [&](int64_t begin, int64_t end) { ... })`：这个 lambda 会在**其他线程**上执行，为什么还能 `[&]`？看 `aten/src/ATen/ParallelOpenMP.h`：

```cpp
template <class F>
inline void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

#pragma omp parallel
  {
    // ...
    if (begin_tid < end) {
      try {
        internal::ThreadIdGuard tid_guard(tid);
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}
```

`#pragma omp parallel` 块是**同步的**：所有工作线程执行完 `f` 之后，调用线程才会离开这个块、继续执行 `if (eptr)` 并返回。也就是说 `parallel_for` 返回时，`f` 的所有调用都已结束；`f` 引用的 `x_c`、`out`、`alpha`、`scalar_t*` 指针在整个过程中都活着。多线程只改变了"谁在跑 `f`"，没有改变"`f` 在什么时候跑完"——生命周期规则关心的是后者。原生线程池版本（`aten/src/ATen/ParallelNative.cpp`）同样在返回前等待所有任务完成。所以 `parallel_for` 的 `[&]` 安全，而 `at::launch` 的 `[&]` 不安全，两者的差别不在"是否多线程"，而在"是否同步"。

不过 `Parallel.h` 里那条 Warning 提醒了另一件事：`parallel_for` 不把调用线程的 TLS 复制到工作线程，所以 lambda 体里只能碰裸指针，不能调 tensor 算子。这与捕获无关，是第六篇的内容。

## 九、回到源码：重读 `scale_shift_cpu`

带着前八节的机制，重读总纲开篇那段扩展代码：

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

- `TORCH_CHECK(x.is_floating_point(), "expected floating point tensor")`：4.4 节的 `torchCheckMsgImpl` 重载——只有一个 `const char*` 参数，直接返回它，不走 `c10::str` 的变参模板。宏本身第五篇讲。
- `auto x_c = x.contiguous();`、`auto out = at::empty_like(x_c);`：2.3 节的 `auto`，类型是 `at::Tensor`。`at::empty_like` 的完整签名里有多个 `std::optional<...>` 参数（7.2 节），这里全部走默认。
- `AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "scale_shift_cpu", [&] { ... })`：6.2–6.5 节。展开成一个 IIFE，`switch (x_c.scalar_type())` 有 `Double` 和 `Float` 两个 `case`，每个 `case` 里 `using scalar_t = ScalarTypeToCPPTypeT<...>;`，然后调用一份 lambda。lambda 的文本被复制两份，各编译一次。传入 `Half` 或 `Long` 的 tensor 会走 `default` 抛 `NotImplementedError`——所以前面那条 `TORCH_CHECK(x.is_floating_point())` 其实挡不住 `Half`/`BFloat16`（它们也是浮点），要支持它们得换 `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)`。
- `x_c.data_ptr<scalar_t>()`：2.2 节，模板参数无法从实参推导，必须显式写；1.3 节，这个成员模板在 `TensorMethods.cpp` 里显式实例化，链接到 `libtorch_cpu.so` 里的那一份；6.6 节，内部用 `CppTypeToScalarType<scalar_t>()` 反查枚举做运行期校验。在 `case` 里 `scalar_t` 与 dtype 必然一致。
- `[&]`（两处）：8.6 节。外层是 IIFE，内层 `parallel_for` 同步返回，两个闭包都不逃逸，引用捕获安全。
- `at::parallel_for(0, x_c.numel(), 4096, lambda)`：2.1 节，`F` 推导为内层闭包类型，这一份 `parallel_for` 实例专属于这个调用点；循环体被内联进去。因为外层有两个 `case`，这里实际有两份 `parallel_for` 实例（`F` 分别是 `double` 版和 `float` 版闭包的类型）。
- `static_cast<scalar_t>(alpha)`：`alpha` 是 `double`，在 `float` 那份代码里要显式转成 `float`，否则 `alpha * in[i]` 会把 `in[i]` 提升成 `double` 再算，抹掉单精度的性能优势。这是 kernel 代码里最常见的 `static_cast` 用途。
- `return out;`：第二篇的按值返回。

整个函数编译后：一个 `scale_shift_cpu` 符号，里面一个 `switch`，两条路径各含一份内联了循环体的 `parallel_for` 实例。运行时执行一次 `switch`，之后热循环里没有任何 dtype 相关的分支。


## 十、mini-c10：`ScalarType` 映射、`MINI_DISPATCH_FLOATING_TYPES` 与第一个模板化 kernel

按系列约定，本篇实现 `minic10/core/ScalarType.h`（补上到 C++ 类型的映射）、`minic10/core/Dispatch.h`、`minic10/util/ArrayRef.h`，把 `TensorImpl`/`Tensor` 的 `sizes()` 改成返回 `IntArrayRef`，然后写第一个 CPU kernel `ops/add.cpp`、`ops/mul.cpp`（直接函数调用，第四篇才有 Dispatcher）。所有片段用 `clang++ -std=c++17 -Wall -Wextra` 在 macOS（arm64）上编译运行过。第二篇的 `intrusive_ptr.h`、`Allocator.h`、`StorageImpl.h`、`DispatchKey.h` 原样沿用。

### 10.1 `core/ScalarType.h`：一张表生成两张映射

```cpp
// minic10/core/ScalarType.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace minic10 {

// 一张表：(C++ 类型, 枚举名)。下面所有映射都由它生成，加 dtype 只改这里。
// 对照 torch/headeronly/core/ScalarType.h 的 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS。
#define MINI_FORALL_SCALAR_TYPES(_) \
  _(float, Float)                   \
  _(double, Double)                 \
  _(int64_t, Long)

enum class ScalarType : int8_t {
#define MINI_DEFINE_ENUM(_, name) name,
  MINI_FORALL_SCALAR_TYPES(MINI_DEFINE_ENUM)
#undef MINI_DEFINE_ENUM
  Undefined,
  NumOptions
};

// 运行期值 -> 编译期类型：只声明主模板，不定义。
// 对没有映射的枚举值（Undefined）使用它会直接编译失败，而不是得到一个错的类型。
template <ScalarType N>
struct ScalarTypeToCPPType;

#define MINI_SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type) \
  template <>                                                       \
  struct ScalarTypeToCPPType<ScalarType::scalar_type> {             \
    using type = cpp_type;                                          \
  };
MINI_FORALL_SCALAR_TYPES(MINI_SPECIALIZE_ScalarTypeToCPPType)
#undef MINI_SPECIALIZE_ScalarTypeToCPPType

template <ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

// 反向：编译期类型 -> 运行期值。继承 integral_constant，因此 CppTypeToScalarType<float>()
// 可以隐式转换成 ScalarType::Float，也可以用 ::value 取。
template <typename T>
struct CppTypeToScalarType;

#define MINI_SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)          \
  template <>                                                                \
  struct CppTypeToScalarType<cpp_type>                                       \
      : std::integral_constant<ScalarType, ScalarType::scalar_type> {};
MINI_FORALL_SCALAR_TYPES(MINI_SPECIALIZE_CppTypeToScalarType)
#undef MINI_SPECIALIZE_CppTypeToScalarType

// 运行期查表的两个函数也从同一张表生成
inline constexpr size_t itemsize(ScalarType t) {
  switch (t) {
#define MINI_CASE(cpp_type, name) \
  case ScalarType::name:          \
    return sizeof(cpp_type);
    MINI_FORALL_SCALAR_TYPES(MINI_CASE)
#undef MINI_CASE
    default:
      return 0;
  }
}

inline constexpr const char* toString(ScalarType t) {
  switch (t) {
#define MINI_CASE(_, name) \
  case ScalarType::name:   \
    return #name;
    MINI_FORALL_SCALAR_TYPES(MINI_CASE)
#undef MINI_CASE
    default:
      return "UNKNOWN_SCALAR";
  }
}

// 编译期自检：两张映射互为逆
static_assert(CppTypeToScalarType<ScalarTypeToCPPTypeT<ScalarType::Float>>::value == ScalarType::Float);
static_assert(std::is_same_v<ScalarTypeToCPPTypeT<CppTypeToScalarType<double>::value>, double>);
static_assert(itemsize(ScalarType::Long) == 8);

}  // namespace minic10
```

与真实源码结构一致：`X-macro` 表 → 枚举、`ScalarTypeToCPPType` 全特化（3.2、4.1 节）、`CppTypeToScalarType` 继承 `integral_constant`、`toString` 的 `switch`。第二篇临时版本的 `itemsize` 保留了签名，实现改为从表生成（加了 `constexpr`，所以能在 `static_assert` 里用）。宏的细节（`#name` 字符串化、`_` 作为宏参数名）第五篇展开，这里只需要知道 `MINI_FORALL_SCALAR_TYPES(F)` 会对表里每一行调用一次 `F(cpp_type, name)`。

### 10.2 `core/Dispatch.h`：`MINI_DISPATCH_FLOATING_TYPES`

```cpp
// minic10/core/Dispatch.h
#pragma once
#include <stdexcept>
#include <string>
#include "minic10/core/ScalarType.h"

// 对照 torch/headeronly/core/Dispatch.h 的 THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL：
// 一个 case 分支 = 把运行期枚举值 enum_type 映射成编译期类型别名 HINT，然后调用 lambda。
#define MINI_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)              \
  case enum_type: {                                                          \
    using HINT [[maybe_unused]] = ::minic10::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                                    \
  }

// scalar_t 这个名字就是在这里定下来的
#define MINI_DISPATCH_CASE(enum_type, ...) \
  MINI_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

// 对照 THO_DISPATCH_SWITCH_TMPL：整个 switch 包在一个立即调用的 lambda 里，
// 所以宏可以出现在表达式位置，case 里的 return 也只是从这个 lambda 返回。
#define MINI_DISPATCH_SWITCH(TYPE, NAME, ...)                                  \
  [&] {                                                                        \
    const auto& the_type = TYPE;                                               \
    constexpr const char* mini_dispatch_name = NAME;                           \
    ::minic10::ScalarType _st = the_type;                                      \
    switch (_st) {                                                             \
      __VA_ARGS__                                                              \
      default:                                                                 \
        throw std::runtime_error(std::string("\"") + mini_dispatch_name +      \
                                 "\" not implemented for '" +                  \
                                 ::minic10::toString(_st) + "'");              \
    }                                                                          \
  }()

#define MINI_DISPATCH_CASE_FLOATING_TYPES(...)                    \
  MINI_DISPATCH_CASE(::minic10::ScalarType::Double, __VA_ARGS__) \
  MINI_DISPATCH_CASE(::minic10::ScalarType::Float, __VA_ARGS__)

#define MINI_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  MINI_DISPATCH_SWITCH(TYPE, NAME, MINI_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define MINI_DISPATCH_CASE_ALL_TYPES(...)           \
  MINI_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)    \
  MINI_DISPATCH_CASE(::minic10::ScalarType::Long, __VA_ARGS__)

#define MINI_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  MINI_DISPATCH_SWITCH(TYPE, NAME, MINI_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))
```

三层结构与 6.2–6.4 节一一对应：`MINI_DISPATCH_FLOATING_TYPES` → `MINI_DISPATCH_SWITCH` + `MINI_DISPATCH_CASE_*` → `MINI_PRIVATE_CASE_TYPE_USING_HINT`。省略的只有 PRELUDE 钩子（选择性构建、profiling）和诊断宏；`TORCH_CHECK_NOT_IMPLEMENTED` 用 `std::runtime_error` 代替（第四篇讨论异常类型，第五篇实现 `MINI_CHECK`）。

### 10.3 `util/ArrayRef.h`

```cpp
// minic10/util/ArrayRef.h
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace minic10 {

// ArrayRef<T>：指针 + 长度的只读视图。不拥有数据，不能存起来，按值传递。
// 对照 c10/util/ArrayRef.h + torch/headeronly/util/HeaderOnlyArrayRef.h。
template <typename T>
class ArrayRef final {
  const T* data_ = nullptr;
  size_t length_ = 0;

 public:
  using value_type = T;
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;

  constexpr ArrayRef() = default;
  // 单个元素：&one 指向调用方的对象
  constexpr ArrayRef(const T& one) : data_(&one), length_(1) {}
  constexpr ArrayRef(const T* data, size_t length) : data_(data), length_(length) {}
  constexpr ArrayRef(const T* begin, const T* end) : data_(begin), length_(end - begin) {}

  // 从 std::vector 隐式构造：vector<bool> 是位域，没有 data()，用 static_assert 挡掉
  template <typename A>
  /* implicit */ ArrayRef(const std::vector<T, A>& vec) : data_(vec.data()), length_(vec.size()) {
    static_assert(!std::is_same_v<T, bool>, "ArrayRef<bool> cannot be constructed from std::vector<bool>");
  }
  // 非类型模板参数 N：std::array<T, N> 和 C 数组 T[N] 的长度是类型的一部分
  template <size_t N>
  /* implicit */ constexpr ArrayRef(const std::array<T, N>& arr) : data_(arr.data()), length_(N) {}
  template <size_t N>
  /* implicit */ constexpr ArrayRef(const T (&arr)[N]) : data_(arr), length_(N) {}
  // 让 f({2, 3}) 能直接调用 f(IntArrayRef)
  /* implicit */ constexpr ArrayRef(const std::initializer_list<T>& il)
      : data_(il.begin() == il.end() ? nullptr : il.begin()), length_(il.size()) {}

  constexpr iterator begin() const { return data_; }
  constexpr iterator end() const { return data_ + length_; }
  constexpr bool empty() const { return length_ == 0; }
  constexpr const T* data() const { return data_; }
  constexpr size_t size() const { return length_; }
  constexpr const T& operator[](size_t i) const { return data_[i]; }

  const T& front() const {
    if (empty()) throw std::out_of_range("ArrayRef: front() of empty list");
    return data_[0];
  }
  const T& back() const {
    if (empty()) throw std::out_of_range("ArrayRef: back() of empty list");
    return data_[length_ - 1];
  }
  const T& at(size_t i) const {
    if (i >= length_) throw std::out_of_range("ArrayRef: index out of range");
    return data_[i];
  }
  ArrayRef<T> slice(size_t n, size_t m) const {
    if (n + m > length_) throw std::out_of_range("ArrayRef: invalid slice");
    return ArrayRef<T>(data_ + n, m);
  }
  ArrayRef<T> slice(size_t n) const { return slice(n, length_ - n); }

  constexpr bool equals(ArrayRef rhs) const {
    if (length_ != rhs.length_) return false;
    for (size_t i = 0; i < length_; ++i)
      if (!(data_[i] == rhs.data_[i])) return false;
    return true;
  }
  // 需要拥有一份时，显式拷出来
  std::vector<T> vec() const { return std::vector<T>(begin(), end()); }

  // 禁止从临时对象赋值：ref = int64_t{5} 或 ref = {int64_t{1}, int64_t{2}} 会立刻悬垂。
  // 写成模板 + enable_if 是为了让 "ref = {}" 仍然选到默认的移动赋值（照抄 c10 的写法）。
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(U&&) = delete;
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(std::initializer_list<U>) = delete;
};

template <typename T>
inline bool operator==(ArrayRef<T> a, ArrayRef<T> b) { return a.equals(b); }
template <typename T>
inline bool operator!=(ArrayRef<T> a, ArrayRef<T> b) { return !a.equals(b); }

using IntArrayRef = ArrayRef<int64_t>;

}  // namespace minic10
```

去掉了 `HeaderOnlyArrayRef` 那层拆分、`SmallVector` 构造函数、推导指引和 `makeArrayRef` 系列，其余是 7.1 节读到的形状。本机验证了 `ref = int64_t{5};` 和 `ref = {int64_t{1}, int64_t{2}};` 报 "overload resolution selected deleted operator '='"，`ref = {};` 正常编译。

### 10.4 `TensorImpl`/`Tensor`：`sizes()` 改为返回 `IntArrayRef`

只列改动的部分。`minic10/core/TensorImpl.h`：

```cpp
#include "minic10/util/ArrayRef.h"
// ...
struct TensorImpl : intrusive_ptr_target {
  // 构造函数参数从 std::vector<int64_t> 改成 IntArrayRef，内部用 .vec() 拷一份拥有
  TensorImpl(intrusive_ptr<StorageImpl> storage, IntArrayRef sizes,
             ScalarType dtype, DispatchKey key)
      : storage_(std::move(storage)), sizes_(sizes.vec()), dtype_(dtype), key_(key) {
    // ... strides_ 计算不变 ...
  }
  // 返回视图而不是 const std::vector&：调用方看不到、也不依赖内部容器是什么
  IntArrayRef sizes() const noexcept { return sizes_; }
  IntArrayRef strides() const noexcept { return strides_; }
  // ...
 private:
  intrusive_ptr<StorageImpl> storage_;
  std::vector<int64_t> sizes_;     // 内部仍用 vector 拥有；换成 SmallVector<int64_t, 5> 时对外接口不变
  std::vector<int64_t> strides_;
  // ...
};
```

`minic10/core/Tensor.h`：

```cpp
class Tensor {
  intrusive_ptr<TensorImpl> impl_;
 public:
  // ...
  IntArrayRef sizes() const { return impl_->sizes(); }
  IntArrayRef strides() const { return impl_->strides(); }
  int64_t dim() const { return static_cast<int64_t>(impl_->sizes().size()); }
  ScalarType dtype() const { return impl_->dtype(); }
  int64_t numel() const { return impl_->numel(); }

  // 成员函数模板：每个 T 一份实例。调用时 T 无法从实参推导，必须显式写 data_ptr<float>()
  template <typename T>
  T* data_ptr() const {
    if (CppTypeToScalarType<T>::value != impl_->dtype()) {
      throw std::runtime_error(std::string("expected scalar type ") +
                               toString(CppTypeToScalarType<T>::value) + " but found " +
                               toString(impl_->dtype()));
    }
    return static_cast<T*>(impl_->data());
  }
  // ...
};

// 参数从 std::vector<int64_t> 改成 IntArrayRef：调用方传 {2, 3}、std::vector、std::array 都行
inline Tensor empty(IntArrayRef sizes, ScalarType dtype) {
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  auto storage = make_intrusive<StorageImpl>(numel * itemsize(dtype), GetCPUAllocator());
  return Tensor(make_intrusive<TensorImpl>(std::move(storage), sizes, dtype, DispatchKey::CPU));
}
```

`data_ptr<T>()` 现在有了 6.6 节的运行期校验：编译期的 `T` 经 `CppTypeToScalarType` 反查成枚举，与 tensor 的 dtype 比较。第二篇的 `main.cpp` 不需要任何修改仍能编译——`for (auto s : t.sizes())`、`t.sizes()[i]`、`empty({2, 3}, ...)` 对 `IntArrayRef` 和 `const std::vector&` 的写法一样，这正是视图类型的价值。

### 10.5 `ops/add.cpp`、`ops/mul.cpp`：第一个模板化 kernel

先给一个声明头（系列布局里 `ops/` 只列了两个 `.cpp`，声明放在 `ops/ops.h`）：

```cpp
// minic10/ops/ops.h
#pragma once
#include "minic10/core/Tensor.h"

namespace minic10 {
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
}  // namespace minic10
```

`add` 用具名函数模板做 kernel，便于在符号表里看到实例：

```cpp
// minic10/ops/add.cpp
#include <stdexcept>
#include <string>
#include "minic10/core/Dispatch.h"
#include "minic10/core/Tensor.h"
#include "minic10/ops/ops.h"

namespace minic10 {
namespace {

// 函数模板：scalar_t 是编译期参数。它本身不是代码，只是生成代码的配方。
template <typename scalar_t>
void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void check_binary_inputs(const Tensor& a, const Tensor& b, const char* name) {
  if (!a.defined() || !b.defined()) {
    throw std::runtime_error(std::string(name) + ": undefined tensor");
  }
  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(std::string(name) + ": dtype mismatch: " +
                             toString(a.dtype()) + " vs " + toString(b.dtype()));
  }
  if (a.sizes() != b.sizes()) {   // IntArrayRef 的 operator==：逐元素比较
    throw std::runtime_error(std::string(name) + ": shape mismatch");
  }
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  check_binary_inputs(a, b, "add");
  Tensor out = empty(a.sizes(), a.dtype());
  // 运行期的 a.dtype() 在这里变成编译期的 scalar_t；lambda 体被实例化两次（double、float）
  MINI_DISPATCH_FLOATING_TYPES(a.dtype(), "add_cpu", [&] {
    add_kernel<scalar_t>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                         out.data_ptr<scalar_t>(), a.numel());
  });
  return out;
}

}  // namespace minic10
```

`mul` 把 kernel 直接写在 lambda 里，并多分发一个 `Long`：

```cpp
// minic10/ops/mul.cpp
#include <stdexcept>
#include <string>
#include "minic10/core/Dispatch.h"
#include "minic10/core/Tensor.h"
#include "minic10/ops/ops.h"

namespace minic10 {

Tensor mul(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype() || a.sizes() != b.sizes()) {
    throw std::runtime_error("mul: dtype or shape mismatch");
  }
  Tensor out = empty(a.sizes(), a.dtype());
  // 这次 kernel 直接写在 lambda 里，并且多分发一个 Long：lambda 体被实例化三次
  MINI_DISPATCH_ALL_TYPES(a.dtype(), "mul_cpu", [&] {
    const scalar_t* pa = a.data_ptr<scalar_t>();
    const scalar_t* pb = b.data_ptr<scalar_t>();
    scalar_t* po = out.data_ptr<scalar_t>();
    for (int64_t i = 0; i < a.numel(); ++i) {
      po[i] = pa[i] * pb[i];
    }
  });
  return out;
}

}  // namespace minic10
```

### 10.6 跑起来

```cpp
// main.cpp
#include <cstdio>
#include <exception>
#include "minic10/core/Tensor.h"
#include "minic10/ops/ops.h"

using namespace minic10;

template <typename T>
Tensor arange_like(IntArrayRef sizes) {
  Tensor t = empty(sizes, CppTypeToScalarType<T>::value);   // 类型 -> 枚举
  T* p = t.data_ptr<T>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = static_cast<T>(i);
  return t;
}

template <typename T>
void print(const char* tag, const Tensor& t) {
  std::printf("%s [%s, dim=%lld]:", tag, toString(t.dtype()), (long long)t.dim());
  const T* p = t.data_ptr<T>();
  for (int64_t i = 0; i < t.numel(); ++i) std::printf(" %g", static_cast<double>(p[i]));
  std::printf("\n");
}

int main() {
  Tensor f = arange_like<float>({2, 3});
  Tensor d = arange_like<double>({2, 3});
  Tensor l = arange_like<int64_t>({6});

  print<float>("add(f, f)", add(f, f));
  print<double>("add(d, d)", add(d, d));
  print<int64_t>("mul(l, l)", mul(l, l));

  try {
    add(l, l);   // Long 没有进 FLOATING_TYPES 的 case 列表 -> default 分支
  } catch (const std::exception& e) {
    std::printf("add(l, l) threw: %s\n", e.what());
  }
  try {
    f.data_ptr<double>();   // 编译期类型 double 与运行期 dtype Float 不一致
  } catch (const std::exception& e) {
    std::printf("f.data_ptr<double>() threw: %s\n", e.what());
  }
  return 0;
}
```

```bash
clang++ -std=c++17 -Wall -Wextra -I. main.cpp minic10/ops/add.cpp minic10/ops/mul.cpp -o demo && ./demo
```

输出：

```text
add(f, f) [Float, dim=2]: 0 2 4 6 8 10
add(d, d) [Double, dim=2]: 0 2 4 6 8 10
mul(l, l) [Long, dim=1]: 0 1 4 9 16 25
add(l, l) threw: "add_cpu" not implemented for 'Long'
f.data_ptr<double>() threw: expected scalar type Double but found Float
```

最后两行分别是 6.3 节 `default` 分支和 6.6 节反向校验的 mini 版。第一条报错的文本格式刻意与 PyTorch 一致——你在 Python 里见到的 `"add_cpu" not implemented for 'Long'` 就是这样来的。

### 10.7 用 `nm` 与 `clang++ -S` 观察：确实生成了多份 kernel

把 `add.cpp` 单独编成目标文件（`-O0`，避免内联把符号吃掉），看符号表：

```bash
clang++ -std=c++17 -O0 -I. -c minic10/ops/add.cpp -o add.o
nm -C add.o | grep -E "minic10::(\(anonymous namespace\)::add_kernel|add\()"
```

```text
0000000000004444 t void minic10::(anonymous namespace)::add_kernel<double>(double const*, double const*, double*, long long)
00000000000049f8 t void minic10::(anonymous namespace)::add_kernel<float>(float const*, float const*, float*, long long)
0000000000000000 T minic10::add(minic10::Tensor const&, minic10::Tensor const&)
0000000000000740 t minic10::add(minic10::Tensor const&, minic10::Tensor const&)::$_0::operator()() const
00000000000043d4 t minic10::add(minic10::Tensor const&, minic10::Tensor const&)::$_0::operator()() const::'lambda0'()::operator()() const
0000000000004364 t minic10::add(minic10::Tensor const&, minic10::Tensor const&)::$_0::operator()() const::'lambda'()::operator()() const
```

逐行对照第六节的结论：

- `add_kernel<double>` 和 `add_kernel<float>`：源码里一个函数模板，目标文件里两个函数，各有各的地址。这就是 1.1 节的"一个模板 + N 组参数 = N 份代码"。不加 `-C` 看到的是 mangled 名 `__ZN7minic1012_GLOBAL__N_110add_kernelIdEEvPKT_S4_PS2_x` 与 `...IfE...`——`Id` 是 `<double>`，`If` 是 `<float>`，模板参数编进了符号名（第七篇 ABI 会再谈 name mangling）。
- `add(...)::$_0::operator()()`：`$_0` 是 `MINI_DISPATCH_SWITCH` 那个 IIFE 的闭包类型（6.3 节），只有一份。
- `$_0::operator()() const::'lambda'()` 和 `'lambda0'()`：**两个**闭包类型，都嵌套在 `$_0` 的 `operator()` 里——就是 `__VA_ARGS__` 被贴进 `Double` 和 `Float` 两个 `case` 后产生的两个 lambda（6.5 节）。源码里写了一次，符号表里有两个。
- `-O2` 下再看：`nm -C add_O2.o` 里只剩 `minic10::add` 一个符号，五个 `t`（局部）符号全部被内联进去了。代码份数没变，只是不再有独立符号。

再看 `mul.cpp`——kernel 直接写在 lambda 里，用 `-S` 输出汇编（arm64，`-O0` 便于对照）：

```bash
clang++ -std=c++17 -O0 -I. -S minic10/ops/mul.cpp -o mul.s
grep -n "fmul\|\tmul\tx8, x8, x9" mul.s
```

```text
6897:	fmul	d0, d0, d1      # double 乘法：'lambda'  的循环体
6956:	fmul	s0, s0, s1      # float  乘法：'lambda0' 的循环体
7015:	mul	x8, x8, x9      # int64  乘法：'lambda1' 的循环体
```

三条不同的乘法指令，分别在三个 `operator()` 里：`fmul d`（双精度浮点）、`fmul s`（单精度浮点）、`mul x`（64 位整数）。x86-64 上对应的是 `mulsd`/`mulss`/`imul`。这就是 6.1 节说的"`float*` 和 `double*` 的解引用是不同的机器指令"——运行期 `switch` 选的是三段早已存在的、各自只认一种类型的代码，循环体内没有任何类型判断。

第四篇会在这两个 kernel 之上加 `DispatchKey` 和类型擦除的 `KernelFunction`，让 `minic10::add(a, b)` 经过一个 Dispatcher 而不是直接调用；第五篇让 `add.cpp`、`mul.cpp` 自己把 kernel 注册进去。


## 十一、工程实践建议与常见错误

**模板基本功**

1. 模板定义放头文件；如果确实要放 `.cpp`，必须显式实例化所有会用到的参数组合（1.3 节 `TensorMethods.cpp` 的做法），否则链接期"未定义的引用"。
2. 依赖模板参数的嵌套类型前写 `typename`，依赖模板参数的成员模板调用前写 `template`。编译器报 "missing 'typename' prior to dependent type name" 时按提示加即可。
3. 读长错误信息：找第一条 `error:`，顺着 `note: in instantiation of ... requested here` 找到自己代码里的那一帧。用 `static_assert` 给自己的模板加前置检查，把错误提前到实例化入口。
4. 无法从实参推导的模板参数（只出现在返回类型里）必须显式写，`data_ptr<scalar_t>()` 是典型。
5. 函数模板不能偏特化，用重载；类模板可以偏特化，traits 类用主模板给默认值、特化给例外。
6. 变参模板的 `Args&&...` + `std::forward<Args>(args)...` 是固定搭配，照抄即可；不要对同一个参数包 `forward` 两次。

**编译期分派**

7. 每个 `AT_DISPATCH` `case` 都是一份完整的 lambda 实例化。选择 dtype 集合时只包含真正需要的：CPU kernel 一般不需要 `Half`，推理 kernel 一般不需要 `Double`（vLLM 的做法）。多一个 `case` 就多一份代码和编译时间。
8. `AT_DISPATCH` 的 `NAME` 必须是字符串字面量（`constexpr const char*`），且最好全局唯一——它既是错误信息也是选择性构建的 key。
9. lambda 体里有不在括号内的逗号（`std::pair<A, B>`、多参数模板）时，旧宏因为 lambda 是最后一个参数所以安全；`AT_DISPATCH_V2` 必须用 `AT_WRAP` 包住。
10. 在 `case` 外面用 `data_ptr<T>()` 要确认 dtype，否则运行期抛 "expected scalar type X but found Y"；在 `case` 里面用 `scalar_t` 永远安全。
11. 把运行期参数变成编译期常量的 if 链（3.3 节 vLLM 的 `BLOCK_Y_SIZE`）要有兜底分支，且分支数与实例化数成正比——不要为每个可能的值都开一个分支。
12. 想按类型走不同代码路径，优先 `if constexpr`（一份函数体、局部差异），其次重载/特化（差异大、结构不同），最后才是 SFINAE（需要让某个重载整体消失时）。项目允许 C++20 时用 concepts 代替 `enable_if`。

**视图类型**

13. `ArrayRef`/`IntArrayRef`/`std::string_view`/`c10::string_view` 只做参数和返回值，不做成员、不做跨语句局部变量。`IntArrayRef ref = {1, 2, 3};` 作为独立语句是悬垂的。要拥有就 `.vec()` 或用 `DimVector`。
14. `sizes()` 返回的 `IntArrayRef` 在 tensor 被 resize 后失效。
15. 可选参数用 `std::optional<T>`（按值传小类型，`const std::optional<Tensor>&` 传 tensor），不要用 `c10::optional` 这个已废弃的别名，也不要用魔法值（`-1`）或 undefined tensor 代替 "没有"。
16. shape 计算的临时结果用 `DimVector`，不用 `std::vector<int64_t>`——前者 5 维以内不分配。

**lambda**

17. 同步且不逃逸的 lambda（`AT_DISPATCH`、`parallel_for`、`cpu_kernel`、立即调用的辅助函数）用 `[&]`；会被存起来或异步执行的（`at::launch`、`std::function` 成员、回调注册、`std::thread`）按值或初始化捕获，`[fn = std::move(f)]` 转移所有权。
18. 优先用具名捕获 `[&x, y]` 而不是 `[&]`/`[=]`，特别是在长 lambda 里——读者一眼看出依赖了什么。ATen kernel 里 `[&]` 泛滥是历史遗留，新代码可以做得更好。
19. 接受可调用对象的接口：热路径、要内联、在头文件里 → 模板参数 `const F&`/`F&&`；要跨 `.so`、要存起来、签名固定 → `std::function`。不要在热循环里每次迭代构造 `std::function`。
20. `parallel_for` 的 lambda 体只碰裸指针和标量，不调 tensor 算子（TLS 不传播，第六篇）。

**Java 直觉需要修正的地方**

21. `vector<int>` 和 `vector<string>` 没有共同基类，不能"向上转型"成一个泛型容器；需要统一处理时用模板函数（编译期）或类型擦除（第四篇）。
22. `<T>` 里可以放值（`array<float, 4>`、`SmallVector<int64_t, 5>`、`ScalarTypeToCPPType<ScalarType::Float>`），这在 Java 里没有对应物，是 CUDA kernel 参数化的基础。
23. 模板参数在运行期不存在，也就没有 `instanceof T` 这种事；要按类型分支，用 `if constexpr (std::is_same_v<T, X>)` 在编译期分。
24. C++ lambda 可以按引用捕获并修改外层变量；换来的代价是你要自己保证它不会活得比外层变量久。


## 十二、总结

本文围绕"一份代码如何服务多种类型"，把 C++ 模板的核心机制和它们在 PyTorch/vLLM 里的用法对应起来。要点：

**模板是配方，实例化是生成代码**。一个模板 + N 组参数 = N 份独立的机器码，各有各的符号（`add_kernel<double>`、`add_kernel<float>`）。Java 泛型擦除类型、只有一份字节码；C++ 恰恰相反，因此能用基本类型、能按类型换实现、能把值当参数，代价是编译慢、二进制大、错误信息长。模板定义要放头文件，例外是显式实例化（`TensorMethods.cpp` 的 `data_ptr<T>`）。

**推导**：函数模板从实参推 `T`，不能从返回值推，所以 `data_ptr<scalar_t>()` 必须显式写；`auto`、`decltype`、`declval`、尾置返回类型、CTAD 推导指引都是同一套规则的延伸。每个 lambda 类型唯一，`parallel_for(const F&)` 每个调用点一份实例。

**非类型模板参数**：`std::array<T, N>`、`SmallVector<T, N>`、`ScalarTypeToCPPType<ScalarType N>`、vLLM kernel 的 `<scalar_t, width, HasWeight>`/`<BLOCK_Y_SIZE>`。值必须是编译期常量，所以运行期的值要经过一组分支才能进入模板——这是 `AT_DISPATCH` 和 vLLM if 链的共同本质。

**特化与变参**：全特化建查表（`ScalarTypeToCPPType`、`_typeConvert`），偏特化按类型模式换实现（`SmallVectorTemplateBase<T, true>`、`CanonicalizeStrTypes<char[N]>`），函数模板用重载代替偏特化（`_str`）；`Args&&...` + `std::forward` 完美转发（`make_intrusive`），递归展开拼字符串（`c10::str`）。

**编译期分支**：`constexpr` 让值编译期可求，`if constexpr` 删掉不选的分支（`generic_sizes<T>`、`_f16Vec`），`static_assert` 给出可读错误，SFINAE/`enable_if` 让重载消失（`HeaderOnlyArrayRef` 容器构造函数、`fused_add_rms_norm_kernel` 两个版本、`execute_op`），C++20 concepts 是它的正式替代但 PyTorch/vLLM 源码尚未使用。

**`AT_DISPATCH_FLOATING_TYPES`**：三层宏——`AT_DISPATCH_FLOATING_TYPES` 生成 `case` 列表；`AT_DISPATCH_SWITCH` → `THO_DISPATCH_SWITCH_TMPL` 是一个 IIFE 里的 `switch`，`default` 抛 `NotImplementedError`；`AT_DISPATCH_CASE` → `AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, ...)` → `THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL` 在每个 `case` 块里 `using scalar_t = ScalarTypeToCPPTypeT<enum_type>;` 然后 `return lambda();`。**`scalar_t` 是 `case` 块作用域里的一个类型别名，名字由 `AT_DISPATCH_CASE` 硬编码，类型由 `ScalarTypeToCPPType` 全特化表查出；lambda 的文本被复制到每个 `case`，N 个 dtype 就是 N 个闭包类型各编译一次**。vLLM 复用同一套宏，只换 dtype 列表并把 `HINT` 改名 `fp8_t`；`Dispatch_v2.h` 用参数计数技巧去掉 `_AND2`/`_AND3` 后缀，`AT_WRAP` 保护 lambda 里的逗号，最终落到同样的 `SWITCH`/`CASE`。

**视图与容器**：`ArrayRef<T>` 是指针 + 长度的不拥有视图，按值传、不存储，一组隐式构造函数统一各种容器；`std::optional<T>` 是值类型的可选值，`c10::optional` 在 v2.13.0 只剩一个废弃别名；`SmallVector<T, N>` 小容量内联不分配，`DimVector` 是 5 维内联的 shape 容器。

**lambda**：闭包类型唯一、可内联；`[&]` 引用捕获不延长寿命，同步且不逃逸时安全（`AT_DISPATCH` 的 IIFE、`parallel_for` 的 `#pragma omp parallel` 同步块），要存起来或异步时按值/初始化捕获；泛型 lambda 的 `operator()` 是模板；模板参数传 lambda 零开销、`std::function` 类型擦除有代价但可跨 `.so`。

最后把 Java 对照集中列一次：

| 概念 | Java | C++ | 类比失效的地方 |
|---|---|---|---|
| `Foo<T>` | 泛型，编译后擦除 | 模板，编译期为每组参数生成代码 | Java 只有一份代码；C++ `vector<int>`/`vector<string>` 是两个类型 |
| `List<Integer>` | 装箱，运行期 `Object` | `vector<int>` 紧密排列的 `int` | Java 泛型不能用基本类型 |
| 类型参数约束 | `<T extends Comparable<T>>`，定义处检查 | 默认无约束；`static_assert`/SFINAE/concepts | C++ 错误在实例化处，信息长 |
| 值参数 `<N>` | 不存在 | `array<T, N>`、`SmallVector<T, N>`、kernel 的 `BLOCK_SIZE` | Java 无法把 tile 大小编进类型 |
| 按类型给不同实现 | 不能（`instanceof` 运行期） | 特化、重载、`if constexpr`，编译期 | — |
| 可变参数 | `Object... args`，运行期数组 | `Args&&... args`，编译期知道每个类型 | 完美转发无对应物 |
| `Optional<T>` | 堆对象包引用 | `std::optional<T>` 值类型，无分配 | Java 更多用 `null` |
| `List<Long>` 只读视图 | 接口 + 虚调用 | `ArrayRef<T>` 指针 + 长度，编译期统一 | `ArrayRef` 不延长生命周期 |
| lambda 捕获 | 只能按值捕获 effectively final | `[&]`/`[=]`/初始化捕获 | Java 靠禁止按引用捕获避免悬垂；C++ 靠程序员 |
| 传 lambda | 函数式接口，接口调用 | 模板参数直接内联；`std::function` 才是类型擦除 | Java 无零开销选项 |
| 类型见证 `Collections.<String>emptyList()` | 少用，通常能推导 | `data_ptr<float>()` 必须写 | C++ 不从返回值推导 |

下一篇进入多态：`AT_DISPATCH` 解决了"按 dtype 选 kernel"，但"按设备（CPU/CUDA）选 kernel"是运行期的事，PyTorch 的 Dispatcher 用虚函数、函数指针、`std::function` 和手写类型擦除（`c10::KernelFunction`）把任意签名的 kernel 装进统一的表里。为什么 `TensorImpl` 有虚函数而 `Tensor` 没有，为什么 `KernelFunction` 同时有 boxed 和 unboxed 两条路径，`IValue` 和 Java 的 `Object` 有什么不同——这些是第四篇的内容。


## 下一篇

[多态与类型擦除：运行时如何选择实现](/cpp-polymorphism-and-type-erasure.html)

