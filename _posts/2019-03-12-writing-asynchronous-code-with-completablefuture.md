---
title: 使用CompletableFuture异步编程
layout: post
tags: [java, jvm, high performance, tunning]
catalog: true
---

### Future vs. CompletableFuture

JDK 5引入了 Future 模式。Future 接口是Java多线程 Future 模式的实现，在`java.util.concurrent`包中，可以来进行异步计算。

Future 模式是多线程设计常用的一种设计模式。所谓Future，顾名思义，就是结果在未来完成，先给你一个凭据。Future的接口很简单，只有五个方法：

```java
public interface Future<V> {

    boolean cancel(boolean mayInterruptIfRunning);

    boolean isCancelled();

    boolean isDone();

    V get() throws InterruptedException, ExecutionException;

    V get(long timeout, TimeUnit unit)
        throws InterruptedException, ExecutionException, TimeoutException;
}
```

**Future 模式的局限性**

* 没有提供通知或者回调的机制，我们无法得知 Future 什么时候完成以进行后续的操作，不只是简单的阻塞等待。（要么使用阻塞，在 `future.get()` 的地方等待 future 返回的结果，这时又变成同步操作。要么使用 `isDone()` 轮询地判断 Future 是否完成，这样会白白耗费CPU的资源。）
* 无法将两个异步计算的结果合并为一个，这两个异步计算之间相互独立，同时第二个又依赖于第一个的结果。
* 无法等待 Future 集合中的所有任务都完成。
* 无法只等待 Future 集合中最快结束的任务完成，并返回它的结果。
* 无法通过编程的方式完成一个 Future 任务的执行（即以手工设定异步操作结果的方式）。

Java 虽然提供了多线程并发编程，但是直到 java 8 CompletableFuture 的出现，才从 JDK 层面真正意义上的支持 基于事件的异步编程范式。当然，之前有很多开源库做了这方面的努力，比如 Guava 的 [ListenableFuture](http://ifeve.com/google-guava-listenablefuture/)，[RxJava](https://github.com/ReactiveX/RxJava)，[Reactor](https://projectreactor.io/)，[Scala promises](https://docs.scala-lang.org/overviews/core/futures.html#promises)。

事实上，Java 8的 CompletableFuture 与 Guava 的 ListenableFuture 是如此的相似，前者正是吸收了所有 Google Guava 中 ListenableFuture 和 SettableFuture 的特征，还提供了其它强大的功能，让 Java 拥有了完整的非阻塞编程模型: Future、Promise 和 Callback（在Java8之前，只有无Callback 的Future）。当然两者有一些细微的差别，我们后面会提到。这里我们只介绍基于 CompletableFuture 实现的异步编程，其他的库提供的异步编程方式（主要是响应式编程范式）我们在后面的文章介绍。


### CompletableFuture 接口概览

CompletableFuture 有59个接口，但是别被它吓到了，同步和异步一区分就少了一半，异步接口中使用指定线程池（而非默认的 `ForkJoinPool.commonPool()`）又少了将近一半。这里我们只介绍异步相关的接口。

#### 1、Async vs. non-Async methods

首先按照同步和异步可以划分为两大类：Async vs. non-Async methods。

> Actions supplied for dependent completions of non-async methods may be performed by the thread that completes the current CompletableFuture, or by any other caller of a completion method.
>
> All async methods without an explicit Executor argument are performed using the ForkJoinPool.commonPool() (unless it does not support a parallelism level of at least two, in which case, a new Thread is created to run each task). 
> And you can therefore also provide a specific Executor to all the Async methods.

在命名上就直接区分开来：所有的异步方法都是以 Async 结尾。下面是所有返回 CompletableFuture 的同步和异步方式列表：


| Non-Async method | Async method |
| --- | --- |
| acceptEither  |  acceptEitherAsync | 
| allOf (waits for all futures to complete)  |   N/A  |
| anyOf (waits for any future to complete)  |   N/A |
| applyToEither  |  applyToEitherAsync |
| completedFuture (converts a value in a future already completed with this value)  |  N/A |
| exceptionally (handles an exception)   |   N/A |
| handle  | handleAsync |
| runAfterBoth   |  runAfterBothAsync |
| runAfterEither |  runAfterEitherAsync |
| N/A | runAsync (initialises a concurrent operation) |
| N/A | supplyAsync (initialises a concurrent operation) |
| thenAccept |  thenAcceptAsync |
| thenAcceptBoth |  thenAcceptBothAsync |
| thenApply  |  thenApplyAsync |
| thenCombine | thenCombineAsync |
| thenCompose | thenComposeAsync |
| thenRun | thenRunAsync |
| whenComplete  |   whenCompleteAsync  |


#### 2、静态工程方法(Start asynchronous operations)
    
可以通过如下两种方式执行一个异步操作:

* using a Runnable with runAsync, or
* using a Supplier with supplyAsync

| 方法名 | 描述 |
| --- | --- |
| `runAsync(Runnable runnable)` | 使用 ForkJoinPool.commonPool() 作为它的线程池执行异步代码。|
| `runAsync(Runnable runnable, Executor executor)` | 使用指定的thread pool执行异步代码。|
| `supplyAsync(Supplier<U> supplier)` | 使用 ForkJoinPool.commonPool() 作为它的线程池执行异步代码，异步操作有返回值 | 
| `supplyAsync(Supplier<U> supplier, Executor executor)` | 使用指定的 thread pool 执行异步代码，异步操作有返回值 |

**TIPS** runAsync 和 supplyAsync 方法的区别是 runAsync 返回的 CompletableFuture 是没有返回值的，这点跟 Future 的 Runable 和 Callable 蛮类似的。


#### 3、串联 (Chain asynchronous operations)

CompletableFuture 更强大的地方在于我们可以基于事件（依赖）将各个任务串联起来构成 DAG/Pipeline 并发执行。 

* 回调 (Callbacks)
    * thenRun/thenRunAsync : Returns a new CompletionStage that, when this stage completes normally, executes the given action.
    * thenAccept/thenAcceptAsync : Returns a new CompletionStage that, when this stage completes normally, is executed with this stage's result as the argument to the supplied action. 
    * thenApply/thenApplyAsync : Returns a new CompletionStage that, when this stage completes normally, is executed with this stage's result as the argument to the supplied function.
    * whenComplete/whenCompleteAsync : 当 CompletableFuture 完成计算结果时对结果进行处理，或者当CompletableFuture产生异常的时候对异常进行处理。
    * handle/handleAsync : 当 CompletableFuture 完成计算结果或者抛出异常的时候，执行提供的fn。
* 组合 (Combining Multiple CompletableFutures)
    * thenCombine/thenCombineAsync : 当两个CompletableFuture都正常完成后，执行提供的fn，用它来组合另外一个CompletableFuture的结果。merge results of two **independent** CompletableFuture。
    * thenCompose/thenComposeAsync : 在异步操作完成的时候对异步操作的结果进行一些操作，并且仍然返回 CompletableFuture 类型。thenCompose可以用于组合多个 CompletableFuture，将前一个结果作为下一个计算的参数，它们之间存在着先后顺序。
    * thenAcceptBoth/thenAcceptBothAsync : like thenCombine but taking a function that returns void. thenAcceptBoth 跟 thenCombine 类似，但是返回 CompletableFuture<Void> 类型。
    * runAfterBoth/runAfterBothAsync : accepts a Runnable for execution after both CompletableFutures complete
    * applyToEither/applyToEitherAsync : takes a unary function, supplying it with the result of whichever CompletableFuture completes first
    * acceptEither/acceptEitherAsync : like applyToEither but taking a unary function with void result
    * runAfterEither/runAfterEitherAsync : accepts a Runnable for execution after either CompletableFuture completes
    * anyOf : 接受由 CompletableFuture 对象构成的数组,返回数组中第一个完成的 CompletableFuture 的返回值 CompletableFuture<Object> 对象。
    * allOf : 接受由 CompletableFuture 对象构成的数组，数组中所有的 CompletableFuture 完成后它返回一个 CompletableFuture<Void> 对象。

详细接口说明如下表格所示: 

| 方法名 | 描述 |
| --- | --- |
| `thenAccept(Consumer<? super T> action)` | 当CompletableFuture完成计算结果，只对结果执行Action，而不返回新的计算值 |
| `thenAcceptAsync(Consumer<? super T> action)` | 当CompletableFuture完成计算结果，只对结果执行Action，而不返回新的计算值，使用ForkJoinPool。| 
| `thenAcceptAsync(Consumer<? super T> action, Executor executor)` | 当CompletableFuture完成计算结果，只对结果执行Action，而不返回新的计算值 |
| `thenApply(Function<? super T,? extends U> fn)` | 接受一个Function<? super T,? extends U>参数用来转换CompletableFuture |
| `thenApplyAsync(Function<? super T,? extends U> fn)` | 接受一个Function<? super T,? extends U> 参数用来转换CompletableFuture，使用ForkJoinPool |
| `thenApplyAsync(Function<? super T,? extends U> fn, Executor executor)` | 接受一个Function<? super T,? extends U>参数用来转换CompletableFuture，使用指定的线程池 |
| `whenComplete(BiConsumer<? super T,? super Throwable> action)` | CompletableFuture完成计算结果时对结果进行处理，或者当CompletableFuture产生异常的时候对异常进行处理。|
| `whenCompleteAsync(BiConsumer<? super T,? super Throwable> action)` |当CompletableFuture完成计算结果时对结果进行处理，或者当CompletableFuture产生异常的时候对异常进行处理。使用ForkJoinPool。 |
| `whenCompleteAsync(BiConsumer<? super T,? super Throwable> action, Executor executor)` | 当CompletableFuture完成计算结果时对结果进行处理，或者当CompletableFuture产生异常的时候对异常进行处理。使用指定的线程池。 |
| `handle(BiFunction<? super T, Throwable, ? extends U> fn)` | 当 CompletableFuture 完成计算结果或者抛出异常的时候，执行提供的fn |
| `handleAsync(BiFunction<? super T, Throwable, ? extends U> fn)` | 当CompletableFuture完成计算结果或者抛出异常的时候，执行提供的fn，使用ForkJoinPool。|
| `handleAsync(BiFunction<? super T, Throwable, ? extends U> fn, Executor executor)` | 当 CompletableFuture 完成计算结果或者抛出异常的时候，执行提供的fn，使用指定的线程池。 |
| `thenCompose(Function<? super T, ? extends CompletionStage<U>> fn)` | 在异步操作完成的时候对异步操作的结果进行一些操作，并且仍然返回 CompletableFuture 类型。 | 
| `thenComposeAsync(Function<? super T, ? extends CompletionStage<U>> fn)` | 在异步操作完成的时候对异步操作的结果进行一些操作，并且仍然返回 CompletableFuture 类型。使用 ForkJoinPool。 | 
| `thenComposeAsync(Function<? super T, ? extends CompletionStage<U>> fn,Executor executor)`  | 在异步操作完成的时候对异步操作的结果进行一些操作，并且仍然返回 CompletableFuture 类型。使用指定的线程池。 |
| `thenCombine(CompletionStage<? extends U> other, BiFunction<? super T,? super U,? extends V> fn)` | 当两个CompletableFuture都正常完成后，执行提供的fn，用它来组合另外一个CompletableFuture的结果。 | 
| `thenCombineAsync(CompletionStage<? extends U> other, BiFunction<? super T,? super U,? extends V> fn)`  | 当两个CompletableFuture都正常完成后，执行提供的fn，用它来组合另外一个CompletableFuture的结果。使用 ForkJoinPool。 |
| `thenCombineAsync(CompletionStage<? extends U> other, BiFunction<? super T,? super U,? extends V> fn, Executor executor)` |当两个CompletableFuture都正常完成后，执行提供的fn，用它来组合另外一个CompletableFuture的结果。使用指定的线程池。|
| `thenAcceptBoth(CompletionStage<? extends U> other, BiConsumer<? super T,? super U> action)` | 当两个CompletableFuture都正常完成后，执行提供的action，用它来组合另外一个CompletableFuture的结果。 
| `thenAcceptBothAsync(CompletionStage<? extends U> other, BiConsumer<? super T,? super U> action)` | 当两个CompletableFuture都正常完成后，执行提供的action，用它来组合另外一个 CompletableFuture 的结果。使用 ForkJoinPool。|
| `thenAcceptBothAsync(CompletionStage<? extends U> other, BiConsumer<? super T,? super U> action, Executor executor)` |当两个 CompletableFuture 都正常完成后，执行提供的action，用它来组合另外一个 CompletableFuture 的结果。使用指定的线程池。|
| `applyToEither(CompletionStage<? extends T> other, Function<? super T,U> fn)` | 当任意一个CompletableFuture完成的时候，fn会被执行，它的返回值会当作新的CompletableFuture<U>的计算结果。|
| `applyToEitherAsync(CompletionStage<? extends T> other, Function<? super T,U> fn)` | 当任意一个CompletableFuture完成的时候，fn会被执行，它的返回值会当作新的 `CompletableFuture<U>` 的计算结果。使用ForkJoinPool |
| `applyToEitherAsync(CompletionStage<? extends T> other, Function<? super T,U> fn, Executor executor)` | 当任意一个CompletableFuture完成的时候，fn会被执行，它的返回值会当作新的 `CompletableFuture<U>` 的计算结果。使用指定的线程池 |
| `acceptEither(CompletionStage<? extends T> other, Consumer<? super T> action)` | 当任意一个 CompletableFuture 完成的时候，action这个消费者就会被执行。 | 
| `acceptEitherAsync(CompletionStage<? extends T> other, Consumer<? super T> action)` | 当任意一个 CompletableFuture 完成的时候，action 这个消费者就会被执行。使用 ForkJoinPool |
| `acceptEitherAsync(CompletionStage<? extends T> other, Consumer<? super T> action, Executor executor)` | 当任意一个CompletableFuture完成的时候，action这个消费者就会被执行。使用指定的线程池。|
| `allOf(CompletableFuture<?>... cfs)` | 在所有 Future 对象完成后结束，并返回一个 future。 |
| `anyOf(CompletableFuture<?>... cfs)` | 在任何一个 Future 对象结束后结束，并返回一个 future。|

**说明** 

1、thenApply() 和 thenCompose() 的区别: thenApply() 是返回的是非CompletableFuture类型，它的功能相当于将 `CompletableFuture<T>` 转换成 `CompletableFuture<U>` ; thenCompose() 可以用于组合多个 CompletableFuture，将前一个结果作为下一个计算的参数，它们之间存在着先后顺序，返回值是新的 CompletableFuture。

2、使用 thenCombine() 的future 之间互相独立并且是并行执行的，最后再将结果汇总。这一点跟 thenCompose() 不同。

3、handle() 的参数是 BiFunction，apply() 方法返回 R，相当于转换的操作。而 whenComplete() 的参数是 BiConsumer，accept() 方法返回void。所以，handle() 相当于 whenComplete() + 转换。

4、thenAccept()是只会对计算结果进行消费而不会返回任何结果的方法。

5、anyOf 和 acceptEither、applyToEither 的区别在于，后两者只能使用在两个 future 中，而 anyOf 可以使用在多个future中。


### 异常处理(Error handling)


可以对 get() 操作进行异常捕获和处理，但这并不是一个最好的方法。

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class AsynchronousExceptions {
    public static void main(final String[] args) throws InterruptedException {
        for (final boolean failure : new boolean[]{false, true}) {

            CompletableFuture<Integer> x = CompletableFuture.supplyAsync(() -> {
                if (failure) {
                    throw new RuntimeException("Oops, something went wrong");
                }
                return 42;
            });

            try {
                // Blocks (avoid this in production code!), and either returns the promise's value, or...
                System.out.println(x.get());
                System.out.println("isCompletedExceptionally = " + x.isCompletedExceptionally());
                // Output[failure=false]: 42
                // Output[failure=false]: isCompletedExceptionally = false
            } catch (ExecutionException e) {
                // ... rethrows the RuntimeException wrapped as an ExecutionException
                System.out.println(e.getMessage());
                System.out.println(e.getCause().getMessage());
                System.out.println("isCompletedExceptionally = " + x.isCompletedExceptionally());
                // Output[failure=true]:  java.lang.RuntimeException: Oops, something went wrong
                // Output[failure=true]:  Oops, something went wrong
                // Output[failure=true]:  isCompletedExceptionally = true
            }
        }
    }
}
```

CompletableFuture 提供了其他几个更优雅的异常处理方式。

#### 1、exceptionally

exceptionally(Function<Throwable,? extends T> fn) : 只有当 CompletableFuture 抛出异常的时候，才会触发这个exceptionally的计算，调用function计算值。非常类似于 try-catch。可以用它来捕获异常并返回一个默认值或者错误代码（handle exception and return a default or error value）。

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class AsynchronousExceptionsHandlingWithExceptionally {
    public static void main(final String[] args) throws InterruptedException, ExecutionException {
        for (final boolean failure : new boolean[]{false, true}) {

            CompletableFuture<Integer> x = CompletableFuture.supplyAsync(() -> {
                if (failure) {
                    throw new RuntimeException("Oops, something went wrong");
                }
                return 42;
            });

            /**
             * Returns a new CompletableFuture that is completed when this CompletableFuture completes,
             * with the result of the given function of the exception triggering this CompletableFuture's completion
             * when it completes exceptionally; otherwise, if this CompletableFuture completes normally,
             * then the returned CompletableFuture also completes normally with the same value.
             */
            CompletableFuture<Integer> tryX = x.exceptionally(ex -> -1); // Note that tryX and x are of same type.

            // Blocks (avoid this in production code!), and either returns the promise's value
            System.out.println(tryX.get());
            System.out.println("isCompletedExceptionally = " + tryX.isCompletedExceptionally());

            // Output[failure=false]: 42
            // Output[failure=false]: isCompletedExceptionally = false

            // Output[failure=true]:  -1
            // Output[failure=true]:  isCompletedExceptionally = false
        }
    }
}
```

**说明** 关于 completeExceptionally

注意不要跟 completeExceptionally 混淆了，completeExceptionally 本质上是一个 fastfail 机制。顾名思义，就是以抛出 CompletionException 异常结束:

> If not already completed, causes invocations of get() and related methods to throw the given exception.


```java
private <R, A extends AsynchronouslyCreatedResource<R>> CompletableFuture<R> pollResourceAsync(
    final Supplier<CompletableFuture<A>> resourceFutureSupplier,
    final CompletableFuture<R> resultFuture,
    final long attempt) {
  resourceFutureSupplier.get().whenComplete((asynchronouslyCreatedResource, throwable) -> {
    if (throwable != null) {
      resultFuture.completeExceptionally(throwable);
    } else {
      if (asynchronouslyCreatedResource.queueStatus().getId() == QueueStatus.Id.COMPLETED) {
        resultFuture.complete(asynchronouslyCreatedResource.resource());
      } else {
        retryExecutorService.schedule(() -> {
          pollResourceAsync(resourceFutureSupplier, resultFuture, attempt + 1);
        }, waitStrategy.sleepTime(attempt), TimeUnit.MILLISECONDS);
      }
    }
  });
  return resultFuture;
}
```

#### 2、exceptionHandler 

exceptionally 只有异常的时候才会被回调，如果我们想异常与否都对结果进行处理，那么可以使用 whenComplete 或者 handler 方法。

##### 2.1 whenComplete

如果你希望不管 CompletableFuture 运行正常与否 都执行一段代码，如释放资源，更新状态，记录日志等，但是同时不影响原来的执行结果。那么你可以使用 whenComplete 方法。exceptionally非常类似于 catch()，而 whenComplete 则非常类似于 finally:

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class AsynchronousExceptionsHandlingWithWhenComplete {
    public static void main(final String[] args) throws InterruptedException, ExecutionException {
        for (final boolean failure : new boolean[]{false, true}) {

            CompletableFuture<Integer> x = CompletableFuture.supplyAsync(() -> {
                if (failure) {
                    throw new RuntimeException("Oops, something went wrong");
                }
                return 42;
            });

            /**
             * Returns a new CompletableFuture with the same result or exception as this CompletableFuture,
             * that executes the given action when this stage completes.
             */
            CompletableFuture<Integer> tryX = x.whenComplete((value, ex) -> { // Note that tryX and x are of same type. This CompletableFuture acts as an invisible "decorator".
                if (value != null) {
                    // We get a chance to transform the result by adding 1...
                    System.out.println("Result: " + value);
                } else {
                    // ... or return an error value:
                    System.out.println("Error code: -1. Root cause: " + ex.getMessage());
                }
            });

            try {
                // Blocks (avoid this in production code!), and either returns the promise's value, or...
                System.out.println(tryX.get());
                System.out.println("isCompletedExceptionally = " + tryX.isCompletedExceptionally());
                // Output[failure=false]: Result: 42
                // Output[failure=false]: 42
                // Output[failure=false]: isCompletedExceptionally = false
            } catch (ExecutionException e) {
                // ... rethrows the RuntimeException wrapped as an ExecutionException
                System.out.println(e.getMessage());
                System.out.println("isCompletedExceptionally = " + tryX.isCompletedExceptionally());
                // Output[failure=true]:  Error code: -1. Root cause: java.lang.RuntimeException: Oops, something went wrong
                // Output[failure=true]:  Oops, something went wrong
                // Output[failure=true]:  isCompletedExceptionally = true
            }
        }
    }
}
```

##### 2.2 handle

handle 方法给你一个机会，可以在一个方法里对正常处理结果进行转换或者对异常进行捕获处理:

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import static java.lang.String.format;

public class AsynchronousExceptionsHandlingWithHandle {
    public static void main(final String[] args) throws InterruptedException, ExecutionException {
        for (final boolean failure : new boolean[]{false, true}) {

            CompletableFuture<Integer> x = CompletableFuture.supplyAsync(() -> {
                if (failure) {
                    throw new RuntimeException("Oops, something went wrong");
                }
                return 42;
            });

            /**
             * Returns a new CompletableFuture that, when this CompletableFuture completes either normally or exceptionally,
             * is executed with this stage's result and exception as arguments to the supplied function.
             */
            CompletableFuture<HttpResponse> tryX = x.handle((value, ex) -> { // Note that tryX and x are of different type.
                if (value != null) {
                    // We get a chance to transform the result...
                    return new HttpResponse(200, format("{\"value\": %s}", value));
                } else {
                    // ... or return details on the error using the ExecutionException's message:
                    return new HttpResponse(500, format("{\"error\": \"%s\"}", ex.getMessage()));
                }
            });

            // Blocks (avoid this in production code!), and either returns the promise's value:
            System.out.println(tryX.get());
            System.out.println("isCompletedExceptionally = " + tryX.isCompletedExceptionally());

            // Output[failure=false]: 200 - {"value": 42}
            // Output[failure=false]: isCompletedExceptionally = false

            // Output[failure=true]:  500 - {"error": "java.lang.RuntimeException: Oops, something went wrong"}
            // Output[failure=true]:  isCompletedExceptionally = false
        }
    }

    private static class HttpResponse {
        private final int status;
        private final String body;

        public HttpResponse(final int status, final String body) {
            this.status = status;
            this.body = body;
        }
        public String toString() { return status + " - " + body; }
    }
}
```

### Asynchronous timeouts 

Java 8 的 CompletableFuture 并没有 timeout 机制，虽然可以在 get 的时候指定 timeout，但是我们知道get 是一个同步堵塞的操作。怎样让 timeout 也是异步的呢？Java 8 内有内建的机制支持，一般的实现方案是启动一个 `ScheduledThreadpoolExecutor` 线程在 timeout 时间后直接调用 `futurn.completeExceptionally(new TimeoutException())`，然后用 `acceptEither()` 或者 `applyToEither` 看是先计算完成还是先超时： 

```java
CompletableFuture.supplyAsync(() -> findBestPrice("LDN - NYC"), executorService)
                         .thenCombine(CompletableFuture.supplyAsync(() -> queryExchangeRateFor("GBP")),  
                                                 this::convert)
                         .acceptEither(timeoutAfter(1, TimeUnit.SECONDS), 
                                       amount -> System.out.println("The price is: " + amount + "GBP"));
```

一个简单的 timeoutAfter 实现如下：

```java
public <T> CompletableFuture<T> timeoutAfter(long timeout, TimeUnit unit) {
    CompletableFuture<T> result = new CompletableFuture<T>();
    delayer.schedule(() -> result.completeExceptionally(new TimeoutException()), timeout, unit);
    return result;
}
```

其中 delayer 是 ScheduledThreadPoolExecutor 的一个实例：

```java
/**
 * Singleton delay scheduler, used only for starting and
 * cancelling tasks.
 */
static final class Delayer {
    static ScheduledFuture<?> delay(Runnable command, long delay,
                                    TimeUnit unit) {
        return delayer.schedule(command, delay, unit);
    }

    static final class DaemonThreadFactory implements ThreadFactory {
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            t.setName("CompletableFutureDelayScheduler");
            return t;
        }
    }

    static final ScheduledThreadPoolExecutor delayer;
    static {
        (delayer = new ScheduledThreadPoolExecutor(
            1, new DaemonThreadFactory())).
            setRemoveOnCancelPolicy(true);
    }
}
```

在 java 9 引入了 `orTimeout` 和 `completeOnTimeOut` 两个方法支持 异步 timeout 机制：

* public CompletableFuture<T> orTimeout(long timeout, TimeUnit unit) : completes the CompletableFuture with a TimeoutException after the specified timeout has elapsed.
* public CompletableFuture<T> completeOnTimeout(T value, long timeout, TimeUnit unit) : provides a default value in the case that the CompletableFuture pipeline times out. 

内部实现上跟我们上面的实现方案是一模一样的，只是现在不需要自己实现了。


### allOf 如果其中一个失败了如何快速结束所有？

默认情况下，allOf 会等待所有的任务都完成，即使其中有一个失败了，也不会影响其他任务继续执行。但是大部分情况下，一个任务的失败，往往意味着整个任务的失败，继续执行完剩余的任务意义并不大。在 谷歌的 Guava 的 [allAsList](https://google.github.io/guava/releases/snapshot/api/docs/com/google/common/util/concurrent/Futures.html#allAsList-java.lang.Iterable-) 如果其中某个任务失败整个任务就会取消执行:

> allAsList(Iterable<ListenableFuture<V>>): Returns a ListenableFuture whose value is a list containing the values of each of the input futures, in order. If any of the input futures fails or is cancelled, this future fails or is cancelled.

那么如何实现 allAsList 的功能呢？

一种做法就是对 allOf 数组中的每个 CompletableFuture 的 exceptionally 方法进行捕获处理：如果有异常，那么整个 allOf 就直接抛出那个异常: 

```java
public static void main(String[] args) throws InterruptedException, ExecutionException {
    CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> {
        System.out.println("-- future1 -->");
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        System.out.println("<-- future1 --");
        return "Hello";
    });

    CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> {
        System.out.println("-- future2 -->");
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        System.out.println("<-- future2 --");
        throw new RuntimeException("Oops!");
    });

    CompletableFuture<String> future3 = CompletableFuture.supplyAsync(() -> {
        System.out.println("-- future3 -->");
        try {
            Thread.sleep(4000);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        System.out.println("<-- future3 --");
        return "world";
    });

    // CompletableFuture<Void> combinedFuture = CompletableFuture.allOf(future1, future2, future3);
    // combinedFuture.join();

    CompletableFuture<Void> allWithFailFast = CompletableFuture.allOf(future1, future2, future3);
    Stream.of(future1, future2, future3).forEach(f -> f.exceptionally(e -> {
        allWithFailFast.completeExceptionally(e);
        return null;
    }));
    
    allWithFailFast.join();

}
```

运行结果：

```java
-- future1 -->
-- future2 -->
-- future3 -->
<-- future1 --
<-- future2 --
Exception in thread "main" java.util.concurrent.CompletionException: java.lang.RuntimeException: Oops!
    at java.util.concurrent.CompletableFuture.encodeThrowable(CompletableFuture.java:273)
    at java.util.concurrent.CompletableFuture.completeThrowable(CompletableFuture.java:280)
    at java.util.concurrent.CompletableFuture$AsyncSupply.run(CompletableFuture.java:1592)
    at java.util.concurrent.CompletableFuture$AsyncSupply.exec(CompletableFuture.java:1582)
    at java.util.concurrent.ForkJoinTask.doExec(ForkJoinTask.java:289)
    at java.util.concurrent.ForkJoinPool$WorkQueue.runTask(ForkJoinPool.java:1056)
    at java.util.concurrent.ForkJoinPool.runWorker(ForkJoinPool.java:1692)
    at java.util.concurrent.ForkJoinWorkerThread.run(ForkJoinWorkerThread.java:157)
Caused by: java.lang.RuntimeException: Oops!
    at QuickTest.lambda$1(QuickTest.java:29)
    at java.util.concurrent.CompletableFuture$AsyncSupply.run(CompletableFuture.java:1590)
    ... 5 more
```

可以看到 `System.out.println("<-- future3 --");` 并没有继续执行。


### 一个有意思的例子

下面我们看一下怎样使用 CompletableFuture 构建和并发执行一个 DAG 任务：

![call-tree-multistages.png](/img/in-post/call-tree-multistages.png)

如上图所示，我们将执行如下一系列操作:

1. we asynchronously compute integers from 1 to 5 – each integer generation takes 2 seconds
2. we sum these together
3. we asynchronously multiply the sum by 1, 2 and 3 – each multiplication takes 2 seconds
4. we take the maximum.

现在让我们看看用 CompletableFuture 怎么实现：

```java
package life.arganzheng.study;

import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static java.util.concurrent.CompletableFuture.completedFuture;
import static java.util.concurrent.TimeUnit.SECONDS;

public class AsynchronousSumAndMax {
    public static void main(final String[] args) {
        stopwatch(() -> {
            Stream<CompletableFuture<Integer>> xs = Stream.of(1, 2, 3, 4, 5).map(
                    x -> CompletableFuture.supplyAsync(() -> compute(x))
            );

            CompletableFuture<Integer> sum = xs.reduce(completedFuture(0), (x, y) -> x.thenCombine(y, (i, j) -> i + j));

            CompletableFuture<Integer>[] ys = Stream.of(1, 2, 3).map(
                    x -> sum.thenApplyAsync(s -> multiply(s, x))
            ).toArray(CompletableFuture[]::new);

            CompletableFuture<Integer> max = CompletableFuture.allOf(ys).thenApply(
                    (Void) -> Arrays.stream(ys)
                            .map(y -> y.getNow(Integer.MAX_VALUE)) // Guaranteed to return y's value, given we synchronise with allOf, and only thenApply this function.
                            .max(Comparator.naturalOrder())
                            .get()
            );

            // Block and wait for results (avoid this in production code!):
            try { println("Result: " + max.get()); } catch (ExecutionException | InterruptedException e) { e.printStackTrace(); }

            // Output:
            // [ForkJoinPool.commonPool-worker-1]: Computing 1...
            // [ForkJoinPool.commonPool-worker-2]: Computing 2...
            // [ForkJoinPool.commonPool-worker-4]: Computing 4...
            // [ForkJoinPool.commonPool-worker-3]: Computing 3...
            // [ForkJoinPool.commonPool-worker-5]: Computing 5...
            // [ForkJoinPool.commonPool-worker-1]: Computed 1.
            // [ForkJoinPool.commonPool-worker-5]: Computed 5.
            // [ForkJoinPool.commonPool-worker-4]: Computed 4.
            // [ForkJoinPool.commonPool-worker-2]: Computed 2.
            // [ForkJoinPool.commonPool-worker-3]: Computed 3.
            // [ForkJoinPool.commonPool-worker-2]: Computing 15 * 1...
            // [ForkJoinPool.commonPool-worker-4]: Computing 15 * 2...
            // [ForkJoinPool.commonPool-worker-5]: Computing 15 * 3...
            // [ForkJoinPool.commonPool-worker-4]: Computed 15 * 2 = 30.
            // [ForkJoinPool.commonPool-worker-5]: Computed 15 * 3 = 45.
            // [ForkJoinPool.commonPool-worker-2]: Computed 15 * 1 = 15.
            // [main]: Result: 45
            // Elapsed time: 4 seconds.
        });
    }

    private static int compute(final int x) {
        println("Computing " + x + "...");
        sleep(2, SECONDS);
        println("Computed " + x + ".");
        return x;
    }

    private static int multiply(final int x, final int y) {
        println("Computing " + x + " * " + y + "...");
        sleep(2, SECONDS);
        final int r = x * y;
        println("Computed " + x + " * " + y + " = " + r + ".");
        return r;
    }

    private static void sleep(final int duration, final TimeUnit unit) {
        try { unit.sleep(duration); } catch (InterruptedException e) { e.printStackTrace(); }
    }

    private static void println(final String message) {
        System.out.println("[" + Thread.currentThread().getName() + "]: " + message);
    }

    private static void stopwatch(final Runnable action) {
        final long begin = System.currentTimeMillis();
        action.run();
        System.out.println("Elapsed time: " + (System.currentTimeMillis() - begin) / 1000 + " seconds.");
    }
}
```

执行结果如下：

```java
[ForkJoinPool.commonPool-worker-1]: Computing 1...
[ForkJoinPool.commonPool-worker-2]: Computing 2...
[ForkJoinPool.commonPool-worker-3]: Computing 3...
[ForkJoinPool.commonPool-worker-2]: Computed 2.
[ForkJoinPool.commonPool-worker-3]: Computed 3.
[ForkJoinPool.commonPool-worker-2]: Computing 4...
[ForkJoinPool.commonPool-worker-3]: Computing 5...
[ForkJoinPool.commonPool-worker-1]: Computed 1.
[ForkJoinPool.commonPool-worker-3]: Computed 5.
[ForkJoinPool.commonPool-worker-2]: Computed 4.
[ForkJoinPool.commonPool-worker-1]: Computing 15 * 3...
[ForkJoinPool.commonPool-worker-3]: Computing 15 * 1...
[ForkJoinPool.commonPool-worker-2]: Computing 15 * 2...
[ForkJoinPool.commonPool-worker-1]: Computed 15 * 3 = 45.
[ForkJoinPool.commonPool-worker-3]: Computed 15 * 1 = 15.
[ForkJoinPool.commonPool-worker-2]: Computed 15 * 2 = 30.
[main]: Result: 45
Elapsed time: 6 seconds.
```

可以看到由于 Step_1: computing integers from 1 to 5 和 Step_3: multiplying the sum by 1, 2 and 3 都是异步执行的，整个执行时间只需要 4秒，对比单线程则需要16秒。而且关键是整个代码可读性非常简洁、内聚和优雅（ 从Java 8开始 Java再也不是那么冗长了:) :

* we summed integers from 1 to 5 using a reducer, combining the neutral element CompletableFuture.completedFuture(0) with other futures as they completed using thenCombine
* we waited for all multiplications to complete using allOf and thenApply-ed Integer’s natural order comparator to find the maximum value.


### CompletableFuture vs. Java8 Stream vs. RxJava1 & RxJava2

CompletableFuture 有很多特性跟RxJava很像，所以将CompletableFuture、Java 8 Stream和RxJava做一个相互的比较。

| 各种对象 | composable | lazy | resuable | async | cached | push | back pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| CompletableFuture | 支持 | 不支持 | 支持 | 支持 | 支持 | 支持 |  不支持 | 
| Stream | 支持 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 | 
| Observable(RxJava1) | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 | 
| Observable(RxJava2) | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 | 不支持 |
| Flowable(RxJava2) | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 | 


### 总结

Java 8提供了一种函数风格的异步和事件驱动编程模型 CompletableFuture，它不仅支持回调还支持各种组合，使用 CompletableFuture，CountDownLatch 等多线程同步机制都可以扔掉了。相比之前的Future 编程，代码更加到简洁、内聚和优雅。


推荐文档
-------

1. [Java 8: Definitive guide to CompletableFuture](https://www.javacodegeeks.com/2013/05/java-8-definitive-guide-to-completablefuture.html)
2. [Java 8: Writing asynchronous code with CompletableFuture](https://www.deadcoderising.com/java8-writing-asynchronous-code-with-completablefuture/)
3. [Java 8 Concurrency - CompletableFuture in practice](http://www.marccarre.com/2016/05/08/java-8-concurrency-completablefuture-in-practice.html)
4. [0 Examples of Using Java’s CompletableFuture](https://dzone.com/articles/what-is-project-amber-in-java-1)
3. [Java 9: Handle timeouts asynchronously using CompletableFuture's orTimeout and completeOnTimeout](https://www.deadcoderising.com/java-9-handle-timeouts-asynchronously-using-completablefutures-ortimeout-and-completeontimeout/)
4. [Asynchronous timeouts with CompletableFutures in Java 8 and Java 9](http://iteratrlearning.com/java9/2016/09/13/java9-timeouts-completablefutures.html)
5. [Asynchronous Timeouts with CompletableFuture](https://dzone.com/articles/asynchronous-timeouts)
6. [mcalavera81/CompletableFutureTest.java](https://gist.github.com/mcalavera81/59121bc6d9c08ea1d7ce)
7. [src/java.base/share/classes/java/util/concurrent/CompletableFuture.java](http://hg.openjdk.java.net/jdk9/jdk9/jdk/file/tip/src/java.base/share/classes/java/util/concurrent/CompletableFuture.java)
8. [](http://www.imooc.com/article/21656)
