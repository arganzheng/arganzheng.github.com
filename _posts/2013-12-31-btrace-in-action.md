---
title: BTrace实战
layout: post
---

### 什么是BTrace

官方文档上有个非常简洁精辟的介绍。

>
BTrace is a safe, dynamic tracing tool for the Java platform. BTrace can be used to dynamically trace a running Java program (similar to DTrace for OpenSolaris applications and OS). BTrace dynamically instruments the classes of the target application to inject tracing code ("bytecode tracing"). Tracing code is expressed in Java programming language. There is also integration with DTrace for the OpenSolaris platform.

关键几点：

1. 动态增强(instrument)运行中的Java程序。普通的AOP之类的都是预先写好的脚本和配置。但是BTrace不同，他可以在运行期间植入切面代码。这也是他最大的作用。这点非常类似于JMockit。
2. 植入代码是字节码级别的跟踪(bytecode tracing，正是BTrace的名称来源)


### 应用场景

正如其名，BytecodeTracing，可以对已经运行的java代码进行动态植入。获取运行时的一些信息。注意，只能Tracing，不能修改目标代码。只能使用`BTraceUtils`类中的方法。

### 编写BTrace脚本

假设我们要拦截的方法如下：

	package me.arganzheng.study.btrace.target.controller;

	@Controller
	public class ApiDispatcher {

		public void dispatch(@PathVariable
		    	int version, @PathVariable
		    	String moduleName, @PathVariable
		    	String apiName, HttpServletRequest request, HttpServletResponse response) {
		    // ...
		    Map<String, Object> context = buildContext(request);
		    // ...
		}

		private Map<String, Object> buildContext(HttpServletRequest request) {
		        Map<String, Object> context = new HashMap<String, Object>();

	        String clientIP = WebUtils.getRealIp(request);

	        // 虚拟需要ClientIP做权重分配
	        context.put("clientIP", clientIP);
	        
	        logger.info("Context.clientIP=" + clientIP);

	        return context;
		}
	}

可以编写如下脚本：

	package me.arganzheng.study.btrace.script;

	import static com.sun.btrace.BTraceUtils.*;

	import java.util.Map;

	import com.sun.btrace.AnyType;
	import com.sun.btrace.annotations.*;

	@BTrace
	public class ApiDispatcherBTrace {

	    @OnMethod(clazz = "me.arganzheng.study.btrace.target.controller.ApiDispatcher", method = "buildContext", location = @Location(value = Kind.ENTRY))
	    public static void onBuildContextMethodEntry(@ProbeClassName
	    String className, @ProbeMethodName
	    String probeMethod, AnyType[] args) {
	        println(Strings.strcat("EnterMethod: ", Strings.strcat(className, Strings.strcat(".", probeMethod))));
	        printArray(args);
	    }

	    /**
	     * @param context
	     */
	    @OnMethod(clazz = "me.arganzheng.study.btrace.target.controller.ApiDispatcher", method = "buildContext", location = @Location(Kind.RETURN))
	    public static void onBuildContextMethodReturn(@Return
	    Map<String, Object> context) {
	        // NOTE: Can Only user methods defined in BTraceUtils...
	        // for (Entry<String, Object> entry : context.entrySet()) {
	        // // ...
	        // }
	        String clientIP = (String) Collections.get(context, "clientIP");
	        println(Strings.strcat("clientIP=", clientIP));
	    }
	}


### 让运行中的应用加载你的BTrace脚本

通过JPS得到target JVM的pid，然后运行btrace脚本： 
	
	webadmin@C2C_206_59_sles10_64:/home/arganzheng> btrace/bin/btrace 28257 /home/arganzheng/btrace_script/ApiDispatcherBTrace.java

当btrace脚本的条件满足时，就会执行：

	EnterMethod: me.arganzheng.study.btrace.target.controller.ApiDispatcher.buildContext
	[org.springframework.web.multipart.support.DefaultMultipartHttpServletRequest@26fc5b01, ]
	clientIP=10.6.206.59


### 原理

[java.lang.instrument.Instrumentation](http://docs.oracle.com/javase/6/docs/technotes/guides/instrumentation/index.html)

### 限制

虽然这样方式很爽，不需要修改目标代码，不需要重启目标程序，完全无侵入性的AOP。但是为了保证trace语句只读, BTrace对trace脚本有一些限制(比如不能改变被trace代码中的状态) 

* BTrace class不能新建类, 新建数组, 抛异常, 捕获异常,
* 不能调用实例方法以及静态方法(com.sun.btrace.BTraceUtils除外)
* 不能将目标程序和对象赋值给BTrace的实例和静态field
* 不能定义外部, 内部, 匿名, 本地类
* 不能有同步块和方法
* 不能有循环
* 不能实现接口, 不能扩展类
* 不能使用assert语句, 不能使用class字面值

所以写起来比较蛋疼。。


### 参考文章

1. [BTrace](https://kenai.com/projects/btrace/pages/Home)
2. [BTrace – a Simple Way to Instrument Running Java Applications](http://piotrnowicki.com/2012/05/btrace-a-simple-way-to-instrument-running-java-applications/)
3. [Study_Java_BTrace ](http://source.hatter.me/p/hatter-source-code/wiki/Study_Java_BTrace)


