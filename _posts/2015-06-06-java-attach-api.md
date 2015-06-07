---
title: Java Attach API
layout: post
---


Dynamic Loading Agent
---------------------

前面在介绍JMX的时候说过应用通过MBean来统计监控数据，然后注册到MBeanServer。但是一个MBeanServer要被外部系统访问得到，还需要通过一个Connector将自己暴露出去，大部分实现都是基于RMI。

从Java 5开始JVM的细节是通过PlatformMBeanServer暴露出去的。而PlatformMBeanServer是通过JDK_HOME/lib/management-agent.jar这个Agent完成RMI的Connector暴露的。将其作为Agent的方式是为了能够让外部监控程序（如JConsole和VisualVM）可以根据需要使用Attach API让目标JVM动态创建这个Connector，然后监控注册到PlatformMBeanServer的MBean数据。

用JD-GUI打开JDK_HOME/lib/management-agent.jar(Mac下是JAVA_HOME/Contents/Home/jre/lib/management-agent.jar)，可以看到其实他就是一个只包含META-INF/MANIFEST.MF的空jar包：

	Manifest-Version: 1.0
	Created-By: 1.7.0_07 (Oracle Corporation)
	Agent-Class: sun.management.Agent
	Premain-Class: sun.management.Agent


这个MANIFEST.MF文件主要是指定了agent的两个入口 [Package java.lang.instrument](https://docs.oracle.com/javase/7/docs/api/java/lang/instrument/package-summary.html)：

> **Premain-Class**
> 	When an agent is specified at JVM launch time this attribute specifies the agent class. That is, the class containing the premain method. When an agent is specified at JVM launch time this attribute is required. If the attribute is not present the JVM will abort. Note: this is a class name, not a file name or path.
> **Agent-Class**
> 	If an implementation supports a mechanism to start agents sometime after the VM has started then this attribute specifies the agent class. That is, the class containing the agentmain method. This attribute is required, if it is not present the agent will not be started. Note: this is a class name, not a file name or path.

也就是说如果目标JVM attach了这个jar包，那么根据这个MANIFEST，它会执行sun.management.Agent的入口方法。根据约定，如果是在JVM启动的时候attach，那么就是premain方法，如果是在JVM启动之后attach，那么就是agentmain方法。

因为sum.management包在JDK里面，所以这个外部agent包就只有一个MANIFEST文件了。现在让我们看一下sum.management.Agent到底做了什么事情：

指定`-Dcom.sun.management.snmp`或者`-Dcom.sun.management.jmxremote`会让其执行想过的逻辑。我们这里只关注JMX。
设置`com.sun.management.jmxremote`选项会让其创建一个local JMX connector，这样允许同一个机器的同一个用户连接。
设置`com.sun.management.jmxremote.port`则会让PlatformMBeanServer暴露给一个可以远程访问的JMX connector。

具体实现，感兴趣的同学可以查看源码 [sun.management.Agent](http://grepcode.com/file/repository.grepcode.com/java/root/jdk/openjdk/6-b14/sun/management/Agent.java#Agent.startAgent%28java.util.Properties%29)。

这个Agent因为是JDK自带的，并且在系统classpath中，所以默认是会被system class loader加载的（事实上所有的Agent都必须通过system class loader加载）。

在JDK5，你只能通过修改启动参数（比如`-Dcom.sun.management.jmxremote`）的方式来加载这个Agent。但是在JDK6之后，这个PlatfromMBeanServer暴露机制是通过Java agent的方式启动, 这样就允许像JConsole，VisualVM这样的外部监控程序通过Attach API动态attach到已经运行的目标JVM，然后命令其动态的load了JDK_HOME/lib/management-agent.jar这个agent包，完成对PlatformMBeanServer进行RMI注册和监听了 [Setting up Monitoring and Management Programmatically](http://docs.oracle.com/javase/6/docs/technotes/guides/management/agent.html#gdhkz):


	/// Example 2-4 Attaching a JMX tool to a connector and getting the agent's address
	static final String CONNECTOR_ADDRESS =
	 "com.sun.management.jmxremote.localConnectorAddress";
	 
	// attach to the target application
	VirtualMachine vm = VirtualMachine.attach(id);
	 
	// get the connector address
	String connectorAddress =
	    vm.getAgentProperties().getProperty(CONNECTOR_ADDRESS);
	 
	// no connector address, so we start the JMX agent
	if (connectorAddress == null) {
	   String agent = vm.getSystemProperties().getProperty("java.home") +
	       File.separator + "lib" + File.separator + "management-agent.jar";
	   vm.loadAgent(agent);
	 
	   // agent is started, get the connector address
	   connectorAddress =
	       vm.getAgentProperties().getProperty(CONNECTOR_ADDRESS);
	}
	 
	// establish connection to connector server
	JMXServiceURL url = new JMXServiceURL(connectorAddress);
	JMXConnector jmxConn = JMXConnectorFactory.connect(url);

就是这么简单。
然后拿到JMXConnector之后，也就是连接上了，我们就可以拿到注册到PlatformMBeanServer中的MBean了：

	MBeanServerConnection mbs = jmxConn.getMBeanServerConnection();
    if(mbs == null) {
        System.err.println("Failed: could not get mbean server connection.");
        System.exit(1);
    }

    ObjectName on = new ObjectName("java.lang:type=Threading");
    ObjectInstance oi = mbs.getObjectInstance(on);
    ThreadMXBean bean = ManagementFactory.newPlatformMXBeanProxy(mbs,
            "java.lang:type=Threading", ThreadMXBean.class);
    if(bean == null) {
        System.err.println("Failed: could not get threading mbean.");
        System.exit(1);
    }
    
    System.out.println("Number of live threads = " + bean.getThreadCount());
    System.exit(0);


instrumentation
---------------

前面介绍的attach API，允许外部JVM attach到一个已经运行的JVM，并且让起加载指定的agent。而agent的premain或者agentmain方法会被相应的触发，我们可以在这里编写一些逻辑。management-agent这个agent只是创建了JMX Connector。但是实际上，我们还可以修改字节码：

	public static void premain(String args, Instrumentation inst);
	public static void main(String args[]);
	public static void agentmain(String args, Instrumentation inst); 

注意到不同于main方法，premain和agentmain方法有第二个参数：Instrumentation inst，这个参数实例是classloader传递过来的，我们可以注册我们的transformers进去，classloader在每次加载新类的时候都会调用所有注册到Instrumentation的transformers，从而达到修改字节码的目的。

Transformer必须实现ClassFileTransformer接口：

	public byte[] transform(ClassLoader loader, 
							String className, 
							Class classBeingRedefined, 
							ProtectionDomain  protectionDomain,  
							byte[] classfileBuffer) throws llegalClassFormatException


例子
----


假设我们现在有个类HelloWorld.java:

	package me.arganzheng.study.instrumentation;

	//to be instrumented java class
	public class HelloWorld {
		public void foo() throws InterruptedException {
			System.out.println("foo is going to run........");
			Thread.sleep(2000L);
		}
	}

然后我们要打印他的每个方法的执行时间。
使用instrumentation和agent方式我们不需要修改代码发布重启。只需要这样子：

编写一个DurationTransformer.java，这个类将修改要增强的类的字节码。

	package com.javapapers.java.instrumentation;

	import java.io.ByteArrayInputStream;
	import java.lang.instrument.ClassFileTransformer;
	import java.lang.instrument.IllegalClassFormatException;
	import java.security.ProtectionDomain;

	import javassist.ClassPool;
	import javassist.CtClass;
	import javassist.CtMethod;

	//this class will be registered with instrumentation agent
	public class DurationTransformer implements ClassFileTransformer {
		public byte[] transform(ClassLoader loader, String className,
				Class classBeingRedefined, ProtectionDomain protectionDomain,
				byte[] classfileBuffer) throws IllegalClassFormatException {
			byte[] byteCode = classfileBuffer;

			// since this transformer will be called when all the classes are
			// loaded by the classloader, we are restricting the instrumentation
			// using if block only for the Lion class
			if (className.equals("me/arganzhengg/study/instrumentation/HelloWorld")) {
				System.out.println("Instrumenting......");
				try {
					ClassPool classPool = ClassPool.getDefault();
					CtClass ctClass = classPool.makeClass(new ByteArrayInputStream(
							classfileBuffer));
					CtMethod[] methods = ctClass.getDeclaredMethods();
					for (CtMethod method : methods) {
						method.addLocalVariable("startTime", CtClass.longType);
						method.insertBefore("startTime = System.nanoTime();");
						method.insertAfter("System.out.println(\"Execution Duration "
								+ "(nano sec): \"+ (System.nanoTime() - startTime) );");
					}
					byteCode = ctClass.toBytecode();
					ctClass.detach();
					System.out.println("Instrumentation complete.");
				} catch (Throwable ex) {
					System.out.println("Exception: " + ex);
					ex.printStackTrace();
				}
			}
			return byteCode;
		}
	}

由于植入代码的特殊性，这里我们简单使用了insertBefore和insertAfter就完成了打印方法执行时间的目的。如果要比较做比较复杂的修改，那么可以使用ASM或者CGLIB。

然后编写DurationAgent.java，在premain中注册我们的DurationTransformer到inst参数中：

	package com.javapapers.java.instrumentation;

	import java.lang.instrument.Instrumentation;

	public class DurationAgent {
		public static void premain(String agentArgs, Instrumentation inst) {
			System.out.println("Executing premain.........");
			inst.addTransformer(new DurationTransformer());
		}
	}


OK，大功告成。不过由于Agent的特殊性，我们还需要做一些事情：

1、将DurationAgent和DurationTransformer编译打包成一个jar包，并且编写好MANIFEST：

	Premain-Class: me.arganzheng.study.instrumentation.DurationAgent
	Can-Redefine-Classes: true
	Can-Retransform-Classes: true
	

2、由于我们是在premain方法植入，所以需要用命令行中使用`-javaagent`加载agent包: 

	TestInstrumentation.java
	package me.arganzheng.study.instrumentation;

	public class TestInstrumentation {
		public static void main(String args[]) throws InterruptedException {
			HelloWorld hw = new HelloWorld();
			hw.foo();
		}
	}

java -javaagent:./instrumentation.jar me.arganzheng.study.instrumentation.TestInstrumentation.	


**TIPS**

1、Lifecycle of a javaagent

* PreMain – The PreMain method is invoked first prior to running the main method. This is the hookpoint for loading the javaagent statically at startup.
* Main – The main method is always invoked after the PreMain invocation.
* AgentMain – The AgentMain method can be invoked at any time before or after the Main method. This hookpoint is for loading the javaagent dynamically at runtime and attaching to a running process.

2、Dynamic loading of a javaagent at runtime

Dynamic loading of a javaagent at runtime can be done quite easily in a programmatic fashion but requires the sun tools jar to be present on the classpath. Certain libraries like jmockit have avoided this by opting to absorb the relevant classes from the sun tools jar into its library under the same package names.

3、BTrace可以简化这一过程，但是有些限制。


参考文档
-------

1. [Package java.lang.instrument](https://docs.oracle.com/javase/7/docs/api/java/lang/instrument/package-summary.html)
2. [Monitoring and Management Using JMX Technology](http://docs.oracle.com/javase/6/docs/technotes/guides/management/agent.html#gdhkz)
3. [Attach API and JMX export](http://blog.fastconnect.fr/?p=385) 非常深入浅出的一片文章，强烈推荐。
4. [Creation, dynamic loading and instrumentation with javaagents](http://dhruba.name/2010/02/07/creation-dynamic-loading-and-instrumentation-with-javaagents/) 强烈推荐！
5. [Java Instrumentation](http://javapapers.com/core-java/java-instrumentation/)

