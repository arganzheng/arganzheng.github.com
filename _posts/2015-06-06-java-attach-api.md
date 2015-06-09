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


实际产品
-------

如上面所介绍，Attach API允许你让一个运行中的JVM加载你的agent，然后在agent的premain或者agentmain方法中你可以通过instrumentation修改二进制代码，或者执行一些操作。一般来说都是做监控。现在有一些商业监控产品就是基于agent的模式开发的。

* [appdynamics](http://www.appdynamics.com/product/how-it-works/)
* [newrelic](http://newrelic.com/java) @see [New Relic for Java](https://docs.newrelic.com/docs/agents/java-agent/getting-started/new-relic-java)

本质上都是AOP埋点统计，但是基于Agent的好处就是不需要修改客户端代码，即使只是一个简单配置或者注解。这里以newrelic为例，说明一下它的威力。下载[newrelic-java-2.21.6.zip](https://oss.sonatype.org/content/repositories/releases/com/newrelic/agent/java/newrelic-java/2.21.6/newrelic-java-2.21.6.zip)，解压之后里面有个jar包：newrelic.jar，这个就是agent包。用JD-GUI打开，可以看到它的META-INF/MANIFEST.MF文件：

	Manifest-Version: 1.0
	Ant-Version: Apache Ant 1.8.4
	Created-By: New Relic, Inc
	Implementation-Title: New Relic Java Agent
	Implementation-Version: 2.21.6
	Built-By: jenkins
	Built-Date: Wed Apr 23 13:51:27 PDT 2014
	Can-Redefine-Classes: false
	Can-Retransform-Classes: true
	Specification-Version: 2.21.6
	Build-Id: true
	Premain-Class: com.newrelic.bootstrap.BootstrapAgent
	Main-Class: com.newrelic.agent.Agent

我们看到它Premain-Class是com.newrelic.bootstrap.BootstrapAgent，这个类做了什么呢？

	package com.newrelic.bootstrap;

	import java.io.IOException;
	import java.io.PrintStream;
	import java.lang.instrument.Instrumentation;
	import java.lang.reflect.Method;
	import java.net.URL;
	import java.net.URLClassLoader;
	import java.net.URLDecoder;
	import java.text.MessageFormat;
	import java.util.jar.JarFile;
	import java.util.zip.ZipEntry;

	public class BootstrapAgent
	{
	  private static final String NEW_RELIC_JAR_FILE = "newrelic.jar";
	  public static final String AGENT_CLASS_NAME = "com.newrelic.agent.Agent";
	  private static final String NEW_RELIC_BOOTSTRAP_CLASSPATH = "newrelic.bootstrap_classpath";
	  
	  private static JarFile getAgentJarFile(URL agentJarUrl)
	  {
	    if (agentJarUrl == null) {
	      return null;
	    }
	    try
	    {
	      return new JarFile(URLDecoder.decode(agentJarUrl.getFile(), "UTF-8"));
	    }
	    catch (IOException e) {}
	    return null;
	  }
	  
	  private static URL getAgentJarUrl()
	  {
	    ClassLoader classLoader = BootstrapAgent.class.getClassLoader();
	    if ((classLoader instanceof URLClassLoader))
	    {
	      URL[] urls = ((URLClassLoader)classLoader).getURLs();
	      for (URL url : urls) {
	        if (url.getFile().endsWith("newrelic.jar")) {
	          return url;
	        }
	      }
	      String agentClassName = BootstrapAgent.class.getName().replace('.', '/') + ".class";
	      for (URL url : urls)
	      {
	        JarFile jarFile = null;
	        try
	        {
	          jarFile = new JarFile(url.getFile());
	          ZipEntry entry = jarFile.getEntry(agentClassName);
	          if (entry != null) {
	            return url;
	          }
	        }
	        catch (IOException e) {}finally
	        {
	          if (jarFile != null) {
	            try
	            {
	              jarFile.close();
	            }
	            catch (IOException e) {}
	          }
	        }
	      }
	    }
	    return null;
	  }
	  
	  public static void premain(String agentArgs, Instrumentation inst)
	  {
	    if (Boolean.getBoolean("newrelic.bootstrap_classpath")) {
	      appendJarToBootstrapClassLoader(inst);
	    }
	    try
	    {
	      Class<?> agentClass = ClassLoader.getSystemClassLoader().loadClass("com.newrelic.agent.Agent");
	      Method premain = agentClass.getDeclaredMethod("premain", new Class[] { String.class, Instrumentation.class });
	      premain.invoke(null, new Object[] { agentArgs, inst });
	    }
	    catch (Throwable t)
	    {
	      System.err.println(MessageFormat.format("Error bootstrapping New Relic agent: {0}", new Object[] { t }));
	      t.printStackTrace();
	    }
	  }
	  
	  private static void appendJarToBootstrapClassLoader(Instrumentation inst)
	  {
	    URL agentJarUrl = getAgentJarUrl();
	    JarFile agentJarFile = getAgentJarFile(agentJarUrl);
	    if (agentJarFile != null) {
	      inst.appendToBootstrapClassLoaderSearch(agentJarFile);
	    }
	  }
	}


可以看到，它做的事情非常简单，就是判断如果系统变量newrelic.bootstrap_classpath为true，那么将自己(newrelic.jar)加入到BootstrapClassLoader的类搜索路径中(appendToBootstrapClassLoaderSearch)。然后通过反射加载和触发com.newrelic.agent.Agent的premain方法。在MANIFEST.MF文件中，我们看到com.newrelic.agent.Agent还是这个jar包的Main-class：

	package com.newrelic.agent;

	import com.newrelic.agent.config.ConfigService;
	import com.newrelic.agent.config.IAgentConfig;
	import com.newrelic.agent.install.ConfigInstaller;
	import com.newrelic.agent.logging.AgentLogManager;
	import com.newrelic.agent.logging.IAgentLogger;
	import com.newrelic.agent.service.AbstractService;
	import com.newrelic.agent.service.ServiceFactory;
	import com.newrelic.agent.service.ServiceManager;
	import com.newrelic.agent.service.ServiceManagerImpl;
	import java.io.PrintStream;
	import java.lang.instrument.Instrumentation;
	import java.net.InetAddress;
	import java.net.UnknownHostException;
	import java.text.MessageFormat;
	import java.util.ResourceBundle;
	import java.util.logging.Level;

	public final class Agent
	  extends AbstractService
	  implements IAgent
	{
	  public static final IAgentLogger LOG = ;
	  private static final String AGENT_ENABLED_PROPERTY = "newrelic.config.agent_enabled";
	  private static final boolean DEBUG = Boolean.getBoolean("newrelic.debug");
	  private static final String VERSION = initVersion();
	  private volatile boolean enabled = true;
	  private final Instrumentation instrumentation;
	  private volatile InstrumentationProxy instrumentationProxy;
	  
	  private Agent(Instrumentation instrumentation)
	  {
	    super(IAgent.class.getSimpleName());
	    this.instrumentation = instrumentation;
	  }
	  
	  protected void doStart()
	  {
	    ConfigService configService = ServiceFactory.getConfigService();
	    IAgentConfig config = configService.getDefaultAgentConfig();
	    AgentLogManager.configureLogger(config);
	    
	    logHostIp();
	    
	    this.enabled = config.isAgentEnabled();
	    if (!this.enabled) {
	      LOG.info("New Relic agent is disabled");
	    }
	    this.instrumentationProxy = InstrumentationProxy.getInstrumentationProxy(this.instrumentation);
	    
	    final long startTime = System.currentTimeMillis();
	    Runnable runnable = new Runnable()
	    {
	      public void run()
	      {
	        Agent.this.jvmShutdown(startTime);
	      }
	    };
	    Thread shutdownThread = new Thread(runnable, "New Relic JVM Shutdown");
	    Runtime.getRuntime().addShutdownHook(shutdownThread);
	  }
	  
	  private void logHostIp()
	  {
	    try
	    {
	      InetAddress address = InetAddress.getLocalHost();
	      LOG.info("Agent Host: " + address.getHostName() + " IP: " + address.getHostAddress());
	    }
	    catch (UnknownHostException e)
	    {
	      LOG.info("New Relic could not identify host/ip.");
	    }
	  }
	  
	  protected void doStop() {}
	  
	  public void shutdownAsync()
	  {
	    Runnable runnable = new Runnable()
	    {
	      public void run()
	      {
	        Agent.this.shutdown();
	      }
	    };
	    Thread shutdownThread = new Thread(runnable, "New Relic Shutdown");
	    shutdownThread.start();
	  }
	  
	  private void jvmShutdown(long startTime)
	  {
	    IAgentConfig config = ServiceFactory.getConfigService().getDefaultAgentConfig();
	    if ((config.isSendDataOnExit()) && (System.currentTimeMillis() - startTime > config.getSendDataOnExitThresholdInMillis())) {
	      ServiceFactory.getHarvestService().harvestNow();
	    }
	    getLogger().info("JVM is shutting down");
	    shutdown();
	  }
	  
	  public synchronized void shutdown()
	  {
	    try
	    {
	      ServiceFactory.getServiceManager().stop();
	      getLogger().info("New Relic Agent has shutdown");
	    }
	    catch (Exception e)
	    {
	      LOG.severe(MessageFormat.format("Error shutting down New Relic Agent: {0}", new Object[] { e }));
	    }
	  }
	  
	  public boolean isEnabled()
	  {
	    return this.enabled;
	  }
	  
	  public InstrumentationProxy getInstrumentation()
	  {
	    return this.instrumentationProxy;
	  }
	  
	  public static String getVersion()
	  {
	    return VERSION;
	  }
	  
	  private static String initVersion()
	  {
	    try
	    {
	      ResourceBundle bundle = ResourceBundle.getBundle(Agent.class.getName());
	      return bundle.getString("version");
	    }
	    catch (Throwable t) {}
	    return "0.0";
	  }
	  
	  public static boolean isDebugEnabled()
	  {
	    return DEBUG;
	  }
	  
	  public static void premain(String agentArgs, Instrumentation inst)
	  {
	    if (ServiceFactory.getServiceManager() != null)
	    {
	      System.err.println("New Relic Agent is already running! Check if more than one -javaagent switch is used on the command line");
	      return;
	    }
	    String enabled = System.getProperty("newrelic.config.agent_enabled");
	    if ((enabled != null) && (!Boolean.parseBoolean(enabled.toString())))
	    {
	      System.out.println("New Relic agent is disabled");
	      return;
	    }
	    try
	    {
	      IAgent agent = new Agent(inst);
	      ServiceManager serviceManager = new ServiceManagerImpl(agent);
	      ServiceFactory.setServiceManager(serviceManager);
	      if (ConfigInstaller.isLicenseKeyEmpty(serviceManager.getConfigService().getDefaultAgentConfig().getLicenseKey()))
	      {
	        LOG.error("license_key is empty in the config, not starting New Relic Agent");
	        return;
	      }
	      if (!serviceManager.getConfigService().getDefaultAgentConfig().isAgentEnabled())
	      {
	        LOG.info("agent_enabled is false in the config, not starting New Relic Agent");
	        return;
	      }
	      serviceManager.start();
	      LOG.info(MessageFormat.format("New Relic Agent v{0} has started", new Object[] { getVersion() }));
	      LOG.config(MessageFormat.format("Java version: {0}", new Object[] { System.getProperty("java.version") }));
	      if (Agent.class.getClassLoader() == null) {
	        LOG.info("Agent class loader is null which typically means the agent is loaded by the bootstrap class loader.");
	      } else {
	        LOG.info("Agent class loader: " + Agent.class.getClassLoader());
	      }
	    }
	    catch (Throwable t)
	    {
	      String msg = MessageFormat.format("Unable to start New Relic agent: {0}", new Object[] { t });
	      try
	      {
	        LOG.log(Level.SEVERE, msg, t);
	      }
	      catch (Throwable t2) {}
	      System.err.println(msg);
	      t.printStackTrace();
	    }
	  }
	  
	  public static void main(String[] args)
	  {
	    new AgentCommandLineParser().parseCommand(args);
	  }
	}

Agent的premain做的事情也很简单，就是创建一个	ServiceManager 然后触发它的start方法。我们继续看ServiceManagerImpl.java的start方法，其实就是启动所有内建的Service:


	public class ServiceManagerImpl extends AbstractService implements ServiceManager {
	  private final Map<String, Service> services = new HashMap();
	  private volatile ExtensionService extensionService;
	  private volatile ProfilerService profilerService;
	  private volatile TracerService tracerService;
	  private volatile TransactionTraceService transactionTraceService;
	  private volatile ThreadService threadService;
	  private volatile HarvestService harvestService;
	  private volatile Service gcService;
	  private volatile TransactionService transactionService;
	  private volatile JmxService jmxService;
	  private volatile CommandParser commandParser;
	  private volatile RPMServiceManager rpmServiceManager;
	  private volatile Service cpuSamplerService;
	  private volatile DeadlockDetectorService deadlockDetectorService;
	  private volatile SamplerService samplerService;
	  private volatile ConfigService configService;
	  private volatile RPMConnectionService rpmConnectionService;
	  private volatile EnvironmentService environmentService;
	  private volatile ClassTransformerService classTransformerService;
	  private volatile StatsService statsService;
	  private volatile SqlTraceService sqlTraceService;
	  private volatile DatabaseService databaseService;
	  private volatile BeaconService beaconService;
	  private volatile JarCollectorService jarCollectorService;
	  private volatile CacheService cacheService;
	  private volatile ServletService servletService;
	  private volatile NormalizationService normalizationService;
	  private volatile ReinstrumentService reinstrumentService;
	  private final IAgent agentService;
	  private volatile IXRaySessionService xrayService;
	  
	  public ServiceManagerImpl(IAgent agent)
	    throws ConfigurationException
	  {
	    super(ServiceManagerImpl.class.getSimpleName());
	    this.agentService = agent;
	    this.configService = ConfigServiceFactory.createConfigService();
	  }
	  
	  protected synchronized void doStart()
	    throws Exception
	  {
	    this.agentService.start();
	    
	    this.classTransformerService = new ClassTransformerServiceImpl();
	    this.jmxService = new JmxService();
	    this.extensionService = new ExtensionService();
	    this.jarCollectorService = new JarCollectorServiceImpl();
	    this.tracerService = new TracerService();
	    this.threadService = new ThreadService();
	    
	    this.cacheService = new CacheServiceImpl();
	    this.extensionService.start();
	    this.classTransformerService.start();
	    
	    boolean realAgent = this.agentService.getInstrumentation() != null;
	    
	    this.statsService = new StatsServiceImpl();
	    this.environmentService = new EnvironmentServiceImpl();
	    this.rpmConnectionService = new RPMConnectionServiceImpl();
	    this.transactionService = new TransactionService();
	    
	    this.rpmServiceManager = new RPMServiceManagerImpl();
	    this.normalizationService = new NormalizationServiceImpl();
	    this.harvestService = new HarvestServiceImpl();
	    this.gcService = (realAgent ? new GCService() : new NoopService("GC Service"));
	    this.transactionTraceService = new TransactionTraceService();
	    this.profilerService = new ProfilerService();
	    this.commandParser = new CommandParser();
	    this.cpuSamplerService = (realAgent ? new CPUSamplerService() : new NoopService("CPU Sampler"));
	    this.deadlockDetectorService = new DeadlockDetectorService();
	    this.samplerService = ((SamplerService)(realAgent ? new SamplerServiceImpl() : new NoopSamplerService()));
	    this.sqlTraceService = new SqlTraceServiceImpl();
	    this.databaseService = new DatabaseServiceImpl();
	    this.beaconService = new BeaconServiceImpl();
	    this.reinstrumentService = new ReinstrumentServiceImpl();
	    
	    this.servletService = new ServletServiceImpl();
	    this.xrayService = new XRaySessionService();
	    
	    this.threadService.start();
	    this.statsService.start();
	    this.environmentService.start();
	    this.rpmConnectionService.start();
	    this.tracerService.start();
	    this.jarCollectorService.start();
	    this.harvestService.start();
	    this.gcService.start();
	    this.transactionService.start();
	    this.transactionTraceService.start();
	    this.profilerService.start();
	    this.commandParser.start();
	    this.jmxService.start();
	    this.cpuSamplerService.start();
	    this.deadlockDetectorService.start();
	    this.samplerService.start();
	    this.sqlTraceService.start();
	    this.beaconService.start();
	    this.cacheService.start();
	    this.servletService.start();
	    this.normalizationService.start();
	    this.databaseService.start();
	    this.configService.start();
	    this.reinstrumentService.start();
	    this.xrayService.start();
	    
	    startServices();
	    
	    this.rpmServiceManager.start();
	  }
	  
	}

这些Service就是针对某些方面的具体监控实现。我们以其中几个Service为例，看一下他们的实现。比如ThreadService:

	package com.newrelic.agent;

	import com.newrelic.agent.logging.IAgentLogger;
	import com.newrelic.agent.service.AbstractService;
	import com.newrelic.agent.util.DefaultThreadFactory;
	import java.lang.management.ManagementFactory;
	import java.lang.management.ThreadMXBean;
	import java.text.MessageFormat;
	import java.util.Collections;
	import java.util.HashSet;
	import java.util.Map;
	import java.util.Map.Entry;
	import java.util.Set;
	import java.util.concurrent.ConcurrentHashMap;
	import java.util.concurrent.Executors;
	import java.util.concurrent.ScheduledExecutorService;
	import java.util.concurrent.ScheduledFuture;
	import java.util.concurrent.ThreadFactory;
	import java.util.concurrent.TimeUnit;

	public class ThreadService extends AbstractService {
	  private static final float HASH_SET_LOAD_FACTOR = 0.75F;
	  private static final String THREAD_SERVICE_THREAD_NAME = "New Relic Thread Service";
	  private static final long INITIAL_DELAY_IN_SECONDS = 300L;
	  private static final long SUBSEQUENT_DELAY_IN_SECONDS = 300L;
	  private volatile ScheduledExecutorService scheduledExecutor;
	  private volatile ScheduledFuture<?> deadThreadsTask;
	  private final Map<Long, Boolean> agentThreadIds;
	  private final Map<Long, Boolean> requestThreadIds;
	  private final Map<Long, Boolean> backgroundThreadIds;
	  private final ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
	  
	  public ThreadService()
	  {
	    super(ThreadService.class.getSimpleName());
	    this.agentThreadIds = new ConcurrentHashMap(6);
	    this.requestThreadIds = new ConcurrentHashMap();
	    this.backgroundThreadIds = new ConcurrentHashMap();
	  }
	  
	  protected void doStart()
	  {
	    if (this.threadMXBean == null) {
	      return;
	    }
	    ThreadFactory threadFactory = new DefaultThreadFactory("New Relic Thread Service", true);
	    this.scheduledExecutor = Executors.newSingleThreadScheduledExecutor(threadFactory);
	    Runnable runnable = new Runnable()
	    {
	      public void run()
	      {
	        try
	        {
	          ThreadService.this.detectDeadThreads();
	        }
	        catch (Throwable t)
	        {
	          String msg = MessageFormat.format("Unexpected exception detecting dead threads: {0}", new Object[] { t.toString() });
	          ThreadService.this.getLogger().warning(msg);
	        }
	      }
	    };
	    this.deadThreadsTask = this.scheduledExecutor.scheduleWithFixedDelay(runnable, 300L, 300L, TimeUnit.SECONDS);
	  }
	  
	  protected void doStop()
	  {
	    if (this.deadThreadsTask != null) {
	      this.deadThreadsTask.cancel(false);
	    }
	    this.scheduledExecutor.shutdown();
	  }
	  
	  protected void detectDeadThreads()
	  {
	    long[] threadIds = this.threadMXBean.getAllThreadIds();
	    int hashSetSize = (int)(threadIds.length / 0.75F) + 1;
	    Set<Long> ids = new HashSet(hashSetSize);
	    for (long threadId : threadIds) {
	      ids.add(Long.valueOf(threadId));
	    }
	    retainAll(this.requestThreadIds, ids);
	    retainAll(this.backgroundThreadIds, ids);
	  }
	  
	  private void retainAll(Map<Long, Boolean> map, Set<Long> ids)
	  {
	    for (Map.Entry<Long, Boolean> entry : map.entrySet()) {
	      if (!ids.contains(entry.getKey())) {
	        map.remove(entry.getKey());
	      }
	    }
	  }
	  
	  ...
	}

其实它就是主要从private final ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean(); 中获取Thread信息。另外还启了一个线程做死锁检测。

再来看看TraceService，这个要复杂的多。看名称就是对方法调用进行跟踪。那么它是怎么实现的。

	package com.newrelic.agent;

	import com.newrelic.agent.extension.ConfigurationConstruct;
	import com.newrelic.agent.extension.ExtensionService;
	import com.newrelic.agent.instrumentation.yaml.PointCutFactory;
	import com.newrelic.agent.service.AbstractService;
	import com.newrelic.agent.service.ServiceFactory;
	import com.newrelic.agent.service.ServiceManager;
	import com.newrelic.agent.tracers.ClassMethodSignature;
	import com.newrelic.agent.tracers.IgnoreTransactionTracerFactory;
	import com.newrelic.agent.tracers.PointCutInvocationHandler;
	import com.newrelic.agent.tracers.RetryException;
	import com.newrelic.agent.tracers.Tracer;
	import com.newrelic.agent.tracers.TracerFactory;
	import java.util.List;
	import java.util.Map;
	import java.util.concurrent.ConcurrentHashMap;

	public class TracerService
	  extends AbstractService
	{
	  private final Map<String, TracerFactory> tracerFactories = new ConcurrentHashMap();
	  private volatile PointCutInvocationHandler[] invocationHandlers = new PointCutInvocationHandler[0];
	  public ITracerService tracerServiceFactory;
	  
	  public TracerService()
	  {
	    super(TracerService.class.getSimpleName());
	    registerTracerFactory(IgnoreTransactionTracerFactory.TRACER_FACTORY_NAME, new IgnoreTransactionTracerFactory());
	    
	    ExtensionService extensionService = ServiceFactory.getExtensionService();
	    for (ConfigurationConstruct construct : PointCutFactory.getConstructs()) {
	      extensionService.addConstruct(construct);
	    }
	    this.tracerServiceFactory = new NoOpTracerService(null);
	  }
	  
	  public Tracer getTracer(TracerFactory tracerFactory, ClassMethodSignature signature, Object object, Object... args)
	  {
	    if (tracerFactory == null) {
	      return null;
	    }
	    return this.tracerServiceFactory.getTracer(tracerFactory, signature, object, args);
	  }
	  
	  public TracerFactory getTracerFactory(String tracerFactoryName)
	  {
	    return (TracerFactory)this.tracerFactories.get(tracerFactoryName);
	  }
	  
	  public void registerTracerFactory(String name, TracerFactory tracerFactory)
	  {
	    this.tracerFactories.put(name.intern(), tracerFactory);
	  }
	  
	  public void registerInvocationHandlers(List<PointCutInvocationHandler> handlers)
	  {
	    if (this.invocationHandlers == null)
	    {
	      this.invocationHandlers = ((PointCutInvocationHandler[])handlers.toArray(new PointCutInvocationHandler[handlers.size()]));
	    }
	    else
	    {
	      PointCutInvocationHandler[] arrayToSwap = new PointCutInvocationHandler[this.invocationHandlers.length + handlers.size()];
	      
	      System.arraycopy(this.invocationHandlers, 0, arrayToSwap, 0, this.invocationHandlers.length);
	      System.arraycopy(handlers.toArray(), 0, arrayToSwap, this.invocationHandlers.length, handlers.size());
	      this.invocationHandlers = arrayToSwap;
	    }
	  }
	  
	  public int getInvocationHandlerId(PointCutInvocationHandler handler)
	  {
	    for (int i = 0; i < this.invocationHandlers.length; i++) {
	      if (this.invocationHandlers[i] == handler) {
	        return i;
	      }
	    }
	    return -1;
	  }
	  
	  public PointCutInvocationHandler getInvocationHandler(int id)
	  {
	    return this.invocationHandlers[id];
	  }
	  
	  protected void doStart() {}
	  
	  protected void doStop()
	  {
	    this.tracerFactories.clear();
	  }
	  
	  public boolean isEnabled()
	  {
	    return true;
	  }
	  
	  private static abstract interface ITracerService
	  {
	    public abstract Tracer getTracer(TracerFactory paramTracerFactory, ClassMethodSignature paramClassMethodSignature, Object paramObject, Object... paramVarArgs);
	  }
	  
	  private class NoOpTracerService
	    implements TracerService.ITracerService
	  {
	    private NoOpTracerService() {}
	    
	    public Tracer getTracer(TracerFactory tracerFactory, ClassMethodSignature signature, Object object, Object... args)
	    {
	      if (ServiceFactory.getServiceManager().isStarted())
	      {
	        TracerService.this.tracerServiceFactory = new TracerService.TracerServiceImpl(TracerService.this, null);
	        return TracerService.this.tracerServiceFactory.getTracer(tracerFactory, signature, object, args);
	      }
	      return null;
	    }
	  }
	  
	  private class TracerServiceImpl
	    implements TracerService.ITracerService
	  {
	    private TracerServiceImpl() {}
	    
	    public Tracer getTracer(TracerFactory tracerFactory, ClassMethodSignature signature, Object object, Object... args)
	    {
	      Transaction transaction = Transaction.getTransaction();
	      if (transaction == null) {
	        return null;
	      }
	      try
	      {
	        return transaction.getTransactionState().getTracer(transaction, tracerFactory, signature, object, args);
	      }
	      catch (RetryException e) {}
	      return getTracer(tracerFactory, signature, object, args);
	    }
	  }
	}

这个类最主要是根据方法前面得到相应的Tracer：public Tracer getTracer(TracerFactory tracerFactory, ClassMethodSignature signature, Object object, Object... args)。Tracer是自定义的抽象接口：

	package com.newrelic.agent.tracers;

	import com.newrelic.agent.config.ITransactionTracerConfig;
	import com.newrelic.agent.database.SqlObfuscator;
	import com.newrelic.agent.trace.TransactionSegment;
	import java.lang.reflect.InvocationHandler;
	import java.util.Collection;
	import java.util.Map;

	public abstract interface Tracer
	  extends InvocationHandler, TimedItem, ExitTracer
	{
	  public abstract long getStartTime();
	  
	  public abstract long getStartTimeInMilliseconds();
	  
	  public abstract long getEndTime();
	  
	  public abstract long getEndTimeInMilliseconds();
	  
	  public abstract long getExclusiveDuration();
	  
	  public abstract long getRunningDurationInNanos();
	  
	  public abstract String getMetricName();
	  
	  public abstract String getTransactionSegmentName();
	  
	  public abstract String getTransactionSegmentUri();
	  
	  public abstract Map<String, Object> getParameters();
	  
	  public abstract Throwable getThrowable();
	  
	  public abstract void childTracerFinished(Tracer paramTracer);
	  
	  public abstract Collection<Tracer> getChildren();
	  
	  public abstract Tracer getParentTracer();
	  
	  public abstract boolean isParent();
	  
	  public abstract boolean isMetricProducer();
	  
	  public abstract ClassMethodSignature getClassMethodSignature();
	  
	  public abstract boolean isTransactionSegment();
	  
	  public abstract boolean isChildHasStackTrace();
	  
	  public abstract TransactionSegment getTransactionSegment(ITransactionTracerConfig paramITransactionTracerConfig, SqlObfuscator paramSqlObfuscator, long paramLong, TransactionSegment paramTransactionSegment);
	}

主要注意的是它继承了[InvocationHandler](http://docs.oracle.com/javase/7/docs/api/java/lang/reflect/InvocationHandler.html)接口，意味着它是能够对方法实现动态代理进行AOP。但是对于没有实现接口的类就无法做代理了。

在InvocationPoint.java中就会调用到：

	package com.newrelic.agent.instrumentation;

	import com.newrelic.agent.Agent;
	import com.newrelic.agent.TracerService;
	import com.newrelic.agent.Transaction;
	import com.newrelic.agent.logging.IAgentLogger;
	import com.newrelic.agent.tracers.ClassMethodSignature;
	import com.newrelic.agent.tracers.Dispatcher;
	import com.newrelic.agent.tracers.EntryInvocationHandler;
	import com.newrelic.agent.tracers.PointCutInvocationHandler;
	import com.newrelic.agent.tracers.TracerFactory;
	import com.newrelic.agent.tracers.WebRequestDispatcher;
	import java.lang.reflect.InvocationHandler;
	import java.lang.reflect.Method;
	import java.text.MessageFormat;
	import java.util.logging.Level;

	final class InvocationPoint
	  implements InvocationHandler
	{
	  private final TracerFactory tracerFactory;
	  private final ClassMethodSignature classMethodSignature;
	  private final TracerService tracerService;
	  private final boolean ignoreApdex;
	  
	  public InvocationPoint(TracerService tracerService, ClassMethodSignature classMethodSignature, TracerFactory tracerFactory, boolean ignoreApdex)
	  {
	    this.tracerService = tracerService;
	    this.tracerFactory = tracerFactory;
	    this.classMethodSignature = classMethodSignature;
	    this.ignoreApdex = ignoreApdex;
	  }
	  
	  public ClassMethodSignature getClassMethodSignature()
	  {
	    return this.classMethodSignature;
	  }
	  
	  public Object invoke(Object proxy, Method method, Object[] args)
	    throws Throwable
	  {
	    try
	    {
	      if (this.ignoreApdex)
	      {
	        Transaction transaction = Transaction.getTransaction();
	        if (transaction != null)
	        {
	          Dispatcher dispatcher = transaction.getDispatcher();
	          if ((dispatcher instanceof WebRequestDispatcher))
	          {
	            ((WebRequestDispatcher)dispatcher).setIgnoreApdex(true);
	            if (Agent.LOG.isLoggable(Level.FINER))
	            {
	              String msg = MessageFormat.format("Set Ignore apdex to \"{0}\"", new Object[] { Boolean.valueOf(true) });
	              Agent.LOG.log(Level.FINER, msg, new Exception());
	            }
	          }
	        }
	      }
	      Object t = this.tracerService.getTracer(this.tracerFactory, this.classMethodSignature, args[0], (Object[])args[1]);
	      if (t == null) {}
	      return null;
	    }
	    catch (Throwable t)
	    {
	      Agent.LOG.log(Level.FINEST, "Tracer invocation error", t);
	    }
	    return null;
	  }
	  
	  public String toString()
	  {
	    return MessageFormat.format("{0} {1}", new Object[] { this.classMethodSignature });
	  }
	  
	  public TracerFactory getTracerFactory()
	  {
	    return this.tracerFactory;
	  }
	  
	  public static InvocationHandler getStacklessInvocationHandler(ClassMethodSignature classMethodSignature, EntryInvocationHandler tracerFactory)
	  {
	    return new StacklessInvocationPoint(classMethodSignature, tracerFactory);
	  }
	  
	  private static final class StacklessInvocationPoint
	    implements InvocationHandler
	  {
	    private final ClassMethodSignature classMethodSignature;
	    private final EntryInvocationHandler tracerFactory;
	    
	    public StacklessInvocationPoint(ClassMethodSignature classMethodSignature, EntryInvocationHandler tracerFactory)
	    {
	      this.classMethodSignature = classMethodSignature;
	      this.tracerFactory = tracerFactory;
	    }
	    
	    public Object invoke(Object proxy, Method method, Object[] args)
	      throws Throwable
	    {
	      this.tracerFactory.handleInvocation(this.classMethodSignature, args[0], (Object[])args[1]);
	      return null;
	    }
	  }
	  
	  public static InvocationHandler getInvocationPoint(PointCutInvocationHandler invocationHandler, TracerService tracerService, ClassMethodSignature classMethodSignature, boolean ignoreApdex)
	  {
	    if ((invocationHandler instanceof EntryInvocationHandler)) {
	      return getStacklessInvocationHandler(classMethodSignature, (EntryInvocationHandler)invocationHandler);
	    }
	    if ((invocationHandler instanceof TracerFactory)) {
	      return new InvocationPoint(tracerService, classMethodSignature.intern(), (TracerFactory)invocationHandler, ignoreApdex);
	    }
	    Agent.LOG.finest("Unable to create an invocation handler for " + invocationHandler);
	    if (ignoreApdex) {
	      return IgnoreApdexInvocationHandler.INVOCATION_HANDLER;
	    }
	    return NoOpInvocationHandler.INVOCATION_HANDLER;
	  }
	}


代码非常多。这里就不展开了。

但是其实使用attach API和auto instrumentation的作用在于尽量减少对客户端的倾入，保持对客户端代码(被监控代码)的透明。对于做成框架或者产品的可能有必要，但是对于内部使用的系统，其实没有太大必要花费这个人力，因为auto instrumentation需要对目标代码具体源码熟悉的情况下才能进行，其实有点case by case的味道。增加监控依赖，配置监控逻辑或者使用注解，或者调用监控API，其实都是可以接受的，也是必须的[Java instrumentation by annotation](https://docs.newrelic.com/docs/agents/java-agent/custom-instrumentation/java-instrumentation-annotation)。


参考文档
-------

1. [Package java.lang.instrument](https://docs.oracle.com/javase/7/docs/api/java/lang/instrument/package-summary.html)
2. [Monitoring and Management Using JMX Technology](http://docs.oracle.com/javase/6/docs/technotes/guides/management/agent.html#gdhkz)
3. [Attach API and JMX export](http://blog.fastconnect.fr/?p=385) 非常深入浅出的一片文章，强烈推荐。
4. [Creation, dynamic loading and instrumentation with javaagents](http://dhruba.name/2010/02/07/creation-dynamic-loading-and-instrumentation-with-javaagents/) 强烈推荐！
5. [Java Instrumentation](http://javapapers.com/core-java/java-instrumentation/)

