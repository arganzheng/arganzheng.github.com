---
title: Java中如何正确的加载配置文件
layout: post
---


取决于配置文件的位置，有不同的获取方式：

1. 文件系统
	* 绝对路径
	* 相对路径
2. classpath：java特有的

对于文件系统路径，那么简单通过File得到相应的文件，进行加载就可以了。

对于classpath中的文件，那么就需要使用ClassLoader进行加载的了。

在Java中主要有如下方法：

* URL java.lang.ClassLoader.getResource(String name)
* URL java.lang.ClassLoader.findResource(String name)
* URL java.lang.ClassLoader.getSystemResource(String name)

我们可以看一下相关的实现，其实非常简单：

	package java.lang;

	public abstract class ClassLoader {

	    /**
	     * Finds the resource with the given name.  A resource is some data
	     * (images, audio, text, etc) that can be accessed by class code in a way
	     * that is independent of the location of the code.
	     *
	     * <p> The name of a resource is a '<tt>/</tt>'-separated path name that
	     * identifies the resource.
	     *
	     * <p> This method will first search the parent class loader for the
	     * resource; if the parent is <tt>null</tt> the path of the class loader
	     * built-in to the virtual machine is searched.  That failing, this method
	     * will invoke {@link #findResource(String)} to find the resource.  </p>
	     *
	     * @apiNote When overriding this method it is recommended that an
	     * implementation ensures that any delegation is consistent with the {@link
	     * #getResources(java.lang.String) getResources(String)} method.
	     *
	     * @param  name
	     *         The resource name
	     *
	     * @return  A <tt>URL</tt> object for reading the resource, or
	     *          <tt>null</tt> if the resource could not be found or the invoker
	     *          doesn't have adequate  privileges to get the resource.
	     *
	     * @since  1.1
	     */
	    public URL getResource(String name) {
	        URL url;
	        if (parent != null) {
	            url = parent.getResource(name);
	        } else {
	            url = getBootstrapResource(name);
	        }
	        if (url == null) {
	            url = findResource(name);
	        }
	        return url;
	    }

	    /**
	     * Find resources from the VM's built-in classloader.
	     */
	    private static URL getBootstrapResource(String name) {
	        URLClassPath ucp = getBootstrapClassPath();
	        Resource res = ucp.getResource(name);
	        return res != null ? res.getURL() : null;
	    }

	    /**
	     * Find a resource of the specified name from the search path used to load
	     * classes.  This method locates the resource through the system class
	     * loader (see {@link #getSystemClassLoader()}).
	     *
	     * @param  name
	     *         The resource name
	     *
	     * @return  A {@link java.net.URL <tt>URL</tt>} object for reading the
	     *          resource, or <tt>null</tt> if the resource could not be found
	     *
	     * @since  1.1
	     */
	    public static URL getSystemResource(String name) {
	        ClassLoader system = getSystemClassLoader();
	        if (system == null) {
	            return getBootstrapResource(name);
	        }
	        return system.getResource(name);
	    }
	}
	    
需要注意的是使用ClassLoader加载资源文件，同样是默认遵循双亲委派模式的。


实际例子
--------

很多开源的java项目都是这样子加载配置文件的，下面我们举两个例子说明一下。

### log4j

在LogManager.java中：

	URL url = Loader.getResource("log4j.xml");

其中Loader.java定义如下：
	
	package org.apache.log4j.helpers;

	/**
	 * Load resources (or images) from various sources
	 */
	public class Loader  { 

	  static final String TSTR = "Caught Exception while in Loader.getResource. This may be innocuous.";

	  // We conservatively assume that we are running under Java 1.x
	  static private boolean java1 = true;
	  
	  static private boolean ignoreTCL = false;
	  
	  static {
	    String prop = OptionConverter.getSystemProperty("java.version", null);
	    
	    if(prop != null) {
	      int i = prop.indexOf('.');
	      if(i != -1) {	
		if(prop.charAt(i+1) != '1')
		  java1 = false;
	      } 
	    }
	    String ignoreTCLProp = OptionConverter.getSystemProperty("log4j.ignoreTCL", null);
	    if(ignoreTCLProp != null) {
	      ignoreTCL = OptionConverter.toBoolean(ignoreTCLProp, true);      
	    }   
	  }
	  

	  /**
	     This method will search for <code>resource</code> in different
	     places. The search order is as follows:

	     <ol>

	     <p><li>Search for <code>resource</code> using the thread context
	     class loader under Java2. If that fails, search for
	     <code>resource</code> using the class loader that loaded this
	     class (<code>Loader</code>). Under JDK 1.1, only the the class
	     loader that loaded this class (<code>Loader</code>) is used.

	     <p><li>Try one last time with
	     <code>ClassLoader.getSystemResource(resource)</code>, that is is
	     using the system class loader in JDK 1.2 and virtual machine's
	     built-in class loader in JDK 1.1.

	     </ol>
	  */
	  static public URL getResource(String resource) {
	    ClassLoader classLoader = null;
	    URL url = null;
	    
	    try {
	  	if(!java1 && !ignoreTCL) {
	  	  classLoader = getTCL();
	  	  if(classLoader != null) {
	  	    LogLog.debug("Trying to find ["+resource+"] using context classloader "
	  			 +classLoader+".");
	  	    url = classLoader.getResource(resource);      
	  	    if(url != null) {
	  	      return url;
	  	    }
	  	  }
	  	}
	  	
	  	// We could not find resource. Ler us now try with the
	  	// classloader that loaded this class.
	  	classLoader = Loader.class.getClassLoader(); 
	  	if(classLoader != null) {
	  	  LogLog.debug("Trying to find ["+resource+"] using "+classLoader
	  		       +" class loader.");
	  	  url = classLoader.getResource(resource);
	  	  if(url != null) {
	  	    return url;
	  	  }
	  	}
	    } catch(IllegalAccessException t) {
	        LogLog.warn(TSTR, t);
	    } catch(InvocationTargetException t) {
	        if (t.getTargetException() instanceof InterruptedException
	                || t.getTargetException() instanceof InterruptedIOException) {
	            Thread.currentThread().interrupt();
	        }
	        LogLog.warn(TSTR, t);
	    } catch(Throwable t) {
	      //  can't be InterruptedException or InterruptedIOException
	      //    since not declared, must be error or RuntimeError.
	      LogLog.warn(TSTR, t);
	    }
	    
	    // Last ditch attempt: get the resource from the class path. It
	    // may be the case that clazz was loaded by the Extentsion class
	    // loader which the parent of the system class loader. Hence the
	    // code below.
	    LogLog.debug("Trying to find ["+resource+
	  		   "] using ClassLoader.getSystemResource().");
	    return ClassLoader.getSystemResource(resource);
	  } 
	  
	  /**
	     Are we running under JDK 1.x?        
	  */
	  public static boolean isJava1() {
	    return java1;
	  }
	  
	  /**
	    * Get the Thread Context Loader which is a JDK 1.2 feature. If we
	    * are running under JDK 1.1 or anything else goes wrong the method
	    * returns <code>null<code>.
	    *
	    *  */
	  private static ClassLoader getTCL() throws IllegalAccessException, 
	    InvocationTargetException {

	    // Are we running on a JDK 1.2 or later system?
	    Method method = null;
	    try {
	      method = Thread.class.getMethod("getContextClassLoader", null);
	    } catch (NoSuchMethodException e) {
	      // We are running on JDK 1.1
	      return null;
	    }
	    
	    return (ClassLoader) method.invoke(Thread.currentThread(), null);
	  }

### [ehcache](http://ehcache.org/)

配置文件：

	<!-- *******************************
	     ***** CACHE CONFIGURATION *****
	     ******************************* -->                
	<bean id="cacheManager" class="org.springframework.cache.ehcache.EhCacheCacheManager">
	    <property name="cacheManager" ref="ehcache"/>
	</bean>
	<bean id="ehcache" class="org.springframework.cache.ehcache.EhCacheManagerFactoryBean">
	    <property name="configLocation" value="classpath:config/ehcache.xml"/>
	    <property name="shared" value="true"/>
	</bean>

具体实现：

	public class EhCacheManagerFactoryBean implements FactoryBean<CacheManager>, InitializingBean, DisposableBean {
		private Resource configLocation;
		/**
		 * Set the location of the EhCache config file. A typical value is "/WEB-INF/ehcache.xml".
		 * <p>Default is "ehcache.xml" in the root of the class path, or if not found,
		 * "ehcache-failsafe.xml" in the EhCache jar (default EhCache initialization).
		 * @see net.sf.ehcache.CacheManager#create(java.io.InputStream)
		 * @see net.sf.ehcache.CacheManager#CacheManager(java.io.InputStream)
		 */
		public void setConfigLocation(Resource configLocation) {
			this.configLocation = configLocation;
		}

		@Override
		public void afterPropertiesSet() throws CacheException, IOException {
			logger.info("Initializing EhCache CacheManager");
			InputStream is = (this.configLocation != null ? this.configLocation.getInputStream() : null);
			try {
				Configuration configuration = (is != null ?
						ConfigurationFactory.parseConfiguration(is) : ConfigurationFactory.parseConfiguration());
				if (this.cacheManagerName != null) {
					configuration.setName(this.cacheManagerName);
				}
				if (this.shared) {
					// Old-school EhCache singleton sharing...
					// No way to find out whether we actually created a new CacheManager
					// or just received an existing singleton reference.
					this.cacheManager = CacheManager.create(configuration);
				}
				else if (this.acceptExisting) {
					// EhCache 2.5+: Reusing an existing CacheManager of the same name.
					// Basically the same code as in CacheManager.getInstance(String),
					// just storing whether we're dealing with an existing instance.
					synchronized (CacheManager.class) {
						this.cacheManager = CacheManager.getCacheManager(this.cacheManagerName);
						if (this.cacheManager == null) {
							this.cacheManager = new CacheManager(configuration);
						}
						else {
							this.locallyManaged = false;
						}
					}
				}
				else {
					// Throwing an exception if a CacheManager of the same name exists already...
					this.cacheManager = new CacheManager(configuration);
				}
			}
			finally {
				if (is != null) {
					is.close();
				}
			}
		}
	}

关键是这一段：

	InputStream is = (this.configLocation != null ? this.configLocation.getInputStream() : null);
	Configuration configuration = (is != null ?
						ConfigurationFactory.parseConfiguration(is) : ConfigurationFactory.parseConfiguration());

我们继续往下看：

	package net.sf.ehcache.config;

	public final class ConfigurationFactory {

	    private static final String DEFAULT_CLASSPATH_CONFIGURATION_FILE = "/ehcache.xml";
	    private static final String FAILSAFE_CLASSPATH_CONFIGURATION_FILE = "/ehcache-failsafe.xml";

	    /**
	     * Configures a bean from an XML input stream.
	     */
	    public static Configuration parseConfiguration(final InputStream inputStream) throws CacheException {

	        LOG.debug("Configuring ehcache from InputStream");

	        Configuration configuration = new Configuration();
	        try {
	            InputStream translatedInputStream = translateSystemProperties(inputStream);
	            final SAXParser parser = SAXParserFactory.newInstance().newSAXParser();
	            final BeanHandler handler = new BeanHandler(configuration);
	            parser.parse(translatedInputStream, handler);
	        } catch (Exception e) {
	            throw new CacheException("Error configuring from input stream. Initial cause was " + e.getMessage(), e);
	        }
	        configuration.setSource(ConfigurationSource.getConfigurationSource(inputStream));
	        return configuration;
	    }

	    /**
	     * Configures a bean from an XML file in the classpath.
	     */
	    public static Configuration parseConfiguration() throws CacheException {
	        ClassLoader standardClassloader = ClassLoaderUtil.getStandardClassLoader();
	        URL url = null;
	        if (standardClassloader != null) {
	            url = standardClassloader.getResource(DEFAULT_CLASSPATH_CONFIGURATION_FILE);
	        }
	        if (url == null) {
	            url = ConfigurationFactory.class.getResource(DEFAULT_CLASSPATH_CONFIGURATION_FILE);
	        }
	        if (url != null) {
	            LOG.debug("Configuring ehcache from ehcache.xml found in the classpath: " + url);
	        } else {
	            url = ConfigurationFactory.class.getResource(FAILSAFE_CLASSPATH_CONFIGURATION_FILE);

	            LOG.warn("No configuration found. Configuring ehcache from ehcache-failsafe.xml "
	                    + " found in the classpath: {}", url);

	        }
	        Configuration configuration = parseConfiguration(url);
	        configuration.setSource(ConfigurationSource.getConfigurationSource());
	        return configuration;
	    }
	}

但是细心的读者可能注意到其实我们对于configLocation其实是配置从classpath中加载的：

	<property name="configLocation" value="classpath:config/ehcache.xml"/>

对应的java属性是

	private Resource configLocation;

然后是调用configLocation.getInputStream()就能够找到 classpath中的 config/ehcache.xml 这个配置文件。这个逻辑其实是Spring封装的资源加载器概念。Spring的资源加载器提供了一个非常通用的getResource()方法可以让我们从文件系统, classpath或者URL中加载文件。

* ResourceLoader
	* DefaultResourceLoader
		* AbstractApplicationContext
			* AbstractRefreshableApplicationContext
				* AbstractRefreshableConfigApplicationContext
					* AbstractXmlApplicationContext
						* ClassPathXmlApplicationContext
						* FileSystemXmlApplicationContext
			* GenericApplicationContext
				* AnnotationConfigApplicationContext
				* GenericGroovyApplicationContext
				* GenericXmlApplicationContext
				* ResourceAdapterApplicationContext
				* StaticApplicationContext
		* ClassRelativeResourceLoader
		* FileSystemResourceLoader
	* ResourcePatternResolver
		* PathMatchingResourcePatternResolver
		* ApplicationContext
			* ConfigurableApplicationContext
				* AbstractApplicationContext
					* AbstractRefreshableApplicationContext
						* AbstractRefreshableConfigApplicationContext
							* AbstractXmlApplicationContext
								* ClassPathXmlApplicationContext
								* FileSystemXmlApplicationContext
					* GenericApplicationContext
						* AnnotationConfigApplicationContext
						* GenericGroovyApplicationContext
						* GenericXmlApplicationContext
						* ResourceAdapterApplicationContext
						* StaticApplicationContext

可以看到ApplicationContext也实现ResourceLoader接口，所以我们可以在ApplicationContext中使用资源加载器：

1、File system

	Resource resource = appContext.getResource("file:c:\\testing.txt");

2、URL path
	
	Resource resource = appContext.getResource("url:http://www.yourdomain.com/testing.txt");

3、Class path

	Resource resource = appContext.getResource("classpath:com/mkyong/common/testing.txt");

下面我们来看看其中最重要的两个类：

	package org.springframework.core.io;

	import org.springframework.util.ResourceUtils;

	/**
	 * Strategy interface for loading resources (e.. class path or file system
	 * resources). An {@link org.springframework.context.ApplicationContext}
	 * is required to provide this functionality, plus extended
	 * {@link org.springframework.core.io.support.ResourcePatternResolver} support.
	 *
	 * <p>{@link DefaultResourceLoader} is a standalone implementation that is
	 * usable outside an ApplicationContext, also used by {@link ResourceEditor}.
	 *
	 * <p>Bean properties of type Resource and Resource array can be populated
	 * from Strings when running in an ApplicationContext, using the particular
	 * context's resource loading strategy.
	 *
	 * @see Resource
	 * @see org.springframework.core.io.support.ResourcePatternResolver
	 * @see org.springframework.context.ApplicationContext
	 * @see org.springframework.context.ResourceLoaderAware
	 */
	public interface ResourceLoader {

		/** Pseudo URL prefix for loading from the class path: "classpath:" */
		String CLASSPATH_URL_PREFIX = ResourceUtils.CLASSPATH_URL_PREFIX;


		/**
		 * Return a Resource handle for the specified resource.
		 * The handle should always be a reusable resource descriptor,
		 * allowing for multiple {@link Resource#getInputStream()} calls.
		 * <p><ul>
		 * <li>Must support fully qualified URLs, e.g. "file:C:/test.dat".
		 * <li>Must support classpath pseudo-URLs, e.g. "classpath:test.dat".
		 * <li>Should support relative file paths, e.g. "WEB-INF/test.dat".
		 * (This will be implementation-specific, typically provided by an
		 * ApplicationContext implementation.)
		 * </ul>
		 * <p>Note that a Resource handle does not imply an existing resource;
		 * you need to invoke {@link Resource#exists} to check for existence.
		 * @param location the resource location
		 * @return a corresponding Resource handle
		 * @see #CLASSPATH_URL_PREFIX
		 * @see org.springframework.core.io.Resource#exists
		 * @see org.springframework.core.io.Resource#getInputStream
		 */
		Resource getResource(String location);

		/**
		 * Expose the ClassLoader used by this ResourceLoader.
		 * <p>Clients which need to access the ClassLoader directly can do so
		 * in a uniform manner with the ResourceLoader, rather than relying
		 * on the thread context ClassLoader.
		 * @return the ClassLoader (only {@code null} if even the system
		 * ClassLoader isn't accessible)
		 * @see org.springframework.util.ClassUtils#getDefaultClassLoader()
		 */
		ClassLoader getClassLoader();

	}

我们来看看他的默认实现：

	package org.springframework.core.io;

	import java.net.MalformedURLException;
	import java.net.URL;

	import org.springframework.util.Assert;
	import org.springframework.util.ClassUtils;
	import org.springframework.util.StringUtils;

	/**
	 * Default implementation of the {@link ResourceLoader} interface.
	 * Used by {@link ResourceEditor}, and serves as base class for
	 * {@link org.springframework.context.support.AbstractApplicationContext}.
	 * Can also be used standalone.
	 *
	 * <p>Will return a {@link UrlResource} if the location value is a URL,
	 * and a {@link ClassPathResource} if it is a non-URL path or a
	 * "classpath:" pseudo-URL.
	 *
	 * @see FileSystemResourceLoader
	 * @see org.springframework.context.support.ClassPathXmlApplicationContext
	 */
	public class DefaultResourceLoader implements ResourceLoader {

		private ClassLoader classLoader;


		/**
		 * Create a new DefaultResourceLoader.
		 * <p>ClassLoader access will happen using the thread context class loader
		 * at the time of this ResourceLoader's initialization.
		 * @see java.lang.Thread#getContextClassLoader()
		 */
		public DefaultResourceLoader() {
			this.classLoader = ClassUtils.getDefaultClassLoader();
		}

		/**
		 * Create a new DefaultResourceLoader.
		 * @param classLoader the ClassLoader to load class path resources with, or {@code null}
		 * for using the thread context class loader at the time of actual resource access
		 */
		public DefaultResourceLoader(ClassLoader classLoader) {
			this.classLoader = classLoader;
		}


		/**
		 * Specify the ClassLoader to load class path resources with, or {@code null}
		 * for using the thread context class loader at the time of actual resource access.
		 * <p>The default is that ClassLoader access will happen using the thread context
		 * class loader at the time of this ResourceLoader's initialization.
		 */
		public void setClassLoader(ClassLoader classLoader) {
			this.classLoader = classLoader;
		}

		/**
		 * Return the ClassLoader to load class path resources with.
		 * <p>Will get passed to ClassPathResource's constructor for all
		 * ClassPathResource objects created by this resource loader.
		 * @see ClassPathResource
		 */
		@Override
		public ClassLoader getClassLoader() {
			return (this.classLoader != null ? this.classLoader : ClassUtils.getDefaultClassLoader());
		}


		@Override
		public Resource getResource(String location) {
			Assert.notNull(location, "Location must not be null");
			if (location.startsWith("/")) {
				return getResourceByPath(location);
			}
			else if (location.startsWith(CLASSPATH_URL_PREFIX)) {
				return new ClassPathResource(location.substring(CLASSPATH_URL_PREFIX.length()), getClassLoader());
			}
			else {
				try {
					// Try to parse the location as a URL...
					URL url = new URL(location);
					return new UrlResource(url);
				}
				catch (MalformedURLException ex) {
					// No URL -> resolve as resource path.
					return getResourceByPath(location);
				}
			}
		}

		/**
		 * Return a Resource handle for the resource at the given path.
		 * <p>The default implementation supports class path locations. This should
		 * be appropriate for standalone implementations but can be overridden,
		 * e.g. for implementations targeted at a Servlet container.
		 * @param path the path to the resource
		 * @return the corresponding Resource handle
		 * @see ClassPathResource
		 * @see org.springframework.context.support.FileSystemXmlApplicationContext#getResourceByPath
		 * @see org.springframework.web.context.support.XmlWebApplicationContext#getResourceByPath
		 */
		protected Resource getResourceByPath(String path) {
			return new ClassPathContextResource(path, getClassLoader());
		}


		/**
		 * ClassPathResource that explicitly expresses a context-relative path
		 * through implementing the ContextResource interface.
		 */
		protected static class ClassPathContextResource extends ClassPathResource implements ContextResource {

			public ClassPathContextResource(String path, ClassLoader classLoader) {
				super(path, classLoader);
			}

			@Override
			public String getPathWithinContext() {
				return getPath();
			}

			@Override
			public Resource createRelative(String relativePath) {
				String pathToUse = StringUtils.applyRelativePath(getPath(), relativePath);
				return new ClassPathContextResource(pathToUse, getClassLoader());
			}
		}

	}

其中最重要的方法就是 Resource getResource(String location); 

	@Override
	public Resource getResource(String location) {
		Assert.notNull(location, "Location must not be null");
		if (location.startsWith("/")) {
			return getResourceByPath(location);
		}
		else if (location.startsWith(CLASSPATH_URL_PREFIX)) {
			return new ClassPathResource(location.substring(CLASSPATH_URL_PREFIX.length()), getClassLoader());
		}
		else {
			try {
				// Try to parse the location as a URL...
				URL url = new URL(location);
				return new UrlResource(url);
			}
			catch (MalformedURLException ex) {
				// No URL -> resolve as resource path.
				return getResourceByPath(location);
			}
		}
	}

如果location是以'/'开头，那么就是取决于子类的实现，默认实现是从classpath中加载。如果是以'classpath:'开头，那么就是从Classpath中加载，否则从URL中加载。然后我们看到其实这个类其实也不做具体的加载，它只是根据location的格式对资源进行定位，创建相应的Resource类而已。具体的加载其实是在Resource类本身。我们来看看Resource类的层级结构：

* InputStreamSource
	* Resource
		* AbstractResource
			* AbstractFileResolvingResource
				* ClassPathResource
					* ClassPathContextResource
					* ClassRelativeContextResource
			* BeanDefinitionResource
			* ByteArrayResource
			* DescriptiveResource
			* FileSystemResource
				* FileSystemContextResource
			* InputStreamResource
			* PathResource
			* VfsResource
		* ContextResource
			* ClassPathContextResource
			* ClassRelativeContextResource
			* FileSystemContextResource
		* WritableResource
			* FileSystemResource
				* FileSystemContextResource
			* PathResource


我们来看看最重要的几个类：

	package org.springframework.core.io;

	import java.io.IOException;
	import java.io.InputStream;

	/**
	 * Simple interface for objects that are sources for an {@link InputStream}.
	 *
	 * <p>This is the base interface for Spring's more extensive {@link Resource} interface.
	 *
	 * <p>For single-use streams, {@link InputStreamResource} can be used for any
	 * given {@code InputStream}. Spring's {@link ByteArrayResource} or any
	 * file-based {@code Resource} implementation can be used as a concrete
	 * instance, allowing one to read the underlying content stream multiple times.
	 * This makes this interface useful as an abstract content source for mail
	 * attachments, for example.
	 *
	 */
	public interface InputStreamSource {

		/**
		 * Return an {@link InputStream}.
		 * <p>It is expected that each call creates a <i>fresh</i> stream.
		 * <p>This requirement is particularly important when you consider an API such
		 * as JavaMail, which needs to be able to read the stream multiple times when
		 * creating mail attachments. For such a use case, it is <i>required</i>
		 * that each {@code getInputStream()} call returns a fresh stream.
		 * @return the input stream for the underlying resource (must not be {@code null})
		 * @throws IOException if the stream could not be opened
		 * @see org.springframework.mail.javamail.MimeMessageHelper#addAttachment(String, InputStreamSource)
		 */
		InputStream getInputStream() throws IOException;
	}


	package org.springframework.core.io;

	import java.io.File;
	import java.io.IOException;
	import java.net.URI;
	import java.net.URL;

	/**
	 * Interface for a resource descriptor that abstracts from the actual
	 * type of underlying resource, such as a file or class path resource.
	 *
	 * <p>An InputStream can be opened for every resource if it exists in
	 * physical form, but a URL or File handle can just be returned for
	 * certain resources. The actual behavior is implementation-specific.
	 *
	 */
	public interface Resource extends InputStreamSource {

		boolean exists();

		boolean isReadable();

		boolean isOpen();

		URL getURL() throws IOException;

		URI getURI() throws IOException;

		File getFile() throws IOException;

		long contentLength() throws IOException;

		long lastModified() throws IOException;

		Resource createRelative(String relativePath) throws IOException;

		String getFilename();

		String getDescription();
	}

然后就是加载classpath资源的Resource：

	package org.springframework.core.io;

	import java.io.FileNotFoundException;
	import java.io.IOException;
	import java.io.InputStream;
	import java.net.URL;

	import org.springframework.util.Assert;
	import org.springframework.util.ClassUtils;
	import org.springframework.util.ObjectUtils;
	import org.springframework.util.StringUtils;

	/**
	 * {@link Resource} implementation for class path resources.
	 * Uses either a given ClassLoader or a given Class for loading resources.
	 *
	 * <p>Supports resolution as {@code java.io.File} if the class path
	 * resource resides in the file system, but not for resources in a JAR.
	 * Always supports resolution as URL.
	 *
	 * @see ClassLoader#getResourceAsStream(String)
	 * @see Class#getResourceAsStream(String)
	 */
	public class ClassPathResource extends AbstractFileResolvingResource {

		private final String path;

		private ClassLoader classLoader;

		private Class<?> clazz;


		/**
		 * Create a new {@code ClassPathResource} for {@code ClassLoader} usage.
		 * A leading slash will be removed, as the ClassLoader resource access
		 * methods will not accept it.
		 * <p>The thread context class loader will be used for
		 * loading the resource.
		 * @param path the absolute path within the class path
		 * @see java.lang.ClassLoader#getResourceAsStream(String)
		 * @see org.springframework.util.ClassUtils#getDefaultClassLoader()
		 */
		public ClassPathResource(String path) {
			this(path, (ClassLoader) null);
		}

		/**
		 * Create a new {@code ClassPathResource} for {@code ClassLoader} usage.
		 * A leading slash will be removed, as the ClassLoader resource access
		 * methods will not accept it.
		 * @param path the absolute path within the classpath
		 * @param classLoader the class loader to load the resource with,
		 * or {@code null} for the thread context class loader
		 * @see ClassLoader#getResourceAsStream(String)
		 */
		public ClassPathResource(String path, ClassLoader classLoader) {
			Assert.notNull(path, "Path must not be null");
			String pathToUse = StringUtils.cleanPath(path);
			if (pathToUse.startsWith("/")) {
				pathToUse = pathToUse.substring(1);
			}
			this.path = pathToUse;
			this.classLoader = (classLoader != null ? classLoader : ClassUtils.getDefaultClassLoader());
		}

		/**
		 * Create a new {@code ClassPathResource} for {@code Class} usage.
		 * The path can be relative to the given class, or absolute within
		 * the classpath via a leading slash.
		 * @param path relative or absolute path within the class path
		 * @param clazz the class to load resources with
		 * @see java.lang.Class#getResourceAsStream
		 */
		public ClassPathResource(String path, Class<?> clazz) {
			Assert.notNull(path, "Path must not be null");
			this.path = StringUtils.cleanPath(path);
			this.clazz = clazz;
		}

		/**
		 * Create a new {@code ClassPathResource} with optional {@code ClassLoader}
		 * and {@code Class}. Only for internal usage.
		 * @param path relative or absolute path within the classpath
		 * @param classLoader the class loader to load the resource with, if any
		 * @param clazz the class to load resources with, if any
		 */
		protected ClassPathResource(String path, ClassLoader classLoader, Class<?> clazz) {
			this.path = StringUtils.cleanPath(path);
			this.classLoader = classLoader;
			this.clazz = clazz;
		}


		/**
		 * Return the path for this resource (as resource path within the class path).
		 */
		public final String getPath() {
			return this.path;
		}

		/**
		 * Return the ClassLoader that this resource will be obtained from.
		 */
		public final ClassLoader getClassLoader() {
			return (this.clazz != null ? this.clazz.getClassLoader() : this.classLoader);
		}


		/**
		 * This implementation checks for the resolution of a resource URL.
		 * @see java.lang.ClassLoader#getResource(String)
		 * @see java.lang.Class#getResource(String)
		 */
		@Override
		public boolean exists() {
			return (resolveURL() != null);
		}

		/**
		 * Resolves a URL for the underlying class path resource.
		 * @return the resolved URL, or {@code null} if not resolvable
		 */
		protected URL resolveURL() {
			if (this.clazz != null) {
				return this.clazz.getResource(this.path);
			}
			else if (this.classLoader != null) {
				return this.classLoader.getResource(this.path);
			}
			else {
				return ClassLoader.getSystemResource(this.path);
			}
		}

		/**
		 * This implementation opens an InputStream for the given class path resource.
		 * @see java.lang.ClassLoader#getResourceAsStream(String)
		 * @see java.lang.Class#getResourceAsStream(String)
		 */
		@Override
		public InputStream getInputStream() throws IOException {
			InputStream is;
			if (this.clazz != null) {
				is = this.clazz.getResourceAsStream(this.path);
			}
			else if (this.classLoader != null) {
				is = this.classLoader.getResourceAsStream(this.path);
			}
			else {
				is = ClassLoader.getSystemResourceAsStream(this.path);
			}
			if (is == null) {
				throw new FileNotFoundException(getDescription() + " cannot be opened because it does not exist");
			}
			return is;
		}

		/**
		 * This implementation returns a URL for the underlying class path resource,
		 * if available.
		 * @see java.lang.ClassLoader#getResource(String)
		 * @see java.lang.Class#getResource(String)
		 */
		@Override
		public URL getURL() throws IOException {
			URL url = resolveURL();
			if (url == null) {
				throw new FileNotFoundException(getDescription() + " cannot be resolved to URL because it does not exist");
			}
			return url;
		}

		/**
		 * This implementation creates a ClassPathResource, applying the given path
		 * relative to the path of the underlying resource of this descriptor.
		 * @see org.springframework.util.StringUtils#applyRelativePath(String, String)
		 */
		@Override
		public Resource createRelative(String relativePath) {
			String pathToUse = StringUtils.applyRelativePath(this.path, relativePath);
			return new ClassPathResource(pathToUse, this.classLoader, this.clazz);
		}

		/**
		 * This implementation returns the name of the file that this class path
		 * resource refers to.
		 * @see org.springframework.util.StringUtils#getFilename(String)
		 */
		@Override
		public String getFilename() {
			return StringUtils.getFilename(this.path);
		}

		/**
		 * This implementation returns a description that includes the class path location.
		 */
		@Override
		public String getDescription() {
			StringBuilder builder = new StringBuilder("class path resource [");
			String pathToUse = path;
			if (this.clazz != null && !pathToUse.startsWith("/")) {
				builder.append(ClassUtils.classPackageAsResourcePath(this.clazz));
				builder.append('/');
			}
			if (pathToUse.startsWith("/")) {
				pathToUse = pathToUse.substring(1);
			}
			builder.append(pathToUse);
			builder.append(']');
			return builder.toString();
		}

		/**
		 * This implementation compares the underlying class path locations.
		 */
		@Override
		public boolean equals(Object obj) {
			if (obj == this) {
				return true;
			}
			if (obj instanceof ClassPathResource) {
				ClassPathResource otherRes = (ClassPathResource) obj;
				return (this.path.equals(otherRes.path) &&
						ObjectUtils.nullSafeEquals(this.classLoader, otherRes.classLoader) &&
						ObjectUtils.nullSafeEquals(this.clazz, otherRes.clazz));
			}
			return false;
		}

		/**
		 * This implementation returns the hash code of the underlying
		 * class path location.
		 */
		@Override
		public int hashCode() {
			return this.path.hashCode();
		}

	}

仔细一看，其实使用的方法跟我们前面说过的一样，就是使用ClassLoader加载。


