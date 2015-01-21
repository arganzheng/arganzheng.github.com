---
title: Java DNS查询内部实现
layout: post
---


源码分析
--------

在Java中，DNS相关的操作都是通过通过InetAddress提供的API实现的。比如

比如查询域名对应的IP地址：

	String dottedQuadIpAddress = InetAddress.getByName( "blog.arganzheng.me" ).getHostAddress();

或者反过来IP对应域名：

	InetAddress[] addresses = InetAddress.getAllByName("8.8.8.8"); // ip or DNS name
	for (int i = 0; i < addresses.length; i++) {
		String hostname = addresses[i].getHostName();
		System.out.println(hostname);
	}

输出：

	google-public-dns-a.google.com

那么InetAddress是如何实现DNS解析的呢？让我们深入代码一步步挖掘下去：


public class InetAddress implements java.io.Serializable {

 	public static InetAddress getByName(String host)
        throws UnknownHostException {
        return InetAddress.getAllByName(host)[0];
    }

	public static InetAddress[] getAllByName(String host)
        throws UnknownHostException {
        return getAllByName(host, null);
    }

    private static InetAddress[] getAllByName(String host, InetAddress reqAddr)
        throws UnknownHostException {

        // ... 省略对于IPV6地址判断，HostName或者IP地址判断
        // hostname, resolve it
        return getAllByName0(host, reqAddr, true);
    }


	private static InetAddress[] getAllByName0 (String host, InetAddress reqAddr, boolean check)
        throws UnknownHostException  {

        /* If it gets here it is presumed to be a hostname */
        /* Cache.get can return: null, unknownAddress, or InetAddress[] */

        /* make sure the connection to the host is allowed, before we
         * give out a hostname
         */
        if (check) {  // 安全检查
            SecurityManager security = System.getSecurityManager();
            if (security != null) {
                security.checkConnect(host, -1);
            }
        }

        // 从DNS Cache中获取
        InetAddress[] addresses = getCachedAddresses(host);

        /* If no entry in cache, then do the host lookup */
        if (addresses == null) {
            addresses = getAddressesFromNameService(host, reqAddr);
        }

        if (addresses == unknown_array)
            throw new UnknownHostException(host);

        return addresses.clone();
    }


这里需要注意的是会先DNS缓存中查询有没有结果。本身没有问题，但是有一个问题在于默认的缓存时间是永远！这个是JDK实现中比较啃爹的地方。

> **InetAddress Caching**
> 
> The InetAddress class has a cache to store successful as well as unsuccessful host name resolutions. 
By default, when a security manager is installed, in order to protect against DNS spoofing attacks, the result of positive host name resolutions are cached forever. When a security manager is not installed, the default behavior is to cache entries for a finite (implementation dependent) period of time. The result of unsuccessful host name resolution is cached for a very short period of time (10 seconds) to improve performance. 
> 
> If the default behavior is not desired, then a Java security property can be set to a different Time-to-live (TTL) value for positive caching. Likewise, a system admin can configure a different negative caching TTL value when needed. 
>
> Two Java security properties control the TTL values used for positive and negative host name resolution caching: 
> 
> networkaddress.cache.ttl 
> Indicates the caching policy for successful name lookups from the name service. The value is specified as as integer to indicate the number of seconds to cache the successful lookup. The default setting is to cache for an implementation specific period of time. 
> A value of -1 indicates "cache forever". 
> 
> networkaddress.cache.negative.ttl (default: 10) 
> Indicates the caching policy for un-successful name lookups from the name service. The value is specified as as integer to indicate the number of seconds to cache the failure for un-successful lookups. 
> A value of 0 indicates "never cache". A value of -1 indicates "cache forever". 

如果Cache miss，那么就会调用配置的nameServices执行真正DNS查询：

 	private static InetAddress[] getAddressesFromNameService(String host, InetAddress reqAddr)
        throws UnknownHostException
    {
        InetAddress[] addresses = null;
        boolean success = false;
        UnknownHostException ex = null;

        // Check whether the host is in the lookupTable.
        // 1) If the host isn't in the lookupTable when
        //    checkLookupTable() is called, checkLookupTable()
        //    would add the host in the lookupTable and
        //    return null. So we will do the lookup.
        // 2) If the host is in the lookupTable when
        //    checkLookupTable() is called, the current thread
        //    would be blocked until the host is removed
        //    from the lookupTable. Then this thread
        //    should try to look up the addressCache.
        //     i) if it found the addresses in the
        //        addressCache, checkLookupTable()  would
        //        return the addresses.
        //     ii) if it didn't find the addresses in the
        //         addressCache for any reason,
        //         it should add the host in the
        //         lookupTable and return null so the
        //         following code would do  a lookup itself.
        if ((addresses = checkLookupTable(host)) == null) {
            try {
                // This is the first thread which looks up the addresses
                // this host or the cache entry for this host has been
                // expired so this thread should do the lookup.

                /*  
                 * 这里可以看到nameServices是链状的，这是JDK7+的逻辑。 
                 * 插入自定义nameservice的逻辑就在这里。
                 */
                for (NameService nameService : nameServices) { 
                    try {
                        /*
                         * Do not put the call to lookup() inside the
                         * constructor.  if you do you will still be
                         * allocating space when the lookup fails.
                         */

                        addresses = nameService.lookupAllHostAddr(host);
                        success = true;
                        break;
                    } catch (UnknownHostException uhe) {
                        if (host.equalsIgnoreCase("localhost")) {
                            InetAddress[] local = new InetAddress[] { impl.loopbackAddress() };
                            addresses = local;
                            success = true;
                            break;
                        }
                        else {
                            addresses = unknown_array;
                            success = false;
                            ex = uhe;
                        }
                    }
                }

                // More to do?
                if (reqAddr != null && addresses.length > 1 && !addresses[0].equals(reqAddr)) {
                    // Find it?
                    int i = 1;
                    for (; i < addresses.length; i++) {
                        if (addresses[i].equals(reqAddr)) {
                            break;
                        }
                    }
                    // Rotate
                    if (i < addresses.length) {
                        InetAddress tmp, tmp2 = reqAddr;
                        for (int j = 0; j < i; j++) {
                            tmp = addresses[j];
                            addresses[j] = tmp2;
                            tmp2 = tmp;
                        }
                        addresses[i] = tmp2;
                    }
                }
                // Cache the address.
                cacheAddresses(host, addresses, success);

                if (!success && ex != null)
                    throw ex;

            } finally {
                // Delete host from the lookupTable and notify
                // all threads waiting on the lookupTable monitor.
                updateLookupTable(host);
            }
        }

        return addresses;
    }


这里有一段非常关键的代码:

	for (NameService nameService : nameServices) { 
        try {
            /*
             * Do not put the call to lookup() inside the
             * constructor.  if you do you will still be
             * allocating space when the lookup fails.
             */
            addresses = nameService.lookupAllHostAddr(host);
            success = true;
            break;
        } catch (UnknownHostException uhe) {
            if (host.equalsIgnoreCase("localhost")) {
                InetAddress[] local = new InetAddress[] { impl.loopbackAddress() };
                addresses = local;
                success = true;
                break;
            }
            else {
                addresses = unknown_array;
                success = false;
                ex = uhe;
            }
        }
    }

这是真正执行DNS查询的地方，而且可以看到是通过NameService链来依次解析的。其中nameServices是InetAddress的一个成员变量：

    /* Used to store the name service provider */
    private static List<NameService> nameServices = null;

通过static块进行初始化的：

    static {
        // create the impl
        impl = InetAddressImplFactory.create();

        // get name service if provided and requested
        String provider = null;;
        String propPrefix = "sun.net.spi.nameservice.provider.";
        int n = 1;
        nameServices = new ArrayList<NameService>();
        provider = AccessController.doPrivileged(
                new GetPropertyAction(propPrefix + n));
        while (provider != null) {
            NameService ns = createNSProvider(provider);
            if (ns != null)
                nameServices.add(ns);

            n++;
            provider = AccessController.doPrivileged(
                    new GetPropertyAction(propPrefix + n));
        }

        // if not designate any name services provider,
        // create a default one
        if (nameServices.size() == 0) {
            NameService ns = createNSProvider("default");
            nameServices.add(ns);
        }
    }

因为是通过InetAddress的static块初始化的，所以必须在使用InetAddress之前配置`sun.net.spi.nameservice.provider.<n>=<default|dns,sun|...>`配置项。否则不会生效。具体可以参见SO上的这篇帖子：[Clean DNS server in JVM](http://stackoverflow.com/questions/17420620/clean-dns-server-in-jvm)。

另外需要注意的是在JDK7之前，只有第一个成功加载的nameservice参与解析。其他的nameservice并不起作用。

> **`sun.net.spi.nameservice.provider.<n>=<default|dns,sun|...>`**
>
> Specifies the name service provider that you can use. By default, Java will use the system configured name lookup mechanism, such as file, nis, etc. You can specify your own by setting this option. <n> takes the value of a positive number, it indicates the precedence order with a small number takes higher precendence over a bigger number. Aside from the default provider, the JDK includes a DNS provider named "dns,sun".
> 
> Prior to JDK 7, the first provider that was successfully loaded was used. In JDK 7, providers are chained, which means that if a lookup on a provider fails, the next provider in the list is consulted to resolve the name.

不过我看一下OpenJDK6的代码，貌似也是链状的。

然后我们来看看NameService的定义：

	package sun.net.spi.nameservice;

	import java.net.UnknownHostException;

	public interface NameService {
	    public java.net.InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;
	    public String getHostByAddr(byte[] addr) throws UnknownHostException;
	}

其实是个很简单的接口。默认有两个实现：

* 匿名类：就是provider名称为default的实现类
* DNSNameService

 	private static NameService createNSProvider(String provider) {
        if (provider == null)
            return null;

        NameService nameService = null;
        if (provider.equals("default")) {
            // initialize the default name service
            nameService = new NameService() {
                public InetAddress[] lookupAllHostAddr(String host)
                    throws UnknownHostException {
                    return impl.lookupAllHostAddr(host);
                }
                public String getHostByAddr(byte[] addr)
                    throws UnknownHostException {
                    return impl.getHostByAddr(addr);
                }
            };
        } else {
            final String providerName = provider;
            try {
                nameService = java.security.AccessController.doPrivileged(
                    new java.security.PrivilegedExceptionAction<NameService>() {
                        public NameService run() {
                            Iterator<NameServiceDescriptor> itr =
                                ServiceLoader.load(NameServiceDescriptor.class)
                                    .iterator();
                            while (itr.hasNext()) {
                                NameServiceDescriptor nsd = itr.next();
                                if (providerName.
                                    equalsIgnoreCase(nsd.getType()+","
                                        +nsd.getProviderName())) {
                                    try {
                                        return nsd.createNameService();
                                    } catch (Exception e) {
                                        e.printStackTrace();
                                        System.err.println(
                                            "Cannot create name service:"
                                             +providerName+": " + e);
                                    }
                                }
                            }

                            return null;
                        }
                    }
                );
            } catch (java.security.PrivilegedActionException e) {
            }
        }

        return nameService;
    }

默认的NameService其实将域名解析操作委托给了impl变量：

    static InetAddressImpl  impl;

InetAddressImpl本身也是一个接口：

	package java.net;
	import java.io.IOException;
	/*
	 * Package private interface to "implementation" used by
	 * {@link InetAddress}.
	 * <p>
	 * See {@link java.net.Inet4AddressImp} and
	 * {@link java.net.Inet6AddressImp}.
	 *
	 * @since 1.4
	 */
	interface InetAddressImpl {

	    String getLocalHostName() throws UnknownHostException;
	    InetAddress[]
	        lookupAllHostAddr(String hostname) throws UnknownHostException;
	    String getHostByAddr(byte[] addr) throws UnknownHostException;

	    InetAddress anyLocalAddress();
	    InetAddress loopbackAddress();
	    boolean isReachable(InetAddress addr, int timeout, NetworkInterface netif,
	                        int ttl) throws IOException;
	}

有两个实现：

* Inet4AddressImpl
* Inet6AddressImpl

这里我们仅仅挑选Inet4AddressImpl说明。

	package java.net;
	import java.io.IOException;

	/*
	 * Package private implementation of InetAddressImpl for IPv4.
	 *
	 * @since 1.4
	 */
	class Inet4AddressImpl implements InetAddressImpl {
	    public native String getLocalHostName() throws UnknownHostException;
	    public native InetAddress[]
	        lookupAllHostAddr(String hostname) throws UnknownHostException;
	    public native String getHostByAddr(byte[] addr) throws UnknownHostException;
	    private native boolean isReachable0(byte[] addr, int timeout, byte[] ifaddr, int ttl) throws IOException;

	    public synchronized InetAddress anyLocalAddress() {
	        if (anyLocalAddress == null) {
	            anyLocalAddress = new Inet4Address(); // {0x00,0x00,0x00,0x00}
	            anyLocalAddress.holder().hostName = "0.0.0.0";
	        }
	        return anyLocalAddress;
	    }

	    public synchronized InetAddress loopbackAddress() {
	        if (loopbackAddress == null) {
	            byte[] loopback = {0x7f,0x00,0x00,0x01};
	            loopbackAddress = new Inet4Address("localhost", loopback);
	        }
	        return loopbackAddress;
	    }

	  public boolean isReachable(InetAddress addr, int timeout, NetworkInterface netif, int ttl) throws IOException {
	      byte[] ifaddr = null;
	      if (netif != null) {
	          /*
	           * Let's make sure we use an address of the proper family
	           */
	          java.util.Enumeration<InetAddress> it = netif.getInetAddresses();
	          InetAddress inetaddr = null;
	          while (!(inetaddr instanceof Inet4Address) &&
	                 it.hasMoreElements())
	              inetaddr = it.nextElement();
	          if (inetaddr instanceof Inet4Address)
	              ifaddr = inetaddr.getAddress();
	      }
	      return isReachable0(addr.getAddress(), timeout, ifaddr, ttl);
	  }
	    private InetAddress      anyLocalAddress;
	    private InetAddress      loopbackAddress;
	}

可以看到这个类基本没有实现什么逻辑，主要实现都是通过JNI调用native方法了。native方法其实就是系统调用了。这里就不展开了，逻辑不外乎就是查看/etc/resolv.conf下配置的nameserver和/etc/hosts下面的配置，然后使用DNS协议查询。

其他NameService通过NameServiceDescriptor来创建相应。NameServiceDescriptor本身也是一个接口：

	package sun.net.spi.nameservice;

	public interface NameServiceDescriptor {
	    /**
	     * Create a new instance of the corresponding name service.
	     */
	    public NameService createNameService () throws Exception ;

	    /**
	     * Returns this service provider's name
	     *
	     */
	    public String getProviderName();

	    /**
	     * Returns this name service type
	     * "dns" "nis" etc
	     */
	    public String getType();
	}

这里我们先看看JDK自带一个实现——DNSNameService。

	package sun.net.spi.nameservice.dns;

	/*
	 * A name service provider based on JNDI-DNS.
	 */
	public final class DNSNameService implements NameService {

	    // List of domains specified by property
	    private LinkedList<String> domainList = null;

	    // JNDI-DNS URL for name servers specified via property
	    private String nameProviderUrl = null;

	    // Per-thread soft cache of the last temporary context
	    private static ThreadLocal<SoftReference<ThreadContext>> contextRef =
	            new ThreadLocal<>();

	    // Simple class to encapsulate the temporary context
	    private static class ThreadContext {
	        private DirContext dirCtxt;
	        private List<String> nsList;

	        ...
	    }   

	}

看一下构造函数：

    public DNSNameService() throws Exception {

        // default domain
        String domain = AccessController.doPrivileged(
            new GetPropertyAction("sun.net.spi.nameservice.domain"));
        if (domain != null && domain.length() > 0) {
            domainList = new LinkedList<String>();
            domainList.add(domain);
        }

        // name servers
        String nameservers = AccessController.doPrivileged(
            new GetPropertyAction("sun.net.spi.nameservice.nameservers"));
        if (nameservers != null && nameservers.length() > 0) {
            nameProviderUrl = createProviderURL(nameservers);
            if (nameProviderUrl.length() == 0) {
                throw new RuntimeException("malformed nameservers property");
            }
        } else {
            // no property specified so check host DNS resolver configured
            // with at least one nameserver in dotted notation.
            //
            List<String> nsList = ResolverConfiguration.open().nameservers();
            if (nsList.isEmpty()) {
                throw new RuntimeException("no nameservers provided");
            }
            boolean found = false;
            for (String addr: nsList) {
                if (IPAddressUtil.isIPv4LiteralAddress(addr) ||
                    IPAddressUtil.isIPv6LiteralAddress(addr)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw new RuntimeException("bad nameserver configuration");
            }
        }
    }

可以看到它主要是解析下面两个配置项，得到nameserver：

* `sun.net.spi.nameservice.domain=<domainname>`
* `sun.net.spi.nameservice.nameservers=<server1_ipaddr,server2_ipaddr ...>`

如果 sun.net.spi.nameservice.nameservers 没有配置，那么会使用 ResolverConfiguration 得到系统配置的nameserver：

	List<String> nsList = ResolverConfiguration.open().nameservers();

ResolverConfiguration的实现逻辑是加载`/etc/resolv.conf`配置文件中的nameserver。具体参见：[ResolverConfigurationImpl](http://grepcode.com/file/repository.grepcode.com/java/root/jdk/openjdk/8-b132/sun/net/dns/ResolverConfigurationImpl.java#ResolverConfigurationImpl)。这里就不赘述。

得到NameServer，DNS的解析就很简单了，对NameServer分别执行DNS查询就可以了。具体代码大家可以参见 [DNSNameService](http://code.metager.de/source/xref/openjdk/jdk8/jdk/src/share/classes/sun/net/spi/nameservice/dns/DNSNameService.java)。

**NOTE** 从代码可以看出，系统配置的nameserver和通过SystemProperty配置的nameserver是或的关系，所以如果配置了sun.net.spi.nameservice.nameservers，那么相当于绕过了系统配置的nameserver了。

前面说过，非默认的NameService是需要通过相应的NameServiceDescriptor传教的，所以DNSNameService也需要有一个对应的NameServiceDescriptor，就是DNSNameServiceDescriptor：

	package sun.net.spi.nameservice.dns;

	import sun.net.spi.nameservice.*;

	public final class DNSNameServiceDescriptor implements NameServiceDescriptor {
	    /**
	     * Create a new instance of the corresponding name service.
	     */
	    public NameService createNameService() throws Exception {
	        return new DNSNameService();
	    }

	    /**
	     * Returns this service provider's name
	     *
	     */
	    public String getProviderName() {
	        return "sun";
	    }

	    /**
	     * Returns this name service type
	     * "dns" "nis" etc
	     */
	    public String getType() {
	        return "dns";
	    }
	}

可以看到DNSNameService的type为dns，name为sun。所以在配置的时候应该配置为`dns,sun`。

但是NameServiceDescriptor本身是怎么加载的呢？回到上面的代码：

 	Iterator<NameServiceDescriptor> itr = ServiceLoader.load(NameServiceDescriptor.class).iterator();

原来是是通过[java.util.ServiceLoader](http://grepcode.com/file/repository.grepcode.com/java/root/jdk/openjdk/8-b132/java/util/ServiceLoader.java#ServiceLoader)。这个类是专门用来加载service-provider的。这个类有非常详细的JavaDoc。其中有一段说明怎么告诉ServiceLoader加载自定义的ServiceProvider：

> A service provider is identified by placing a provider-configuration file in the resource directory META-INF/services. The file's name is the fully-qualified binary name of the service's type. The file contains a list of fully-qualified binary names of concrete provider classes, one per line. Space and tab characters surrounding each name, as well as blank lines, are ignored. The comment character is '#' ('\u0023', NUMBER SIGN); on each line all characters following the first comment character are ignored. The file must be encoded in UTF-8.
> 
> **Example** Suppose we have a service type com.example.CodecSet which is intended to represent sets of encoder/decoder pairs for some protocol. In this case it is an abstract class with two abstract methods:
>
> 	  public abstract Encoder getEncoder(String encodingName);
> 	  public abstract Decoder getDecoder(String encodingName);
>
> Each method returns an appropriate object or null if the provider does not support the given encoding. Typical providers support more than one encoding.
> If com.example.impl.StandardCodecs is an implementation of the CodecSet service then its jar file also contains a file named
 META-INF/services/com.example.CodecSet
> This file contains the single line:
>
> 	  com.example.impl.StandardCodecs    # Standard codecs
>
> The CodecSet class creates and saves a single service instance at initialization:
> 	
>	  private static ServiceLoader<CodecSet> codecSetLoader = ServiceLoader.load(CodecSet.class);
>
> To locate an encoder for a given encoding name it defines a static factory method which iterates through the known and available providers, returning only when it has located a suitable encoder or has run out of providers.
> 	
>	  public static Encoder getEncoder(String encodingName) {
>      	  for (CodecSet cp : codecSetLoader) {
>       	  Encoder enc = cp.getEncoder(encodingName);
>             if (enc != null)
>                return enc;
>     	  }
>         return null;
> 	  }
>
> A getDecoder method is defined similarly.

所以对于DNSNameServiceDescriptor，在`sun/net/spi/nameservice/dns/META-INF/services`目录下有一个名称为`sun.net.spi.nameservice.NameServiceDescriptor`的文件：

	# dns service provider descriptor
	sun.net.spi.nameservice.dns.DNSNameServiceDescriptor

这种机制后来广泛用于Spring的自定义便签（[Creating a Custom Spring 3 XML Namespace](http://java.dzone.com/articles/creating-custom-spring-3-xml)），估计是受这个影响。


实际应用
--------

知道这个有什么用呢？比如说我们想让我们的应用走谷歌的公共DNS服务器：8.8.8.8。但是我们又不想让用户去修改系统配置，这对用户是个负担，而且我们不希望影响到其他应用。怎么处理呢？只需要在应用启动之前简单配置一下：
		
	System.setProperty("sun.net.spi.nameservice.provider.1", "dns,sun");
	System.setProperty("sun.net.spi.nameservice.nameservers", "8.8.8.8");
	System.setProperty("sun.net.spi.nameservice.provider.2", "default");


再比如，如果你只想针对某些域名做特殊的解析，那么你可以自定义一个NameServiceProvider，实现对应的NameServiceDescriptor，还有相应的META-INF说明。然后在应用启动的时候配置一下：

	System.setProperty("sun.net.spi.nameservice.provider.1", "dns,yourProviderName");
	System.setProperty("sun.net.spi.nameservice.provider.2", "default");


参考文章
--------

1. [DNS: Java Glossary](http://mindprod.com/jgloss/dns.html) 深入浅出的一篇介绍性文章，强烈推荐。
2. [Understanding host name resolution and DNS behavior in Java](http://www.myhowto.org/java/42-understanding-host-name-resolution-and-dns-behavior-in-java/)
3. [Networking Properties](http://docs.oracle.com/javase/1.5.0/docs/guide/net/properties.html)
4. [Local Managed DNS (Java)](http://rkuzmik.blogspot.com/2006/08/local-managed-dns-java_11.html) 被墙了，但是可以从Google Cache中获取到历史页面。
