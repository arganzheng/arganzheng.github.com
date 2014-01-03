---
title: JVM编码
layout: post
---


    package java.nio.charset;
    
    public abstract class Charset implements Comparable<Charset>{
    
        private static volatile Charset defaultCharset;
    
    
        /**
         * Returns the default charset of this Java virtual machine.
         *
         * <p> The default charset is determined during virtual-machine startup and
         * typically depends upon the locale and charset of the underlying
         * operating system.
         *
         * @return  A charset object for the default charset
         *
         * @since 1.5
         */
        public static Charset defaultCharset() {
            if (defaultCharset == null) {
    	    synchronized (Charset.class) {
    		java.security.PrivilegedAction pa =
    		    new GetPropertyAction("file.encoding");
    		String csn = (String)AccessController.doPrivileged(pa);
    		Charset cs = lookup(csn);
    		if (cs != null)
    		    defaultCharset = cs;
                    else 
    		    defaultCharset = forName("UTF-8");
                }
    	}
    	return defaultCharset;
        }
    
    }
    
就是一个lazy-init的单例模式实现。另外，其实就是取自file.encoding环境变量，取不到，默认是"UTF-8"。

**Tips** 可以在eclipse中用Display View中，执行java语句：

    System.getProperty("file.encoding");
    
非常实用的功能。   
