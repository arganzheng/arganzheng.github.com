---
title: 如何往HttpServletRequest中塞请求参数
layout: post
---


### 需求

假如用户没有传递cooperatorId，那么默认等于uin。

常规做法是在interceptor或者filter或者action中判断如果没有cooperatorId参数，则设置到request的parameterMap中：

    String cooperatorId = request.getParameter("cooperatorId");
    if(cooperatorId == null){
        request.getParameterMap().put("cooperatorId",request.getParameter("uin")) ;
    }

## 问题

    public interface ServletRequest {
        
        ...
        
        /** Returns a java.util.Map of the parameters of this request.
         * Request parameters
         * are extra information sent with the request.  For HTTP servlets,
         * parameters are contained in the query string or posted form data.
         *
         * @return an immutable java.util.Map containing parameter names as
         * keys and parameter values as map values. The keys in the parameter
         * map are of type String. The values in the parameter map are of type
         * String array.
         *
         */
        public Map getParameterMap();

        ...
    }

但是HttpServletRequest.getParameterMap()返回的是immutable的java.util.Map，所以不能直接修改parameterMap。只能另辟蹊径了。

## 解决方案

虽然request本身的属性是不可修改的，但是方法确是可以重载的，假如我们定义一个HttpServerRequest的子类，覆盖他的getParameter()和getParameterMap()方法，让其好像有cooperatorId这请求参数一般，也可以达到我们的目的。J2EE提供了一个HttpServletRequestWrapper可以帮我们很方便实现这个子类：

    /**
     *
     * Provides a convenient implementation of the HttpServletRequest interface that
     * can be subclassed by developers wishing to adapt the request to a Servlet.
     * This class implements the Wrapper or Decorator pattern. Methods default to
     * calling through to the wrapped request object.
     *
     *
     * @see   javax.servlet.http.HttpServletRequest
      * @since    v 2.3
     *
     */
    public class HttpServletRequestWrapper extends ServletRequestWrapper implements HttpServletRequest {
       ...
    }


代码实现：

    package me.arganzheng.study;

    import java.util.Collections;
    import java.util.Enumeration;
    import java.util.HashMap;
    import java.util.Map;

    import javax.servlet.http.HttpServletRequest;
    import javax.servlet.http.HttpServletRequestWrapper;

    /**
    * <pre>
    * Api的 HttpServletRequest，主要是完成如下功能：
    * 背景：现在我们所有的API都是要求传递cooperatorId，而且cooperatorId的定位是卖家QQ号。但是现在逐渐有针对买家的三方应用出现。
    * 这些应用是以买家为纬度的，不需要传递cooperatorId。但是我们现在的鉴权体系包括后台的IDL都是要求传递cooperatorId，这里提出了这么一个逻辑：
    * 如果没有传递cooperatorId，那么默认就是授权者，即uin。这样cooperatorId只有在和uin不同的情况下才需要传递。
    * 通过这样，给买家用户造成一个不需要传递cooperatorId的假象。而且这个逻辑也是简化参数传递不错的逻辑。
    *
    * @author arganzheng
    * @date 2012-9-10
    */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public class ApiHttpServletRequest extends HttpServletRequestWrapper {

        private static final String UIN = "uin";
         private static final String COOPERATOR_ID = "cooperatorId";

         public ApiHttpServletRequest(HttpServletRequest request){
            super(request);
        }

        @Override
        public String getParameter(String name) {
            String value = super.getParameter(name);
            if(value == null && COOPERATOR_ID.equals(name)){
                 value = super.getParameter(UIN);
            }
            return value;
        }

         @Override
        public Map getParameterMap() {
             Map map = super.getParameterMap();
             if(!map.containsKey(COOPERATOR_ID) && map.containsKey(UIN)){
                  map = new HashMap(map);
                  map.put(COOPERATOR_ID, map.get(UIN));
             }
             return map;

        }

         @Override
         public Enumeration getParameterNames() {
              if(super.getParameter(COOPERATOR_ID) == null && super.getParameter(UIN) != null){
                   return Collections.enumeration(getParameterMap().keySet());
              }else{
                   return super.getParameterNames();
              }
         }

         @Override
         public String[] getParameterValues(String name) {
              String[] values = super.getParameterValues(name);
              if(values == null && COOPERATOR_ID.equals(name)){
                   values = super.getParameterValues(UIN);
              }
              return values;
         }
    }

    package me.arganzheng.study.filter;

    import java.io.IOException;

    import javax.servlet.Filter;
    import javax.servlet.FilterChain;
    import javax.servlet.FilterConfig;
    import javax.servlet.ServletException;
    import javax.servlet.ServletRequest;
    import javax.servlet.ServletResponse;
    import javax.servlet.http.HttpServletRequest;

    import me.arganzheng.study.ApiHttpServletRequest;

    /**
    * 如果没有传递cooperatorId，那么默认就是授权者，即uin。这样cooperatorId只有在和uin不同的情况下才需要传递。
    *
    * @author arganzheng
    * @date 2012-9-10
    */

    public class ApiResetServletRequestFilter implements Filter {

        @Override
        public void destroy() {
        }

        @Override
        public void doFilter(ServletRequest request, ServletResponse response, 
                FilterChain filterChain) throws IOException, ServletException {
            filterChain.doFilter(new ApiHttpServletRequest((HttpServletRequest) request), response);
        }

        @Override
        public void init(FilterConfig arg0) throws ServletException {
        }

    }


然后在web.xml中配置一个filter：

    <filter>
        <filter-name>ApiResetServletRequestFilter</filter-name>
            <filter-class>
                me.arganzheng.study.filter.ApiResetServletRequestFilter
            </filter-class>
        </filter>
    <filter-mapping>
        <filter-name>ApiResetServletRequestFilter</filter-name>
        <url-pattern>/*</url-pattern>
    </filter-mapping>


补充：关于unmodifiableMap
------------------------

其实Java的unmodifiableMap，其实现原理就是利用Java的final变量的immutable特性+函数重载。final特性保证了变量引用的immutable，而将写接口重载使得内容不可更改了。具体代码如下：

    public class Collections {
        // Suppresses default constructor, ensuring non-instantiability.
        private Collections() {
        }
        /**
         * Returns an unmodifiable view of the specified map.  This method
         * allows modules to provide users with "read -only" access to internal
         * maps.  Query operations on the returned map "read through"
         * to the specified map, and attempts to modify the returned
         * map, whether direct or via its collection views, result in an
         * <tt>UnsupportedOperationException</tt> .<p>
         *
         * The returned map will be serializable if the specified map
         * is serializable.
         *
         * @param  m the map for which an unmodifiable view is to be returned.
         * @return an unmodifiable view of the specified map.
         */
        public static <K,V> Map<K,V> unmodifiableMap(Map<? extends K, ? extends V> m) {
         return new UnmodifiableMap<K,V>(m);
        }


        /**
         * @serial include
         */
        private static class UnmodifiableMap <K,V> implements Map<K,V>, Serializable {
         // use serialVersionUID from JDK 1.2.2 for interoperability
         private static final long serialVersionUID = -1034234728574286014L;


         private final Map<? extends K, ? extends V> m;


         UnmodifiableMap(Map<? extends K, ? extends V> m) {
                if (m==null )
                    throw new NullPointerException();
                this.m = m;
            }


         public int size()                { return m .size();}
         public boolean isEmpty()              {return m.isEmpty();}
         public boolean containsKey(Object key)   {return m.containsKey(key);}
         public boolean containsValue(Object val) {return m.containsValue(val);}
         public V get(Object key)              { return m.get(key);}


         public V put(K key, V value) {
             throw new UnsupportedOperationException();
            }
         public V remove(Object key) {
             throw new UnsupportedOperationException();
            }
         public void putAll(Map<? extends K, ? extends V> m) {
             throw new UnsupportedOperationException();
          }
         public void clear() {
             throw new UnsupportedOperationException();
          }
                   ...
        }
    }

也可以通过反射改变unmodifiedMap的特性。

具体例子：参见 http://vangjee.wordpress.com/2009/02/25/how-to-modify-request-headers-in-a-j2ee-web-application/

不过貌似只能通过Cache访问了。
