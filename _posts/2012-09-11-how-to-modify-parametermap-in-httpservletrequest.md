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


补记：另一个实战例子 2015-07-24
----------------------------

最近在做一个新项目，出于对用户隐私的考虑，需要对所有提交的参数进行加密，不让用户知道我们到底提交了什么，特别是统计参数（语言、国际、版本、uid等）。原来浏览器那边也有类似的需求，于是打算直接用他们那一套加密机制。参数就约定为_p=加密后的JSON串。比如

	_p=3lg62PMuu5QhfQAn8HxmjUQndEdunnPa2JjrnPHT2IEjqNVsDvin2z0UhUypPXGvif4hXibzuCbjJu6RUX9%252Bs7rq8VslqGZAr4XdsKs6XnWjnG8DXKtad3TyQlDTux9MQgESqZXlWGjvxzK0NHb5I995HzBvMmmqtZzbl0O4%252FwouMT4xZuNxelACNceoQkno6IEF%252BUIFclwDyRiNE6QqVNupOR5cYdqE75u1kEpMRePZivOlrX%252FeFBXf%252FhWC0xVHQdbFAuJGSgUr%252BzxcdC05%252BX74nufbkL8GJzNJktS47tWVJHiMArwk2ziXIcvDNDfdfJF%252F02Cq8Lt4in1oBFPR0mTZIKYariNNlv1I%252BuueZMUzjg1YFvDTvalFmkyeChWWNfAoLrQNf9vfHKRro8de7AknPTdh0UgE5kfauFq%252FAzi6b1zOIRCih9gZcCHXw477YdDw8gtrYQX0X2ACBDut%252BHliMF7CCO4STi9cdS85HCKl9L5yzZXTqBNrwlWiGr8Y6QSITvxFj5jne1sokTOkR%252FAZbOjqUnoQAhGV5G%252Bt32R3SkM4tZkO6rLMJ4j40dw7XY%252FCWfKX9JGG2Pd6ZJHwyfxCPFPXOOcdfSrgstTnFu4mbrIEXFhl9YwI%252Fd04IA%252F3%252FdFWx7sAc2vnDOfCpzQbQvPWdjQLv%252BlwVS6kn%252BAUzYME%253D

解密之后就变成了：

	_p={"pt":"dl","uid":"1234556","cf":"gp","co":"us","la":"en","of":"gp","pr":"","sv":"a_22","av":"1.1.0.1002","packageNames":["com.example.android.apis","com.dianxinos.powermanager","com.android.gesture.builder","com.hexin.plat.android","com.google.android.apps.plus","com.google.android.inputmethod.pinyin"]}

然后将JSON串反序列化成Java POJO对象，进行相应的操作，实现起来大概是这个样子：

 	@RequestMapping(value = "/getAppCategory")
    @ResponseBody
    public RestResponse<Map<String, String>> getAppCategory(HttpServletRequest request){
    	String params = request.getParameter("_p");
        Assert.notEmpty(params);

        String reqString = params;
        if (ConfigurationTool.decodeParameter()) { // if need decode
            reqString = ParameterUtil.decodeParameter(params);
        }

        GetAppCategoryRequest getAppCategoryRequest = objectMapper.readValue(reqString, GetAppCategoryRequest.class);

        // do string
        ...
    }

虽然，我们把加密过程封装了，但是有如下三个问题：

1. 每个方法都要做解密，反序列化过程，麻烦
2. 接收参数只能是通用的HttpServletRequest，无法自动使用Spring MVC的参数注入
3. 不方便测试，即使这里用配置参数决定要不要加密，提交一个JSON串仍然没有传统的k-v方式简单友好

于是想一下能不能将这一切透明化，让服务端以为客户端提交的还是传统的k-v方式呢？一个简单的想法就是通过在filter或者interceptor中将_p参数提取出来，然后反序列化，再将参数一个个重新塞入HttpServletRequest中。

如上讨论，传统的HttpServletRequest是不可修改的，于是写了这么一个类：

	package me.arganzheng.study.sever.launcher.common;

	import java.io.IOException;
	import java.lang.reflect.Field;
	import java.util.ArrayList;
	import java.util.Collection;
	import java.util.Collections;
	import java.util.Enumeration;
	import java.util.HashMap;
	import java.util.List;
	import java.util.Map;
	import java.util.TreeMap;

	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpServletRequestWrapper;

	import org.apache.commons.lang.StringUtils;
	import org.codehaus.jackson.annotate.JsonProperty;
	import org.codehaus.jackson.map.ObjectMapper;
	import org.codehaus.jackson.map.annotate.JsonSerialize.Inclusion;
	import org.codehaus.jackson.type.TypeReference;

	import me.arganzheng.study.server.launcher.request.BaseRequest;
	import me.arganzheng.study.server.launcher.util.ConfigurationTool;
	import me.arganzheng.study.server.launcher.util.ParameterUtil;

	public class ApiHttpServletRequest extends HttpServletRequestWrapper {

	    private static ObjectMapper objectMapper = new ObjectMapper();

	    private static Map<String, String> baseReqKeyMap = new HashMap<String, String>();

	    static {
	        objectMapper.setSerializationConfig(objectMapper.getSerializationConfig().withSerializationInclusion(
	                Inclusion.NON_NULL));

	        Field[] fileds = BaseRequest.class.getDeclaredFields();
	        for (Field f : fileds) {
	            String fname = f.getName();
	            JsonProperty jsonPropertyAnnotation = f.getAnnotation(JsonProperty.class);
	            if (jsonPropertyAnnotation != null) {
	                String shortName = jsonPropertyAnnotation.value();
	                baseReqKeyMap.put(shortName, fname);
	            }
	        }
	    }

	    // modified parameters map
	    private final Map<String, String[]> parameters = new TreeMap<String, String[]>();

	    public ApiHttpServletRequest(HttpServletRequest request) {
	        super(request);

	        // 请求参数是特殊的_p=加密的JSON字符串
	        String params = request.getParameter("_p");
	        if (StringUtils.isBlank(params)) {
	            parameters.putAll(super.getParameterMap());
	            return;
	        }

	        // for API requests
	        String reqString = params;
	        if (ConfigurationTool.decodeParameter()) {
	            reqString = ParameterUtil.decodeParameter(params);
	        }

	        // parse json and put it to request parameters map.
	        try {
	            Map<String, Object> req = objectMapper.readValue(reqString, new TypeReference<Map<String, Object>>() {
	            });
	            if (req != null) {
	                addMapEntries(req);
	            }
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	        // put the original param value back
	        parameters.put("_p", new String[] { reqString });
	    }

	    private void addMapEntries(Map<String, Object> req) {
	        if (req != null && !req.isEmpty()) {
	            for (Map.Entry<String, Object> prop : req.entrySet()) {
	                Object value = prop.getValue();
	                if (value != null) {
	                    if (value instanceof Collection<?>) {
	                        Collection<?> a = (Collection<?>) value;
	                        List<String> v = new ArrayList<String>();
	                        for (Object e : a) {
	                            v.add(e.toString());
	                        }
	                        parameters.put(getKey(prop.getKey()), v.toArray(new String[0]));
	                    } else {
	                        String svalue = value.toString();
	                        parameters.put(getKey(prop.getKey()), new String[] { svalue });
	                    }
	                }
	            }
	        }
	    }

	    private String getKey(String key) {
	        String v = baseReqKeyMap.get(key);
	        if (v == null) {
	            return key;
	        }
	        return v;
	    }

	    @Override
	    public String getParameter(String name) {
	        String[] strings = getParameterMap().get(name);
	        if (strings != null) {
	            return strings[0];
	        }
	        return super.getParameter(name);
	    }

	    @Override
	    public Map<String, String[]> getParameterMap() {
	        // Return an unmodifiable collection because we need to uphold the interface contract.
	        return Collections.unmodifiableMap(parameters);
	    }

	    @Override
	    public Enumeration<String> getParameterNames() {
	        return Collections.enumeration(getParameterMap().keySet());
	    }

	    @Override
	    public String[] getParameterValues(final String name) {
	        return getParameterMap().get(name);
	    }

	}

然后写个filter偷梁换柱一下：


	package me.arganzheng.study.server.launcher.filter;

	import java.io.IOException;

	import javax.servlet.Filter;
	import javax.servlet.FilterChain;
	import javax.servlet.FilterConfig;
	import javax.servlet.ServletException;
	import javax.servlet.ServletRequest;
	import javax.servlet.ServletResponse;
	import javax.servlet.http.HttpServletRequest;

	import me.arganzheng.study.server.launcher.common.ApiHttpServletRequest;

	public class ApiResetServletRequestFilter implements Filter {

	    @Override
	    public void destroy() {
	    }

	    @Override
	    public void doFilter(ServletRequest request, ServletResponse response, FilterChain filterChain) throws IOException,
	            ServletException {
	        filterChain.doFilter(new ApiHttpServletRequest((HttpServletRequest) request), response);
	    }

	    @Override
	    public void init(FilterConfig arg0) throws ServletException {
	    }

	}

然后在web.xml中注册一下：

	<filter>
		<filter-name>ApiResetServletRequestFilter</filter-name>
		<filter-class>me.arganzheng.study.server.launcher.filter.ApiResetServletRequestFilter</filter-class>
	</filter>
	<filter-mapping>
		<filter-name>ApiResetServletRequestFilter</filter-name>
		<url-pattern>/*</url-pattern>
	</filter-mapping>


Done！现在我们可以这样子接受请求参数了：

	@RequestMapping(value = "/getAppCategory")
    @ResponseBody
    public RestResponse<Map<String, String>> getAppCategory(GetAppCategoryRequest getAppCategoryRequest) {
    	// do string
        ...
    }

GetAppCategoryRequest定义如下：

    package me.arganzheng.study.server.launcher.request;

	import java.util.List;

	public class GetAppCategoryRequest extends BaseRequest {

	    private List<String> packageNames;

	    public List<String> getPackageNames() {
	        return packageNames;
	    }

	    public void setPackageNames(List<String> packageNames) {
	        this.packageNames = packageNames;
	    }
	}

其中BaseRequest定义如下：

	public class BaseRequest {

	    // API版本
	    private String api;

	    // 产品，固定值 dl
	    @JsonProperty("pt")
	    private String product = "*";

	    // App语言标识
	    @JsonProperty("la")
	    private String language = "*";

	    // 当前渠道号
	    @JsonProperty("cf")
	    private String currentFrom = "*";

	    // 原始渠道
	    @JsonProperty("of")
	    private String oldFrom = "*";

	    // App版本
	    @JsonProperty("av")
	    private String appVersion = "*";

	    @JsonProperty("sv")
	    private String sdkVersion = "*";

	    // 系统国家标识 (两位，大写字母) http://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
	    @JsonProperty("co")
	    private String country = "*";

	    // 运营商
	    @JsonProperty("pr")
	    private String provider = "*";

	    // ...
	} 

并且可以使用_p=加密的JSON串提交，也可以用打散的key-value方式提交请求了，方便测试。
