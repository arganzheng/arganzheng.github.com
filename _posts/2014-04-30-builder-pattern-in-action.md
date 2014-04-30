---
title: 优雅的Builder模式
layout: post
---


Builder模式是GoF中的创建型设计模式中的一种。原来的意图是将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。不过现在基本主要都是用于流水接口进行优雅构建。另外，在Groovy这样的动态语言中，是一种简单方便的DSL。下面分别介绍这两种场景。


### 流水接口

这个例子来源于笔者在做动态页面缓存需求时的一个实现。具体可以参见笔者的另一篇博文：[动态页面缓存方案](http://blog.arganzheng.me/posts/http-caching.html)。其中应用控制静态页面的缓存机制可以通过如下两种方式控制：

1. 对于静态的配置，可以使用@CacheControl注解进行配置。

  @CacheControl(isPublic = true, maxAge = 300, sMaxAge = 300)

  会自动生成：Cache-Control: public, max-age=300, s-maxage=300的HTTP头。

2. 对于动态的配置，可以使用CacheControlHeader工具类方便的控制：

		int age = calculateLeftTiming();
	  	String cacheControlValue = CacheControlHeader.newBuilder()
	          .setCacheType(CacheType.PUBLIC)
	          .setMaxAge(age)
	          .setsMaxAge(age).build().stringValue();
	  	if (StringUtils.isNotBlank(cacheControlValue)) {
	    	response.addHeader("Cache-Control", cacheControlValue);
	  	}

具体实现如下：

	package me.arganzheng.study.builder.annotation;

	import java.lang.annotation.ElementType;
	import java.lang.annotation.Retention;
	import java.lang.annotation.RetentionPolicy;
	import java.lang.annotation.Target;

	/**
	 * <pre>
	 * 设置HTTP Response Header中关于Cache的控制头——Cache-Control和Expires(For HTTP 1.0)。
	 * 
	 * Cache-Control 
	 * 	public = true | false 
	 * 	max-age = delta-seconds (0 ==> no-cache | no-store )
	 *  s-maxage = delta-seconds. 
	 *  must-revalidate = true | false 
	 *  no-transform = true | false
	 *  Expires = 日期字符串，格式为yyyy-MM-dd HH:mm:ss。并且自动推算匹配的max-age。
	 * 
	 * 说明：使用annotation意味着只能支持静态配置。而不是动态的信息。所以如果value值需要动态设置的话就需要程序自己在handler method中处理了。
	 * 比如 Last-Modified和Etag。max-age和Expires有可能也需要动态，这种情况下就不能使用这个annotation。
	 * 可以考虑使用JAX-RS框架[HTTP Header Caching in JAX-RS](https://devcenter.heroku.com/articles/jax-rs-http-caching)。
	 * </pre>
	 * 
	 * @author zhengzhibin
	 * 
	 */
	@Target(ElementType.METHOD)
	@Retention(RetentionPolicy.RUNTIME)
	public @interface CacheControl {

		/**
		 * public和private都是关键字。。
		 * 
		 */
		boolean isPublic() default true;

		/**
		 * max-age = delta-seconds (0 ==> no-cache | no-store )。
		 * 
		 */
		int maxAge() default -1;

		/**
		 * s-maxage = delta-seconds.
		 */
		int sMaxAge() default -1;

		/**
		 * must-revalidate = true | false
		 */
		boolean mustRevalidate() default false;

		/**
		 * proxy-revalidate = true | false
		 */
		boolean proxyRevalidate() default false;

		/**
		 * no-transform = true | false
		 */
		boolean noTransform() default true;

		/**
		 * <pre>
		 * Expires = 日期字符串，并且自动推算匹配的max-age。
		 * HTTP标准是RFC 1123 date format, e.g., Thu, 01 Dec 1994 16:00:00 GMT.
		 * public static final String PATTERN_RFC1123 = "EEE, dd MMM yyyy HH:mm:ss zzz";
		 * 但是这个格式对于我们中国人非常不友好，还是改成这个格式方便些：yyyy-MM-dd HH:mm:ss。
		 * 
		 * </pre>
		 */
		String expires() default "";
	}


	package me.arganzheng.study.builder.common;

	import org.apache.commons.lang.StringUtils;

	/**
	 * <pre>
	 * Cache-Control Header
	 *  
	 * NOTE: 这个不会顺便增加Expires头。
	 * </pre>
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public class CacheControlHeader {
		public enum CacheType {
			PUBLIC, PRIVATE;
		}

		/**
		 * public | private
		 * 
		 */
		CacheType cacheType = null;
		/**
		 * max-age = delta-seconds (0 ==> no-cache | no-store )。
		 * 
		 */
		boolean noCache;
		boolean noStore;

		int maxAge = -1;

		/**
		 * s-maxage = delta-seconds.
		 */
		int sMaxAge = -1;

		/**
		 * must-revalidate = true | false
		 */
		boolean mustRevalidate = false;

		/**
		 * proxy-revalidate = true | false
		 */
		boolean proxyRevalidate = false;

		/**
		 * no-transform = true | false
		 */
		boolean noTransform = false;

		private CacheControlHeader() {
		}

		public static Builder newBuilder() {
			return Builder.create();
		}

		public CacheType getCacheType() {
			return cacheType;
		}

		public boolean isNoCache() {
			return noCache;
		}

		public boolean isNoStore() {
			return noStore;
		}

		public int getMaxAge() {
			return maxAge;
		}

		public int getsMaxAge() {
			return sMaxAge;
		}

		public boolean isMustRevalidate() {
			return mustRevalidate;
		}

		public boolean isProxyRevalidate() {
			return proxyRevalidate;
		}

		public boolean isNoTransform() {
			return noTransform;
		}

		public String stringValue() {
			StringBuilder sb = new StringBuilder();

			if (cacheType == CacheType.PUBLIC) {
				append(sb, "public");
			} else if (cacheType == CacheType.PRIVATE) {
				append(sb, "private");
			}

			if (mustRevalidate) {
				append(sb, "must-revalidate");
			}

			if (noTransform) {
				append(sb, "no-transform");
			}

			if (proxyRevalidate) {
				append(sb, "proxy-revalidate");
			}

			if (maxAge == 0) {
				append(sb, "max-age=0");
				noCache = noStore = true;
			} else if (maxAge > 0) {
				append(sb, "max-age=" + maxAge);
			}

			if (sMaxAge == 0) {
				append(sb, "s-maxage=0");
				noCache = noStore = true;
			} else if (sMaxAge > 0) {
				append(sb, "s-maxage=" + sMaxAge);
			}

			if (noCache || noStore) {
				append(sb, "no-cache,no-store");
			}

			return sb.toString();
		}

		private StringBuilder append(StringBuilder sb, String str) {
			if (sb == null) {
				sb = new StringBuilder();
				sb.append(str);
			} else {
				String s = StringUtils.trimToEmpty(sb.toString());
				if (StringUtils.isEmpty(s) || s.charAt(s.length() - 1) == ',') {
					sb.append(str);
				} else {
					sb.append(',').append(str);
				}
			}

			return sb;
		}


	package me.arganzheng.study.builder.interceptor;

	import java.text.ParseException;
	import java.util.Date;

	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpServletResponse;

	import org.apache.commons.lang.StringUtils;
	import org.apache.commons.lang.time.DateUtils;
	import org.apache.commons.lang.time.FastDateFormat;
	import org.apache.log4j.Logger;
	import org.springframework.web.method.HandlerMethod;
	import org.springframework.web.servlet.ModelAndView;
	import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;
	import org.springframework.web.servlet.resource.ResourceHttpRequestHandler;

	import me.arganzheng.study.builder.annotation.CacheControl;
	import me.arganzheng.study.builder.common.CacheControlHeader;
	import me.arganzheng.study.builder.common.CacheControlHeader.CacheType;

	/**
	 * 处理@CacheControl注解
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public class CacheControlInterceptor extends HandlerInterceptorAdapter {
		public static final String EXPIRES_PATTERN = "yyyy-MM-dd HH:mm:ss";
		public static final String PATTERN_RFC1123 = "EEE, dd MMM yyyy HH:mm:ss zzz";

		private static final Logger logger = Logger.getLogger(CacheControlInterceptor.class);

		/**
		 * 根据@CacheControl注解设置对应的HTTP Response Header
		 */
		public void postHandle(HttpServletRequest request, HttpServletResponse response,
				Object handler, ModelAndView modelAndView) throws Exception {
			if (handler instanceof HandlerMethod) {
				HandlerMethod method = (HandlerMethod) handler;
				CacheControl cacheControlAnnotation = method.getMethodAnnotation(CacheControl.class);
				if (cacheControlAnnotation == null
						|| StringUtils.isNotEmpty(response.getHeader("Cache-Control"))) {
					return;
				}

				if (logger.isDebugEnabled()) {
					logger.debug("Response with CacheControl Set!");
				}

				CacheControlHeader.Builder cc = CacheControlHeader.newBuilder();

				if (cacheControlAnnotation.isPublic()) {
					cc.setCacheType(CacheType.PUBLIC);
				} else {
					cc.setCacheType(CacheType.PRIVATE);
				}

				Date expiresDate = null;
				if (cacheControlAnnotation.mustRevalidate()) {
					cc.setMustRevalidate(true);
				}

				if (cacheControlAnnotation.noTransform()) {
					cc.setNoTransform(true);
				}

				if (cacheControlAnnotation.proxyRevalidate()) {
					cc.setProxyRevalidate(true);
				}

				if (cacheControlAnnotation.maxAge() >= 0) {
					cc.setMaxAge(cacheControlAnnotation.maxAge());
				}
				if (cacheControlAnnotation.sMaxAge() >= 0) {
					cc.setsMaxAge(cacheControlAnnotation.sMaxAge());
				}

				if (StringUtils.isNotEmpty(cacheControlAnnotation.expires())) {
					try {
						expiresDate = DateUtils.parseDate(cacheControlAnnotation.expires(),
								new String[] { EXPIRES_PATTERN, PATTERN_RFC1123 });

						Date now = new Date();

						long age = expiresDate.getTime() - now.getTime();
						long maxAge = age / 1000;

						if (cacheControlAnnotation.maxAge() < 0) { // maxAge not set
							cc.setMaxAge(Long.valueOf(maxAge).intValue());
						}
						if (cacheControlAnnotation.sMaxAge() < 0) { // sMaxAge not
																	// set
							cc.setsMaxAge(Long.valueOf(maxAge).intValue());
						}
					} catch (ParseException e) {
						logger.error("parse expires date failed! Expired Date = "
								+ cacheControlAnnotation.expires());
					}
				}

				String value = cc.build().stringValue();
				if (StringUtils.isNotBlank(value)) {
					response.addHeader("Cache-Control", value);
				}

				if (expiresDate != null) {
					response.addHeader("Expires",
							FastDateFormat.getInstance(PATTERN_RFC1123).format(expiresDate));
				}
			}
		}
	}


		/**
		 * <pre>
		 * Builder for CacheControlHeader. 
		 * 
		 * 这里没有处理字段是否有被显示设置问题。可以考虑每个字段增加一个标志字段，表面是否被显示设值。
		 * </pre>
		 * 
		 * @author zhengzhibin
		 * 
		 */
		public static class Builder {
			private CacheControlHeader cacheControlHeader;

			public static Builder create() {
				return new Builder();
			}

			private Builder() {
				this.cacheControlHeader = new CacheControlHeader();
			}

			public Builder setCacheType(CacheType cacheType) {
				cacheControlHeader.cacheType = cacheType;
				return this;
			}

			public Builder setNoCache(boolean noCache) {
				cacheControlHeader.noCache = noCache;
				return this;
			}

			public Builder setNoStore(boolean noStore) {
				cacheControlHeader.noStore = noStore;
				return this;
			}

			public Builder setMaxAge(int maxAge) {
				cacheControlHeader.maxAge = maxAge;
				return this;
			}

			public Builder setsMaxAge(int sMaxAge) {
				cacheControlHeader.sMaxAge = sMaxAge;
				return this;
			}

			public Builder setMustRevalidate(boolean mustRevalidate) {
				cacheControlHeader.mustRevalidate = mustRevalidate;
				return this;
			}

			public Builder setProxyRevalidate(boolean proxyRevalidate) {
				cacheControlHeader.proxyRevalidate = proxyRevalidate;
				return this;
			}

			public Builder setNoTransform(boolean noTransform) {
				cacheControlHeader.noTransform = noTransform;
				return this;
			}

			public CacheControlHeader build() {
				return cacheControlHeader;
			}
		}
	}
	

常见的Builder模式案例：

1. Quartz
2. SQL 对象查询
3. android AlertDialog.Builder && NotificationCompat.Builder

### DSL

具体参见笔者以前的一篇文章：[Groovy元编程——使用invokeMethod和闭包构建DSL和Builder](http://blog.arganzheng.me/posts/groovy-builder-in-action.html)。这里就不赘述了。


参考文章
--------

1. [Another builder pattern for Java](http://blog.crisp.se/2013/10/09/perlundholm/another-builder-pattern-for-java)
2. [make-it-easy](https://code.google.com/p/make-it-easy/)