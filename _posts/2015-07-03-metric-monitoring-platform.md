---
title: Metric监控系统
layout: post
---

背景
---

对系统运行中的一些情况进行统一上报，监控，统计和展示。允许业务自定义监控项。有助于了解系统的运行情况，如性能，访问频率，Cache命中率等。


实现
----

具体参见笔者写的另一篇文章:[java服务端监控平台设计](http://blog.arganzheng.me/posts/java-monitor-platform.html)


使用
---

首先需要配置pom.xml，引入monitor-client包：

	<dependency>
		<groupId>com.baidu.global.mobile.server.monitor.client</groupId>
		<artifactId>monitor-client</artifactId>
		<version>1.0.0</version>
	</dependency>	


然后创建agent。这里有两种方式。第一种是通过Spring配置，这也是推荐的方式：

	<!-- monitor agent -->
	<bean id="agent" class="com.baidu.global.mobile.server.monitor.agent.AgentImpl">
		<constructor-arg value="classpath:${env}_agent.xml" />
	</bean>

其中agent.xml的配置如下：

	<agent>
		<region>HK</region>
		<application>guanxing</application>
		<threadPoolSize>2</threadPoolSize>
		<loadBalancerName>WRR</loadBalancerName>
		<!-- <servers>10.242.112.38,10.242.116.27,10.242.117.36</servers> -->
		<endpoints>
			<endpoint>
				<host>10.242.112.38</host>
				<port>4567</port>
				<!-- <weight>1</weight> <maxFails>1</maxFails> <failTimeout>10</failTimeout> 
					<timeout>6</timeout> -->
			</endpoint>
			<endpoint>
				<host>10.242.116.27</host>
				<port>4567</port>
			</endpoint>
			<endpoint>
				<host>10.242.117.36</host>
				<port>4567</port>
			</endpoint>
		</endpoints>
	</agent>	

从配置文件可以大概知道我们agent上报的实现细节：

1. 采用线程池异步上报
2. 支持多个metric收集服务，客户端进行负载均衡。目前只支持Weighted Round Robin方式。从Nginx实现挖过来的。
3. 目前不支持server的自动发现和注销。


也可以通过AgentFactory获取agent:

	package com.baidu.global.mobile.server.monitor.agent;

	/**
	 * <pre>
	 * 因为Agent的本身是有线程池和网络连接池，开销比较大，不适合于每次都New一个，最好是一个应用共用一个，所以提供这么一个单例工厂方法。
	 * 
	 * 建议如果使用Spring的话，直接在Spring配置就可以了。默认就是单例，而且可以很方便的注入。
	 * 
	 * <bean id="agent"
	 * 		class="com.baidu.global.mobile.server.monitor.agent.AgentImpl">
	 * 		<constructor-arg value="classpath:${env}_agent.xml" />
	 * </bean>
	 * 
	 * </pre>
	 * 
	 * @author argan
	 */
	public class AgentFactory {
	    private static volatile Agent instance = null;
	
	    public static Agent getInstance(String configLocation) {
	        if (instance == null) {
	            synchronized (AgentFactory.class) {
	                if (instance == null) {
	                    instance = new AgentImpl(configLocation);
	                }
	            }
	        }
	        return instance;
	    }
	
	    public static Agent getInstance(AgentConfig config) {
	        if (instance == null) {
	            synchronized (AgentFactory.class) {
	                if (instance == null) {
	                    instance = new AgentImpl(config);
	                }
	            }
	        }
	        return instance;
	    }
	
	}


创建Agent之后，就可以调用Agent接口进行metric上报了：

	package com.baidu.global.mobile.server.monitor.agent;

	import java.util.Map;
	
	public interface Agent {
	    // Counter
	    void increment(String name);
	
	    void increment(String name, long delta);
	
	    void increment(String category, String name);
	
	    void increment(String category, String name, long delta);
	
	    void increment(String category, String name, long delta, Map<String, String> tags);
	
	    // Guage
	    void guage(String name, double value);
	
	    void guage(String category, String name, double value);
	
	    void guage(String category, String name, double value, Map<String, String> tags);
	
	    // Label
	    void label(String name, String value);
	
	    void label(String category, String name, String value);
	
	    void label(String category, String name, String value, Map<String, String> tags);
	
	    void addMetrics(Metric...metrics);
	}


Agent的接口很简单，其实就一个：void addMetrics(Metric... metrics);
其中Metric定义如下：

	package com.baidu.global.mobile.server.monitor.agent;

	import static com.baidu.global.mobile.server.monitor.agent.utils.Preconditions.checkNotNullOrEmpty;
	import static com.google.common.base.Preconditions.checkArgument;
	import static com.google.common.base.Preconditions.checkNotNull;
	
	import java.util.Collections;
	import java.util.HashMap;
	import java.util.Map;
	import java.util.zip.DataFormatException;
	
	import org.springframework.util.Assert;
	import org.springframework.util.StringUtils;
	
	import com.baidu.global.mobile.server.agent.common.MetricType;
	import com.baidu.global.mobile.server.agent.common.Region;
	import com.baidu.global.mobile.server.monitor.agent.utils.WebUtils;
	
	/**
	 * <pre>
	 * 
	 * Thrift生成的类型不好用。
	 * 
	 * 	1. 不支持Object类型。
	 * 	2. 无法进行校验
	 *  3. 没有Builder模式。
	 * 所以定义了一个，再进行转换。
	 * 
	 * 但是builder模式有个问题，就是不好处理NULL，所以这里仍然提供setter方法。
	 * 
	 * </pre>
	 * 
	 * @author argan
	 *
	 */
	public class Metric {
	    // 哪个应用产生的日志：比如Nantianmen, Guanxing, Spider,
	    // Nginx等。这个需要应用配置，由client发送给Server。
	    private String application;
	
	    // 服务器区域，目前主要有HK，ID。有这个可以进行区域范围的筛选，没有的话只能根据host判断区域。
	    private Region region = Region.All;
	
	    // 这里不采用灵活的Map<String, String> tags，而是每增加一个tag需要显式的定义一下。
	    // 日志从哪台机器打过来的。可以动态获取到。如果client没有指定，那么log server以client IP地址为准。
	    private String host;
	
	    // MetricType, 会影响后面的body格式。
	    private MetricType type = MetricType.Metric;
	
	    // Category，用于过滤筛选，比如要找出Top10
	    // 访问量的URL。定义成String类型，用户自己区别开，常见的有URI，Service，DataSource, Memory, Thread等。
	    // 因为底层TSDB不支持Key的模糊查询，所以这个还是很有必要的。
	    private String category = "All";
	
	    // metricName，建议是package格式，如：sys.cpu.nice
	    private String name;
	
	    // 如果没有上传，则为server收集到该日志的时间。
	    private long timestamp;
	
	    // 目前只支持Long(Counter)和Double(Guage)和String(lable)。
	    private Object value;
	
	    // custom tags，比如Creator等
	    private Map<String, String> tags = new HashMap<String, String>();
	
	    public Metric() {
	    }
	
	    public Metric(String name) {
	        this.name = checkNotNullOrEmpty(name);
	    }
	
	    public void setApplication(String application) {
	        this.application = application;
	    }
	
	    public void setRegion(Region region) {
	        this.region = region;
	    }
	
	    public void setHost(String host) {
	        this.host = host;
	    }
	
	    public void setType(MetricType type) {
	        this.type = type;
	    }
	
	    public void setCategory(String category) {
	        this.category = category;
	    }
	
	    public void setName(String name) {
	        this.name = name;
	    }
	
	    public void setTimestamp(long timestamp) {
	        this.timestamp = timestamp;
	    }
	
	    public void setValue(Object value) {
	        this.value = checkNotNull(value);
	    }
	
	    public void setTags(Map<String, String> tags) {
	        this.tags = tags;
	    }
	
	    /**
	     * Time when the data point was measured.
	     *
	     * @return time when the data point was measured
	     */
	    public long getTimestamp() {
	        return timestamp;
	    }
	
	    public Object getValue() {
	        return value;
	    }
	
	    public String stringValue() {
	        return (String) value;
	    }
	
	    public long longValue() throws DataFormatException {
	        try {
	            return ((Number) value).longValue();
	        } catch (Exception e) {
	            throw new DataFormatException("Value is not a long");
	        }
	    }
	
	    public double doubleValue() throws DataFormatException {
	        try {
	            return ((Number) value).doubleValue();
	        } catch (Exception e) {
	            throw new DataFormatException("Value is not a double");
	        }
	    }
	
	    public boolean isDoubleValue() {
	        if (value instanceof Number) {
	            return !(((Number) value).doubleValue() == Math.floor(((Number) value).doubleValue()));
	        }
	        return false;
	    }
	
	    public boolean isIntegerValue() {
	        if (value instanceof Number) {
	            return ((Number) value).doubleValue() == Math.floor(((Number) value).doubleValue());
	        }
	        return false;
	    }
	
	    public boolean isStringValue() {
	        return value.getClass().equals(String.class);
	    }
	
	    public static void main(String[] args) {
	        long l = 122L;
	        int i = 10;
	        Double d = 12.50;
	        String s = "hello world";
	
	        Metric m = new Metric();
	        m.setValue(l);
	        System.out.println(m.getValue().getClass());
	
	        m.setValue(i);
	        System.out.println(m.getValue().getClass());
	
	        m.setValue(d);
	        System.out.println(m.getValue().getClass());
	
	        m.setValue(s);
	        System.out.println(m.getValue().getClass());
	
	        System.out.println(m.isStringValue());
	    }
	
	    /**
	     * Returns the metric name.
	     *
	     * @return metric name
	     */
	    public String getName() {
	        return name;
	    }
	
	    /**
	     * Returns the tags associated with the data point.
	     *
	     * @return tag for the data point
	     */
	    public Map<String, String> getTags() {
	        return Collections.unmodifiableMap(tags);
	    }
	
	    public String getApplication() {
	        return application;
	    }
	
	    public Region getRegion() {
	        return region;
	    }
	
	    public String getHost() {
	        return host;
	    }
	
	    public MetricType getType() {
	        return type;
	    }
	
	    public String getCategory() {
	        return category;
	    }
	
	    public void validate() {
	        if (this.timestamp == 0) {
	            this.timestamp = System.currentTimeMillis();
	        }
	
	        if (StringUtils.isEmpty(this.host)) {
	            this.setHost(WebUtils.getLocalNetWorkIp());
	        }
	
	        Assert.notNull(this.name, "metric name is required!");
	        Assert.notNull(this.application, "application is required!");
	        Assert.notNull(this.type, "type is required!");
	        Assert.notNull(this.category, "category is required!");
	        Assert.notNull(this.region, "region is required!");
	        Assert.notNull(this.host, "host is required!");
	    }
	
	    /**
	     * This method starts the building process for Metrics.
	     */
	    public static Builder start() {
	        return new Builder();
	    }
	
	    public static class Builder {
	        private final Metric metric = new Metric();
	
	        public Builder application(String application) {
	            checkNotNullOrEmpty(application);
	            metric.application = application;
	
	            return this;
	        }
	
	        public Builder host(String host) {
	            checkNotNullOrEmpty(host);
	            metric.host = host;
	
	            return this;
	        }
	
	        public Builder category(String category) {
	            checkNotNullOrEmpty(category);
	            metric.category = category;
	
	            return this;
	        }
	
	        public Builder region(Region region) {
	            Assert.notNull(region);
	            metric.region = region;
	
	            return this;
	        }
	
	        public Builder type(MetricType type) {
	            Assert.notNull(type);
	            metric.type = type;
	
	            return this;
	        }
	
	        public Builder name(String name) {
	            checkNotNullOrEmpty(name);
	            metric.name = name;
	
	            return this;
	        }
	
	        public Builder timestamp(long timestamp) {
	            metric.timestamp = timestamp;
	
	            return this;
	        }
	
	        public Builder value(Object value) {
	            metric.value = checkNotNull(value);
	
	            return this;
	        }
	
	        /**
	         * Adds a tag to the data point.
	         *
	         * @param name tag identifier
	         * @param value tag value
	         * @return the metric the tag was added to
	         */
	        public Builder addTag(String name, String value) {
	            checkNotNullOrEmpty(name);
	            checkNotNullOrEmpty(value);
	            metric.getTags().put(name, value);
	
	            return this;
	        }
	
	        /**
	         * Adds tags to the data point.
	         * 
	         * @param tags map of tags
	         * @return the metric the tags were added to
	         */
	        public Builder addTags(Map<String, String> tags) {
	            checkNotNull(tags);
	            metric.getTags().putAll(tags);
	
	            return this;
	        }
	
	        /**
	         * set the data point for the metric.
	         *
	         * @param timestamp when the measurement occurred
	         * @param value the measurement value
	         * @return the metric
	         */
	        protected Builder innerAddDataPoint(long timestamp, Object value) {
	            checkArgument(timestamp > 0);
	            metric.timestamp = timestamp;
	            metric.value = checkNotNull(value);
	
	            return this;
	        }
	
	        /**
	         * Adds the data point to the metric with a timestamp of now.
	         *
	         * @param value the measurement value
	         * @return the metric
	         */
	        public Builder setDataPoint(long value) {
	            return innerAddDataPoint(System.currentTimeMillis(), value);
	        }
	
	        public Builder setDataPoint(long timestamp, long value) {
	            return innerAddDataPoint(timestamp, value);
	        }
	
	        /**
	         * Adds the data point to the metric.
	         *
	         * @param timestamp when the measurement occurred
	         * @param value the measurement value
	         * @return the metric
	         */
	        public Builder setDataPoint(long timestamp, double value) {
	            return innerAddDataPoint(timestamp, value);
	        }
	
	        /**
	         * Adds the data point to the metric with a timestamp of now.
	         *
	         * @param value the measurement value
	         * @return the metric
	         */
	        public Builder setDataPoint(double value) {
	            return innerAddDataPoint(System.currentTimeMillis(), value);
	        }
	
	        /**
	         * Adds the data point to the metric.
	         *
	         * @param timestamp when the measurement occurred
	         * @param value the measurement value
	         * @return the metric
	         */
	        public Builder setDataPoint(long timestamp, String value) {
	            return innerAddDataPoint(timestamp, value);
	        }
	
	        /**
	         * Adds the data point to the metric with a timestamp of now.
	         *
	         * @param value the measurement value
	         * @return the metric
	         */
	        public Builder setDataPoint(String value) {
	            return innerAddDataPoint(System.currentTimeMillis(), value);
	        }
	
	        public Metric build() {
	            metric.validate();
	            return metric;
	        }
	
	    }
	
	}


Counter和Guage和Label其实是特殊的Metric，这里只是提供了一个便利的接口。

比如，我们要统计接口的调用频率和耗时，那么可以在Spring MVC的Interceptor中这么做：



	/**
	 * 每个方法处理时间监控。
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public class PerformanceMonitorInterceptor extends HandlerInterceptorAdapter {

		private static final Logger logger = Logger.getLogger(PerformanceMonitorInterceptor.class);

		@Autowired(required = false)
		private Agent agent;

		@Override
		public boolean preHandle(HttpServletRequest request,
				HttpServletResponse response, Object handler) throws Exception {
			long startTime = System.currentTimeMillis();
			request.setAttribute("startTime", startTime);

			return super.preHandle(request, response, handler);
		}

		public void postHandle(HttpServletRequest request,
				HttpServletResponse response, Object handler,
				ModelAndView modelAndView) throws Exception {
			if (handler instanceof HandlerMethod) {
				HandlerMethod method = (HandlerMethod) handler;
				long startTime = (Long) request.getAttribute("startTime");
				long processTime = System.currentTimeMillis() - startTime;

				if (agent == null) {
					logger.warn("Agent is null, metric report disabled!");
				}
				// send to monitor server
				String controllerMethod = method.getBean().getClass()
						.getCanonicalName()
						+ "." + method.getMethod().getName();
				Map<String, String> tags = getGeneralParams(request);
				tags.put("controller_method", controllerMethod);
				agent.guage("URI", request.getRequestURI(), processTime, tags);
			}

		}

		private Map<String, String> getGeneralParams(HttpServletRequest request) {
			Map<String, String> gp = new HashMap<String, String>();

			putIfNotEmpty(gp, "language", request.getParameter("la"));
			putIfNotEmpty(gp, "country", request.getParameter("co"));
			putIfNotEmpty(gp, "uuid", request.getParameter("uuid"));

			return gp;
		}

		public void putIfNotEmpty(Map<String, String> map, String key, String value) {
			if (StringUtils.isNotEmpty(value)) {
				map.put(key, value);
			}
		}
	}

然后在监控平台就可以看到每一次调用的情况了。可以查看某个时间段的某个metric的统计数据。不过现在的数据展示方面还是比较弱的。metric之间没有关联性查询。

另外，尽管Agent上报方式其实蛮简单的，不过更简单的方式是注解上报，后面会考虑加上。

发挥你的想象力，把它用起来:)

感兴趣的同学可以参考这篇文章: [Observability at Twitter](https://blog.twitter.com/2013/observability-at-twitter)。介绍了Twitter的监控体系，其中就有我们现在在做的metric监控平台。


