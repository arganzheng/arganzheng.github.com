---
layout: post
title: 使用拦截器做简单的性能监控
---

需求
----

监控所有的Controller Handler Method，对处理时间超过阈值的邮件报警。


实现
----

Spring MVC提供了interceptor的机制，可以方便的进行对handler method进行AOP。具体实现如下。

首先在Spring MVC配置文件配置一下interceptor：

    <!-- Configures Handler Interceptors -->
	<mvc:interceptors>
		<mvc:interceptor>
			<mvc:mapping path="/**" />
			<bean class="me.arganzheng.study.springmvc.interceptor.PerformanceMonitorInterceptor" />
		</mvc:interceptor>
	</mvc:interceptors>

PerformanceMonitorInterceptor的实现非常简单直观：

    package me.arganzheng.study.springmvc.interceptor;

    import java.util.HashMap;
    import java.util.Map;
    import java.util.Map.Entry;
    
    import javax.servlet.http.HttpServletRequest;
    import javax.servlet.http.HttpServletResponse;
    
    import org.apache.log4j.Logger;
    import org.springframework.web.method.HandlerMethod;
    import org.springframework.web.servlet.ModelAndView;
    import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;
    
    import me.arganzheng.study.springmvc.Constants;
    import me.arganzheng.study.springmvc.utils.WebUtils;
    import em.arganzheng.study.springmvc.utils.AlarmManager;
    
    /**
     * 每个方法处理时间监控，如果超过阈值，则邮件报警。
     * 
     * @author zhengzhibin
     * 
     */
    public class PerformanceMonitorInterceptor extends HandlerInterceptorAdapter {
    
    	private static final Logger logger = Logger.getLogger(PerformanceMonitorInterceptor.class);
    
    	private static final long MAX_REQUEST_NEED = 10000L;
    	private static final int MB = 1024 * 1024;
    
    	@Override
    	public boolean preHandle(HttpServletRequest request, HttpServletResponse response,
    			Object handler) throws Exception {
    		long startTime = System.currentTimeMillis();
    		request.setAttribute("startTime", startTime);
    
    		return super.preHandle(request, response, handler);
    	}
    
    	public void postHandle(HttpServletRequest request, HttpServletResponse response,
    			Object handler, ModelAndView modelAndView) throws Exception {
    		if (handler instanceof HandlerMethod) {
    			HandlerMethod method = (HandlerMethod) handler;
    			long startTime = (Long) request.getAttribute("startTime");
    			long processTime = System.currentTimeMillis() - startTime;
    			if (processTime > MAX_REQUEST_NEED) {
    				String message = generateAlarmMessage(processTime, method, request.getRequestURI(),
    						request.getParameterMap());
    				AlarmManager.alarmEmailAsync("arganzheng@arganzheng.me",
    						Constants.AlarmEmail.ALARM_TITLE, Constants.AlarmEmail.SERVER_PEOLE,
    						message);
    				logger.warn("Process Timeout! " + message);
    			}
    		}
    	}
    
    	private String generateAlarmMessage(long processTime, HandlerMethod method, String requestUri,
    			Map<String, String[]> parameterMap) {
    		StringBuilder sb = new StringBuilder();
    		sb.append("RequestUri: ").append(requestUri + "\n");
    		sb.append("ControllerMethod: ")
    				.append(method.getBean().getClass().getCanonicalName() + "."
    						+ method.getMethod().getName());
    		for (Entry<String, String[]> en : parameterMap.entrySet()) {
    			sb.append("   ");
    			sb.append(en.getKey() + "=");
    			String[] v = en.getValue();
    			if (v.length > 1) {
    				for (String vv : v) {
    					sb.append(vv + ",");
    				}
    			} else {
    				sb.append(v[0]);
    			}
    			sb.append("\n");
    		}
    		sb.append("ProcessTime: ").append(processTime).append(" ms\n");
    
    		sb.append("LocalNetWorkIp: ").append(WebUtils.getLocalNetWorkIp() + "\n");
    
            // Total number of processors or cores available to the JVM
            sb.append("Available processors (cores): ")
                    .append(Runtime.getRuntime().availableProcessors()).append("\n");

            long totalMemory = Runtime.getRuntime().totalMemory();
            long freeMemory = Runtime.getRuntime().freeMemory();

            // Total memory currently available to the JVM
            sb.append("Total memory available to JVM (MB): ").append(totalMemory / MB).append("\n");
            // used memory
            sb.append("Used Memory (MBs): ").append((totalMemory - freeMemory) / MB).append("\n");
            // Total amount of free memory available to the JVM
            sb.append("Free memory (MBs): ").append(freeMemory / MB).append("\n");
            // Maximum amount of memory the JVM will attempt to use
            sb.append("Maximum memory (MBs): ").append(Runtime.getRuntime().maxMemory() / MB)
                    .append("\n");

    		return sb.toString();
    	}
    }

