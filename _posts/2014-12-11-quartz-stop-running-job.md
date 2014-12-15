---
title: Quartz突然停止执行问题
layout: post
---


### 现象

线上的API监控系统有一段时间没有发报警邮件了，觉得有点异常，测试同学特意写了一个抛出错误的python校验器，结果发现也是没有收到邮件。


### 原因

表面现象是没有收到报警邮件，那么有可能是：

1. job没有执行
2. 校验器没有错误：根据前面的验证，可以直接排除。
3. 邮件没有发送

通过在job执行前后打日志，发现一个问题：

有些job只有start日志，没有end日志，也就是说卡住了。然后整个Quartz的QRTZ_FIRED_TRIGGERS表就占满了处于EXECUTING状态的job（我们的Quartz线程池个数配置为8个）。然后线程池空了，没有线程可以执行新的job。

那么job是卡在哪里呢？

通过`jstack -l pid`发现，所有的Quartz执行线程都卡在IO上：

7个是网络IO：

![quartz-jstack-blocking-in-file-io](/media/images/quartz-jstack-blocking-in-net-io.jpg)

还有一个是文件IO：

![quartz-jstack-blocking-in-file-io](/media/images/quartz-jstack-blocking-in-file-io.jpg)

根据堆栈信息，很容易找到有问题的代码。

网络IO那个是因为HttpURLConnection没有设置ReadTimeout，那么默认是block forever。还有貌似在getResponseCode之前要先获取一下链接的InputStream。

而文件IO那个，就比较不好处理。这是一个Java调用外部python脚本，然后读取python脚本返回的信息。如果python卡住，那么java线程会一直卡住。问了一下测试同学，说python脚本没有耗时操作，应该很快就返回才是。所以可以等待如果python脚本20分钟没有执行完，就kill掉它。另外，对于读取python返回的数据block问题，由于Java1.7之前的文件IO都是block IO，而且不支持timeout，所以需要特别处理。具体参见笔者前面写的一篇文章 [Java文件读取支持timeout](http://blog.arganzheng.me/posts/java-file-reading.html)。

最后的解决方案是：

	package me.arganzheng.study.quartz.monitor.service;

	import java.io.BufferedReader;
	import java.io.IOException;
	import java.io.InputStream;
	import java.io.InputStreamReader;
	import java.io.StringWriter;
	import java.util.ArrayList;
	import java.util.List;

	import org.apache.commons.lang.StringUtils;
	import org.apache.log4j.Logger;

	import me.arganzheng.study.quartz.constants.Constants;

	public class ScriptExecutor {
		private static final Logger logger = Logger.getLogger(ScriptExecutor.class);

		private static final int TIMEOUT = 20000;

		/**
		 * @param exec python, e.g.
		 * @param path script path
		 * @return STDOUT STDERR的內容以異常形式拋出。
		 * @throws IOException
		 * @throws ApiValidatorException
		 */
		public static String executeScript(String exec, String path, String... args)
				throws ApiValidatorException, IOException {
			List<String> cmd = new ArrayList<String>();

			cmd.add(exec);
			cmd.add(path);

			for (String arg : args) {
				// java会自动加上单引号，避免shell转义
				cmd.add(arg);
			}

			Runtime rt = Runtime.getRuntime();
			Process p = rt.exec(cmd.toArray(new String[cmd.size()]));
			// retrieve output from python script
			InputStream es = p.getErrorStream();
			if (es != null) {
				String error = null;
				try {
					error = getStreamAsStringWithTimeout(p.getErrorStream(), Constants.CHARSET_UTF8,
							TIMEOUT);
				} catch (IOException e) { // read TIMEOUT, kill the process
					logger.error("IOException for getStreamAsString with timeout=" + TIMEOUT, e);
					// kill the script process
					p.destroy();
					return null;
				}
				if (StringUtils.isNotEmpty(error)) {
					throw new ApiValidatorException(error);
				}
			}

			// 带timeout的read，如果timeout，则kill掉process进程。
			String result = null;
			try {
				result = getStreamAsStringWithTimeout(p.getInputStream(), Constants.CHARSET_UTF8,
						TIMEOUT);
			} catch (IOException e) { // read TIMEOUT, kill the process
				logger.error("IOException for getStreamAsString with timeout=" + TIMEOUT, e);
			}
			// kill the script process
			p.destroy();

			return result;
		}

		// InputStream => InputStreamReader(with charset) => String
		private static String getStreamAsStringWithTimeout(InputStream is, String charset,
				int timeoutMillis) throws IOException {
			BufferedReader reader = new BufferedReader(new InputStreamReader(is, charset));
			StringWriter writer = new StringWriter();

			char[] chars = new char[256];

			int count = 0;
			long maxTimeMillis = System.currentTimeMillis() + timeoutMillis;
			while (System.currentTimeMillis() < maxTimeMillis) {
				int readLength = java.lang.Math.min(is.available(), 256);
				count = reader.read(chars, 0, readLength);
				if (count == -1) {
					break;
				}
				writer.write(chars, 0, count);
			}
			return writer.toString();
		}
	}


修复后，观察了一段时间，发现在也没有quartz线程block住的问题了，同时也没有python进程一直存活的情况。

