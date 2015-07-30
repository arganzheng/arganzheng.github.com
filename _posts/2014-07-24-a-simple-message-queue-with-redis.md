---
title: 使用Redis做简单的消息队列
layout: post
---


消息队列的好处直接，就是 1. 解耦；2. 削峰；3. 异步化。基本上和缓存一样是居家必备之良药。然而消息队列虽然重要，但是同时其实是蛮重的一个组件。

所以就想能不能先简单的通过Redis来实现消息队列呢？不考虑PubSub、分布式、持久化、事务等复杂的情况。就像JDK的各种Queue一样。答案当然是可以的，因为Redis提供的list数据结构就非常适合做消息队列。

> ### [List](https://github.com/springside/springside4/wiki/redis#24-list)
> 
> List是一个双向链表，支持双向的Pop/Push，江湖规矩一般从左端Push，右端Pop——LPush/RPop，而且还有Blocking的版本BLPop/BRPop，客户端可以阻塞在那直到有消息到来，所有操作都是O(1)的好孩子，可以当Message Queue来用。当多个Client并发阻塞等待，有消息入列时谁先被阻塞谁先被服务。任务队列系统Resque是其典型应用。
>
> 还有RPopLPush/ BRPopLPush，弹出来返回给client的同时，把自己又推入另一个list，LLen获取列表的长度。
> 
> 还有按值进行的操作：LRem(按值删除元素)、LInsert(插在某个值的元素的前后)，复杂度是O(N)，N是List长度，因为List的值不唯一，所以要遍历全部元素，而Set只要O(log(N))。
>
> 按下标进行的操作：下标从0开始，队列从左到右算，下标为负数时则从右到左。
>
> * LSet ，按下标设置元素值。
> * LIndex，按下标返回元素。
> * LRange，不同于POP直接弹走元素，只是返回列表内一段下标的元素，是分页的最爱。
> * LTrim，限制List的大小，比如只保留最新的20条消息。
> 
> 复杂度也是O(N)，其中LSet的N是List长度，LIndex的N是下标的值，LRange的N是start的值+列出元素的个数，因为是链表而不是数组，所以按下标访问其实要遍历链表，除非下标正好是队头和队尾。LTrim的N是移除元素的个数。
>
> 在消息队列中，并没有JMS的ack机制，如果消费者把job给Pop走了又没处理完就死机了怎么办？
> * 解决方法之一是加多一个sorted set，分发的时候同时发到list与sorted set，以分发时间为score，用户把job做完了之后要用ZREM消掉sorted set里的job，并且定时从sorted set中取出超时没有完成的任务，重新放回list。
> * 另一个做法是为每个worker多加一个的list，弹出任务时改用RPopLPush，将job同时放到worker自己的list中，完成时用LREM消掉。如果集群管理(如zookeeper)发现worker已经挂掉，就将worker的list内容重新放回主list。


大概实现如下。

首先定义了一个消息处理接口：

	package me.arganzheng.study.message.queue.processor;

	/**
	 * 消息处理接口。这里不引入MessageConverter的概念，只接收textMessage，一般来说是JSON。
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public interface MessageHandler {
		void onMessage(String message);
	}

业务只需要实现这个接口就可以了。

然后定义一个QueueProcessor，处理从队列中获取消息调用业务MessageHandler的骨架逻辑：

	package me.arganzheng.study.message.queue.processor;

	import java.util.List;
	import java.util.concurrent.Executor;

	import javax.annotation.PostConstruct;

	import org.apache.commons.lang.StringUtils;
	import org.apache.log4j.Logger;
	import org.springframework.util.Assert;

	import redis.clients.jedis.Jedis;
	import redis.clients.jedis.JedisPool;

	/**
	 * 消息监听器。
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public class QueueProcessor {
		public static final Logger logger = Logger.getLogger(QueueProcessor.class);

		private Jedis jedis;

		private JedisPool jedisPool;

		private String host;
		private int port;
		private int timeout;
		private String password;

		private String queueName;

		private MessageHandler messageHandler;

		public void setJedis(Jedis jedis) {
			this.jedis = jedis;
		}

		public void setJedisPool(JedisPool jedisPool) {
			this.jedisPool = jedisPool;
		}

		public void setHost(String host) {
			this.host = host;
		}

		public void setPort(int port) {
			this.port = port;
		}

		public void setTimeout(int timeout) {
			this.timeout = timeout;
		}

		public void setPassword(String password) {
			this.password = password;
		}

		public void setQueueName(String queueName) {
			this.queueName = queueName;
		}

		public void setMessageHandler(MessageHandler messageHandler) {
			this.messageHandler = messageHandler;
		}

		private Executor executor;

		public void setExecutor(Executor executor) {
			this.executor = executor;
		}

		@PostConstruct
		public void init() throws Exception {
			// init不能卡住，影响Spring的启动，需要在新的线程处理。 @see
			// DefaultMessageListenerContainer.AsyncMessageListenerInvoker
			new Thread(new AsyncMessageListener()).start();
		}

		public void destroy() {
			// TODO
		}

		// -------------------------------------------------------------------------
		// Inner classes used as internal adapters
		// -------------------------------------------------------------------------

		/**
		 * Runnable that performs looped.
		 */
		private class AsyncMessageListener implements Runnable {

			@Override
			public void run() {
				while (true) {
					try {
						process();
					} catch (Exception e) {
						logger.error("exception occur while process.", e);
					}
				}
			}
		}

		/**
		 * Default constructor for convenient dependency injection via setters.
		 */
		public QueueProcessor() {
		}

		public QueueProcessor(final String host, final int port, final int timeout,
				final String password, final String queueName,
				MessageHandler messageProcessor) throws Exception {
			Assert.notNull(host);
			Assert.isTrue(port > 0);
			Assert.notNull(queueName);
			Assert.notNull(messageProcessor);

			this.host = host;
			this.port = port;
			this.timeout = timeout;
			this.password = password;
			this.queueName = queueName;

			this.jedis = createJedis();
			this.messageHandler = messageProcessor;
		}

		private Jedis createJedis() throws Exception {
			final Jedis jedis = new Jedis(this.host, this.port, this.timeout);

			jedis.connect();
			if (null != this.password) {
				jedis.auth(this.password);
			}
			return jedis;
		}

		private Jedis getJedis() throws Exception {
			if (jedisPool != null) {
				return jedisPool.getResource();
			} else if (jedis == null) {
				this.jedis = createJedis();
			}
			return jedis;
		}

		public QueueProcessor(final Jedis jedis, final String queueName,
				MessageHandler messageProcessor) {
			Assert.notNull(jedis);
			Assert.notNull(queueName);
			Assert.notNull(messageProcessor);

			this.jedis = jedis;
			this.messageHandler = messageProcessor;
		}

		public QueueProcessor(final JedisPool jedisPool, final String queueName,
				MessageHandler messageProcessor) {
			Assert.notNull(jedisPool);
			Assert.notNull(queueName);
			Assert.notNull(messageProcessor);

			this.jedisPool = jedisPool;
			this.queueName = queueName;
			this.messageHandler = messageProcessor;
		}

		/**
		 * 同步执行。队列监听器与消息处理器是同一个线程。
		 * 
		 * @param executor
		 * @return
		 * @throws Exception
		 */
		public void process() throws Exception {
			logger.info("Start to process..");
			if (this.executor == null) {
				Jedis jedis = getJedis();
				while (true) {
					// 默认是blocking
					List<String> messages = jedis.blpop(this.timeout, queueName);
					String payload = messages.get(1);
					if (StringUtils.isNotBlank(payload)) {
						messageHandler.onMessage(payload);
					}
				}
			} else {
				process(this.executor);
			}

		}

		/**
		 * 异步执行。消息处理器在executor执行。
		 * 
		 * @param executor
		 * @throws Exception
		 */
		public void process(Executor executor) throws Exception {
			logger.info("Start to process with executor..");

			Jedis jedis = getJedis();
			while (true) {
				List<String> messages = jedis.blpop(this.timeout, queueName);
				final String payload = messages.get(1);
				submitTask(executor, payload);
			}
		}

		private void submitTask(Executor executor, final String payload) {
			if (StringUtils.isNotBlank(payload)) {
				executor.execute(new Runnable() {

					@Override
					public void run() {
						messageHandler.onMessage(payload);
					}
				});
			}
		}

	}

可以看到其实我们是简单使用了Redis的List数据结构作为消息队列：`List<String> messages = jedis.blpop(this.timeout, queueName);`。

然后业务要使用的话在Spring中配置一下就可以了：

	<bean id="jedisPool" class="redis.clients.jedis.JedisPool"
		destroy-method="destroy">
		<constructor-arg index="0" ref="jedisPoolConfig" />
		<constructor-arg index="1" value="${redis.host}" />
		<constructor-arg index="2" value="${redis.port}" />
		<constructor-arg index="3" value="${redis.timeout}" />
		<constructor-arg index="4" value="${redis.password}" />
	</bean>

	<!-- registration for message push -->
	<bean id="subscribeQueueProcessor"
		class="me.arganzheng.study.message.queue.processor.QueueProcessor"
		destroy-method="destroy">
		<property name="jedisPool" ref="jedisPool" />
		<property name="queueName" value="queue:message:subscribe" />
		<property name="messageHandler" ref="subscribeMessageHandler" />
	</bean>

	<bean id="subscribeMessageHandler"
		class="me.arganzheng.study.message.queue.processor.handler.SubscribeMessageHandler">
	</bean>


其中subscribeMessageHandler定义如下：

	package me.arganzheng.study.message.queue.processor.handler;

	import org.apache.log4j.Logger;
	import org.springframework.beans.factory.annotation.Autowired;

	import me.arganzheng.study.message.model.RegistrationInfo;
	import me.arganzheng.study.message.service.RegistrationInfoService;
	import me.arganzheng.study.message.queue.processor.MessageHandler;
	import com.google.gson.Gson;
	import com.google.gson.GsonBuilder;

	/**
	 * 处理订阅消息
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public class SubscribeMessageHandler implements MessageHandler {

		private static final Logger logger = Logger.getLogger(SubscribeMessageHandler.class);

		@Autowired
		private RegistrationInfoService registrationInfoService;

		@Override
		public void onMessage(String message) {
			try {
				logger.info("Received subscribed message: " + message);
				Gson gson = new GsonBuilder().create();
				RegistrationInfo regInfo = gson.fromJson(message, RegistrationInfo.class);
				boolean result = registrationInfoService.save(regInfo);
				if (!result) {
					logger.error("save registrationInfo failed! regInfo=" + regInfo);
				}
			} catch (Exception e) {
				logger.error("Handle message failed!message=" + message, e);
			}
		}
	}

这是消息消费者的封装。在消息生产者那边，只需要往list里面插入消息就可以了：

    public static final String QUEUE_MESSAGE_SUBSCRIBE = "queue:message:subscribe";

    @Value("${queueSize}")
    private long queueSize = 1000;

    public boolean pushToQueue(Map<String, String> registrationInfo) {
        Gson gson = new GsonBuilder().create();
        try {
            final String jsonPayload = gson.toJson(registrationInfo);
            redisUtils.doTask(new JedisTemplate<Long>() {
                @Override
                public Long doJedis(Jedis jedis) {
                    Long result = jedis.rpush(QUEUE_MESSAGE_SUBSCRIBE, jsonPayload);
                    // 为了避免队列内存溢出，只保留1000个消息
                    jedis.ltrim(QUEUE_MESSAGE_SUBSCRIBE, -queueSize, -1);
                    return result;
                }
            });
            return true;
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            return false;
        }
    }


整个实现只花了不到一个小时。成本非常低，满足业务的需要，等后面再考虑选择一个合适的消息中间件。

推荐阅读
-------

1. [The Architecture and Design of a Publish & Subscribe Messaging System Tailored for Big Data Collecting and Analytics - Abstraction Builder](http://bulldog2011.github.io/blog/2013/03/27/the-architecture-and-design-of-a-pub-sub-messaging-system/)
