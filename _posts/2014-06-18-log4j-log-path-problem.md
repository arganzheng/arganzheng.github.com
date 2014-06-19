---
layout: post
title: log4j日志路径问题
---

问题
----

原来的log4j.properties配置文件中配置日志文件路径如下：

	## File output...
	log4j.appender.system=org.apache.log4j.RollingFileAppender
	log4j.appender.system.File=../logs/system.log
	log4j.appender.defaultLogger.DatePattern='.'yyyy-MM-dd
	log4j.appender.system.layout=org.apache.log4j.PatternLayout
	log4j.appender.system.layout.ConversionPattern=%d [%p] [%t] %c (%F\:%L): %m%n


可以看到这里使用了相对路径：`log4j.appender.system.File=../logs/system.log`。相对路径有个问题，就是最终路径取决于执行shell脚本时所在的目录路径。比如，你在`/home/arganzheng/tomcat/tomcat-reading`路径下执行bin/startup.sh，则日志路径为：`/home/arganzheng/tomcat/logs/system.log`，而在`/home/arganzheng/tomcat/tomcat-reading/bin`目录下执行startup.sh，则日志路径为：`/home/arganzheng/tomcat/tomcat-reading/logs/system.log`。后者才是我们想要的。但是我们无法强制大家一定要在某个目录下执行startup.sh。使用绝对路径更不行，因为我们的代码是一套的。


解决方案
--------

其实我们要的效果是在`/home/arganzheng/tomcat/tomcat-reading/logs`下记录日志。相对路径是没有问题的，但是参照物不能是动态的，比如执行shell时所在的目录。相反，我们可以找静态的参照物——startup.sh，因为我们是执行它，所以我们可以得到它的路径`/home/arganzheng/tomcat/tomcat-reading/bin/startup.sh`。这样，很容易我们就可以得到`/home/arganzheng/tomcat/tomcat-reading/logs`目录了。脚本很简单：

	[work@arganzheng-ubuntu apache-tomcat-6.0.41-mbrowser]$ cat bin/setenv.sh 
	# determine base directory; preserve where you're running from
	#echo "Path to $(basename $0) is $(readlink -f $0)"
	realpath=$(readlink -f "$0")
	filepath=$(dirname "$realpath")
	basedir=${filepath%/*}

	LOG_DIR=${basedir}/logs
	mkdir -p ${LOG_DIR}
	GC_LOG_DIR=${basedir}/logs/gc
	mkdir -p ${GC_LOG_DIR}
	GC_FILE_PATH="${GC_LOG_DIR}/gc-$(date +%s).log"

	JAVA_OPTS="$JAVA_OPTS -server -Xmn1024M -Xms4096M -Xmx4096M -Djava.net.preferIPv4Stack=true -Djava.awt.headless=true -Dorg.apache.catalina.connector.RECYCLE_FACADES=false -XX:PermSize=512M -XX:MaxPermSize=512M -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSCompactAtFullCollection -XX:CMSFullGCsBeforeCompaction=0 -XX:+CMSClassUnloadingEnabled -XX:+UseFastAccessorMethods -XX:+UseCMSInitiatingOccupancyOnly  -XX:SurvivorRatio=65536 -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=81 -XX:SoftRefLRUPolicyMSPerMB=0 -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -verbose:gc -Xloggc:${GC_FILE_PATH}"
	JAVA_OPTS="$JAVA_OPTS -DLOG_DIR=${LOG_DIR} -Denv=idc"


TIPS: 这里还看到，我们也对gc log文件进行了相同的处理。

然后在我们的log4j.properties文件就可以使用这个环境变量了：

	# Rules reminder:
	# TRACE < DEBUG < INFO < WARN < ERROR < FATAL

	# Global logging configuration
	log4j.rootLogger=info,console, system 

	## Console output...
	log4j.appender.console=org.apache.log4j.ConsoleAppender
	log4j.appender.console.layout=org.apache.log4j.PatternLayout
	log4j.appender.console.layout.ConversionPattern=[%p]\t%d\t[%t] %c (%F\:%L) - %m%n

	## File output...
	log4j.appender.system=org.apache.log4j.DailyRollingFileAppender
	log4j.appender.system.File=${LOG_DIR}/system.log
	log4j.appender.system.Append=true 
	#log4j.appender.system.Threshold=INFO
	log4j.appender.defaultLogger.DatePattern='.'yyyy-MM-dd
	log4j.appender.system.layout=org.apache.log4j.PatternLayout
	log4j.appender.system.layout.ConversionPattern=%d [%p] [%t] %c (%F\:%L): %m%n

	## File output...
	log4j.appender.mbrowser=org.apache.log4j.DailyRollingFileAppender
	log4j.appender.mbrowser.File=${LOG_DIR}/mbrowser.log
	log4j.appender.mbrowser.Append=true
	#log4j.appender.mbrowser.Threshold=INFO
	log4j.appender.defaultLogger.DatePattern='.'yyyy-MM-dd
	log4j.appender.mbrowser.layout=org.apache.log4j.PatternLayout
	log4j.appender.mbrowser.layout.ConversionPattern=%d [%p] [%t] %c (%F\:%L): %m%n

	# 3rdparty logging configuration
	log4j.logger.org.springframework=WARN

	# My logging(Application) logging configuration
	log4j.logger.me.arganzheng.study.mbrowser=INFO, mbrowser
	log4j.additivity.me.arganzheng.study.mbrowser=false

	# GCM
	log4j.logger.com.google.android.gcm.server=INFO, mbrowser
	log4j.additivity.com.google.android.gcm.server=false

	### mybatis loggers ###
	#log4j.logger.com.ibatis=DEBUG
	#log4j.logger.com.ibatis.common.jdbc.SimpleDataSource=DEBUG
	#log4j.logger.com.ibatis.common.jdbc.ScriptRunner=DEBUG
	#log4j.logger.com.ibatis.sqlmap.engine.impl.SqlMapClientDelegate=DEBUG

	### sql loggers
	#log4j.logger.java.sql.Connection=DEBUG
	#log4j.logger.java.sql.Statement=DEBUG
	#log4j.logger.java.sql.PreparedStatement=DEBUG
	#log4j.logger.java.sql.ResultSet=DEBUG


但是注意到其实这个方案只是解决了日志文件的路径问题。但是没有解决多个环境（dev, test, idc）下不同的log4j配置需求。log4j不像其他的配置文件，可以通过环境变量来标识。可能autogen是最终极的解决方案。这里后面有时间可以研究一下，先简单解决目前的问题。