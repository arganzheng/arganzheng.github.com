---
title: 自动化部署平台设计
layout: post
---


我们一般通过jenkins做持续构建，在代码提交之后自动构建maven任务，输出jar包或者war包。但是我们还需要把jar/war包发布到线上机器。这个操作能不能自动化呢？

要实现自动化部署，我们需要解决下面三个问题：

1. WHAT to deploy: jar/war/zip, etc. 如果要支持js/css/html，甚至class文件，还需要支持文件发布。
2. WHERE to deploy: 
	* 目标机器
	* 目标目录
3. HOW to deploy
	1. backup: 一般来说只需要保留上一个可用版本就可以了。当然也可以保留多个，回滚的时候选择最新的进行回滚。
	2. remove service from LB: 对于用长连接监控SP可用性的RPC框架，这个过程是自动化的；对于Nginx，则需要将其从upstream中摘除（有些LB支持动态upstream）
	3. shutdown old running service: 这个简单，简单的kill -9。
	4. startup the newly deploy service: 这个最好部署服务有自己的startup脚本，直接调用就可以了。
	5. check if service is OK?
		1. if OK: resume service to LB: 同
		2. if NOT OK: rollback to backup


使用Jenkins的节点管理实现自动化部署
-------------------------------

我们用Jenkins做持续构建，是不是可以用jenkins做持续部署呢？Jenkins有节点管理可以实现简单的分布式部署功能。


### 1、配置Node节点

关键在认证登陆这块，Linux机器一般采用SSH，Windows机器则采用Java Web Start。

这里建议直接打通SSH免登陆。具体参见笔者以前的文章：[shell如何实现ssh免密码登陆](http://blog.arganzheng.me/posts/ssh-login-without-password.html)。

	Jenkins =>  Manage Jenkins => Manage Nodes => New Node:

	Name: web01
	Remote root directory: /home/work/jenkins
	Labels: web01 production
	Launch method: Launch slave agents on Unix machines via SSh
		Host: xxx
		Credentials: SSH Username with private key（如果是免登陆处理了，直接勾选 `From the Jenkins master ~/.ssh` 就可以了）


### 2、配置任务选择label

跟配置普通的任务基本一样，除了一点，就是要选择`Restrict where this project can be run`，在Label Expression中输入相应的label。


另外，因为Jenkins本质上只是一个持续构建工具，要实现自动化部署，还需要配置一下`Post-build Actions`，执行本地部署shell脚本。

因为通过前面的节点管理和label限制，这个job其实就是跑在要发布的服务器上，所以我们不需要远程执行脚本，只需要本地执行脚本就就可以了，这个Jenkins默认就支持了。

任务配置面板中 Build => Add build step => Execute shell: 

输入以下脚本：

	#!/bin/sh

	APP_DIR="/home/work/apps/mobopay/user"
	JENKINS_WORKSPACE="/home/work/jenkins/workspace"

	APP_JAR="${JENKINS_WORKSPACE}/user-impl/target/user-impl.jar"
	APP_BIN="${JENKINS_WORKSPACE}/user-impl/bin/start_production.sh"

	## 1. backup and copy
	[ -d $APP_DIR ] || mkdir -p $APP_DIR

	cd $APP_DIR

	[ -d target ] || mkdir target
	[ -d bin ] || mkdir bin

	if [ -f target/user-impl.jar ]; then
		echo "start backup..." 
		mv target/user-impl.jar target/user-impl.jar.bak
	fi

	echo "cp jar and bin to target..."
	cp $APP_JAR $APP_DIR/target

	[ -f APP_BIN ] || cp $APP_BIN $APP_DIR/bin

	## 2. shutdown old running service
	APP_PARAMS="/home/work/apps/mobopay/user/target/user-impl.jar"
	APP_PID=`ps aux | grep java| grep "$APP_PARAMS" | grep -v grep | awk '{ print $2}'`

	for i in $APP_PID; do
		echo "Kill PID [ $APP_PID ] contains $APP_PARAMS"
		kill -9 $APP_PID
	done

	## 3. startup the newly deploy service
	echo "start uping..."
	nohup bin/start_production.sh &
	echo "start up!"

**NOTES && TIPS**

需要注意的是shell的执行错误并不会影响job的执行结果。所以脚本中要多echo一些有用的信息，同时要进入Cosole Output中查看。


### 3、运行任务


配置后简单的启动构建任务：

	Started by user anonymous
	Building remotely on web03 (production) in workspace /home/work/jenkins/workspace/user-impl
	Checking out a fresh workspace because there's no workspace at /home/work/jenkins/workspace/user-impl
	Cleaning local Directory .
	Checking out https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl at revision '2016-03-30T09:46:29.503 +0800'
	
	...

	[INFO] Installing /home/work/jenkins/workspace/user-impl/target/user-impl.jar to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.jar
	[INFO] Installing /home/work/jenkins/workspace/user-impl/pom.xml to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.pom
	[INFO] ------------------------------------------------------------------------
	[INFO] BUILD SUCCESS
	[INFO] ------------------------------------------------------------------------
	[INFO] Total time: 02:30 min
	[INFO] Finished at: 2016-03-30T09:49:11+08:00
	[INFO] Final Memory: 43M/1119M
	[INFO] ------------------------------------------------------------------------
	[user-impl] $ /bin/sh /tmp/hudson8177948577949318327.sh
	cp jar and bin to target...
	start uping...
	start up!
	/home/work/apps/mobopay/user
	Finished: SUCCESS `Building remotely on web03 (production) in workspace /home/work/jenkins/workspace/user-impl

看起来是成功了。但是从日志很容易看出问题：Jenkins并不是在所有`label=production`的节点上执行构建任务，而是随机挑选了一台！仔细一想当然是合理的，毕竟人家的初衷就是slave作为构建集群，并不是发布集群。

看来在Jenkins上搞自动化部署，比较靠谱的做法是"一次构建，多次发布"。这其实并不难，只需要把我们jar包scp过去，然后远程执行脚本就可以了。当然，最好有插件可以支持像label一样的选择机器发布，而不是写死在脚本中。

谷歌了一下，发现这篇文章: [使用Jenkins实现多平台并行集成](http://tonybai.com/2012/02/15/intergating-on-multiple-platforms-simultaneously-using-jenkins/)。跟我遇到的问题是一模一样的，里面提到Jenkins的Multi-configuration project: Suitable for projects that need a large number of different configurations, such as testing on multiple environments, platform-specific builds, etc. 用这个就可以实现多slave同时构建了。

与free-style project相比，Multi-Configuration project的配置页面中没有"Restrict where this project can be run"配置选项，但却多出了一个"Configuration Matrix"配置区域。在该区域中，我们可以选择Slaves，在Node/Label中，我们可以看到当前Jenkins中配置的所有Label和Nodes。注意选择label跟前面"Restrict where this project can be run"效果是一样，Jenkins只会从该label中的若干个节点中选择一个来执行构建。所以这里要选择Nodes，然后勾选你需要构建的节点。


保持点击Build Now，在看到任务在所有选中的节点上同时执行了。同时master的输出如下日志：

	Started by user anonymous
	Building remotely on web02 (production) in workspace /home/work/jenkins/workspace/user-impl-production
	Updating https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl at revision '2016-03-30T15:01:42.878 +0800'
	At revision 1985
	no change for https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl since the previous build
	Triggering user-impl-production » web03
	Triggering user-impl-production » web02
	Triggering user-impl-production » web01
	user-impl-production » web03 completed with result SUCCESS
	user-impl-production » web02 completed with result SUCCESS
	user-impl-production » web01 completed with result SUCCESS
	Finished: SUCCESS

点击到具体的Slave上看，也能看到相应的日志：

	Started by upstream project "user-impl-production" build number 4
	originally caused by:
	 Started by user anonymous
	Building remotely on web03 (production) in workspace /home/work/jenkins/workspace/user-impl-production/label/web03
	Updating https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl at revision '2016-03-30T15:01:42.878 +0800'
	At revision 1985
	no change for https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl since the previous build
	[web03] $ /home/work/jenkins/tools/hudson.tasks.Maven_MavenInstallation/system/bin/mvn -Dlabel=web03 clean install -Dmaven.test.skip
	[INFO] Scanning for projects...
	[INFO]                                                                         
	[INFO] ------------------------------------------------------------------------
	[INFO] Building user-impl 1.0.0-SNAPSHOT
	[INFO] ------------------------------------------------------------------------
	[INFO] 
	[INFO] --- maven-clean-plugin:2.5:clean (default-clean) @ user-impl ---
	[INFO] 
	[INFO] --- maven-resources-plugin:2.6:resources (default-resources) @ user-impl ---
	[INFO] Using 'UTF-8' encoding to copy filtered resources.
	[INFO] Copying 1 resource
	[INFO] Copying 7 resources
	[INFO] 
	[INFO] --- maven-compiler-plugin:3.1:compile (default-compile) @ user-impl ---
	[INFO] Changes detected - recompiling the module!
	[INFO] Compiling 2 source files to /home/work/jenkins/workspace/user-impl-production/label/web03/target/classes
	[INFO] 
	[INFO] --- maven-resources-plugin:2.6:testResources (default-testResources) @ user-impl ---
	[INFO] Not copying test resources
	[INFO] 
	[INFO] --- maven-compiler-plugin:3.1:testCompile (default-testCompile) @ user-impl ---
	[INFO] Not compiling test sources
	[INFO] 
	[INFO] --- maven-surefire-plugin:2.18.1:test (default-test) @ user-impl ---
	[INFO] Tests are skipped.
	[INFO] 
	[INFO] --- maven-jar-plugin:2.5:jar (default-jar) @ user-impl ---
	[INFO] Building jar: /home/work/jenkins/workspace/user-impl-production/label/web03/target/user-impl.jar
	[INFO] 
	[INFO] --- spring-boot-maven-plugin:1.3.2.RELEASE:repackage (default) @ user-impl ---
	[INFO] 
	[INFO] --- maven-install-plugin:2.5.2:install (default-install) @ user-impl ---
	[INFO] Installing /home/work/jenkins/workspace/user-impl-production/label/web03/target/user-impl.jar to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.jar
	[INFO] Installing /home/work/jenkins/workspace/user-impl-production/label/web03/pom.xml to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.pom
	[INFO] ------------------------------------------------------------------------
	[INFO] BUILD SUCCESS
	[INFO] ------------------------------------------------------------------------
	[INFO] Total time: 2.814 s
	[INFO] Finished at: 2016-03-30T15:01:50+08:00
	[INFO] Final Memory: 39M/1473M
	[INFO] ------------------------------------------------------------------------
	[web03] $ /bin/sh /tmp/hudson7762667286947684737.sh
	cp jar and bin to target...
	cp: cannot stat `/home/work/jenkins/workspace/user-impl/target/user-impl.jar': No such file or directory
	cp: cannot stat `/home/work/jenkins/workspace/user-impl/bin/start_production.sh': No such file or directory
	start uping...
	start up!
	nohup: failed to run command `bin/start_production.sh': No such file or directory
	Finished: SUCCESS

效果还不错！但是有三个问题：

1、Slave的workspace构建jar包路径跟master不一样：

	[INFO] Installing /home/work/jenkins/workspace/user-impl-production/label/web03/target/user-impl.jar to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.jar

看到路径中居然还有label信息。。这意味着脚本要不修改，要不就就不要使用workspace路径，直接使用maven本地仓库路径（但是需要注意的是仓库中的构建是带SNAPSHOT的，而且我们的bin目录并不会跟着install过去。。）。

不过我想Jenkins作为这么通用的开放框架，应该有环境变量，谷歌了一下，果然在每次构建的时候Jenkins会设置$WORKSPACE环境变量：[Environtment variables for jenkins when script running](http://blog.tmtk.net/post/2012-11-19-environtment_variables_for_jenkins_when_script_running/)。整个世界一下子清静很多～

2、脚本构建出错，在Master最终状态或者日志都是显示成功：

	cp: cannot stat `/home/work/jenkins/workspace/user-impl/target/user-impl.jar': No such file or directory
	cp: cannot stat `/home/work/jenkins/workspace/user-impl/bin/start_production.sh': No such file or directory

这个简单，修改我们的deploy脚本，检查文件是否存在，如果不存在就以非0状态码退出就可以了：

	if [ -f $APP_JAR ]; then
		cp $APP_JAR $APP_DIR/target
		echo "cp $APP_JAR to $APP_DIR/target success"
	else
		echo "$APP_JAR not exist!"
		exit 1
	fi

再Build一下：

	...

	start backup...
	cp jar and bin to target...
	/home/work/jenkins/workspace/user-impl/target/user-impl.jar not exist!
	Build step 'Execute shell' marked build as failure
	Finished: FAILURE

OK，不管是从slave的日志或者从管理界面都可以看到失败的slave构建。

3、第三个问题，也是他妈最蛋疼的问题：就是Jenkins会在构建结束之后把它创建的所有进程都干掉！！！所以对于通过Jenkins Execute Shell调起的需要长时间运行的外部任务，即使使用nohup或者&放在后台运行，也一样会被kill掉！！结果就是我们的应用一直没有起来！谷歌了很久（关键词很重要：jenkins shell script background），终于发现解决方案：[Spawning processes from build](https://wiki.jenkins-ci.org/display/JENKINS/Spawning+processes+from+build):

> Note that this will set the BUILD_ID environment variable for the process being spawned to something other than the current BUILD_ID. Or you can start jenkins with `-Dhudson.util.ProcessTree.disable=true` - see [ProcessTreeKiller](https://wiki.jenkins-ci.org/display/JENKINS/ProcessTreeKiller) for details.

也就是说我们要不在启动Jenkins的时候指定不要kill创建的子进程：
	
	java -Dhudson.util.ProcessTree.disable=true -jar jenkins.war

要不在脚本中通过把BUILD_ID环境变量设置成一个与父进程不一样的数值来让Jenkins无法kill掉它：

	BUILD_ID=dontKillMe /usr/apache/bin/httpd

综上，最终的deploy脚本如下：

	#!/bin/sh

	BUILD_ID=dontKillMe

	APP_DIR="/home/work/apps/mobopay/user"
	JENKINS_WORKSPACE=$WORKSPACE

	APP_JAR="${JENKINS_WORKSPACE}/target/user-impl.jar"
	APP_BIN="${JENKINS_WORKSPACE}/bin/start_production.sh"

	## 1. backup and copy
	[ -d $APP_DIR ] || mkdir -p $APP_DIR

	cd $APP_DIR

	[ -d target ] || mkdir target
	[ -d bin ] || mkdir bin

	if [ -f target/user-impl.jar ]; then
		echo "start backup..." 
		mv target/user-impl.jar target/user-impl.jar.bak
	fi

	echo "cp jar and bin to target..."

	if [ -f $APP_JAR ]; then
		cp $APP_JAR $APP_DIR/target
		echo "cp $APP_JAR to $APP_DIR/target success"
	else
		echo "$APP_JAR not exist!"
		exit 1
	fi

	if [ -f $APP_BIN ]; then
		cp $APP_BIN $APP_DIR/bin
		echo "cp $APP_BIN $APP_DIR/bin success"
	else
		echo "$APP_BIN not exist!"
		exit 1
	fi

	## 2. shutdown old running service
	APP_PARAMS="${APP_DIR}/target/user-impl.jar"
	APP_PID=`ps aux | grep java| grep "$APP_PARAMS" | grep -v grep | awk '{ print $2}'`

	for i in $APP_PID; do
		echo "Kill PID [ $APP_PID ] contains $APP_PARAMS"
		kill -9 $APP_PID
	done

	## 3. startup the newly deploy service
	echo "Starting up with nohup bin/start_production.sh & ..."
	nohup bin/start_production.sh &
	echo "Start up success!"

其中start_production.sh脚本如下：

	#!/bin/bash

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

	echo $basedir

	JAVA_OPTS="-Dspring.profiles.active=production -Dlogback.configurationFile=logback-production.xml -DLOG_DIR=${LOG_DIR}"
	JAVA_OPTS="$JAVA_OPTS -server -Xmn256M -Xms1024M -Xmx1024M -Djava.net.preferIPv4Stack=true -Djava.awt.headless=true -Dorg.apache.catalina.connector.RECYCLE_FACADES=false -Djava.util.Arrays.useLegacyMergeSort=true -XX:MetaspaceSize=64M -XX:MaxMetaspaceSize=256M -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSCompactAtFullCollection -XX:CMSFullGCsBeforeCompaction=1 -XX:+CMSClassUnloadingEnabled -XX:+UseFastAccessorMethods -XX:+UseCMSInitiatingOccupancyOnly  -XX:SurvivorRatio=65536 -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=80 -XX:SoftRefLRUPolicyMSPerMB=0 -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -verbose:gc -Xloggc:${GC_FILE_PATH} -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=10 -XX:GCLogFileSize=100M -XX:+HeapDumpOnOutOfMemoryError"
	#JAVA_OPTS="${JAVA_OPTS} -Xdebug -Xrunjdwp:server=y,transport=dt_socket,address=8000,suspend=n" 
	echo $JAVA_OPTS

	java $JAVA_OPTS -jar ${basedir}/target/user-impl.jar

兴高采烈的运行，发现还是不行。。妹，但是发现如果把start_production.sh的脚本内容跟deploy脚本合并成一个就成功了。。也就是说`BUILD_ID=dontKillMe java $JAVA_OPTS -jar ${basedir}/target/user-impl.jar &`就生效，但是`BUILD_ID=dontKillMe nohup bin/start_production.sh`就没有用。。把`BUILD_ID=dontKillMe`放在start_production.sh中也不行。。哭死。。

经过一个下午的试验和谷歌，终于发现了一个解决方案：[Nohup doesn't work when executing script from Jenkins](http://superuser.com/questions/922841/nohup-doesnt-work-when-executing-script-from-jenkins)。

实验证明，一定要把start_production.sh的stdout输出重定向到/dev/null，否则就会启动不了。也就是说我们的脚本要改成：

	nohup bin/start_production.sh >/dev/null &

这样就可以了：

	Started by upstream project "user-impl-production" build number 23
	originally caused by:
	 Started by user anonymous
	Building remotely on web02 (production) in workspace /home/work/jenkins/workspace/user-impl-production/label/web02
	Updating https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl at revision '2016-03-31T20:55:45.084 +0800'
	At revision 2044
	no change for https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl since the previous build
	
	...

	[web02] $ /bin/sh /tmp/hudson7126890291497807782.sh
	start backup...
	cp jar and bin to target...
	cp /home/work/jenkins/workspace/user-impl-production/label/web02/target/user-impl.jar to /home/work/apps/mobopay/user/target success
	cp /home/work/jenkins/workspace/user-impl-production/label/web02/bin/start_production.sh /home/work/apps/mobopay/user/bin success
	Starting up with nohup bin/start_production.sh & ...
	Start up success!
	Java HotSpot(TM) 64-Bit Server VM warning: UseCMSCompactAtFullCollection is deprecated and will likely be removed in a future release.
	Java HotSpot(TM) 64-Bit Server VM warning: CMSFullGCsBeforeCompaction is deprecated and will likely be removed in a future release.
	Finished: SUCCESS

可以看到stderr输出是没有关系的，不过我们也可以把它屏蔽掉：

	nohup bin/start_production.sh >/dev/null 2>&1 &

至此，终于完全搞定，肉留满面啊，感觉还是自己写一个自动发布平台简单一些。。


**TIPS** 

如果遇到如下错误：

	Started by user anonymous
	Building remotely on web01 in workspace /home/work/jenkins/workspace/user-impl
	Checking out a fresh workspace because there's no workspace at /home/work/jenkins/workspace/user-impl
	Cleaning local Directory .
	Checking out https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl at revision '2016-03-29T16:07:22.391 +0800'
	A         src
	A         src/test
	...
	A         bin
	AU        bin/start_test.sh
	AU        bin/start_development.sh
	AU        bin/start_production.sh
	A         pom.xml
	 U        .
	At revision 1974
	[user-impl] $ mvn clean install -Dmaven.test.skip
	FATAL: command execution failed
	java.io.IOException: Cannot run program "mvn" (in directory "/home/work/jenkins/workspace/user-impl"): error=2, No such file or directory
		at java.lang.ProcessBuilder.start(ProcessBuilder.java:1048)
		at hudson.Proc$LocalProc.<init>(Proc.java:244)
		at hudson.Proc$LocalProc.<init>(Proc.java:216)
		at hudson.Launcher$LocalLauncher.launch(Launcher.java:815)
		at hudson.Launcher$ProcStarter.start(Launcher.java:381)
		at hudson.Launcher$RemoteLaunchCallable.call(Launcher.java:1148)
		at hudson.Launcher$RemoteLaunchCallable.call(Launcher.java:1113)
		at hudson.remoting.UserRequest.perform(UserRequest.java:120)
		at hudson.remoting.UserRequest.perform(UserRequest.java:48)
		at hudson.remoting.Request$2.run(Request.java:326)
		at hudson.remoting.InterceptingExecutorService$1.call(InterceptingExecutorService.java:68)
		at java.util.concurrent.FutureTask.run(FutureTask.java:266)
		at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
		at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
		at java.lang.Thread.run(Thread.java:745)
		at ......remote call to web01(Native Method)
		at hudson.remoting.Channel.attachCallSiteStackTrace(Channel.java:1416)
		at hudson.remoting.UserResponse.retrieve(UserRequest.java:220)
		at hudson.remoting.Channel.call(Channel.java:781)
		at hudson.Launcher$RemoteLauncher.launch(Launcher.java:928)
		at hudson.Launcher$ProcStarter.start(Launcher.java:381)
		at hudson.Launcher$ProcStarter.join(Launcher.java:388)
		at hudson.tasks.Maven.perform(Maven.java:332)
		at hudson.tasks.BuildStepMonitor$1.perform(BuildStepMonitor.java:20)
		at hudson.model.AbstractBuild$AbstractBuildExecution.perform(AbstractBuild.java:782)
		at hudson.model.Build$BuildExecution.build(Build.java:205)
		at hudson.model.Build$BuildExecution.doRun(Build.java:162)
		at hudson.model.AbstractBuild$AbstractBuildExecution.run(AbstractBuild.java:534)
		at hudson.model.Run.execute(Run.java:1738)
		at hudson.model.FreeStyleBuild.run(FreeStyleBuild.java:43)
		at hudson.model.ResourceController.execute(ResourceController.java:98)
		at hudson.model.Executor.run(Executor.java:410)
	Caused by: java.io.IOException: error=2, No such file or directory
		at java.lang.UNIXProcess.forkAndExec(Native Method)
		at java.lang.UNIXProcess.<init>(UNIXProcess.java:248)
		at java.lang.ProcessImpl.start(ProcessImpl.java:134)
		at java.lang.ProcessBuilder.start(ProcessBuilder.java:1029)
		at hudson.Proc$LocalProc.<init>(Proc.java:244)
		at hudson.Proc$LocalProc.<init>(Proc.java:216)
		at hudson.Launcher$LocalLauncher.launch(Launcher.java:815)
		at hudson.Launcher$ProcStarter.start(Launcher.java:381)
		at hudson.Launcher$RemoteLaunchCallable.call(Launcher.java:1148)
		at hudson.Launcher$RemoteLaunchCallable.call(Launcher.java:1113)
		at hudson.remoting.UserRequest.perform(UserRequest.java:120)
		at hudson.remoting.UserRequest.perform(UserRequest.java:48)
		at hudson.remoting.Request$2.run(Request.java:326)
		at hudson.remoting.InterceptingExecutorService$1.call(InterceptingExecutorService.java:68)
		at java.util.concurrent.FutureTask.run(FutureTask.java:266)
		at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
		at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
		at java.lang.Thread.run(Thread.java:745)
	Build step 'Invoke top-level Maven targets' marked build as failure
	Finished: FAILURE


谷歌了一下，根据这个说法 [Cannot run program “mvn” error=2, No such file or directory](http://stackoverflow.com/questions/26906972/cannot-run-program-mvn-error-2-no-such-file-or-directory)，让Jenkins自动安装Maven就可以了。


### 总结

利用jenkins的master-slave节点管理实现分布式构建，同时jenkins的labels可以支持（逐个机器）灰度和全量构建，整个过程只需要简单的几步配置就可以了。

但是jenkins毕竟是一个持续构建工具，不是部署工具，要实现自动部署还需要编写shell脚本。另外，每个job都是要走 svn/git checkout => maven构建 的步骤，一个是没必要的构建，另一个是如果不小心提交了新的代码也会被构建部署上线（当然，我们可以通过RELEASE分支避免，但是这样子的话，每次发布都需要修改jenkins任务）。

个人感觉还是将构建和部署分开比较合理。


---


其他方式讨论
----------

### 1. 使用[Node and Label parameter plugin](https://wiki.jenkins-ci.org/display/JENKINS/NodeLabel+Parameter+Plugin)插件

[Node and Label parameter plugin](https://wiki.jenkins-ci.org/display/JENKINS/NodeLabel+Parameter+Plugin): This plugin adds two new parameter types to job configuration - node and label, this allows to dynamically select the node where a job/project should be executed.

跟前面一样，需要先配置好多个Node节点。但是跟前面不一样的是，不需要创建Multi-Configuration project，安装好Node and Label parameter插件后，勾选`This build is parameterized`选项之后，Add Paramter下拉框中会增加Label和Node两个parameter选项。

勾选：Allow multi node selection for concurrent builds

注意同时勾选`Execute concurrent builds if necessary`，否则会提示错误："Execute concurrent builds" and "Allow multi node selection for concurrent builds" only make sense together!

然后会发现Build Now变成了Build with Parameters，点击之后会先进入到parameters选择页面，也就是节点选择。因为我们允许多个节点同时构建，所以下拉框支持多选。选择你需要的构建的节点，点击Build就可以了。

	Started by user anonymous
	Building remotely on web01 (production) in workspace /home/work/jenkins/workspace/user-impl
	Updating https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl at revision '2016-03-30T18:02:38.977 +0800'
	At revision 1996
	no change for https://svn.baidu.com/app/gensoft/mobopay/trunk/backend/mobopay/remote/user/user-impl since the previous build
	Next nodes: [web02, web03]
	Schedule build on node web02
	Schedule build on node web03
	[user-impl] $ /home/work/jenkins/tools/hudson.tasks.Maven_MavenInstallation/system/bin/mvn -D=web01 clean install -Dmaven.test.skip
	[INFO] Scanning for projects...
	[INFO]                                                                         
	[INFO] ------------------------------------------------------------------------
	[INFO] Building user-impl 1.0.0-SNAPSHOT
	[INFO] ------------------------------------------------------------------------
	[INFO] 
	[INFO] --- maven-clean-plugin:2.5:clean (default-clean) @ user-impl ---
	[INFO] Deleting /home/work/jenkins/workspace/user-impl/target
	[INFO] 
	[INFO] --- maven-resources-plugin:2.6:resources (default-resources) @ user-impl ---
	[INFO] Using 'UTF-8' encoding to copy filtered resources.
	[INFO] Copying 1 resource
	[INFO] Copying 7 resources
	[INFO] 
	[INFO] --- maven-compiler-plugin:3.1:compile (default-compile) @ user-impl ---
	[INFO] Changes detected - recompiling the module!
	[INFO] Compiling 2 source files to /home/work/jenkins/workspace/user-impl/target/classes
	[INFO] 
	[INFO] --- maven-resources-plugin:2.6:testResources (default-testResources) @ user-impl ---
	[INFO] Not copying test resources
	[INFO] 
	[INFO] --- maven-compiler-plugin:3.1:testCompile (default-testCompile) @ user-impl ---
	[INFO] Not compiling test sources
	[INFO] 
	[INFO] --- maven-surefire-plugin:2.18.1:test (default-test) @ user-impl ---
	[INFO] Tests are skipped.
	[INFO] 
	[INFO] --- maven-jar-plugin:2.5:jar (default-jar) @ user-impl ---
	[INFO] Building jar: /home/work/jenkins/workspace/user-impl/target/user-impl.jar
	[INFO] 
	[INFO] --- spring-boot-maven-plugin:1.3.2.RELEASE:repackage (default) @ user-impl ---
	[INFO] 
	[INFO] --- maven-install-plugin:2.5.2:install (default-install) @ user-impl ---
	[INFO] Installing /home/work/jenkins/workspace/user-impl/target/user-impl.jar to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.jar
	[INFO] Installing /home/work/jenkins/workspace/user-impl/pom.xml to /home/work/.m2/repository/com/baidu/mobojoy/mobopay/remote/user/user-impl/1.0.0-SNAPSHOT/user-impl-1.0.0-SNAPSHOT.pom
	[INFO] ------------------------------------------------------------------------
	[INFO] BUILD SUCCESS
	[INFO] ------------------------------------------------------------------------
	[INFO] Total time: 2.859 s
	[INFO] Finished at: 2016-03-30T18:02:45+08:00
	[INFO] Final Memory: 39M/1477M
	[INFO] ------------------------------------------------------------------------
	[user-impl] $ /bin/sh /tmp/hudson7869689904044983757.sh
	start backup...
	cp jar and bin to target...
	cp /home/work/jenkins/workspace/user-impl/target/user-impl.jar to /home/work/apps/mobopay/user/target success
	cp /home/work/jenkins/workspace/user-impl/bin/start_test.sh /home/work/apps/mobopay/user/bin success
	Start uping...
	Start up success!
	/home/work/apps/mobopay/user
	Finished: SUCCESS

但是看不到其他节点的日志输出，不知道是否构建成功。效果上不如第一种方法。


参考文章
-------

1. [Jenkins on-demand slave selection through labels](http://michael-prokop.at/blog/2014/03/01/jenkins-on-demand-slave-selection-through-labels/)
2. [Continuous Integration Using Docker, Maven and Jenkins](https://www.wouterdanes.net/2014/04/11/continuous-integration-using-docker-maven-and-jenkins.html)
3. [Building Gradle Projects with Jenkins CI](https://guides.codepath.com/android/building-gradle-projects-with-jenkins-ci)
4. [Using Jenkins to run remote deployment scripts over SSH](http://julianhigman.com/blog/2012/02/25/using-jenkins-to-run-remote-deployment-scripts-over-ssh/)
5. [Jenkins多服务器自动部署，发布到多台服务器](http://www.2cto.com/os/201501/370149.html)
6. [JUMPING INTO CONTINUOUS DELIVERY WITH JENKINS AND SSH](http://blog.akquinet.de/2012/01/10/jumping-into-continuous-delivery-with-jenkins-and-ssh/)


