---
title: 用ant创建可执行的jar包
layout: post
---


今天打算将Swing桌面程序打包成一个可以执行的jar包发布，方便用户使用。
可执行的jar包其实原理很简单，就是在jar包的manifest文件中用Main-Class标明你的应用程序入口: `Main-Class: classname`。其中classname是类的全路径名，不包含.class后缀，跟Spring的配置一样。

然后你就可以直接鼠标双击该jar包，或者在命令行中执行如下命令启动该jar了：
     
     java -jar JAR-name

在ant脚本中这通过一个简单的脚本就可以完成了：

     <project name="me.arganzheng.study.util" basedir="." default="usage">
          <!-- 变量设置 -->
          <property file="build.properties" />
          <property name="project.root" value="." />

          <!-- 代码目录 -->
          <property name="src.dir" value="${project.root}/java" />

          <!-- lib目录 -->
          <property name="lib.dir" value="${project.root}/lib" />    
          <!-- 临时编译目录 -->
          <property name="build.dir" value="${project.root}/build" />

          <!--目标jar文件 -->
          <property name="jarfile" value="${project.root}/me.arganzheng.study.util.jar" />

          <!-- Java编译CLASSPATH -->
          <path id="master-classpath">
               <fileset dir="${lib.dir}" />
          </path>

          <target name="clean" description="清空所有输出文件包括build和部署目录">
               <delete dir="${build.dir}" />
          </target>

          <target name="compile" description="编译Java文件">
               <mkdir dir="${build.dir}" />
               <javac destdir="${build.dir}" source="1.6" target="1.6"
                    encoding="utf-8" debug="false" deprecation="false" optimize="true"
                    failonerror="true">
                    <src path="${src.dir}" />
                    <classpath refid="master-classpath" />
                    <!-- <compilerarg line="-g:none" /> -->
               </javac>
          </target>

          <!-- 帮助信息 -->
          <target name="usage">
               <echo message=" Application Build File" />
               <echo message="用法：ant -[target]" />
               <echo message="------------------------------------------------------" />
               <echo message="[target]" />
               <echo message="clean        --> 清空所有输出文件包括build和部署目录" />
               <echo message="compile      --> 编译Java文件" />
               <echo message="jar          --> 创建用于发布的JAR包文件" />
               <echo message="------------------------------------------------------" />
          </target>

          <target name="jar" depends="compile" description="generate the distribution" >
               <jar basedir="${build.dir}" destfile="${jarfile}" >
                <manifest>
                     <attribute name="Main-Class" value="me.arganzheng.study.util.ApiTestUI" />
                 </manifest>
               </jar>
          </target>
     </project>

然而打出来的jar包，双击执行，弹出如下错误框：

     Could not find the main class: me.arganzheng.study.util.ApiTestUI.
     Program will exit.

一开始以为是Main-Class没有打进去，或者路径写错了。但是解压后查看MANIFEST.MF，发现是正确的：

     Manifest-Version: 1.0
     Ant-Version: Apache Ant 1.8.4
     Created-By: 1.6.0_25-b06 (Sun Microsystems Inc.)
     Main-Class: com.paipai.api.ui.ApiTestUI

用java -jar执行能够得到一些信息：

     D:\code\api_sdk_proj\java>java -jar me.arganzheng.study.util.jar
     Exception in thread "main" java.lang.NoClassDefFoundError: org/dyno/visual/swing/layouts/Alignment
     Caused by: java.lang.ClassNotFoundException: org.dyno.visual.swing.layouts.Alignment
             at java.net.URLClassLoader$1.run(Unknown Source)
             at java.security.AccessController.doPrivileged(Native Method)
             at java.net.URLClassLoader.findClass(Unknown Source)
             at java.lang.ClassLoader.loadClass(Unknown Source)
             at sun.misc.Launcher$AppClassLoader.loadClass(Unknown Source)
             at java.lang.ClassLoader.loadClass(Unknown Source)
     Could not find the main class: me.arganzheng.study.util.ApiTestUI. Program will exit.

也就是说实际上并不是找不到main class（这是一个误导人的提示，强烈BS一下）。而是运行时有些类找不到。
原因很简单，jar包中依赖到的lib没有一并打包进去（类似于war）。但是jar本身有一定的局限性，虽然它的manifest文件支持指定Class-Path头指定运行时classpath：

>Class-Path: jar1-name jar2-name directory-name/jar3-name
>
> The Class-Path header points to classes or JAR files on the local network, not JAR files within the JAR file or classes accessible over internet protocols. To load classes in JAR files within a JAR file into the class path, you must write custom code to load those classes. For example, if MyJar.jar contains another JAR file called MyUtils.jar, you cannot use the Class-Path header in MyJar.jar's manifest to load classes in MyUtils.jar into the class path.


因此，简单的方式就是将引用的jar包解压然后将其源码一并打包成一个jar。

Ant的[zipfileset](http://ant.apache.org/manual/Types/zipfileset.html)就是做这样的事情的：

> A <zipfileset> is a special form of a <fileset> which can behave in 2 different ways :
1. When the src attribute is used - or a nested resource collection has been specified (since Apache Ant 1.7), the zipfileset is populated with zip entries found in the file src.
2. When the dir attribute is used, the zipfileset is populated with filesystem files found under dir.
<zipfileset> supports all attributes of <fileset> in addition to those listed below.

当使用src的时候，是会解压并且重新打包，并且可以使用includes/excludes过滤zip/jar包中的文件。因此我们可以这么写：

    <target name="jar" depends="compile" description="generate the distribution">
          <jar destfile="${jarfile}">
               <manifest>
                    <attribute name="Main-Class" value="me.arganzheng.study.util.ApiTestUI" />
                    <attribute name="Class-Path" value="." />
               </manifest>
               <fileset dir="${build.dir}" />
               <zipfileset includes="**/*.class" src="${lib.dir}/httpclient-4.0.1.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/commons-codec-1.3.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/commons-logging-1.1.1.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/httpcore-4.0.1.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/httpmime-4.0.1.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/apache-mime4j-0.6.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/com.paipai.netframework-3.0.2.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/com.paipai.util-3.0.9.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/com.paipai.component-3.1.9.MODIFY.FOR.MSGQ.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/commons-lang-2.5.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/ezmorph-1.0.4.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/json-lib-2.2.3.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/commons-collections-3.2.1.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/commons-beanutils-1.7.0.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/grouplayout.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/kxml2-2.3.0.jar" />
               <zipfileset includes="**/*.class" src="${lib.dir}/xstream-1.4.2.jar" />
          </jar>
     </target>

但是有个问题就是这样每次增加删除一个jar包，都需要同时维护这里。google了一下，发现了[zipgroupfileset](http://stackoverflow.com/questions/6291525/how-can-i-use-a-zipfileset-src-attribute-without-having-to-specify-it-manually-f)：

> **zipgroupfileset**
> A <zipgroupfileset> allows for multiple zip files to be merged into the archive. Each file found in this fileset is added to the archive the same way that zipfileset src files are added.
> <zipgroupfileset> is a fileset and supports all of its attributes and nested elements.


因此可以简化成这样子：

     <target name="jar" depends="compile" description="generate the distribution">
          <jar destfile="${jarfile}">
               <manifest>
                    <attribute name="Main-Class" value="me.arganzheng.study.util.ApiTestUI" />
                    <attribute name="Class-Path" value="."/>
               </manifest>
               <fileset dir="${build.dir}"/> 
               <zipgroupfileset dir="${lib.dir}" includes="*.jar" />
          </jar>
     </target>


但是这样发现zipgroupfileset有个比较严重的问题，就是它不支持zipfileset对压缩文件内容进行过滤功能（includes/excludes）

> To the best of my knowledge there's no way to filter when using <zipgroupfileset>: the include/excludes used there apply to the zips to be merged, not the content within them.

解决方法有两种:

1. 恢复成上面枚举的zipfileset模式。
2. 先将所有的jar包解压到一个临时目录，再对这个目录进行过滤和压缩，即 unzip==>clean-up==>jar。这样的控制力最强大了。

这篇文章提供了一个简单和强大的做法：[Ant: Exclude files from merged jar file](http://stackoverflow.com/questions/1274879/ant-exclude-files-from-merged-jar-file)。即先用zipgroupfileset将所有的jar包打包到一个临时汇总jar。然后用zipfileset对这个汇总的临时jar包进行过滤打包。

于是用这样的ant脚本搞定：

     <target name="jar" depends="compile" description="generate the distribution">
          <jar jarfile="temp/external-libs.jar">
               <zipgroupfileset dir="${lib.dir}" includes="*.jar" />
          </jar>
          <sleep seconds="1"/>
          <jar destfile="${jarfile}">
               <manifest>
                    <attribute name="Main-Class" value="me.arganzheng.study.util.ApiTestUI" />
                    <attribute name="Class-Path" value="."/>
               </manifest>
               <fileset dir="${build.dir}"/>
               <zipfileset src="temp/external-libs.jar">
                   <exclude name="*"/>
               </zipfileset>
          </jar>
     </target>

最终的build.xml脚本如下：

     <?xml version="1.0" encoding="utf-8"?>
     <project name="me.arganzheng.study.util" basedir="." default="usage">
          <!-- 变量设置 -->
          <property file="build.properties" />
          <property name="project.root" value="." />

          <!-- 代码目录 -->
          <property name="src.dir" value="${project.root}/java" />

          <!-- lib目录 -->
          <property name="lib.dir" value="${project.root}/lib" />    
          <!-- 临时编译目录 -->
          <property name="build.dir" value="${project.root}/build" />

          <!--目标jar文件 -->
          <property name="jarfile" value="${project.root}/me.arganzheng.study.util.jar" />

          <!-- Java编译CLASSPATH -->
          <path id="master-classpath">
               <fileset dir="${lib.dir}" />
          </path>

          <target name="clean" description="清空所有输出文件包括build和部署目录">
               <delete dir="${build.dir}" />
          </target>

          <target name="compile" description="编译Java文件">
               <mkdir dir="${build.dir}" />
               <javac destdir="${build.dir}" source="1.6" target="1.6"
                    encoding="utf-8" debug="false" deprecation="false" optimize="true"
                    failonerror="true">
                    <src path="${src.dir}" />
                    <classpath refid="master-classpath" />
                    <!-- <compilerarg line="-g:none" /> -->
               </javac>
          </target>

          <!-- 帮助信息 -->
          <target name="usage">
               <echo message=" Application Build File" />
               <echo message="用法：ant -[target]" />
               <echo message="------------------------------------------------------" />
               <echo message="[target]" />
               <echo message="clean        --> 清空所有输出文件包括build和部署目录" />
               <echo message="compile      --> 编译Java文件" />
               <echo message="jar          --> 创建用于发布的JAR包文件" />
               <echo message="------------------------------------------------------" />
          </target>

          <target name="jar" depends="compile" description="generate the distribution">
               <jar jarfile="temp/external-libs.jar">
               <zipgroupfileset dir="${lib.dir}" includes="*.jar" />
               </jar>
               <sleep seconds="1"/>
               <jar destfile="${jarfile}">
                    <manifest>
                         <attribute name="Main-Class" value="me.arganzheng.study.util.ApiTestUI" />
                         <attribute name="Class-Path" value="."/>
                    </manifest>
                    <fileset dir="${build.dir}"/>
                    <zipfileset src="temp/external-libs.jar">
                        <exclude name="*"/>
                    </zipfileset>
               </jar>
          </target>
     </project>


---


使用maven创建可执行的jar包（补充于2014-07-27）
---------------------------------------------


可以看到前面的打包方式是使用ant，而不是maven。这是因为这是一个遗留的项目，原来采用的就是ant脚本，lib目录下的依赖jar也是手动维护的。但是在真实的J2EE环境下，我们经常是采用maven来管理依赖和打包的。那么怎样用maven来打一个可以运行的jar包呢？

这里提供了几个方法供大家参考。

### 法一、复用前面的ant脚本。使用maven的maven-antrun-plugin可以在maven中执行ant脚本。依赖的jar包可以使用maven-dependency-plugin插件得到。

具体可以参考：[使用Spring跟踪应用异常（6）—构建一个可运行JAR包 ](http://www.importnew.com/11886.html)


### 法二、使用[maven-shade-plugin](http://maven.apache.org/plugins/maven-shade-plugin/)打CLI包。

在这篇文章：[Maven实战（九）——打包的技巧](http://www.infoq.com/cn/news/2011/06/xxb-maven-9-package) 的 可执行CLI包 一节提到了一个强大的maven插件——[maven-shade-plugin](http://maven.apache.org/plugins/maven-shade-plugin/)。这个插件可以让用户配置Main-Class的值，然后在打包的时候将值填入/META-INF/MANIFEST.MF文件。关于项目的依赖，它很聪明地将依赖JAR文件全部解压后，再将得到的.class文件连同当前项目的.class文件一起合并到最终的CLI包中，这样，在执行CLI JAR文件的时候，所有需要的类就都在Classpath中了。


### 法三、将目录结构打成一个tar包，发布解压运行。在运行脚本中设置好classpath和Main Class：

     $ cat StartApp.sh

     #!/bin/bash

     # preserve current working directory
     cd `dirname $0`/..
     BASE=`pwd`
     SYSTEM_ROOT=/usr/local
     export JAVA_HOME=$SYSTEM_ROOT/java
     LIBCLASSPATH=`echo $BASE/lib/*.jar | tr ' ' ':'`
     export CLASSPATH=$LIBCLASSPATH:$BASE/conf:$BASE/data
     echo $CLASSPATH
     echo $BASE

     JAVA_OPTS=" -Xmx1024m -Dlog4j.configuration=$BASE/conf/log4j.properties"

     echo $JAVA_OPTS
      
     $JAVA_HOME/bin/java $JAVA_OPTS -Dintl.standalone.alicall.dataSync.config.path=conf/dataSync.properties me.arganzheng.study.standalone.dataSync.HelloWorld

进一步的，我们可以使用maven的 [maven-assembly-plugin](http://maven.apache.org/plugins/maven-assembly-plugin/)。它可以让我们很方便的自定义包的格式，并且支持各种打包文件格式，包括zip、tar.gz、tar.bz2等等。这对于windows下开发的情况，就特别的方便（Windows不方便打tar包或者gzip包）。

 
**TIPS && NOTES**

如果是采用了前面两种方式，发布上去之后不要简单的执行: `java -jar hello-world-1.0-cli.jar`。请务必给你的standalone程序设置一个`-Xmx`值！


-----


补记
----

后来在做一个standalone的时候，发现打成一个jar包还是很不方便的。不利于动态修改配置。使用目录结构分明的类似于war包这种还是比较好。具体参考笔者的另一篇文章 [java standalone模板](http://blog.arganzheng.me/posts/java-standalone-template.html)。


参考文章
-------

1. [创建可执行的jar包](http://www.cnblogs.com/az19870227/archive/2011/09/29/2195540.html)
2. [使用Spring跟踪应用异常（6）—构建一个可运行JAR包 ](http://www.importnew.com/11886.html)
3. [Maven实战（九）——打包的技巧](http://www.infoq.com/cn/news/2011/06/xxb-maven-9-package) 
