---
title: maven的resources插件
layout: post
---


maven的resources插件职责是将项目的资源文件拷贝到目标目录。maven将资源划分为两类：main resources 和 test resources。因此有如下三个相应的goal:


#### 1. resources:resources 

* command: mvn resources:resources
* bound by default to the process-resources life-cycle phase
* copies the resources for the main source code to the main output directory
* the resources for your main source code, is by default specified by the `<resources>` element (`project.build.resources`, 默认是`src/main/resources`) 
* the main build output directory,  is by default specified by the `project.build.outputDirectory` element


#### 2. resources:testResources 

* command: mvn resources:testResources
* bound by default to the process-test-resources life-cycle phase
* copies the resources for the test source code to the test output directory
* the resources for your test source code, is by default specified by the `<testResources>` (`project.build.testResources`) element
* the test build output directory,  is by default specified by the `project.build.testOutputDirectory` element


#### 3. resources:copy-resources 

* command: mvn resources:copy-resources
* requires that you configure the phase to be bound to (specified as part of the plugin configuration)
* copies resources to an output directory
* Rather than using the <resources> or <testResources> elements or their defaults, this uses a <resources> element that is specified as part of the plugin configuration
* For more details, see [Copy Resources](http://maven.apache.org/plugins/maven-resources-plugin/examples/copy-resources.html)


Some Tips
---------

### [Specifying resource directories](http://maven.apache.org/plugins/maven-resources-plugin/examples/resource-directory.html)

maven默认`src/main/resources`下的资源。如果你有资源文件不是存放在这个目录下，那么可以通过`<resources>`元素指定：

     <project>
          ...
          <build>
               ...
               <resources>
                    <resource>
                         <directory>${basedir}/apk</directory>
                         <includes>
                         <include>**/*.apk</include>
                       </includes>
                         <targetPath>META-INF/apk</targetPath>
                    </resource>
                    <resource>
                         <directory>${basedir}/src/main/resources</directory>
                    </resource>
               </resources>
               ...
          </build>
          ...
     </project>


### [Including and excluding files and directories](http://maven.apache.org/plugins/maven-resources-plugin/examples/include-exclude.html)


除了指定原资源文件和目标目录，我们还可以指定包含或者排除哪些目录或者文件。

例如：

     <project>
          ...
          <build>
               ...
               <resources>
                    <resource>
                         <directory>${basedir}/sample</directory>
                         <targetPath>META-INF/sample</targetPath>
                    </resource>
                    <resource>
                         <directory>${basedir}/docs</directory>
                         <targetPath>META-INF/docs</targetPath>
                    </resource>
                    <!-- 把源代码打在一起，方便后面SDK打包 -->
                    <resource>
                         <directory>${project.build.sourceDirectory}</directory>
                         <includes>
                              <include>com/qq/buy/api/client/**/*.java</include>
                         </includes>
                         <targetPath>META-INF/source</targetPath>
                    </resource>
                    <resource>
                         <directory>${basedir}/src/main/resources</directory>
                    </resource>
               </resources>
          </build>
     </project>     


### [Resources Filter](http://maven.apache.org/plugins/maven-resources-plugin/examples/filter.html)

最后，我们还可以对资源文件进行变量替换，这个maven称之为resources filter。变量值可以来自 system properties, project properties, filter resources 和 command line.

resources的三个goals都支持resources filter。

例如：

     <project>
          ...
          <build>
               ...
               <resources>
                    <resource>
                         <directory>src/main/webapp/WEB-INF</directory>
                         <filtering>true</filtering>
                         <includes>
                              <include>**/*.xml</include>
                         </includes>
                    </resource>
                    <resource>
                         <directory>src/main/resources</directory>
                         <filtering>false</filtering>
                    </resource>
               </resources>

               <filters>
                    <filter>config.properties</filter>
               </filters>
          </build>
     </project>     


### 最后的绝招——the maven antrun plugin

如果resources插件还是不能满足你的需求。那么还有最后一个绝招——maven-antrun-plugin。可以通过在maven中运行ant脚本。

例如，假设有个RPC框架，需要将IDL文件生成java源代码，打包成依赖的jar包。可以这么处理：

     <project>
          ...
          <properties>
               <idl.dir>src/main/idl</idl.dir>
               <idl.package>me.arganzheng.project.api.metadata.idl</idl.package>
               <idl.dist_dir>/usr/local/dist_dir/api/idllib</idl.dist_dir>
          </properties>
          ...
          <profiles>
               <profile>
                    <id>production</id>
                    <build>
                         <plugins>
                              <plugin>
                                   <artifactId>maven-antrun-plugin</artifactId>
                                   <executions>
                                        <execution>
                                             <id>generate-idl-protocal-sources</id>
                                             <phase>generate-sources</phase>
                                             <goals>
                                                  <goal>run</goal>
                                             </goals>
                                             <configuration>
                                                  <tasks>
                                                       <mkdir dir="${project.basedir}/target/idl-temp" />

                                                       <!-- 把IDL java文件编译成class -->
                                                       <javac
                                                            fork="true" executable="${env.JAVA_HOME}/bin/javac"
                                                            srcdir="${idl.dir}"
                                                            encoding="UTF-8"
                                                            destdir="${project.basedir}/target/idl-temp">
                                                            <classpath refid="maven.compile.classpath" />
                                                       </javac>

                                                       <!-- 生成IDL的Request和Response等java文件 -->
                                                       <java classname="com.paipai.platform.tools.AutoGenForCi">
                                                            <classpath refid="maven.compile.classpath" />
                                                            <classpath path="${project.basedir}/target/idl-temp" />
                                                            <arg value="${idl.package}"/>
                                                            <arg path="${project.basedir}/target/idl-temp"/>
                                                            <arg value="app"/>
                                                            <arg value="UTF-8"/>
                                                       </java>
                                                  </tasks>
                                             </configuration>
                                        </execution>
                                        <execution>
                                             <id>transfer</id>
                                             <phase>install</phase>
                                             <goals>
                                                  <goal>run</goal>
                                             </goals>
                                             <configuration>
                                                  <tasks>
                                                       <copy todir="${idl.dist_dir}" file="target/${project.name}-${project.version}.jar" />
                                                  </tasks>
                                             </configuration>
                                        </execution>
                                   </executions>
                              </plugin>
                              <!-- 把上一步生成的IDL的Request和Response等java文件包含进后面编译时的源码范围 -->
                              <plugin>
                                   <groupId>org.codehaus.mojo</groupId>
                                   <artifactId>build-helper-maven-plugin</artifactId>
                                   <version>1.7</version>
                                   <executions>
                                        <execution>
                                             <id>register-idl-protocal-sources</id>
                                             <phase>generate-sources</phase>
                                             <goals>
                                                  <goal>add-source</goal>
                                             </goals>
                                             <configuration>
                                                  <sources>
                                                       <source>${project.basedir}/target/idl-temp</source>
                                                  </sources>
                                             </configuration>
                                        </execution>
                                   </executions>
                              </plugin>
                         </plugins>
                    </build>
               </profile>
          </profiles>
     </project>

这里放在了profiles中，执行命令的时候需要通过`-Pproduction`指定。

如果ant脚本比较复杂，可以将其独立成一个文件：

     <project>
          <build>
               <plugins>
                    <plugin>
                         <artifactId>maven-antrun-plugin</artifactId>
                         <executions>
                              <execution>
                                   <id>transfer</id>
                                   <phase>pre-integration-test</phase>
                                   <configuration>
                                        <tasks>
                                             <echo>Transfer api war...</echo>
                                             <property name="current_version" value="${project.version}" />
                                             <ant antfile="transfer.xml">
                                                  <target name="copyFileCycle" />
                                             </ant>
                                        </tasks>
                                   </configuration>
                                   <goals>
                                        <goal>run</goal>
                                   </goals>
                              </execution>
                         </executions>
                    </plugin>
               </plugins>
          </build>
     </project>

transfer.xml脚本就是一个简单的ant脚本：

     <?xml version="1.0" encoding="UTF-8"?>
     <project name="me.arganzheng.project.api" default="copyFileCycle" basedir=".">
          
          <property name="project.root" value="." />
          <property name="env.default_dist_dir" value="/usr/local/dist_dir" />
          <property name="target.dir" value="${env.default_dist_dir}/api" />
          
          <target name="copyFileCycle"  description="copy files for cycle">
               <echo message="copy api-metadata war to ${target.dir}"/>
               <mkdir dir="${target.dir}" />
               <copy tofile="${target.dir}/meta.war" file="${project.root}/target/api-sdk-v2-1.0.war" preservelastmodified="true" />
          </target>
          
     </project>

参考文档
--------

1. [Maven Resources Plugin](http://maven.apache.org/plugins/maven-resources-plugin/)

