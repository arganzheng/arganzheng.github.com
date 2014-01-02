---
title: scala with maven
layout: post
---


scala官方比较prefer SBT，不过maven也可以的。关键还是IDE问题啊。

### pom.xml

    <project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
    	<modelVersion>4.0.0</modelVersion>
    	<groupId>me.arganzheng.study.scala</groupId>
    	<artifactId>idl-tools</artifactId>
    	<version>1.0</version>
    	<packaging>jar</packaging>
    	<name>idl-tools</name>
    
    	<!-- Shared version number properties -->
    	<properties>
    		<scala.version>2.10.1</scala.version>
    		<project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    	</properties>
    
    	<repositories>
    		<repository>
    			<id>scala-tools.org</id>
    			<name>Scala-Tools Maven2 Repository</name>
    			<url>http://scala-tools.org/repo-releases</url>
    		</repository>
    	</repositories>
    
    	<pluginRepositories>
    		<pluginRepository>
    			<id>scala-tools.org</id>
    			<name>Scala-Tools Maven2 Repository</name>
    			<url>http://scala-tools.org/repo-releases</url>
    		</pluginRepository>
    	</pluginRepositories>
    
    	<dependencies>
    		<dependency>
    			<groupId>org.scala-lang</groupId>
    			<artifactId>scala-compiler</artifactId>
    			<version>${scala.version}</version>
    			<scope>provided</scope>
    		</dependency>
    		<dependency>
    			<groupId>org.scala-lang</groupId>
    			<artifactId>scala-library</artifactId>
    			<version>${scala.version}</version>
    		</dependency>
    		<dependency>
    			<groupId>com.google.protobuf</groupId>
    			<artifactId>protobuf-java</artifactId>
    			<version>2.5.0</version>
    		</dependency>
    	</dependencies>
    
    	<build>
    		<plugins>
    			<plugin>
    				<groupId>org.scala-tools</groupId>
    				<artifactId>maven-scala-plugin</artifactId>
    				<version>2.15.2</version>
    				<executions>
    					<execution>
    						<id>compile</id>
    						<goals>
    							<goal>compile</goal>
    						</goals>
    						<phase>process-resources</phase>
    					</execution>
    					<execution>
    						<id>test-compile</id>
    						<goals>
    							<goal>testCompile</goal>
    						</goals>
    						<phase>process-test-resources</phase>
    					</execution>
    				</executions>
    				<configuration>
    					<scalaVersion>${scala.version}</scalaVersion>
    				</configuration>
    			</plugin>
    			<plugin>
    				<groupId>org.apache.maven.plugins</groupId>
    				<artifactId>maven-compiler-plugin</artifactId>
    				<version>2.3.2</version>
    				<configuration>
    					<source>1.6</source>
    					<target>1.6</target>
    					<encoding>${project.build.sourceEncoding}</encoding>
    				</configuration>
    			</plugin>
    		</plugins>
    	</build>
    </project>
    
### 目录结构

    project/
        pom.xml   -  Defines the project
        src/
            main/
                java/ - Contains all java code that will go in your final artifact.  
                      See maven-compiler-plugin for details
                scala/ - Contains all scala code that will go in your final artifact.  
                       See maven-scala-plugin for details
                resources/ - Contains all static files that should be available on the classpath 
                           in the final artifact.  See maven-resources-plugin for details
                webapp/ - Contains all content for a web application (jsps, css, images, etc.)  
                        See maven-war-plugin for details
        site/ - Contains all apt or xdoc files used to create a project website.  
                 See maven-site-plugin for details       
        test/
            java/ - Contains all java code used for testing.   
                     See maven-compiler-plugin for details
            scala/ - Contains all scala code used for testing.   
                      See maven-scala-plugin for details
            resources/ - Contains all static content that should be available on the 
                          classpath during testing.   See maven-resources-plugin for details

### 使用maven的archetype:generate生成项目骨架

    mvn archetype:generate \
          -DarchetypeGroupId=org.scala-tools.archetypes \
          -DarchetypeArtifactId=scala-archetype-simple  \
          -DremoteRepositories=http://scala-tools.org/repo-releases \
          -DgroupId=me.arganzheng.study.scala \
          -DartifactId=scala-helloworld \
          -Dversion=1.0-SNAPSHOT


## 参考文章

1. [Maven For Beginners](http://www.scala-lang.org/old/node/345.html)
2. [Introduction to maven](http://www.scala-blogs.org/2008/01/maven-for-scala.html)
3. [Scala and Maven - Getting Started Guide using maven-scala-plugin (TOTD #170)](https://blogs.oracle.com/arungupta/entry/scala_and_maven_getting_started)
