---
title: 解决Ant编译乱码问题
layout: post
---


今天在用ant编译一个老的java工程，发现第一个遇到的问题就是编码问题。首先是报“警告： 
    
    编码 GBK 的不可映射字符”

一看就是典型的采用GBK去解码utf-8文件了。在javac中增加命令行参数-encoding UTF-8就可以解决了。

    <target name="compile" description="编译Java文件">
          <mkdir dir="${build.dir}" />
          <javac destdir="${build.dir}" source="1.6" target="1.6" 
               debug="false" deprecation="false" optimize="true"
               failonerror="true">
               <src path="${src.dir}" />
               <classpath refid="master-classpath" />
               <compilerarg line="encoding utf-8" />
          </javac>
     </target>

Ant支持直接通过encoding属性声明：

    <target name="compile" description="编译Java文件">
          <mkdir dir="${build.dir}" />
          <javac destdir="${build.dir}" source="1.6" target="1.6" encoding="utf-8" debug="false" deprecation="false" optimize="true" failonerror="true">
               <src path="${src.dir}" />
               <classpath refid="master-classpath" />
          </javac>
     </target>

但是接着又遇到一个诡异的问题：

    Ant编译utf-8非法字符：/65279

Google了一下，发现这个问题是由于utf-8 BOM标识导致的：

> 一般用UE或记事本编辑过的UTF-8的文件头会加入BOM标识，该标识由3个char组成。在UTF－8的标准里该BOM标识是可有可无的,Sun 的javac 在编译带有BOM的UTF-8的格式的文件时会出现“非法字符：/65279”的错误，但是用Eclipse进行编译却没有问题，原因在于Eclipse 使用的是自己的JDT，而非javac，关于JDT的描述可以到eclipse的官网上去查看。

如果文件比较少，那么可以用UE、Editplus等文本编辑器重新保存文件为不带BOM的UTF-8格式。如果文件比较多，建议还是写个程序自动处理一下。其实也很简单，就是简单的对每个文件进行一个BOM判断，如果是BOM，那么就跳过BOM标识，copy剩下的字节。

    import java.io.BufferedInputStream;
    import java.io.ByteArrayOutputStream;
    import java.io.File;
    import java.io.FileInputStream;
    import java.io.FileOutputStream;
    import java.io.IOException;
    import java.io.InputStream;
    import java.io.OutputStream;

    public class UTF8Parser {

        public static void main(String[] args) {
            if (args.length < 1) {
                return;
            }

            File file = new File(args[0]);

            System.out.println("java代码位置：" + file.getAbsolutePath());

            UTF8Parser.clearUTF8Mark(file);
        }

        private static void clearUTF8Mark(File file) {
            if (file.isDirectory()) {
                for (File f : file.listFiles()) {
                    clearUTF8Mark(f);
                }
            } else {
                FileInputStream fis = null;
                InputStream is = null;
                ByteArrayOutputStream baos = null;
                OutputStream out = null;
                try {
                    fis = new FileInputStream(file);
                    is = new BufferedInputStream(fis);
                    baos = new ByteArrayOutputStream();
                    byte b[] = new byte[3];
                    is.read(b);
                    if (-17 == b[0] && -69 == b[1] && -65 == b[2]) {
                        System.out.println(file.getAbsolutePath());
                        b = new byte[1024];
                        while (true) {
                            int bytes = 0;
                            try {
                                bytes = is.read(b);
                            } catch (IOException e) {
                            }
                            if (bytes == -1) {
                                break;
                            }
                            baos.write(b, 0, bytes);
                            b = baos.toByteArray();
                        }
                        file.delete();
                        out = new FileOutputStream(file);
                        baos.writeTo(out);
                    }
                } catch (Exception e) {
                    System.exit(0);
                } finally {
                    try {
                        if (fis != null) {
                            fis.close();
                        }
                        if (out != null) {
                            out.close();
                        }
                        if (is != null) {
                            is.close();
                        }
                        if (baos != null) {
                            baos.close();
                        }
                    } catch (Exception e) {
                        System.exit(0);
                    }
                }
            }
        }
    }

javac编译一下，然后运行之。

也可以在ant中运行Java程序，类似于maven中的exec插件。如下：

    <!-- 清除utf-8标记 -->
    <target name="cleanBOM">
        <echo>清除utf-8标记</echo>
        <java dir="./bin" classname="UTF8Parser" fork="true" failonerror="true" maxmemory="128m">
            <arg line="${src.dir}" />
        </java>
    </target>












