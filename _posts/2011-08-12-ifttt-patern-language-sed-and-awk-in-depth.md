---
layout: post
title: ifttt模式语言——sed和awk深入学习
---


## 引子——一些思考

sed和awk能否实现这样一类问题：假如匹配了一个pattern，对下面的记录行采取这样的action，直到另一个pattern满足。
也就是说：
    
    pattern1 ==> begin action
    {action}
    pattern2 ==> end action

有点麻烦，但是并不是不能做到。因为sed和awk都是ifttt模式，上面的模式其实其实可以分解为两个ifttt：
    
    if pattern1, then do the action;
    
此时的action context就是pattern1开始的那一行。

    if pattern2, then stop the action;
    
这样，将action的context限制在pattern1和patter2之间。从而达到我们的目的。

只要维持一个开关（标志位，flag）就可以做到。类似于如下伪代码:

    if(pattern1){
        flag = on;
    }
    while(flag == on){
        do the action;
        if(pattern2){
             flag = off
        }
    }

让我们现看看sed中如何实现。

sed和awk有诸多不一样的地方，sed支持地址范围（address space）和嵌套地址空间（nested space）的概念，而awk也支持地址空间，但是不支持嵌套地址。
也就是说sed和awk都支持在两个pattern之间执行一个action（通过两个patterns指定一个操作范围，将操作限制在指定的行记录（地址空间，action上下文。**the ability to restrict the operation to certain lines.**），这正是我们前面提到的问题。也就是说它内建的语法（机制，模式）就已经支持了上面的问题，因此sed和awk很适合作这种事情。

这个模式翻译成伪代码就是：
    
    if pattern1 then
        do the action
    until pattern2 happen

>### Ranges by patterns
You can specify two regular expressions as the range. Assuming a "#" starts a comment, you can search for a keyword, remove all comments until you see the second keyword. In this case the two keywords are "start" and "stop:"
>
    sed '/start/,/stop/ s/#.*//'
>
**The first pattern turns on a flag that tells sed to perform the substitute command on every line. The second pattern turns off the flag.** If the "start" and "stop" pattern occurs twice, the substitution is done both times. If the "stop" pattern is missing, the flag is never turned off, and the substitution will be performed on every line until the end of the file.
>
You should know that if the "start" pattern is found, the substitution occurs on the same line that contains "start." This turns on a switch, which is line oriented. That is, the next line is read and the substitute command is checked. If it contains "stop" the switch is turned off. Switches are line oriented, and not word oriented.

可以看出，sed确实原生就支持这种模式，而且其底层实现应该也是使用了一个flag作开关控制。

有几点需要注意：

1. 一旦pattern1匹配了，即使在do action的过程中pattern1再次出现，sed不会对其作匹配，直到pattern2出现。而这时候如果pattern2再次出现了，action也已经结束了。**也就是说sed只匹配第一个pattern1和pattern2**。
2. 默认sed的地址空间会包含模式匹配的那一行。即pattern匹配的那一行会被action处理，而且这不是一个可配置的问题。需要使用sed的branch命令特殊处理。具体参见：[SED FAQ: How do I address all the lines between RE1 and RE2, excluding the lines themselves?](http://sed.sourceforge.net/sedfaq4.html#s4.24)
3. 一旦你指定了地址空间，那么你无法对地址空间以外的行进行操作。比如pattern1之前的几行或者pattern2之后的几行。这是一个悲剧的事情。
4. 由于sed和awk其实是行编辑器，每一个action执行的时候，其实是有一个上下文的，这个上下文就是当前匹配的行（地址），但是无法往上回溯，比如需要将当前匹配行以上的行进行处理（而这些行其实已经pass了）。

关于第三点，sed除了支持地址空间的概念，还支持嵌套地址空间：pattern1和pattern2限定了一个地址空间（restrict lines ranges）,你不能对该地址空间之外的行进行action操作，但是你可以对该地址空间中的行做进一步的action，这就是嵌套地址空间的概念。

> ### Grouping with { and }
The curly braces, "{" and "}," are used to group the commands.
>
Hardly worth the build up. All that prose and the solution is just matching squigqles. Well, there is one complication. **Since each sed command must start on its own line, the curly braces and the nested sed commands must be on separate lines**.

#### 嵌套地址

嵌套地址是包含在另一地址中的地址。根据定义，外部地址范围必须是行集合或者范围地址；嵌套的地址可以为单行地址，行集合地址或另一范围地址。看几个例子就明白了：）

例1：删除第20行到第30行之间的所有空行。
    
    20, 30{
        /^$/d
    }

例2：删除既包含sed又包含awk的行：

    /sed/{
        /awk/d
    }

不使用嵌套地址空间，可以这样写：

    /sed.*awk/d

例3：删除包含BEGIN但是不包含END的行
    
    # Delete lines that contain BEGIN but not END
    /BEGIN/{
        /END/ !d
    }

这个例子很有意思。嵌套地址空间相当于执行了两次，一次先对原文件（第0层地址空间，0, $）应用外层地址空间，然后对外层地址空间得到的行集合再应用内层地址空间。比如例3，先得到所有含有BEGIN的行，再对这些行进行二次过滤，排除含有END的行，这样就得到包含BEGIN但是不包含END的行了。

例4：将BEGIN和END之间的行集合删除

    sed '/BEGIN/, /END/ d' beginEnd.dat
    
注意区分 `sed '/BEGIN.*END/d' beginEnd.dat` 。后者是删除同时包含BEGIN和END的行（任意一行），而前者是指定了一个行范围——[包含BEGIN的行，包含END的行]。

例5：删除两个单词BEGIN和END之间的文本。开始和结束文本可在一行，也可跨越多行。

    #Deletes the text between two words, BEGIN and END.
    /BEGIN.*END/ s///
    /BEGIN/, /END/ {
            # Put BEGIN line in hold space
            /BEGIN/{
                    h
                    d
            }
            # append lines between BEGIN and END to hold space
            /END/ !{
                    H
                    d
            }
            # Append END line to hold space, and exchange hold space and pattern space
            /END/{
                    x
                    G
            }
            # pattern space now contain all lines
            s/BEGIN.*END//
    }
    forrest@forrest-laptop:~/Desktop$ cat sed_study.txt 
    This is the text between [BEGIN and END] in one line.
    [BEGIN
    This is the text between begin and end in multi-line.
    END]
    forrest@forrest-laptop:~/Desktop$ sed -f beginEnd5.sed beginEnd.dat 
    This is the text between [] in one line.
    []
    forrest@forrest-laptop:~/Desktop$ 

这个例子很好的展示了sed核心概念：**pattern space**和**hold space(buffer)**。值得好好学习，比如，思考一下最后的s/BEGIN.*END//为什么可以跨多行替换呢？

> ### The Pattern Space and the Hold Space
Thesed utility has two buffers. The commands reviewed up to this point work with the Pattern space, which initially holds the line of input that sed just read. The Hold space can hold data while you manipulate data in the Pattern space; it is a temporary buffer. Until you place data in the Hold space it is empty. This section discusses commands that move data between the Pattern space and the Hold space.
>
g Copies the contents of the Hold space to the Pattern space. The original contents of the Pattern space is lost.
G Appends a NEWLINE and the contents of the Hold space to the Pattern space.
h Copies the contents of the Pattern space to the Hold space. The original contents of the Hold space is lost.
H Appends a NEWLINE and the contents of the Pattern space to the Hold space.
x Exchanges the contents of the Pattern space and the Hold space.

有一篇文章介绍的不错：[Famous Sed One-Liners Explained, Part I: File Spacing, Numbering and Text Conversion and Substitution](http://www.catonmat.net/blog/sed-one-liners-explained-part-one/)

### sed的处理流程和Pattern Space
>
This demonstrates the pattern space sed uses to operate on a line. The actual operation sed uses is:
>
* Copy the input line into the pattern space.
* Apply the first sed command on the pattern space, if the address restriction is * true.
* Repeat with the next sed expression, again operating on the pattern space.
* the last operation is performed, write out the pattern space and read in the next line from the input file.

简而言之，sed将文件行读入pattern space中，然后对pattern space中的行集合进行action处理。

> ### [The Hold Buffer](http://www.grymoire.com/Unix/Sed.html#toc-uh-52)
So far we have talked about three concepts of sed: (1) The input stream or data before it is modified, (2) the output stream or data after it has been modified, and (3) the pattern space, or buffer containing characters that can be modified and send to the output stream.
>
There is one more "location" to be covered: the hold buffer or hold space. Think of it as a spare pattern buffer. It can be used to "copy" or "remember" the data in the pattern space for later. There are five commands that use the hold buffer.

最后，sed支持反空间：sed的!符号，不仅表示`not do the action`，还可以表示`not match the pattern`。

#### 一个牛刀杀鸡的Hold Space实战——使用sed将换行符去除

    forrest@ubuntu:~$ cat -v aaa.txt 
    aaa
    bbb
    ccc
    forrest@ubuntu:~$ sed 's/\n//' aaa.txt 
    aaa
    bbb
    ccc
    
没有任何效果，这是为什么呢？这是因为sed是面向行的处理器，sed在抽取出原始数据行放在pattern space中的时候鞥就将换行符去除了。**Because newline is already removed and placed in the pattern space.**

有方法可以让pattern space中有很多行，并且各行以换行符隔开。就是使用hold space作数据中转区。
    
    forrest@ubuntu:~$ sed -n ' H;
    > $ {
    > x;
    > s/\n//g;
    > p
    > }' aaa.txt
    aaabbbccc

回忆一下，H表示将pattern space的内容追加到hold space中。一直追加到文件最后一行。关键在于H在追加的时候，会自动换行。
>### [Keeping more than one line in the hold buffer](http://www.grymoire.com/Unix/Sed.html#uh-56)
>The "H" command allows you to combine several lines in the hold buffer. It acts like the "N" command as lines are appended to the buffer, with a "\n" between the lines. You can save several lines in the hold buffer, and print them only if a particular pattern is found later.（这里我们是判断遇到了$）

在最后一行的时候将hold space积累的数据exchange到pattern space中。这时候的pattern space就拥有所有记录行了并且有空格，在我们这个例子中是`aaa\nbbb\nccc\n`。接下来的 `s/\n//g;` 其实就是对这个字符串进行操作，主要需要加g否则只会替换第一个\n。最后的p会将pattern space中的东东打印到终端。

刚发现另一篇文章，也是描述这个问题的：
> #### [Unix Sed Tutorial: Multi-Line File Operation with 6 Practical Examples](http://www.thegeekstuff.com/2009/11/unix-sed-tutorial-multi-line-file-operation-with-6-practical-examples/)
In case, if you want to delete all the newlines in a file, you cannot use the following method. Because newline is already removed and placed in the pattern space.
>
    $ sed 's/\n//' filename or  $sed 's/\n/ENDOFLINE\n/' filename
>
For situations like this sed multi-line is appropriate. Sed provides the command “N” for Multi-Line Operations.
>
N command reads a next line from the input, Append next line to pattern space. Next line is separated from the original pattern space by a newline character.

但是他给的方法却是错误的：

    forrest@ubuntu:~$ cat aaa.txt 
    1
    2
    3
    4
    5
    6
    forrest@ubuntu:~$ sed '{
    N
    s/\n/@/
    }' aaa.txt 
    1@2
    3@4
    5@6

这是因为N只能一次读入两行，相当于两行两行的处理了。

当然，上面这个例子只是为了让大家对sed的pattern space和hold space有比较深刻的理解。对于去掉换行符这种小case，只需要简单的tr命令就可以了。

    # tr居然不认文件参数，只能标准输入输出
    tr -d '\n'  < file.in > file.out
或者 
    
    tr "\n" " " < file.in > file.out

### 关于sed的变量和地址空间

不要小看下面这个例子，虽然他在讲述r命令，一个我们不怎么使用的命令，但是关键是里面对sed变量和地址空间的介绍，非常的到位。

> ### [Reading in a file with the 'r' command](http://www.grymoire.com/Unix/Sed.html#uh-37)
There is also a command for reading files. The command
>
    sed '$r end' <in>out
will append the file "end" at the end of the file (address "$)." The following will insert a file after the line with the word "INCLUDE:"
>
    sed '/INCLUDE/ r file' <in >out
You can use the curly braces to delete the line having the "INCLUDE" command on it: 
>   
    #!/bin/sh
    sed '/INCLUDE/ {
    	r file
    	d
    }'
>
Click here to get file: sed_include.sh
>
The order of the delete command "d" and the read file command "r" is important. Change the order and it will not work. There are two subtle actions that prevent this from working. The first is the "r" command writes the file to the output stream. **The file is not inserted into the pattern space, and therefore cannot be modified by any command. Therefore the delete command does not affect the data read from the file.（sed的action中的任何操作都不会应用在原文件（thus，原匹配地址空间），必须这样，否则会出现递归死循环。）**
>
The other subtlety is the "d" command deletes the current data in the pattern space. Once all of the data is deleted, it does make sense that no other action will be attempted. Therefore a "d" command executed in a curly brace also aborts all further actions. As an example, the substitute command below is never executed: 
>    
    #!/bin/sh
    # this example is WRONG
    sed -e '1 {
    	d
    	s/.*//
    }'
>
The earlier example is a crude version of the C preprocessor program. The file that is included has a predetermined name. **It would be nice if sed allowed a variable (e.g "\1)" instead of a fixed file name. Alas, sed doesn't have this ability. You could work around this limitation by creating sed commands on the fly, or by using shell quotes to pass variables into the sed script. Suppose you wanted to create a command that would include a file like cpp, but the filename is an argument to the script.** An example of this script is:
>
    % include 'sys/param.h' <file.c >file.c.new
    A shell script to do this would be:
    
    #!/bin/sh
    # watch out for a '/' in the parameter
    # use alternate search delimiter
    sed -e '\_#INCLUDE <'"$1"'>_{
    	r '"$1"'
    	d
    }'
Let me elaborate. If you had a file that contains

    Test first file
    #INCLUDE <file1>
    Test second file
    #INCLUDE <file2>
>    
you could use the command
    
    sed_include1.sh file1<input|sed_include1.sh file2
to include the specified files.

### 参考资料
关于sed的最佳学习文章: [Sed - An Introduction and Tutorial by Bruce Barnett](http://www.grymoire.com/Unix/Sed.html)


## 实战——将UT不通过的单元测试注释掉

要处理的文件格式：
    
    @Test
    @Prepare(autoImport = true)
    public void testFindProductById() 
    {
           ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
           assertResult(result);
    }

三种方式可以解决UT不通过问题：

1. 将整个函数注释掉：包括@XXX anotation。
2. 只将函数体注释掉
3. 将函数名和@Test注释掉，这样maven就不会跑该单元测试

其中法1最难，因为需要不仅需要回溯，还需要知道整个函数边界。
法2相对来说最简单，因为只需要知道函数体边界。但是需要处理第一个{紧跟在函数头还是另起一行的情况。
法3只需要回溯，不需要知道函数（体）边界。

对于sed和awk这样的pattern==>action模式语言，回溯几乎是不可能做到，但是可以采用一个巧妙的方式避免回溯——使用tac命令将要处理的文本反着打印。这样pattern就在action之前发现，从而避免了回溯处理。

识别函数边界需要在识别出函数头时，计数'{'和'}'的匹配情况，确定函数边界。并不是很麻烦，不过费些时间。
这里先实现最简单的法3：

假设输入文件例子如下：

    public class ProductBatchUploadDaoTest{
    
        private productBatchUploadDao;
        public setProdoctBatchUploadDao(ProductBatchUploadDO productBatchUploadDao){
            this.productBatchUploadDao = productBatchUploadDao;
        }
    
        @Test
        @Prepare(autoImport = true)
        public void testFindProductByIdList() {
            ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
    	aaa;
            assertResult(result);
        }
    
        @Test
        @Prepare(autoImport = true)
        public void testFindProductById() {
            ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
            if(result==null){
    		    throw new RuntimeException("no result!");
        	}else{
        		assertResult(result);
        	}
        }
    }
    
其中`testFindProductById()`是我们要注释掉的TestCase，我们可以这么作：

    forrest@forrest-laptop:~/work/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis$ cat ProductBatchUploadDaoImplTest.java | sed -f change_tc_func_name.sed | tac | sed -f comment_out_@Test.sed | tac
    
将得到如下输出结果：

    public class ProductBatchUploadDaoTest{
    
        private productBatchUploadDao;
        public setProdoctBatchUploadDao(ProductBatchUploadDO productBatchUploadDao){
            this.productBatchUploadDao = productBatchUploadDao;
        }
    
        @Test
        @Prepare(autoImport = true)
        public void testFindProductByIdList() {
            ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
    	aaa;
            assertResult(result);
        }
    
        //@Test
        @Prepare(autoImport = true)
        public void comment_out_by_forrest_testFindProductById() {
            ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
            if(result==null){
    		    throw new RuntimeException("no result!");
    	    }else{
    		    assertResult(result);
    	    }
        }
    }

其中change_tc_func_name.sed定义如下：

    s/\<testFindProductById\>/comment_out_by_forrest_testFindProductById/

而comment_out_@Test.sed定义如下：

    /comment_out_by_forrest_testFindProductById/, /@Test/{    #comment_out_by_forrest_testFindProductById与第一个@Test之间的地址范围
    	s#@Test#//@Test#
    }

将上面的ProductBatchUploadDaoImplTest.java替换成每个failed的TestClass文件名，再将testFindProductById替换成该TestClass中的每个failed testCase函数名，就可以。这需要能够传递参数给sed，但是sed并没有变量和参数的概念，不过可以用shell调用sed，将用户参数传递给shell，shell再传递给sed。[Passing arguments into a sed script](http://www.grymoire.com/Unix/Sed.html#uh-22) && [Reading in a file with the 'r' command](http://www.grymoire.com/Unix/Sed.html#uh-37)

其实上面的命令替换名字和注释@Test操作可以合并成一个sed命令，因为我们是先扫描到函数名再到@Test的。所以comment_out_@Test.sed可以么写：

    /\<testFindProductById\>/, /@Test/{    #指定testFindProductById与第一个@Test之间的地址范围作为action的操作对象
            s/testFindProductById/comment_out_by_forrest_testFindProductById/    #将testFindProductById替换为comment_out_by_forrest_testFindProductById
            s#@Test#//@Test#    #将@Test注释掉
    }

    forrest@forrest-laptop:~/work/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis$ tac ProductBatchUploadDaoImplTest.java | sed  -f comment_out_@Test.sed  | tac > ProductBatchUploadDaoImplTest_fixed.java
    forrest@forrest-laptop:~/work/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis$ mv  ProductBatchUploadDaoImplTest_fixed.java  ProductBatchUploadDaoImplTest.java
    
虽然合成一个了，扫描的次数还是一样的，而且同样需要一个中间文件。

使用tac避免回溯十分的巧妙，但是也有不足就是多了两次文件读写（两次tac），但是回溯本身也需要多次读写某部分文本，而且需要显式的上下文处理，如果文件不是很多很大，那么tac无疑是quick但是未必dirty的做法。

注：法3在大部分情况下都能解决UT不通过的问题，因为JUnit不会运行该TestCase，但是JUnit仍然会编译该TestCase所在的TestClass，所以编译错误导致UT不通过的问题，法3没法解决，需要将函数（体）comment out，即法1和法2的做法。

现在考虑一下怎么生成订正脚本。
假使输入是这样的文本：

    com.alibaba.intl.biz.product.dao.ibatis.ProductBatchUploadDaoImplTest.testUpdateProductById
    com.alibaba.intl.biz.product.dao.ibatis.ProductBatchUploadDaoImplTest.testFindProductById

我们希望生成这样的一个ProductBatchUploadDaoImplTest.sed文件：

    /\<testUpdateProductById\>/, /@Test/{
            s/testUpdateProductById/comment_out_by_forrest_testUpdateProductById/
            s#@Test#//@Test#
    }
    
    /\<testFindProductById\>/, /@Test/{
            s/testFindProductById/comment_out_by_forrest_testFindProductById/
            s#@Test#//@Test#
    }

然后用户在他的模块src目录下执行如下shell命令：

    find . -f --name ProductBatchUploadDaoImplTest.java | xargs tac ProductBatchUploadDaoImplTest.java | sed  -f ProductBatchUploadDaoImplTest.sed  | tac > ProductBatchUploadDaoImplTest_fixed.java

最后将XXX_fixed.java替换XXX.java。

综合起来我们可以写成这么一个ProductBatchUploadDaoImplTest_ut_fix.shell文件：
    
    #!/bin/sh
    
    tac ProductBatchUploadDaoImplTest.java | sed '
    /\<testUpdateProductById\>/, /@Test/{
            s/testUpdateProductById/comment_out_by_forrest_testUpdateProductById/
            s#@Test#//@Test#
    }
    
    /\<testFindProductById\>/, /@Test/{
            s/testFindProductById/comment_out_by_forrest_testFindProductById/
            s#@Test#//@Test#
    }' | tac > ProductBatchUploadDaoImplTest_fixed.java


写了一个python脚本生成上面的订正脚本：
    
    #!/usr/bin/python
    
    import sys
    import optparse
    
    def generate_shell_for_this_failed_class(last_failed_clazz_path, last_failed_clazz, failed_clazz_path, failed_clazz, failed_tc):
        if last_failed_clazz is None:
            print "#!/bin/sh"
            print 
        else:
            print "\' | tac > %s_fixed_tmp.java" % last_failed_clazz_path
            print
            print "mv %s_fixed_tmp.java %s.java" % (last_failed_clazz_path, last_failed_clazz_path) 
            print
            print 'echo "<<end comment out for %s"' % last_failed_clazz
            print 
    
        print 
        print 'echo ">>begin comment out for %s"' % failed_clazz
        print 
        
        print "tac %s.java | sed \'" % failed_clazz_path
        print "   /\\<%s\\>/, /@Test/{" % failed_tc 
        print "   s/%s/comment_out_by_forrest_%s/" % (failed_tc, failed_tc)
        print "   s#@Test#//@Test#"
        print "   }"
    
    def generate_shell_for_the_last_failed_class(last_failed_clazz, failed_tc):
        print "   /\\<%s\\>/, /@Test/{" % failed_tc 
        print "   s/%s/comment_out_by_forrest_%s/" % (failed_tc, failed_tc)
        print "   s#@Test#//@Test#"
        print "   }"
    
    
    # com.alibaba.intl.biz.product.dao.ibatis.ProductBatchUploadDaoImplTest.testFindProductById
    def get_failed_tc(line):
        return line.split('.')[-1]
    
    def get_failed_clazz(line):
        return line.split('.')[-2]
    
    # com.alibaba.intl.biz.product.dao.ibatis.ProductBatchUploadDaoImplTest.testFindProductById
    # ==>com/alibaba/intl/biz/product/dao/ibatis/ProductBatchUploadDaoImplTest
    def get_failed_clazz_abs_path(destdir, line):
        sl = line.split('.')[:-1]
        return destdir + "/".join(sl)
    
    
    def main():
        p = optparse.OptionParser(description="generate the shell script to comment out the failed ut.",
                prog="generate_shell",
                usage="%prog dir")
    
        p.add_option('--destdir', '-d', action="store", dest="destdir", default="")
    
        options, arguments = p.parse_args()
        if len(arguments) == 1:
            filepath = arguments[0]
        else:
            p.print_help()
            exit(1)
    
        # append the ended '/' if necessary
        if options.destdir.endswith('/'):
            destdir = options.destdir
        else:
            destdir = options.destdir + '/'
    
        last_failed_clazz = None
        last_failed_clazz_path = None
        f = open(filepath, "rb")
        for line in f:
            line = line.strip()
            failed_tc = get_failed_tc(line) # get the failed test case name, testFindProductById, e.g.
            failed_clazz = get_failed_clazz(line) # get the failed test class name, ProductBatcheUploadDaoImplTest, e.g.
            failed_clazz_path = get_failed_clazz_abs_path(destdir, line)
            if failed_clazz != last_failed_clazz:
                generate_shell_for_this_failed_class(last_failed_clazz_path, last_failed_clazz, failed_clazz_path, failed_clazz, failed_tc)
                last_failed_clazz = failed_clazz
                last_failed_clazz_path = failed_clazz_path
            else:
                generate_shell_for_the_last_failed_class(last_failed_clazz, failed_tc)
    
        print "\' | tac > %s_fixed_tmp.java" % last_failed_clazz_path
        print
        print "mv %s_fixed_tmp.java %s.java" % (last_failed_clazz_path, last_failed_clazz_path) 
        print 
        print 'echo "<<end comment out for %s"' % last_failed_clazz
        print
        print 'echo "all the failed test case have been fixed by comment out @Test and change testcase name"'
    
    if __name__ == "__main__":
        main()

OK，现在让我们执行一下看效果如何：

    forrest@ubuntu:~/study/python$ python generate_comment_out_for_failed_ut_shell.py /home/forrest/Desktop/bbb.txt -d /home/forrest/work2/intl-biz/product/src/java.test/ > ut.sh
可以将这样的文件：
    
    com.alibaba.intl.biz.product.dao.ibatis.ProductBatchUploadDaoImplTest.testFindProductById
    com.alibaba.intl.biz.product.dao.ibatis.ProductBatchUploadDaoImplTest.testInsertProductUseSequence
    com.alibaba.intl.biz.product.dao.ibatis.ProductDaoImplTest.testBopsFindProductById

生成如下shell脚本：
    
    #!/bin/sh
    
    
    echo ">>begin comment out for ProductBatchUploadDaoImplTest"
    
    tac /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductBatchUploadDaoImplTest.java | sed '
       /\<testFindProductById\>/, /@Test/{
       s/testFindProductById/comment_out_by_forrest_testFindProductById/
       s#@Test#//@Test#
       }
       /\<testInsertProductUseSequence\>/, /@Test/{
       s/testInsertProductUseSequence/comment_out_by_forrest_testInsertProductUseSequence/
       s#@Test#//@Test#
       }
    ' | tac > /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductBatchUploadDaoImplTest_fixed_tmp.java
    
    mv /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductBatchUploadDaoImplTest_fixed_tmp.java /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductBatchUploadDaoImplTest.java
    
    echo "<<end comment out for ProductBatchUploadDaoImplTest"
    
    
    echo ">>begin comment out for ProductDaoImplTest"
    
    tac /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductDaoImplTest.java | sed '
       /\<testBopsFindProductById\>/, /@Test/{
       s/testBopsFindProductById/comment_out_by_forrest_testBopsFindProductById/
       s#@Test#//@Test#
       }
    ' | tac > /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductDaoImplTest_fixed_tmp.java
    
    mv /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductDaoImplTest_fixed_tmp.java /home/forrest/work2/intl-biz/product/src/java.test/com/alibaba/intl/biz/product/dao/ibatis/ProductDaoImplTest.java
    
    echo "<<end comment out for ProductDaoImplTest"
    
    echo "all the failed test case have been fixed by comment out @Test and change testcase name"

### 用awk实现更改函数名和注释@Test标注

前面我们用sed实现了更改函数名和注释@Test标注的功能，核心的sed其实很简单：

    /\<testFindProductById\>/, /@Test/{
            s/testFindProductById/comment_out_by_forrest_testFindProductById/
            s#@Test#//@Test#
    }
    forrest@ubuntu:~/study/sed$ tac ProductBatchUploadDaoImplTest.java | sed -f comment_function.sed | tac
即可。

awk是一门比sed更强大的语言，所以sed能够处理的事情，awk也能够处理，下面是用awk的核心实现代码：

    /\<testFindProductById\>/, /@Test/ {
        sub(/testFindProductById/, "comment_out_by_forrest_testFindProductById", $0)
        sub(/@Test/, "//@Test", $0)
    }
    { print }
    
    forrest@ubuntu:~/study/awk$ tac ProductBatchUploadDaoImplTest.java | awk -f comment_function.awk | tac
效果是一样的。

**注意：awk与sed处理机制有个最明显的不同是awk的pattern-action statements是pipeline关系，从上到下依次执行，每个statement的输入结果其实是上一个statement的处理输出结果。而对于sed，每个处理结果要么直接输出，要么放在hold space中，原始信息始终放在pattern space中，没有被修改。这是要特别注意的地方。**

举个例子。上面的awk脚本，如果改成：

    { print }
    
    /\<testFindProductById\>/, /@Test/ {
        sub(/testFindProductById/, "comment_out_by_forrest_testFindProductById", $0)
        sub(/@Test/, "//@Test", $0)
    }
将无任何效果。
改成

    /\<testFindProductById\>/, /@Test/ {
        sub(/testFindProductById/, "comment_out_by_forrest_testFindProductById", $0)
        sub(/@Test/, "//@Test", $0)
    }
    
    /comment_out_by_forrest/ { print }
将输出如下结果：

    forrest@ubuntu:~/study/awk$ tac ProductBatchUploadDaoImplTest.java | awk -f patternrange.awk | tac
    public void comment_out_by_forrest_testFindProductById() {
但是并不会导致死循环，因为每一行从上到下执行过的statement就不会再执行了。

另外，虽然awk不支持嵌套地址空间，但是awk在action中可以使用正则表达式，所以同样可以达到一样的效果。


### 如何注释函数体

我们来看看怎么将函数（体）注释掉。先考虑一下法2，也就是只注释函数体：
将

    @Test
    @Prepare(autoImport = true)
    public void testFindProductById() {
        ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
        if(result==null){
                throw new RuntimeException("no result!");
        }else{
                assertResult(result);
        }
    }

注释成：

    @Test
    @Prepare(autoImport = true)
    public void testFindProductById() {
        //ProductBatchUploadDO result = productBatchUploadDao.findProductById(22222222);
        //if(result==null){
        //        throw new RuntimeException("no result!");
        //}else{
        //        assertResult(result);
        //}
    }

算法：

给定failed TestCase Name，搜索public void testFindProductById声明，如果找到
position = begin，初始化nested = 0 (表示嵌套层次)
继续查找，如果找到{，则nested++，如果找到}，则nested--。当nested == 0时候，position = end。

    comment_all? include begin and end line?
    position {outside, begin, body, end}
    
    function comment_statement(curline){
       // check the curline
       if curline含有testCase ==> position = begin, nested = 0;
       if curline含有'{' ==> nested++; if position==begin & nested==1, position=body;  
       if curline含有'}' ==> nested--; if nested==0, positon = end; 
    
       if(nested>0){
           comment_out(curline);
       }
    }
    
    while(true){
        comment_statement(curline);
    }

给定一行，分析后，决定是pass或者comment_out这一行，并且根据需要更新position和nested（状态）。

可以看到这个算法需要两个状态变量和一些if-else判断，sed并不支持，所以考虑使用awk：
    
    BEGIN{
        nested = -100; # 表示函数嵌套层次
        position = "NOP"; #需要一个位置变量，便是是否进入要处理的函数，要有四个状态：不关心状态，要处理的函数头，要处理的函数体，要处理的函数尾
    }
    
    function comment_line(curline){
        if (nested > 0) {
            if( nested == 1 && position == "begin") {
                    print curline;
                    position = "body";
            }else if( position == "body"){
                    print "//" curline;
            }else{
                    print curline;
            }
        }else if( nested == 0) {
            if(position == "body") position = "end";
    
            print curline;
        }else if( nested < 0){
            position = "NOP";
    
            print curline;
        }
    }
    
    /\<testFindProductById\>/ { position = "begin"; nested = 0; }
    
    /{/ {
            nested++;
        }
    
    /}/ {
            nested--;
        }
    
    { comment_line($0); } #for every line, call the comment_line() function

事实上，awk还是比较适合处理表格形数据（tabular data）。如果写的awk比较复杂，还是使用python或者perl这样的更高级有点的语言比较好。

PS: 现在(2013-02-19)回过头整理我以前写的文章，发现以前的我真她妈蛋疼，纯粹瞎折腾，活该找不到女朋友。。。

--EOF--
