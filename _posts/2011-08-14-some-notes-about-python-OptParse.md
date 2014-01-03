---
title: python2.x的一个需要注意的地方
layout: post
---


今天晚上过来跑图片dump python脚本，无意中发现python的OptParse的一个小“问题”，很容易中招，就像python的range一样，不小心就容易出问题。
程序主体逻辑如下：
def main():
    p = optparse.OptionParser(description="dump summ img in a dir that matches the given patten",
            prog="dump_summ_img",
            usage="%prog dir")

    p.add_option('--destdir', '-o', action="store", dest="destdir", default="")

    p.add_option('--fromnum', '-f', action="store", dest="fromnum", default=0)
    p.add_option('--tonum', '-t', action="store", dest="tonum", default=0)

    p.add_option('--depth', '-d', action="store", dest="depth", default=10)

    options, arguments = p.parse_args()
    if len(arguments) == 1:
        srcdir = arguments[0]
    else:
        p.print_help()
        exit(1)

    # remove the ended '/'
    global destdir
    if options.destdir.endswith('/'):
        destdir = options.destdir[:-1]
    else:
        destdir = options.destdir

    if options.fromnum <= 0 or options.tonum <= 0 or options.fromnum > options.tonum:
        print "Please input the range you want to dump, -f 1 -t 10, will dump [1, 10)."
        exit(1)
    else:
        fromnum = int(options.fromnum)
        tonum = int(options.tonum)

    for dir in range(fromnum, tonum):
        curdir = os.path.join(srcdir, str(dir))
        if not os.path.exists(curdir):
            print "ERROR: %s not exist, ignore and continue try next dir!" % curdir
            continue
        print "INFO: Begin dump summ img for %s" % curdir
        dump_files_with_pattern_for_dir(curdir, options.depth)

if __name__ == "__main__":
    main()

由于我们的产品图片一级目录是以产品Id进行两位划分的，所以其实是10~99的目录名，为了dump 80到99这20个目录，我执行如下命令：
forrest@ubuntu:~/study/python$ python dump_summ_img.py /mnt/product_repository/product/ -o /home/forrest/test/ -f 80 -t 100
但是发现程序总是打印：
Please input the range you want to dump, -f 1 -t 10, will dump [1, 10).
然后退出。
也就是说：
    if options.fromnum <= 0 or options.tonum <= 0 or options.fromnum > options.tonum:
        print "Please input the range you want to dump, -f 1 -t 10, will dump [1, 10)."
        exit(1)
这个if判断成立了，但是不可能啊。
而且奇怪的是将100改成99就没有问题了：
forrest@ubuntu:~/study/python$ python dump_summ_img.py /mnt/product_repository/product/ -o /home/forrest/test/ -f 80 -t 99
后来仔细看了一下程序，发现了问题：
    if options.fromnum <= 0 or options.tonum <= 0 or options.fromnum > options.tonum:
        print "Please input the range you want to dump, -f 1 -t 10, will dump [1, 10)."
        exit(1)
    else:
        fromnum = int(options.fromnum)
        tonum = int(options.tonum)

其实我是有注意到options.fromnum和options.tonum得到的其实是str，而不是int，因为我一开始没有强制类型转换，后面的for dir in range(fromnum, tonum): 报错了，说range期望输入是int类型而不是str类型。所以我将它们强制类型转换了。但是后来增加的判断却忘记了这点，但是python在这里却没有报错，这才是我写这篇文章的原因。
 if options.fromnum <= 0 or options.tonum <= 0 or options.fromnum > options.tonum:
在python看来，这是一个合法的表达式，对于options.fromnum > options.tonum，因为两者都是字符串，所以python执行的是字符串的大小比较。所以当你输入80和100的时候，"80" > "100:，所以options.fromnum > options.tonum成立了。
而比较诡异的是options.fromnum <= 0 or options.tonum <= 0，这是字符串与整数的比较，python没有报错，相反，它搞了个非常恶心的规则，让人十分惊讶：

How does Python compare string and int?
http://stackoverflow.com/questions/3270680/how-does-python-compare-string-and-int

From the manual:

CPython implementation detail: Objects of different types except numbers are ordered by their type names; objects of the same types that don’t support proper comparison are ordered by their address.

When you order two strings or two numeric types the ordering is done in the expected way (lexicographic ordering for string, numeric ordering for integers).

When you order a string and an integer the type names are ordered. "str" is lexicographically after "int", "float", "long", "list", "bool", etc. However a tuple will order higher than a string because "tuple" > "str":

0 > 'foo'
False
[1, 2] > 'foo'
False
(1, 2) > 'foo'
True
Is this behavior mandated by the language spec, or is it up to implementors?
There is no language specification. The language reference says:

Otherwise, objects of different types always compare unequal, and are ordered consistently but arbitrarily.

So it is an implementation detail.

Are there differences between any of the major Python implementations?
I can't answer this one because I have only used the official CPython implementation, but there are other implementations of Python such as PyPy.

Are there differences between versions of the Python language?
In Python 3.x the behaviour has been changed so that attempting to order an integer and a string will raise an error:

>>> '10' > 5
Traceback (most recent call last):
  File "", line 1, in 
    '10' > 5
TypeError: unorderable types: str() > int()

简单来说：
Strings are compared lexicographically, and dissimilar types are compared by the name of their type ("int" < "string"). 3.x fixes the second point by making them non-comparable.

写python写多了，越来越发现它的缺点，哎，语言就像女人，距离产生美，朦胧产生美。

PS: 很多动态语言（脚本语言）在类型方面都有类似的缺点，比如AWK中，对于字符串连接，不是使用用+运算符，而是空格。如果你是用+号连接，会产生意想不到的结果。比如下面这段AWK脚本:
BEGIN{
    nested = -10
}

function comment_line(curline){
    if (nested > 0) { 
        print "//"  + curline
    }else { 
        print curline 
    }
}

/testFindProductById/ { nested = 0; comment_line($0); }
/{/ { nested = nested + 1; comment_line($0); }
/}/ { nested = nested - 1; comment_line($0); }

会打印出0，而不是//xxx，因为在awk中+号只用于Number，字符串的+号操作返回的一个无结果的Number，就是0。将

  print "//"  + curline ==>  print "//"  curline 

就可以了。
