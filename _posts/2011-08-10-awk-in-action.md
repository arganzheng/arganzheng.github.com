---
title: AWK学习与实战
layout: post
---


### 前言

AWK的`pattern { action }`语法结构已经影响到很多后来的东东，当然更早的是`grep`和`sed`。
以至于现在有个网站叫做[ifttt](http://ifttt.com/)，其名字意思就是if this then that，做的就是这个事情。虽然这门语言有点过时，还是有必要介绍一下。

### What is AWK?

An awk program is a sequence of patterns and actions that tell what to look for in the input data and what to do when it's found. Awk searches a set of files for lines matched by any of the patterns; when a matching line is found, the corresponding action is performed. A pattern can select lines by combinations of regular expressions and comparison operations on strings, numbers, fields, variables, and array elements. Actions may perform arbitrary processing on selected lines; the action language looks like C but there are no declarations, and strings and numbers are built-in data types.

Awk scans the input files and splits each input line into fields automatically. Because so many things are automatic - input, field splitting, storage management, initialization - awk programs are usually much smaller than they would be in a more conventional language. Thus one common use of awk is for the kind of data manipulation suggested above.

Retrieval, transformation, reduction and validation of data, is the tasks that awk was originally designed for.


### The Structure of an AWK Program

Each awk program is a sequence of one or more pattern-action statements:

    pattern  { action }
    pattern  { action }
    ...

The basic operation of awk is to scan a sequence of input lines one after another, searching for lines that are matched by any of the patterns in the program. 

Every input line is tested against each of the patterns in turn. For each pattern that matches, the corresponding action (which may involve multiple steps)
is performed. Then the next line is read and the matching starts over. This continues until all the input has been read.

For example:

    $3 == 0  { print $1 }

is a single pattern-action statement; for every line in which the third field is zero, the first field is printed.

pattern and action are both optional, 如果没有指定pattern，那么就是针对每一行无条件执行action。如果没有指定action，那么默认是打印匹配行。

Since patterns and actions are both optional, actions are enclosed in braces to distinguish them from patterns.

### Running an AWK Program

1. awk 'program' input files
    
    for example:
        
        awk '$3 == 0 { print $1 }' file1 file2
    
    to print the first field of every line of file 1 and file2 in which the third field is zero.

2. awk 'program '

    In this case awk will apply the program to whatever you type next on your terminal until you type an end-of-file signal (control-d on Unix systems). 

3. awk -f progfile optional list of input files

    用过-f参数指定要运行的AWK程序。

### AWK's Data Type

There are only two types of data in awk: numbers and strings of characters.
这点跟shell差不多，不过shell的numbers只有int。另外，正如后来很多shell都支持array，AWK后面也是支持数组的。

Awk provides arrays for storing groups of related values.

EXAMPLE: The following program prints its input in reverse order by line.

    # reverse - print input in reverse order by line
    
    { line[NR] = $0 } # remember each input line
    END { i = NR # print lines in reverse order
        while (i > 0) {
                print line[i]
                i = i -1
        }
    }

### AWK's XXX

关于AWK的其他东东请参考陈皓写的[AWK 简明教程](http://coolshell.cn/articles/9070.html#jtss-tsina)。这是我见过的最简单明了的AWK教程了。为此，我很开心的删掉了我自己写的部分:)

### 使用AWK做简单统计和分析

#### 1. 计数（Counting）

使用emp变量统计工作时间大于15个小说的员工个数：

    $3 > 15 { emp = emp + 1 }
    END { print emp, "employees worked more than 15 hours." }

AWK variables used as numbers begin life with the value 0, so we didn't need to initialize emp.

#### 2. 求和与平均（Sums and Averages）

Here is a program that uses NR to compute the average pay:

    { pay = pay + $2 * $3 }
        
    END { print NR, "employees"
          print "total pay is", pay
          print "average pay is", pay/NR
    }

#### 3. 还能找出最大值：

AWK variables can hold strings of characters as well as numbers. This program finds the employee who is paid the most per hour:

    $2 > maxrate { maxrate = $2; maxemp = $1 }
    END { print "highest hourly rate:", maxrate, "for", maxemp }

In this program the variable maxrate holds a numeric value, while the variable maxemp holds a string. (If there are several employees who all make the same maximum pay, this program finds only the first.}


### 实战——使用awk分析反重复铺货数据

最近在做反重复铺货二期项目，其中需要根据ASC给的数据分析一下线上的产品的重复情况，比如：
一共有多少家公司，每家公司的重复产品组发布情况（x%的公司有xx个重复产品组。。)，每个重复产品组下重复产品组的分布情况（如x%的分组下有xx个重复产品。）。最多有XX个重复产品组，一个重复产品分组最多有XXX个重复产品。。。类似这样的数据。

ASC的同学最终扔给我这么一个TXT文件：

    forrest@ubuntu:~/Desktop$ ls -lsh group_total.txt 
    484M -rw-r--r-- 1 forrest forrest 484M 2011-08-22 09:27 group_total.txt
    forrest@ubuntu:~/Desktop$ wc -l group_total.txt 
    24845015 group_total.txt
    forrest@ubuntu:~/Desktop$ head group_total.txt 
    0,10000057,1
    1,100001141,2
    2,100001141,1
    3,100001141,1
    4,100001141,1
    5,100001426,1
    6,100001426,1
    7,100003003,1
    8,100003056,1
    9,10000357,1
    
每一行表示一个重复产品组，其中field依次是new_cluster_id, vaccount_id, group_count。
group_count的级别只到公司级别，不是member级别。所以上线时重复分组数会更多，囧。

让我们一个问题一个问题来分析和处理：

#### 1. 一共有多少个重复分组

分析：重复分组就是group_count>1的分组
于是可以这么写：

    BEGIN{ 
        FS = ","
        groupcount = 0 
    }
    
    $3 > 1 { groupcount++ }
    
    END{ print groupcount }

运行一下：

    forrest@ubuntu:~/Desktop$ awk -f how_many_group.awk group_total.txt 
    9239808

#### 2. 一共有多少个公司有重复产品

分析：找出那些group_count>1的分组，提取其vaccount_id，去除重复结果就是了。

关键在于重复的vaccount_id，对于每一行记录，需要判断当前行的vaccount_id是不是已经存在，如果不存在，vaccountcount++;

awk并不是很适合做这样的事情，因为他没有map或set数据结构。不过可以先将记录按照vaccount_id排序一下，这样，当检查到vaccountId变化了，说明是另一个公司了。

不管用哪种方式，可以顺便统计一下每个公司有多少个重复分组。使用map需要考虑map的大小，以免OOM。

    forrest@ubuntu:~/Desktop$ sort -nr -k 2  group_total.txt -o group_total_sorted_by_vaccountId_2.txt
    forrest@ubuntu:~/Desktop$ tail group_total_sorted_by_vaccountId_2.txt
    15749303,1810,1
    15215638,1810,1
    15215637,1810,1
    11929142,1810,1
    11163978,1810,1
    1772922,1775,6
    13382971,1703,1
    11246939,24,1
    4437936,23,1
    16637830,23,1

发现sort还是有点问题的。最好还是将数据调整一下：

    forrest@ubuntu:~/Desktop$ awk -F, '{print $2, $1, $3}' group_total.txt > group_total_2.txt
    forrest@ubuntu:~/Desktop$ sort -n group_total_2.txt -o group_total_sorted_by_vaccountId.txt
    forrest@ubuntu:~/Desktop$ head group_total_sorted_by_vaccountId.txt 
    23 16637830 1
    23 4437936 1
    24 11246939 1
    1703 13382971 1
    1775 1772922 6
    1810 11163978 1
    1810 11429190 3
    1810 11929142 1
    1810 15215637 1
    1810 15215638 1
    
我将field顺序调整了一下，变成了：vaccount_id, cluster_id, group_count，然后直接根据数值排序，这样sort才比较正常（注：现在笔者终于知道前面sort为什么结果不正确，是因为`-k 2`，其实是从第二列开始比较排序，而不是只比较第二列）。下面我们就直接处理group_total_2.txt，也不需要设置FS=","了。
    
    forrest@ubuntu:~/Desktop$ cat how_many_company_spam.awk
    BEGIN{
            last_vaccount_id = 0;
            last_group_count = 0;
            last_total_count = 0;
    }

    $3>1 {
            this_vaccount_id = $1
            if(last_vaccount_id == 0){
                    last_vaccount_id=this_vaccount_id;
            }
            if(last_vaccount_id == this_vaccount_id){ # the same vaccount
                    last_group_count++;
                    last_total_count+=$3
            }else{ #a new vacount begin, print the statis
                    printf("%s %s %s\n", last_vaccount_id, last_group_count, last_total_count);
                    last_vaccount_id = this_vaccount_id;
                    last_group_count = 1;
                    last_total_count = $3;
            }
    }

    END{#don't forget to print the last vaccount id
            printf("%s %s %s\n", last_vaccount_id, last_group_count, last_total_count);
    }
    
稍微有点复杂，由于AWK所有变量默认都是初始化为0,所以上面的BEGIN其实可以省略。运行结果如下：
    
    forrest@ubuntu:~/Desktop$ awk -f how_many_company_spam.awk group_total_2.txt > group_count_static_for_company.txt
    forrest@ubuntu:~/Desktop$ head group_count_static_for_company.txt 
    100001141 1 2
    100005800 1 18
    100007438 1 15
    100008161 3 8
    100008694 3 12
    10001167 1 2
    100051088 1 2
    10006316 1 2
    100069413 2 7
    100070717 2 24
    
得出如下统计结果——每个公司的重复产品组数量和重复产品数量。

现在要统计有多少个公司spam了，只需要统计行数就可以了。

    forrest@ubuntu:~/Desktop$ wc -l group_count_static_for_company.txt
    1392336 group_count_static_for_company.txt

有个这个信息，还可以得到很多信息了。比如重复产品分组最多的公司和数量。

### 结束语

虽然笔者曾经很认真的研究过AWK和sed，但是还是不推荐过分使用AWK。笔者认为AWK比较适合用于如下场景：
1. one line command
2. 简单数值统计和分析
3. 简单条件过滤

如果在熟悉AWK的情况下，10分钟之内不能写出来，那么建议不要使用AWK，用其他你熟悉的高级脚本语言吧，比如python或者ruby:)



