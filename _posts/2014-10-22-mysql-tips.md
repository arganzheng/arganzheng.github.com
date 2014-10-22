---
title: MySQL常用命令
layout: post
---


用户授权
--------

MySQL的用户创建和授权基本是通过一个命令解决：

　　GRANT privileges ON what TO user IDENTIFIED BY "password" WITH GRANT OPTION　 

要使用该语句，你需要填写下列部分： 

#### 1. privileges 授予用户的权限，下表列出可用于GRANT语句的权限指定符：

* Alter 修改表和索引 
* Create 创建数据库和表 
* Delete 删除表中已有的记录 
* Drop 抛弃（删除）数据库和表 
* INDEX 创建或抛弃索引 
* Insert 向表中插入新行 
* REFERENCE 未用 
* Select 检索表中的记录 
* Update 修改现存表记录 
* FILE 读或写服务器上的文件 
* PROCESS 查看服务器中执行的线程信息或杀死线程 
* RELOAD 重载授权表或清空日志、主机缓存或表缓存
* SHUTDOWN 关闭服务器 
* ALL 所有，ALL PRIVILEGES同义词
* USAGE 无权限，即创建用户，但不授予权限

#### 2. what　权限运用的级别。权限可以是全局的（适用于所有数据库和所有表）、特定数据库（适用于一个数据库中的所有表）或特定表的。可以通过指定一个columns字句是权限是列特定的。

#### 3. user　权限授予的用户，它由一个用户名和主机名组成。在MySQL中，你不仅指定谁能连接，还有从哪里连接。这允许你让两个同名用户从不同地方连接。 

#### 4. password 赋予用户的口令，它是可选的。如果你对新用户没有指定IDENTIFIED BY子句，该用户不赋给口令（不安全）。对现有用户，任何你指定的口令将代替老口令。如果你不指定口令，老口令保持不变，当你用IDENTIFIED BY时，口令字符串用改用口令的字面含义，GRANT将为你编码口令，不需要用SET PASSWORD 那样使用password()函数。 当然你也可以使用IDENTIFIED BY PASSWORD指定编码后的密码。

#### 5. WITH GRANT OPTION子句是可选的。如果你包含它，用户可以授予权限通过GRANT语句授权给其它用户。你可以用该子句给与其它用户授权的能力。


**NOTES**

用户名、口令、数据库和表名在授权表记录中是大小写敏感的，主机名和列名不是。 


### 一些例子

mysql> GRANT ALL ON `reading`.* TO 'arganzheng'@'localhost' IDENTIFIED BY 'xxxx';

给来自localhost的用户arganzheng分配可对数据库reading的所有操作权限，并且密码设置为xxxx。


mysql> GRANT ALL ON `reading`.* TO 'arganzheng'@'%' IDENTIFIED BY '密码';

给来自任何机器的用户arganzheng分配可对数据库reading的所有操作权限，并且密码设置为xxxx。

mysql> GRANT ALL ON `reading`.* TO 'arganzheng'@'localhost' IDENTIFIED BY '密码' with GRANT OPTION


mysql> GRANT select, insert, delete, update ON `reading`.* TO 'arganzheng'@'localhost'

给来自localhost的用户arganzheng分配可对数据库reading的所有表进行select,insert,update,delete操作的权限，因为arganzheng前面已经设置密码了，所以这里不需要再次通过IDENTIFIED BY设置密码。


**TISP**

1、当你使用GRANT和REVOKE语句时，会自动重载权限，但是如果是直接修改授权表则需要手动执行 FLUSH PRIVILEGES 命令。

2、复制老数据库的权限

这个在做MySQL slave的时候特别有用，可以使用下面这个脚本将主机的权限导出：

	MYSQL_CONN="-uroot -pXXXXX"
	mysql ${MYSQL_CONN} --skip-column-names -A -e"SELECT CONCAT('SHOW GRANTS FOR ''',user,'''@''',host,''';') FROM mysql.user WHERE user<>''" | mysql ${MYSQL_CONN} --skip-column-names -A | sed 's/$/;/g' > MySQLUserGrants.sql

然后将其导入到新的slave:

	mysql -uroot -p -A < MySQLUserGrants.sql










