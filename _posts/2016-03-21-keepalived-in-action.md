---
title: keepalived实战
layout: post
catalog: true
---


理论介绍
-------

Keepalived uses the IP Virtual Server (IPVS) kernel module to provide transport layer (Layer 4) load balancing, redirecting requests for network-based services to individual members of a server cluster. IPVS monitors the status of each server and uses the Virtual Router Redundancy Protocol (VRRP) to implement high availability.


### 配置文件说明

keepalived默认使用`/etc/keepalived/keepalived.conf`做完配置文件。配置文件分为下面几个配置段：

global_defs

	Defines global settings such as the email addresses for sending notification messages, the IP address of an SMTP server, the timeout value for SMTP connections in seconds, a string that identifies the host machine, the VRRP IPv4 and IPv6 multicast addresses, and whether SNMP traps should be enabled.

static_ipaddress, static_routes

	Define static IP addresses and routes, which VRRP cannot change. These sections are not required if the addresses and routes are already defined on the servers and these servers already have network connectivity.

vrrp_sync_group

	Defines a VRRP synchronization group of VRRP instances that fail over together.

vrrp_instance

	Defines a moveable virtual IP address for a member of a VRRP synchronization group's internal or external network interface, which accompanies other group members during a state transition. Each VRRP instance must have a unique value of virtual_router_id, which identifies which interfaces on the master and backup servers can be assigned a given virtual IP address. You can also specify scripts that are run on state transitions to BACKUP, MASTER, and FAULT, and whether to trigger SMTP alerts for state transitions.

vrrp_script

	Defines a tracking script that Keepalived can run at regular intervals to perform monitoring actions from a vrrp_instance or vrrp_sync_group section.

virtual_server_group

	Defines a virtual server group, which allows a real server to be a member of several virtual server groups.

virtual_server

	Defines a virtual server for load balancing, which is composed of several real servers.



### 安装和启动

法一：系统安装

	# yum install keepalived

法二：源码编译安装	

	# yum -y install kernel-headers kernel-devel

	$ cd /home/work/downloads/keepalived-1.2.19

	$ ./configure --prefix=/home/work/keepalived/ exec_prefix=/home/work/keepalived --with-kernel-dir=/lib/modules/$(uname -r)/build 

	...

	Keepalived configuration
	------------------------
	Keepalived version       : 1.2.19
	Compiler                 : gcc
	Compiler flags           : -g -O2
	Extra Lib                : -lssl -lcrypto -lcrypt
	Use IPVS Framework       : Yes
	IPVS sync daemon support : Yes
	IPVS use libnl           : No
	fwmark socket support    : Yes
	Use VRRP Framework       : Yes
	Use VRRP VMAC            : Yes
	SNMP support             : No
	SHA1 support             : No
	Use Debug flags          : No

	$ make
	
	...

	$ make install
	
	...

	$ cd /home/work/keepalived
	$ ll keepalived/
	total 16
	drwxr-xr-x 2 work work 4096 Mar 22 10:24 bin
	drwxr-xr-x 5 work work 4096 Mar 22 10:24 etc
	drwxr-xr-x 2 work work 4096 Mar 22 10:24 sbin
	drwxr-xr-x 3 work work 4096 Mar 22 10:24 share

这样就安装成功了。网上有文档说还要开启`net.ipv4.ip_nonlocal_bind=1`开启允许绑定非本机的IP，但是我实战结果是不需要的：

	# vim /etc/sysctl.conf

添加：

	net.ipv4.ip_nonlocal_bind=1 	# 开启允许绑定非本机的IP

然后执行：

	# sysctl -p


实战
----

### 实战1：VIP Failover

A typical Keepalived high-availability configuration consists of one master server and one or more backup servers. One or more virtual IP addresses, defined as VRRP instances, are assigned to the master server's network interfaces so that it can service network clients. The backup servers listen for multicast VRRP advertisement packets that the master server transmits at regular intervals. The default advertisement interval is one second. If the backup nodes fail to receive three consecutive VRRP advertisements, the backup server with the highest assigned priority takes over as the master server and assigns the virtual IP addresses to its own network interfaces. If several backup servers have the same priority, the backup server with the highest IP address value becomes the master server.

#### 配置

	$ vim etc/keepalived/keepalived.conf

	vrrp_instance VRRP1 {  

		interface xgbe0 		# Specify the network interface to which the virtual address is assigned

		state MASTER

		priority 101           	# 101 on master, 100 on backup

		nopreempt               # Don't fail back

		virtual_router_id 51 	# Needs to be the same on both instances, 
								# and needs to be unique if using multicast, does not matter with unicast

		authentication {
	       	auth_type PASS
	        auth_pass 10086
	    }

		virtual_ipaddress {  	# The VIP that keepalived will monitor
			10.244.81.8/25  dev xgbe0
		}

	}


backup的配置其实跟master差不多，主要state, priority不同：


	$ vim etc/keepalived/keepalived.conf

	vrrp_instance VRRP1 {  

		interface xgbe0 			# Specify the network interface to which the virtual address is assigned

		state BACKUP

		priority 100           	# 101 on master, 100 on backup

		nopreempt               # Don't fail back

		virtual_router_id 51 	# Needs to be the same on both instances, 
								# and needs to be unique if using multicast, does not matter with unicast

		authentication {
	       	auth_type PASS
	        auth_pass 10086
	    }

		virtual_ipaddress {  	# The VIP that keepalived will monitor
			10.244.81.8/25  dev xgbe0
		}

	}

In the event that the master server (svr1) fails, keepalived assigns the virtual IP address 10.0.0.100/24 to the eth0 interface on the backup server (svr2), which becomes the master server.

你可以通过`ip`命令确定VIP是否处于活跃状态：

	[work@hkg02-mj-data01.hkg02.arganzheng.me keepalived]$ ip add list
	1: lo: <LOOPBACK,UP,LOWER_UP> mtu 16436 qdisc noqueue state UNKNOWN
	    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
	    inet 127.0.0.1/8 scope host lo
	2: xgbe0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
	    link/ether 80:38:bc:0e:78:49 brd ff:ff:ff:ff:ff:ff
	    inet 10.244.81.25/25 brd 10.244.81.127 scope global xgbe0
	    inet 10.244.81.8/25 scope global secondary xgbe0
	3: xgbe1: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN qlen 1000
	    link/ether 80:38:bc:0e:78:4a brd ff:ff:ff:ff:ff:ff
	4: eth0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN qlen 1000
	    link/ether 9c:37:f4:8c:15:46 brd ff:ff:ff:ff:ff:ff
	5: eth1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN qlen 1000
	    link/ether 9c:37:f4:8c:15:47 brd ff:ff:ff:ff:ff:ff

你也可以通过`/var/log/messages`日志文件得知：

	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived[19254]: Starting Keepalived v1.2.19 (03/22,2016)
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived[19255]: Starting Healthcheck child process, pid=19256
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived[19255]: Starting VRRP child process, pid=19257
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: Registering Kernel netlink reflector
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: Registering Kernel netlink command channel
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: Opening file '/home/work/keepalived/etc/keepalived/keepalived.conf'.
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: Registering Kernel netlink reflector
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: Registering Kernel netlink command channel
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: Registering gratuitous ARP shared channel
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: Opening file '/home/work/keepalived/etc/keepalived/keepalived.conf'.
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: Configuration is using : 45872 Bytes
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: Configuration is using : 101010 Bytes
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: ------< Global definitions >------
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:  Router ID = hkg02-mj-data01.hkg02.arganzheng.me
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:  VRRP IPv6 mcast group = ff02::12
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: ------< VRRP Topology >------
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:  VRRP Instance = VRRP1
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Using VRRPv2
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Want State = MASTER
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Runing on device = xgbe0
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Gratuitous ARP repeat = 5
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Gratuitous ARP refresh repeat = 1
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Virtual Router ID = 51
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Priority = 101
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Advert interval = 1 sec
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Accept disabled
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Preempt disabled
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Authentication type = SIMPLE_PASSWORD
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Password = 10086
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:    Virtual IP = 1
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]:      10.244.81.8/25 dev xgbe0 scope global
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: Using LinkWatch kernel netlink reflector...
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: ------< Global definitions >------
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]:  Router ID = hkg02-mj-data01.hkg02.arganzheng.me
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]:  VRRP IPv6 mcast group = ff02::12
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: ------< SSL definitions >------
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]:  Using autogen SSL context
	Mar 22 15:25:28 hkg02-spi-spi2sys4 Keepalived_healthcheckers[19256]: Using LinkWatch kernel netlink reflector...
	Mar 22 15:25:29 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: VRRP_Instance(VRRP1) Transition to MASTER STATE
	Mar 22 15:25:30 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: VRRP_Instance(VRRP1) Entering MASTER STATE
	Mar 22 15:25:32 hkg02-spi-spi2sys4 ntpd[2594]: Listening on interface #3 xgbe0, 10.244.81.8#123 Enabled

同样启动backup上的keepalived，查看/var/log/message日志：

	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived[46389]: Starting Keepalived v1.2.19 (03/22,2016)
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived[46390]: Starting Healthcheck child process, pid=46391
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived[46390]: Starting VRRP child process, pid=46392
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]: Registering Kernel netlink reflector
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]: Registering Kernel netlink command channel
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Registering Kernel netlink reflector
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Registering Kernel netlink command channel
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Registering Kernel netlink command channel
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]: Opening file '/home/work/keepalived/etc/keepalived/keepalived.conf'.
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Registering gratuitous ARP shared channel
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Registering gratuitous ARP shared channel
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Opening file '/home/work/keepalived/etc/keepalived/keepalived.conf'.
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]: Configuration is using : 45872 Bytes
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Configuration is using : 101010 Bytes
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: ------< Global definitions >------
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:  Router ID = hkg02-mj-data02.hkg02.arganzheng.me
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: ------< Global definitions >------
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:  Router ID = hkg02-mj-data02.hkg02.arganzheng.me
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:  VRRP IPv6 mcast group = ff02::12
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: ------< VRRP Topology >------
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:  VRRP Instance = VRRP1
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Using VRRPv2
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Want State = BACKUP
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Using VRRPv2
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Want State = BACKUP
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Runing on device = xgbe0
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Gratuitous ARP repeat = 5
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Gratuitous ARP refresh repeat = 1
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Virtual Router ID = 51
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Priority = 100
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Advert interval = 1 sec
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Accept disabled
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Preempt disabled
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Authentication type = SIMPLE_PASSWORD
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Password = 10086
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Password = 10086
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:    Virtual IP = 1
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:      10.244.81.8/25 dev xgbe0 scope global
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]:      10.244.81.8/25 dev xgbe0 scope global
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: Using LinkWatch kernel netlink reflector...
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: VRRP_Instance(VRRP1) Entering BACKUP STATE
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]: ------< Global definitions >------
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]:  Router ID = hkg02-mj-data02.hkg02.arganzheng.me
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]:  VRRP IPv6 mcast group = ff02::12
	Mar 22 16:06:20 hkg02-spi-spi2sys14 Keepalived_healthcheckers[46391]:  VRRP IPv6 mcast group = ff02::12


现在可以测试一下VIP是不是有切换。把master上的keepalived进程干掉：
	
	[root@hkg02-mj-data01.hkg02.arganzheng.me ~]# killall keepalived

发现data01的日志变化：

	Mar 22 16:10:35 hkg02-spi-spi2sys4 Keepalived[19255]: Stopping Keepalived v1.2.19 (03/22,2016)
	Mar 22 16:10:35 hkg02-spi-spi2sys4 Keepalived_vrrp[19257]: VRRP_Instance(VRRP1) sending 0 priority
	Mar 22 16:10:36 hkg02-spi-spi2sys4 ntpd[2594]: Deleting interface #3 xgbe0, 10.244.81.8#123, interface stats: received=0, sent=0, dropped=0, active_time=2704 secs	

VIP也已经被摘除了：

	[root@hkg02-mj-data01.hkg02.arganzheng.me ~]# ip addr list xgbe0
	2: xgbe0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
	    link/ether 80:38:bc:0e:78:49 brd ff:ff:ff:ff:ff:ff
	    inet 10.244.81.25/25 brd 10.244.81.127 scope global xgbe0

data02的日志也跟着变化：

	Mar 22 16:10:35 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: VRRP_Instance(VRRP1) Transition to MASTER STATE
	Mar 22 16:10:36 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: VRRP_Instance(VRRP1) Entering MASTER STATE
	Mar 22 16:10:37 hkg02-spi-spi2sys14 ntpd[2577]: Listening on interface #3 xgbe0, 10.244.81.8#123 Enabled	

VIP也跟着挂载到xgbe0上了：

	[work@hkg02-mj-data02.hkg02.arganzheng.me keepalived]$ ip addr list  xgbe0
	2: xgbe0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
	    link/ether 80:38:bc:0e:79:31 brd ff:ff:ff:ff:ff:ff
	    inet 10.244.81.24/25 brd 10.244.81.127 scope global xgbe0
	    inet 10.244.81.8/25 scope global secondary xgbe0

然后把data01的keepalived重新启动:

	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived[41476]: Starting Keepalived v1.2.19 (03/22,2016)
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived[41477]: Starting Healthcheck child process, pid=41478
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived[41477]: Starting VRRP child process, pid=41479
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: Registering Kernel netlink reflector
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: Registering Kernel netlink command channel
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: Opening file '/home/work/keepalived/etc/keepalived/keepalived.conf'.
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: Registering Kernel netlink reflector
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: Registering Kernel netlink command channel
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: Registering gratuitous ARP shared channel
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: Opening file '/home/work/keepalived/etc/keepalived/keepalived.conf'.
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: Configuration is using : 45872 Bytes
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: Configuration is using : 101010 Bytes
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: ------< Global definitions >------
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:  Router ID = hkg02-mj-data01.hkg02.arganzheng.me
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:  VRRP IPv6 mcast group = ff02::12
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: ------< VRRP Topology >------
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:  VRRP Instance = VRRP1
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Using VRRPv2
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Want State = MASTER
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Runing on device = xgbe0
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Gratuitous ARP repeat = 5
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Gratuitous ARP refresh repeat = 1
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Virtual Router ID = 51
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Priority = 101
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Advert interval = 1 sec
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Accept disabled
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Preempt disabled
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Authentication type = SIMPLE_PASSWORD
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Password = 10086
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:    Virtual IP = 1
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]:      10.244.81.8/25 dev xgbe0 scope global
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: Using LinkWatch kernel netlink reflector...
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: ------< Global definitions >------
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]:  Router ID = hkg02-mj-data01.hkg02.arganzheng.me
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]:  VRRP IPv4 mcast group = 224.0.0.18
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]:  VRRP IPv6 mcast group = ff02::12
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: ------< SSL definitions >------
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]:  Using autogen SSL context
	Mar 22 16:32:06 hkg02-spi-spi2sys4 Keepalived_healthcheckers[41478]: Using LinkWatch kernel netlink reflector...
	Mar 22 16:32:07 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: VRRP_Instance(VRRP1) Transition to MASTER STATE
	Mar 22 16:32:07 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: VRRP_Instance(VRRP1) Received lower prio advert, forcing new election
	Mar 22 16:32:08 hkg02-spi-spi2sys4 Keepalived_vrrp[41479]: VRRP_Instance(VRRP1) Entering MASTER STATE
	Mar 22 16:32:09 hkg02-spi-spi2sys4 ntpd[2594]: Listening on interface #4 xgbe0, 10.244.81.8#123 Enabled

发现VIP又被抢占回去了：

	[root@hkg02-mj-data01.hkg02.arganzheng.me keepalived]# ip addr list xgbe0
	2: xgbe0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
	    link/ether 80:38:bc:0e:78:49 brd ff:ff:ff:ff:ff:ff
	    inet 10.244.81.25/25 brd 10.244.81.127 scope global xgbe0
	    inet 10.244.81.8/25 scope global secondary xgbe0

data02的日志：

	Mar 22 16:32:07 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: VRRP_Instance(VRRP1) Received higher prio advert
	Mar 22 16:32:07 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: VRRP_Instance(VRRP1) Entering BACKUP STATE
	Mar 22 16:32:07 hkg02-spi-spi2sys14 Keepalived_vrrp[46392]: VRRP_Instance(VRRP1) Entering BACKUP STATE
	Mar 22 16:32:08 hkg02-spi-spi2sys14 ntpd[2577]: Deleting interface #3 xgbe0, 10.244.81.8#123, interface stats: received=0, sent=0, dropped=0, active_time=1291 secs

VIP已经丢失：

	[work@hkg02-mj-data02.hkg02.arganzheng.me keepalived]$ ip addr list  xgbe0
	2: xgbe0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
	    link/ether 80:38:bc:0e:79:31 brd ff:ff:ff:ff:ff:ff
	    inet 10.244.81.24/25 brd 10.244.81.127 scope global xgbe0

但是我明明配置了`nopreempt`模式，理论上来说，master的keepalived恢复，VIP不应该被抢占回去啊。谷歌了一下，原来nopreempt是可以阻止master恢复之后重新抢占VIP，但是前提是你两个节点的state都配置成BACKUP（http://article.gmane.org/gmane.linux.keepalived.devel/1537）。

**NOTES && TIPS**

1、keepalived的pid文件默认是写到`/var/run/keepalived.pid`文件，所以，即使你使用了非root自定义目录安装，最后启动还是要以root身份启动。

2、如果不是默认安装，或者使用的配置文件不是默认的配置文件，在启动Keepalived的时候一定要使用`-f`参数指定配置文件启动。而且**一定要使用绝对路径**。keepalived是不认相对路径的，而且不会提示找不到配置文件。可以使用下面的启动项启动：

	[root@hkg02-mj-data01.hkg02.arganzheng.me keepalived]# sbin/keepalived -d -D -f /home/work/keepalived/etc/keepalived/keepalived.conf

3、 默认情况下ARRP工作在“抢占模式”下，如果发现一个节点的服务停止了，另一个节点会立即把VIP和VMAC“抢过来”。但是我们一般都希望VIP能够保持在同一个节点，只要那个节点没有挂掉。keepalived有个`nopreempt`参数，可以阻止master恢复之后重新抢占VIP，但是前提是你两个节点的state都配置成BACKUP（http://article.gmane.org/gmane.linux.keepalived.devel/1537）。

4、如果Keepalived所在网络不允许使用组播（或者有防火墙之类的），可以使用VRRP单播 [Keepalived using unicast, track and notify scripts](http://kaivanov.blogspot.com/2015/02/keepalived-using-unicast-track-and.html): 

	unicast_src_ip 10.244.97.4 # local-IP
	unicast_peer {
		10.244.97.5 # peer-IP
	}

5、需要注意主备的state, weight和priority的值，这些值如果设置不合理可能会影响VIP的切换。

6、keepalived并没有自己的日志文件，所有的日志都打在`/var/log/message`中，开启-d -D选项日志会更详细一些。


### 实战2：MySQL Master HA


如果要做MySQL的Master HA，那么只需要在前面VIP failover的基础上增加对后端MySQL的检查，即原来是keepalived挂了，VIP failover，现在是MySQL挂了触发VIP failover，从而实现MySQL master failover。

keepalived的Track Script机制就是为这个准备的。

#### Keepalived Tracking Script

检查脚本可以是shell或者python等语言，keepalived只对脚本的返回值做了规定：

* 0 表示 “everything is fine”
* 1 (其实是非0) 表示 “something went wrong”

检查脚本可以检查任何逻辑，下面是常见的检查点：

* 进程有没有在运行?
* IP地址是不是ping得通?
* 应用是不是正常运行（比如redis是不是PING返回正常的PONG，MySQL的select心跳，等等）


如果除了VIP failover，你还需要做其他事情，那么这时候就需要用到Keepalived的另一个非常有用的机制——notify script。

#### Keepalived Notify Script

Keepalived Notify Script提供了通过脚本来实现状态变更时的回调处理的机制。这样我们就可以做VIP failover（将VIP从网口上添加或者摘除）之外更多的事情。它的配置格式大概如下：

	vrrp_instance MyVRRPInstance {
		
		...

		notify_master "/home/work/userbin/keepalived/master-backup.sh MASTER"
		notify_backup "/home/work/userbin/keepalived/master-backup.sh BACKUP"
		notify_fault "/home/work/userbin/keepalived/master-backup.sh FAULT"
	}

* notify_mater - 当节点变成master状态时触发
* nofity_backup - 当节点变成backup状态时触发
* notify_fault - 当网络异常或者我们的检查脚本错误次数达到fault的阈值时触发


跟检查脚本一样，通知脚本也可以是任何脚本语言(shell, perl, python, ruby, etc.)，Keepalived会传递下面三个参数给通知脚本:

* $1 = “GROUP” or “INSTANCE” 
* $2 = name of group or instance 
* $3 = target state of transition (“MASTER”, “BACKUP”, “FAULT”) 

通知脚本可以根据这些参数做相应的逻辑处理。一个简单通用的master-backup.sh模板一般如下：

	#!/bin/bash

	ENDSTATE=$3
	NAME=$2
	TYPE=$1

	case $ENDSTATE in
	    "BACKUP") # Perform action for transition to BACKUP state
	              exit 0
	              ;;
	    "FAULT")  # Perform action for transition to FAULT state
	              exit 0
	              ;;
	    "MASTER") # Perform action for transition to MASTER state
	              exit 0
	              ;;
	    *)        echo "Unknown state ${ENDSTATE} for VRRP ${TYPE} ${NAME}"
	              exit 1
	              ;;
	esac


具体参见: [About Keepalived Notification and Tracking Scripts](https://docs.oracle.com/cd/E37670_01/E41138/html/section_hxz_zdw_pr.html)

#### 具体配置

	$ vim etc/keepalived/keepalived.conf

	global_defs {
	   	notification_email {
	    	zhengzhibin@arganzheng.me
	     	yanglibei@arganzheng.me
	   }

	   notification_email_from keepalived@data01
	   # smtp_server 127.0.0.1 # smtp服务器地址,如果安装并开启了postfix的话,可以填写本机地址.
	   # smtp_connect_timeout 30 # 连接邮件服务器超时时间
	}

	vrrp_script check_mysql {   # Requires keepalived-1.1.13
		script 	"/home/work/userbin/keepalived/check_mysql.sh"
	
		interval 2      # Run script every 2 seconds
		fall 2		# If script returns non-zero f times in succession, enter FAULT state
		rise 3		# If script returns zero r times in succession, exit FAULT state
		timeout 3   	# Wait up to t seconds for script before assuming non-zero exit code
		# weight w      # Reduce priority by w on fall or add w points of priority if OK
	}

	vrrp_instance VI_MYSQL {  

		interface xgbe0			# Specify the network interface to which the virtual address is assigned

		state BACKUP            # both backup to make nopreempt works

		priority 101           	# 101 on master, 100 on backup

		nopreempt               # Don't fail back

		virtual_router_id 51 	# Needs to be the same on both instances, 
								# and needs to be unique if using multicast, does not matter with unicast

		authentication {
	       	auth_type PASS
	        auth_pass 10086
	    }

		virtual_ipaddress {  	# The VIP that keepalived will monitor
			10.244.81.8/25  dev xgbe0
		}

		track_script {
			check_mysql
		}
	}


其中/home/work/userbin/keepalived/check_mysql.sh对本机的MySQL进行检查：

	#!/bin/bash
	# Check if mysql is running ok, return 1 if not.
	# Used by keepalived to initiate a failover in case mysql is down

	MYSQL_DIR=/home/work/mysql/mysql-5.7.11-linux-glibc2.5-x86_64

	cd $MYSQL_DIR
	MYSQL_CONN='-uxxx -pxxxx'
	bin/mysql ${MYSQL_CONN} --skip-column-names -A -e "select version()" >/dev/null 2>&1

	if [ $? -eq 0 ]; then
	 	echo "MySQL is running OK."
	  	exit 0
	else
	  	echo "MySQL is NOT running properly. Setting keepalived state to FAULT."
	  	exit 1
	fi


这里我们只是做了简单的心跳检查。


> If a configured script returns a non-zero exit code f times in succession, Keepalived changes the state of the VRRP instance or group to FAULT, removes the virtual IP address 10.0.0.10 from eth0, reduces the priority value by w and stops sending multicast VRRP packets. If the script subsequently returns a zero exit code r times in succession, the VRRP instance or group exits the FAULT state and transitions to the MASTER or BACKUP state depending on its new priority.


#### 测试

启动master和slave所在的MySQL和keepalived进程。这时候VIP指向master所在的data01。在另一台机器web04上check这时候的master serverId是哪个：

	[work@hkg02-mj-web04.hkg02.arganzheng.me work]$ mysql -h10.244.81.8 -uxxx -pxxxxx --skip-column-names -A -e "select @@server_id"
	+------+
	| 1    |
	+------+

可以看到确实是data01的master在响应请求。

然后我们把data01上的master关闭：

	bin/mysqladmin -u root -p shutdown 

很快就可以看到data01的keepalived日志显示：

	Mar 23 15:39:37 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:39 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Script(check_mysql) failed
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Instance(VI_MYSQL) Entering FAULT STATE
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Instance(VI_MYSQL) removing protocol VIPs.
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Instance(VI_MYSQL) removing protocol VIPs.
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_healthcheckers[55232]: Netlink reflector reports IP 10.244.81.8 removed
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Instance(VI_MYSQL) Now in FAULT state
	Mar 23 15:39:39 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Instance(VI_MYSQL) Now in FAULT state
	Mar 23 15:39:40 hkg02-mj-data01 ntpd[5338]: Deleting interface #3 xgbe0, 10.244.81.8#123, interface stats: received=0, sent=0, dropped=0, active_time=646 secs
	Mar 23 15:39:41 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:43 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:45 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:47 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:49 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:51 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:53 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:55 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:57 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:39:59 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.	
	....

然后满屏的日志都是check_mysql打出来的错误日志：MySQL is NOT running properly. Setting keepalived state to FAULT.

data02的keepalived日志则相应的显示：

	Mar 23 15:39:40 hkg02-spi-spi2sys14 Keepalived_vrrp[6916]: VRRP_Instance(VI_MYSQL) Transition to MASTER STATE
	Mar 23 15:39:41 hkg02-spi-spi2sys14 Keepalived_vrrp[6916]: VRRP_Instance(VI_MYSQL) Entering MASTER STATE
	Mar 23 15:39:41 hkg02-spi-spi2sys14 Keepalived_vrrp[6916]: VRRP_Instance(VI_MYSQL) setting protocol VIPs.
	Mar 23 15:39:41 hkg02-spi-spi2sys14 Keepalived_vrrp[6916]: VRRP_Instance(VI_MYSQL) Sending gratuitous ARPs on xgbe0 for 10.244.81.8
	Mar 23 15:39:41 hkg02-spi-spi2sys14 Keepalived_healthcheckers[6914]: Netlink reflector reports IP 10.244.81.8 added
	Mar 23 15:39:42 hkg02-spi-spi2sys14 ntpd[2577]: Listening on interface #7 xgbe0, 10.244.81.8#123 Enabled
	Mar 23 15:39:46 hkg02-spi-spi2sys14 Keepalived_vrrp[6916]: VRRP_Instance(VI_MYSQL) Sending gratuitous ARPs on xgbe0 for 10.244.81.8

这时候在web04再次请求`select @@server_id`就返回2了：

	[work@hkg02-mj-web04.hkg02.arganzheng.me work]$ mysql -h10.244.81.8 -uxxx -xxxx --skip-column-names -A -e "select @@server_id"
	+------+
	| 2    |
	+------+

整个过程大概持续3秒钟吧。


然后这时候我们再把data01的master重启，看看结果怎样：

	[work@hkg02-mj-data01.hkg02.arganzheng.me mysql-5.7.11-linux-glibc2.5-x86_64]$ bin/mysqld_safe &
	[1] 49418
	[work@hkg02-mj-data01.hkg02.arganzheng.me mysql-5.7.11-linux-glibc2.5-x86_64]$ 2016-03-23T07:44:29.790230Z mysqld_safe Logging to '/home/work/mysql/mysql-5.7.11-linux-glibc2.5-x86_64/log/mysqld.log'.
	2016-03-23T07:44:29.814254Z mysqld_safe Starting mysqld daemon with databases from /ssd/mysql/data

data01的keepalived马上检测到这种情况：

	Mar 23 15:44:27 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:44:29 hkg02-mj-data01 root: MySQL is NOT running properly. Setting keepalived state to FAULT.
	Mar 23 15:44:35 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Script(check_mysql) succeeded
	Mar 23 15:44:35 hkg02-mj-data01 Keepalived_vrrp[55233]: VRRP_Instance(VI_MYSQL) Entering BACKUP STATE

可以看到，由于我们配置了非抢占模式，所以master恢复之后也还是BACKUP状态了。这正是我们想要的结果。`select @@server_id`仍然返回2：

	[work@hkg02-mj-web04.hkg02.arganzheng.me work]$ mysql -h10.244.81.8 -uxxx -pxxxx --skip-column-names -A -e "select @@server_id"
	+------+
	| 2    |
	+------+

注意：这里backup master必须设置为可写，否则还需要做额外的操作把read-only设置为false。

我们这里并没有引入MHA自动调整拓扑结构，所以还需要一些人工操作，把主从关系调整一下。


### 实战3: 负载均衡

Keepalived的loadbalancing是通过配置一个virtual_server来实现的，跟HAProxy的配置挺像的。负载均衡的策略是通过LVS实现的，具体参见: [The Keepalived Solution](http://www.linuxvirtualserver.org/docs/ha/keepalived.html)。

下面是一个简单的例子：

	virtual_server 10.244.81.8 3306 { # 对外服务的VIP和端口
		delay_loop 2 				# 每隔2秒检查一次real_server状态  
	  	lb_algo wrr 				# LVS算法
	  	lb_kind DR 				# LVS算法
	  	persistence_timeout 60    # 会话保持时间
	  	protocol TCP

	  	real_server 10.244.81.24 3306 { # backup server1
	  		weight 3  
	  		# notify_down /home/work/userbin/keepalived/mysql_down.sh  # 检测到服务down后执行的脚本
	    	TCP_CHECK {
	    	  	connect_timeout 10		# 连接超时时间  
	    	  	nb_get_retry 3       	# 重连次数  
	    	  	delay_before_retry 3    # 重连间隔时间  
	    	  	connect_port 3306   	# 健康检查端口
	    	}
	  	}

	  	real_server 10.244.81.25 3306 { # backup server2
	    	weight 3  
	  		# notify_down /home/work/userbin/keepalived/mysql_down.sh  # 检测到服务down后执行的脚本
	    	TCP_CHECK {
	    	  	connect_timeout 10		# 连接超时时间  
	    	  	nb_get_retry 3       	# 重连次数  
	    	  	delay_before_retry 3    # 重连间隔时间  
	    	  	connect_port 3306   	# 健康检查端口
	    	}
	  	}
	}


当然，同时提供服务的情况，比较适合MySQL的Slave集群提供读服务，不适合多个master同时提供写服务，因为数据容易冲突。

需要在两台LB机器上都开启允许ip地址转发：

	# vim /etc/sysctl.conf

添加：

	net.ipv4.ip_forward=1

让配置生效：

	# sysctl -p


总结
----

不像HAProxy等LB，本身引入了单点问题，keepalived天然就是分布式的，通过组播或者单播进行集群通讯，通过VRRP+VIP实现HA，通过LVS+VIP实现LB，整个配置非常简单，切换非常快速。可能唯一的缺点就是对运行环境有要求，需要root权限，而且必须在同一个网段。


### 补充：跨网段使用Keealived 

使用Keepalived做VIP failover要求VIP和后端RIPs必须在同一个网段，比如我们上面的例子：

	VIP: 10.244.81.8/25
	MASTER: 10.244.81.25/25
	BACKUP: 10.244.81.24/25

VIP和RIPs都是在同一个子网中。

但是有时候MASTER和BACKUP可能就不在一个网段，比如出于容灾的需要，MASTER和BACKUP位于不同的机架，这时候他们就不会处于同一个子网中。那么这种情况下，怎样使用Keepalived做VIP failover呢？

首先，使用VRRP默认的组播方式肯定是不行的，这样是通知不到对方，会导致MASTER和BACKUP都认为自己是master，导致两边都抢占了VIP，有点类似于闹裂现象。

改成单播可以解决这个问题：

MASTER:

	vrrp_instance VI_MYSQL {  

		interface xgbe0			# Specify the network interface to which the virtual address is assigned

		state BACKUP            # both backup to make nopreempt works

		priority 101           	# 101 on master, 100 on backup

		nopreempt               # Don't fail back

		mcast_src_ip 10.244.81.25
		unicast_peer {
			10.242.110.46
		}

		virtual_router_id 51 	# Needs to be the same on both instances, 
								# and needs to be unique if using multicast, does not matter with unicast

		authentication {
	       	auth_type PASS
	        auth_pass 10086
	    }

		virtual_ipaddress {  	# The VIP that keepalived will monitor
			10.244.81.8/25  dev xgbe0
		}

		track_script {
			check_mysql
		}
	}

BACKUP:

	vrrp_instance VI_MYSQL {  

		interface xgbe0			# Specify the network interface to which the virtual address is assigned

		state BACKUP            # both backup to make nopreempt works

		priority 100           	# 101 on master, 100 on backup

		nopreempt               # Don't fail back

		mcast_src_ip 10.242.110.46
		unicast_peer {
			10.244.81.25
		}

		virtual_router_id 51 	# Needs to be the same on both instances, 
								# and needs to be unique if using multicast, does not matter with unicast

		authentication {
	       	auth_type PASS
	        auth_pass 10086
	    }

		virtual_ipaddress {  	# The VIP that keepalived will monitor
			10.244.81.8/25  dev xgbe0
		}

		track_script {
			check_mysql
		}
	}


实验证明单播确实可以实现VIP的漂移。但是关键在于VIP只能跟其中一个RS在同个网段，上面的例子中VIP就跟MASTER在同一个网段，但是跟BACKUP就不是同一个网段。结果就是导致如果BACKUP抢占到了VIP，但是这个VIP对外是无法访问的！也就是ping不通。网上说是要修改路由表之类的，反正会很麻烦：[节省公网ip keepalived 另类用法](http://dngood.blog.51cto.com/446195/870821)。

但是呢？公司的BGW是支持跨网段负载均衡的，所以理论上结合LVS肯定是可以做到的，具体有待后面研究了。


参考文章
-------

1. [How To Set Up Highly Available Web Servers with Keepalived and Floating IPs on Ubuntu 14.04](https://www.digitalocean.com/community/tutorials/how-to-set-up-highly-available-web-servers-with-keepalived-and-floating-ips-on-ubuntu-14-04)
2. [Keepalived using unicast, track and notify scripts](http://kaivanov.blogspot.com/2015/02/keepalived-using-unicast-track-and.html) 
3. [17.6 Configuring Simple Virtual IP Address Failover Using Keepalived](https://docs.oracle.com/cd/E37670_01/E41138/html/section_uxg_lzh_nr.html)
4. [Kamailio High Availability Done Right with Keepalived](http://blog.unicsolution.com/2015/01/kamailio-high-availability-with.html)
5. [Deploying Highly Available Virtual Interfaces With Keepalived](http://prefetch.net/articles/linuxkeepalivedvrrp.html) 非常通俗易懂的一篇介绍文章。
6. [Linux Virtual Server Tutorial](http://www.ultramonkey.org/papers/lvs_tutorial/html/)
