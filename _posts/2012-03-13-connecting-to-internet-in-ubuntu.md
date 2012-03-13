---
layout: post
title: dhcp client config in ubuntu
---

#### use PPPoEConf to connect to Internet

It's quite simple in ubuntu. Just simply type:
    $ sudo pppoeconf
then flow the instruction.

see [使用 PPPoEConf 連接 Internet](http://wiki.linux.org.hk/w/Connect_Internet_with_PPPoEConf) for a step-by-step tutorial.

#### config DHCP in ubuntu

in ubuntu it's all in the /etc/network/interfaces config file.

    $ sudo vi /etc/network/interfaces

*DHCP*

    auto eth0
    iface eth0 inet dhcp

*Static IP*

    auto eth0
    iface eth0 inet static
    address 192.168.1.100
    netmask 255.255.255.0
    network 192.168.1.0
    broadcast 192.168.1.255
    gateway 192.168.1.1

Then restart the network

    $ sudo /etc/init.d/networking restart 

see [使用 DHCP 連接網絡或 Internet](http://wiki.linux.org.hk/w/Connect_network_with_DHCP) for detail.

