---
title: kubernetes初体验
layout: post
catalog: true
tags: [k8s, docker]
category: [技术]
---


### 一、本地安装部署Kubernetes集群

今天想本地搭建一下k8s环境体验一下。看了一下官方文档 [Local-machine Solutions](https://kubernetes.io/docs/setup/pick-right-solution/#local-machine-solutions) 提供了不少方式。其中大部分资料推荐的都是基于 [Minikube](https://kubernetes.io/docs/setup/minikube/) 方式搭建。但是这种方式有个蛋疼的地方，就是需要自己先安装一个虚拟机，像 VirtualBox，或者 VMWare Fusion，还要自己安装 kubectl，麻烦。于是试着按照 [Docker Desktop](https://www.docker.com/products/docker-desktop) 的方式安装，果然很方便。只要下载 Docker Desktop 的 dmg 安装文件，点击安装，一切都是自动化的。安装完成之后。在工具栏右上角会有一个鲸鱼的图标，那个就是 docker desktop。这时候命令行已经可以执行docker的相关命令了，但是默认并没有安装kubernetes环境，需要安装。点击 Preferences ，点击 Kubernetes，选择所有（Enable Kubernetes，Deploy Docker Stacks to Kubernetes by default，Show System Containters），点击 Apply。这样就开始安装和启动 一个 Single-node Kubernetes Cluster。


### 二、使用初体验

网上有大把的文章介绍，这里就不赘述。读者可以参考这篇文章，写的还不错：[Kubernetes快速入门实战](https://www.jianshu.com/p/b0ff456c5518)。


### 三、安装Kubernetes Dashboard

[k8s Dashboard](https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/) 默认并不安装，需要自己部署：

```shell
kubectl create -f https://raw.githubusercontent.com/kubernetes/dashboard/master/aio/deploy/recommended/kubernetes-dashboard.yaml
```

部署很简单，但是访问还需要费一些劲。首先通过`kubectl proxy`启动一个命令行代理。这样 k8s Dashboard 就可以在 `http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/` 这个URL被访问。

但是还有一步需要设置一下，就是访问账户和token。通过这篇文章设置一个简单的用户 [Creating sample user](https://github.com/kubernetes/dashboard/wiki/Creating-sample-user)。然后这才可以进入到 dashboard 页面。

