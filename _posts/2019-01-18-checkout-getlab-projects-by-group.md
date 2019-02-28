---
title: gitlab如何checkout某个group中的所有项目
layout: post
tags: [gitlab, git]
catalog: true
---

gitlab 是一个非常棒的基于git的代码版本控制平台，跟 github 非常类似，但是有一个非常重要的功能是缺失的，就是搜索代码。这时候只能将所有的分支拉到本地，进行grep。为了方便管理，我们的项目都是按照 group，也就是 namespace 分开管理的，问题在于 gitlab 并没有在界面上直接提供下载整个 group 中的所有项目的功能，谷歌了一下，发现大家基本都是利用 Gitlab API 提供的查询所有项目的接口，然后封装一个 shell 脚本来实现这个功能。

比如 这个 [sync-projects](https://gist.github.com/JonasGroeger/1b5155e461036b557d0fb4b3307e1e75) 。但是写的稍微有点复杂，我做了简单的优化：

1. 使用 `"${BASE_PATH}api/v3/groups/$GROUP_NAME/projects` 替代 `${BASE_PATH}api/v3/projects` 接口
2. 移除对 jq 的依赖
3. 把 groupName 作为命令行参数传递，而不是写死在脚本中（private_token 出于安全考虑，保留原来通过环境变量传递的方式）

最终简化后的shell脚本如下：

```shell
#!/usr/bin/env bash

BASE_PATH="http://10.21.6.54:6060/"

if [ -z "$1" ]
  then
    echo "group name is required."
    exit 1;
fi

GROUP_NAME="$1"

# export GITLAB_PRIVATE_TOKEN=Rq6EnxsQydfxgUyH8v2J

if [ -z "$GITLAB_PRIVATE_TOKEN" ]; then
    echo "Please set the environment variable GITLAB_PRIVATE_TOKEN"
    echo "See ${BASE_PATH}profile/account"
    exit 1
fi

FIELD_NAME="ssh_url_to_repo"

echo "Cloning all git projects in group $GROUP_NAME";

REPO_SSH_URLS=$(curl -s "${BASE_PATH}api/v3/groups/$GROUP_NAME/projects?private_token=$GITLAB_PRIVATE_TOKEN&per_page=999" \
   | grep -o "\"$FIELD_NAME\":[^ ,]\+" | awk -F'"' '{print $4}' | grep $GROUP_NAME)

for REPO_SSH_URL in $REPO_SSH_URLS; do
    THEPATH=$(echo "$REPO_SSH_URL" | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')

    if [ ! -d "$THEPATH" ]; then
        echo "Cloning $THEPATH ( $REPO_SSH_URL )"
        git clone "$REPO_SSH_URL" --quiet &
    else
        echo "Pulling $THEPATH"
        (cd "$THEPATH" && git pull --quiet) &
    fi
done
```

使用命令如下：

```
export GITLAB_PRIVATE_TOKEN=Rq6EnxsQydfxgUyH8v2J
./checkout-gitlab-projects.sh ai-platform
```

