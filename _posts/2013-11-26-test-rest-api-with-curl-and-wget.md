---
layout: post
title: 使用curl和wget模拟REST请求
---


模拟HTTP请求一般使用curl，例如有这么一个REST接口 [会员卡推送接口文档](http://open.weigou.qq.com/doc/wecard-push-api)

> #### 1. 绑卡

>  开放平台POST消息格式:
>
    {
        "topic": "wecard",
        "event": "bindCard",
        "uin": 商家的uin,
        "data": {
            "openId": "用户的唯一标识",
            "mobile": "手机号",
            "cardNo": "卡号"
        }
    }
>    
>应用应该返回的消息格式:
>
HTTP Status Code:200
>
    {
        "errorCode": 0, // 必填。错误码(10000以上(不含)，0为正常)
        "errorMessage": "", // 必填。错误信息
        "data": { // 必填。
            "cardId": "123-abc", // 必填。卡的唯一id，可以和卡号一样
            "cardNo": "123-456", // 必填。卡号
            "expiryDate": "2013-10-31 20:18:20", // 卡有效期 格式yyyy-MM-dd HH-mm-ss
            "cardLevelId": "1", // 必填。卡等级id(数值)
            "points": "1234", //  必填。积分
            "name": "张三",　//　持卡人姓名
            "mobile": "13700000000", // 持卡人手机号
            "birthday": "2000-12-12", // 持卡人生日，格式yyyy-MM-dd
            "sex": "M ", // 持卡人性别，M（男）或F（女）
            "cardExtension": { // 扩展信息，key、value形式
                "store": "常去门店名称"
            }
        }
    }

使用curl可以测试：

    arganzheng-mbp:~ argan$ curl --header "Content-Type: application/json;charset=utf-8" -d '{"topic":"wecard","event":"bindCard","uin":795019782,"data":{"openId":"123456","mobile":"138000000","cardNo":"卡号"}}' "http://ec2.arganzheng.me/sample/v1/wecard?signature=1e10b9e5e9fe023d7ae79d38e847b99223613a52&nonce=nonce&timestamp=timestamp" -o bindCardResp.json
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   472    0   353  100   119    674    227 --:--:-- --:--:-- --:--:--  1597
    arganzheng-mbp:~ argan$ cat bindCardResp.json 
    {
      "errorCode" : 0,
      "errorMessage" : "",
      "data" : {
        "openId" : "123456",
        "mobile" : "138000000",
        "cardId" : "卡号",
        "cardNo" : "卡号",
        "expiryDate" : "2013-10-31 20:18:20",
        "cardLevelId" : 1,
        "points" : 100,
        "name" : "arganzheng",
        "birthday" : "1985-11-16",
        "sex" : null,
        "cardExtension" : null
      }
    }

也可以使用wget进行测试，语法差不多：
    
    arganzheng-mbp:~ argan$ wget --header="Content-Type: application/json;charset=utf-8" --post-data='{"topic":"wecard","event":"bindCard","uin":795019782,"data":{"openId":"123456","mobile":"138000000","cardNo":"卡号"}}' "http://ec2.arganzheng.me/sample/v1/wecard?signature=1e10b9e5e9fe023d7ae79d38e847b99223613a52&nonce=nonce&timestamp=timestamp" -O bindCardResp.json
    --2013-11-26 14:45:35--  http://ec2.arganzheng.me/sample/v1/wecard?signature=1e10b9e5e9fe023d7ae79d38e847b99223613a52&nonce=nonce&timestamp=timestamp
    Resolving ec2.arganzheng.me... 54.201.85.167
    Connecting to ec2.arganzheng.me|54.201.85.167|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [application/json]
    Saving to: ‘bindCardResp.json’

        [ <=>                                                                         ] 353         --.-K/s   in 0s      

    2013-11-26 14:45:36 (25.9 MB/s) - ‘bindCardResp.json’ saved [353]

    arganzheng-mbp:~ argan$ cat bindCardResp.json 
    {
      "errorCode" : 0,
      "errorMessage" : "",
      "data" : {
        "openId" : "123456",
        "mobile" : "138000000",
        "cardId" : "卡号",
        "cardNo" : "卡号",
        "expiryDate" : "2013-10-31 20:18:20",
        "cardLevelId" : 1,
        "points" : 100,
        "name" : "arganzheng",
        "birthday" : "1985-11-16",
        "sex" : null,
        "cardExtension" : null
      }
    } 