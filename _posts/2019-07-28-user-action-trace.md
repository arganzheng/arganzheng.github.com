---
layout: post
title: 用户行为串联方案
tags: [埋点]
catalog: true
---


## 背景

目前商城的所有行为日志埋点上报都缺少一个唯一的用户行为ID。它的作用是串联用户行为的会话ID，用户行为包括曝光、点击、加购、下单、分享、点赞等，用户行为按时序依次产生，需要业务传递 traceId，标识点击、效果转换的动作是哪一次的曝光、点击产生的。

带来的问题就是用户行为日志没法很好的串联起来，正负样本生产和报表统计都会有偏差。


## 业界解决方案

### 1、阿里云推荐平台

##### 1) [行为表（USER_BEHAVIOR）](https://help.aliyun.com/document_detail/54476.html?spm=a2c4g.11186623.2.16.58fe46f0j3RERr#h2--user_behavior-)

> trace_id: 返回的推荐列表用于跟踪效果。如果对item_id 的行为不是来自推荐引导，则为NULL

##### 2) [全流程规范](https://help.aliyun.com/document_detail/30373.html?spm=a2c4g.11186623.6.593.763e374a8DLoX3)

> 日志埋点规范在内容方面要求客户记录交易行为和准备行为，并且在记录用户行为日志时区分开这两种行为。和通常的日志一样，日志埋点规范也要求记录行为发生的一些上下文参数，比如时间，用户，物品等信息。此外为了便于计算由推荐服务带来的转换率，日志埋点规范要求客户记录traceid。traceid将用户在推荐坑位的点击与最终成交关联起来，用于判断成交是否是由推荐服务带来的。traceid由RecEng在返回实时返回推荐结果时返回，细节详见日志埋点规范。

##### 3) [测试场景 - 指标设置](https://help.aliyun.com/document_detail/61028.html?spm=5176.11065259.1996646101.searchclickresult.50b13240v5gG3t#h2-1-1)

> 确保您在收集行为日志埋点时，带上了trace_id字段信息。trace_id用于打穿从曝光、点击直到后续各转化环节的链路，例如当曝光行为、点击行为的trace_id相同时，则认为该点击是由该次曝光带来。日志埋点规范请点击这里。


### 2、腾讯云推荐平台

##### 1) [API 使用说明 - 术语表](https://cloud.tencent.com/document/product/587/9824#.E6.9C.AF.E8.AF.AD.E8.A1.A8)

> TraceId：用户行为 ID，串联用户行为的会话 ID，用户行为包括曝光、点击、效果转换、点赞等，用户行为按时序依次产生，需要业务传递 traceId，标识点击、效果转换的动作是哪一次的曝光、点击产生的。

##### 2) [API 使用说明 - Action 上报](https://cloud.tencent.com/document/product/587/9824#action-.E4.B8.8A.E6.8A.A5)

> 上报某一用户在特定场景下的行为，用户的行为包括曝光、点击、转换、点赞等， 上报用户行为时，必须指定用户行为的会话 id。用户行为可以在客户端和服务端上报，建议在客户端上报，可控性更强些，遇到协议变更或者问题排查时，更容易处理。

> trace_id : 跟踪点击和曝光的自定义会话 ID，为了保证点击跟曝光是同一个用户对同一个 item 的操作行为；强烈建议每次曝光分配一个 trace_id。

##### 3) [常见问题 - 数据上报 API 字段有哪些？](https://cloud.tencent.com/document/product/587/9827#.E6.95.B0.E6.8D.AE.E4.B8.8A.E6.8A.A5-api-.E5.AD.97.E6.AE.B5.E6.9C.89.E5.93.AA.E4.BA.9B.EF.BC.9F)

> action 上报：表示哪个用户（uid），在哪个场景下（scene_id），对哪些 item（item_id），进行了点击/曝光等操作（action_type），即 uid，item_id，action_type，scene_id 这个四元组需要对应起来。
如果没有分配 scene_id 的点击或者成交等，scene_id 可以暂时不填写。但要注意的是，上报过来的场景 A 的点击和曝光数据，一定是场景 A 下产生的点击和曝光，不能是其他场景的点击和曝光。可以简单理解为“某个场景下的曝光和点击数据，要么点击和曝光一起上报，要么都不上报（包括成交、购买等）”。

> trace_id : 跟踪点击和曝光的自定义会话 ID，用来关联“曝光-点击” session 的，是由业务上报方生成的，一般是由 uid_时戳_ 随机数生成的一个 ID，为了保证点击跟曝光是同一个用户对同一个 item 的操作行为；强烈建议每次曝光分配一个 trace_id，如果点击来自这次曝光，赋之相同的 trace_id，这样就能识别 session 了。


### 3、百度智能推荐BRS

##### 1) [电商业务上传数据接口 - 2.1用户行为数据](https://cloud.baidu.com/doc/BRS/s/Yjwvxjpzt#21%E7%94%A8%E6%88%B7%E8%A1%8C%E4%B8%BA%E6%95%B0%E6%8D%AE%EF%BC%88action%EF%BC%89)

```json
{
    "common": {
        "logid": "18102949407735214587", //log全局唯一id，由客户自主生成，依据此标志去除重复请求数据，必填
        "timestamp": 1489471203000, //发送时间戳，毫秒,int，必填
        "ip": "xxx.xxx.xxx.xxx" //发送端的ip，必填
    },
    "data": {
        "he": // 报文头部
        {
            "o": // 操作系统类型 "Android" 和 "iOS"，
                "s": // 操作系统 API Level,string "21"
                "sv": // 操作系统版本 Release,string,"4.2"
                "a": // App版本 int,42
                "n": // App版本 string,"4.0.0"
                "d": // Device Id，MAC 地址 
                "dd": // Android IMEI号 string
                "ad": // iOS IDFA string
                "w": // 宽度,int， 分辨率像素
                "h": // 高度,int， 分辨率像素
                "k": // app key,string, 标示token，全局唯一，需申请， 必填
                "c": // 发行渠道, string, "AppStore"
                "lang": // 系统语言,string， 参考系统标准API[1]
                "op": // 运营商,string
                "m": // 终端型号,string,"Le X620"，Build.MODEL
                "ma": // 终端制造商，string,"LeMobile"，Build.MANUFACTURER
                "cl": // 基站定位,string
                "gl": // gps 定位,string "时间戳|经度_纬度"，经纬度为小数
                "wl": // wifi 定位, string ""
                "l": // 联网方式,"wifi","3G"，"4G"
                "t": // unix时间戳，毫秒,
                "z": // 时区, string， 参考系统标准API[2]
        },
        "ev": // 自定义事件统计
            [{
                "i": // id，事件Id，必填，使用event名称
                    "t": // timestamp，事件开始时间戳, 单位为毫秒
                    "d": // duration，事件持续时间
                    "ext": {
                        "uid": // 用户id,
                            "iid": // 物料id,
                            "src": //source， 用来区分物料item展示的位置信息，方便数据统计；例如：首页推荐、详情页推荐、用户中心推荐等
                            "man": //manual_edit， 用来区分当前事件操作对象是否是人工运营的结果；
                            "trace_id": //trace_id， 用来作为百度推荐结果的识别标记，用来关联推荐展现和点击行为；
                            "sample_name": // sample_name， 用于后端优化效果，来源于预测接口中返回的sample_name
                    }
            }, ...]
    }
}
```

其中事件字段中也是有 trace_id 字段：

> trace_id": // trace_id， 用来作为百度推荐结果的识别标记，用来关联推荐展现和点击行为；

##### 2) [电商业务上传数据接口 - 3.3 事件信息](https://cloud.baidu.com/doc/BRS/s/Yjwvxjpzt#32%E4%BA%8B%E4%BB%B6%E4%BF%A1%E6%81%AF)

> trace_id : 若不传无法在console上看统计报表中相关统计数据。
> trace_id，标示了属于百度推荐系统的推荐结果，用于关联从预测服务中获取的推荐结果，可以从response_body中获得。
> 备注：分为两种情况：
> 1. 客户内部可以生成用于关联每次推荐请求和结果的展现点击等行为的id，则将其透传即可，推荐服务会在返回结果中复用该id值并以trace_id的形式返回；
> 2. 客户内部不生成该id，则推荐服务会在内部生成该trace_id，并在返回结果中回传给客户端，以供客户端使用；
> 客户需要在内部有一套机制，保证对同一次请求获得的Item推荐结果后续在展现和点击行为匹配的一致性，并在用户行为数据中使用对应的固定trace_id进行返回；以便服务端进行推荐效果数据统计、小流量实验分层效果统计等。


### 4、vivo 商业化变现

了解到商业化那边的做法是每次请求广告推荐接口时，服务端会产生一个reqId，返回给客户端，类似于如下数据格式：

```json
{
    "code":"SUCCESS",
    "data":{
        "req_id":"10086007",
        "rec":[["i1","0.9"],["i2","0.8"]],
        "abtag":"p1"
    }
    "message":null
}
```

客户端拿到这个 reqId 之后，需要一路透传下去，确保每次曝光、点击、下载等日志上报都带上这个reqId。以曝光点击日志为例：

上面返回了推荐结果 reqId = 10086007，当用户点击了其中的一个 i1 的时候

1. 上报一个点击埋点日志：`userId=007, itemId=xxx, eventId='click', screen='guess-what-you-like', reqId=10086007, ...`
2. i1的详情页URL中会透传reqId过去，即：`https://shop.vivo.com.cn/product/10000611?skuId=100797&reqId=10086007`，这样可以确保在详情页的行为日志可以继续上报这个reqId。


## 难点

商业化变现用户的行为只有曝光、点击和下载，而且整个行为序列是连贯无中断的。reqId 方案很容易透传。但是电商场景用户行为会复杂很多表现在：

1、行为更加多样化：曝光、点击、加购、结算、支付、分享、收藏等。

2、行为具有不连续性：因为购物车的存在，用户的行为被分成了两个明显的链路：

1. 曝光、点击和加购
2. 结算、支付。

用户可能浏览了好几个商品之后加入到购物车之后再进入购物车去一起结算，这时候怎样把之前购物车的商品的reqId跟后面的结算支付流程串联起来呢？

而且电商场景用户还经常把购物车当做收藏夹使用，导致整个行为链路时间跨度非常的长。

这方面有一些解决方案，但是实施起来比较麻烦，对业务埋点要求和侵入性也比较高。一期可以先不考虑这个问题。


