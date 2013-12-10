---
layout: post
title: 一个简单的性能优化和防止DOS攻击的示例
---


需求
----

提供一个根据商品ID(itemID)返回质检报告图片URL的链接。

问题与相应的解决方案
--------------------

问题1. 根据商品ID获取质检报告图片URL需要查询DB
解决方案： 因为质检报告基本就是不会更新，另外，实时性也不是很高。所以可以考虑用Cache。本地化Cache或者Memcachd都可以。

问题2. 恶意用户使用随机商品ID拼凑URL进行调用攻击
解决方案：对于错误的商品ID，往Cache存入null值，下次直接返回，避免走DB。不过恶意用户可能可以拉取到真正的商品ID，所以中级解决方案要做一个调用鉴权（比如签名），不过这样对接入用户是一个伤害，所以这里只考虑错误商品ID的处理情况。

    
    /**
     * 提供查询QT报告图片URL给商品组，不需要验证。
     * 
     * @author arganzheng
     * @date 2013-12-9
     */
    @Controller
    @RequestMapping("/ui/qt")
    public class QtReportFrontController {

        private static final Logger logger                  = Logger.getLogger(QtReportFrontController.class);

        private static final String KEY_PREFIX              = "QT_REPORT_URL";

        // 24小时
        private static int          ExpireInterval          = 24 * 3600 * 1000;
        // 反正DOS攻击
        private static int          Anti_Dos_ExpireInterval = 1 * 3600 * 1000;

        @Autowired
        private QtReportService     qtService;

        @Autowired
        private CacheServiceHelper  cacheServiceHelper;

        @RequestMapping(value = "/{itemId}", method = GET)
        public String qtReportDetail(@PathVariable
        String itemId) {
            String qtReportUrlKey = KEY_PREFIX + "_" + itemId;

            if (cacheServiceHelper.exist(qtReportUrlKey)) {
                String reportUrl = (String) cacheServiceHelper.get(qtReportUrlKey);
                return "redirect:" + reportUrl;
            } else {
                List<QtReport> reports = qtService.getQtReportByItemId(itemId);
                if (CollectionUtils.isEmpty(reports)) {
                    logger.error("Can not find Qt Report for ItemId: " + itemId);
                    cacheServiceHelper.put(qtReportUrlKey, null, Anti_Dos_ExpireInterval);
                    return "appstore/error";
                }
                // 重定向到reportUrl去
                String reportUrl = reports.get(0).getReportUrl();
                cacheServiceHelper.put(qtReportUrlKey, reportUrl, ExpireInterval);
                return "redirect:" + reportUrl;
            }
        }
    }

结论
----

做什么事情多想一步，才能成为优秀的程序员:)

