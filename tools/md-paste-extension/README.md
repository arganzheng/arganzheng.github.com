# 粘贴博客 Markdown（上传图片）

这是给作者使用的 Chrome MV3 扩展：把博客 Markdown 中指向
`arganzheng.life` 的图片下载成文件，再按顺序粘贴到目标编辑器，让编辑器用自己的粘贴上传流程处理图片。

## 安装

1. 打开 `chrome://extensions`，开启「开发者模式」。
2. 点「加载已解压的扩展程序」。
3. 选择本目录 `tools/md-paste-extension/`。

## 使用

在博客文章的「导出」菜单点「复制 Markdown」，然后在公司编辑器的源码编辑区放置光标。右键选择「粘贴博客 Markdown（上传图片）」，或按 `Alt+Shift+V`；也可以点击扩展工具栏图标。上传期间会逐张显示进度，结束后报告已上传、未确认和保留外链的数量。

## 限制

- 只下载 `arganzheng.life` 上的 HTTP(S) 图片；其他主机的图片 Markdown 会原样保留。
- 引用式图片（例如 `![说明][image-id]`）不会转换。
- 图片是否上传成功取决于目标编辑器是否接受粘贴/拖放的图片文件，并由它自己的上传流程确认。
- 代码围栏和反引号代码中的图片不会转换。

本目录没有构建步骤或运行时依赖。Node 解析器测试在 `test/split.test.cjs`；Playwright 端到端 harness 在 `test/harness/`。
