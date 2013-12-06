---
layout: post
title: 用Photoshop磨皮
---


需求：色斑、痘痘如何PS掉呢？

思路：斑点是由于色调和明度相对于光滑部分的皮肤而言比较暗淡，只要将他们的色调和明度调高就可以了。

关键：如何选择色斑。

方法：通道与选区的关系：黑不选中，白选中。


实战
----

![Before-PS](/media/images/Before-PS.jpg)

步骤一：在通道中选择色斑比较明显的通道，如这里的green通道，复制该通道，得到green copy通道。

步骤二：对该通道进行更进一步的对比度强化。

1. 使用FilterOtherHigh Pass(保留高反差)
2. 使用ImageCalculations，在弹出菜单中将Blending（混合模式）选择为Hard Light（强光），这样以后得到一个Alpha1通道。
3. 重复多次Calculations命令，知道色斑已经凸出的很明显为止。(还可以再调整色阶或曲线提高通道中的对比度，不过这里不需要)


步骤三：将通道转化为选区，由于色斑是黑色的，所以还要Invert选区。

步骤四：调整色斑的HSB值，通过色阶或曲线都可以。这里用曲线（图像==>调整==>曲线）更容易。边调整边预览。

**TIP** 选区还在预览效果不好，可以把鼠标点击Photoshop软件窗口之外的其他地方去除选区预览，这样焦点回去的时候选区还在。）

效果图：

![After-PS](/media/images/After-PS.jpg)

