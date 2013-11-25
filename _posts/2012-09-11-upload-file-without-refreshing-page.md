---
layout: post
title: 如何不刷新页面上传文件
---


### 场景：

一个表单页面：有很多字段，其中有一个是上传图片字段，如何提交这个表单页面。

### 解决方案

方案1. 为了不刷新页面，一般来说要不就是对文件标签使用iframe和单独的form来提交。提交到iframe指定的action处理完成后，将文件上传路径回写到父页面的某个字段。需要在iframe中区分是否已经上传了文件。
如：http://apitest.buy.qq.com/apitools/imgUpload.xhtml?imgName=pic

    <html> 
     <head></head> 
     <body> 
     	<form name="form1" id="form1" method="post" enctype="multipart/form-data" action="uploadImg.xhtml" onsubmit="return upload();"> 
     		<input type="file" name="image" id="image"> 
     		<input type="submit" name="submit" value="上传"> 
     	</form>   
     	<script type="text/javascript"> 
     		var fileName = "1345429941610"; 
     		function upload() {
     			var strFileName = document.getElementById("image").value; 
     			if (strFileName == "") { 
     				alert("请选择要上传的文件"); 
     				return false; 
     			} 
     			var strtype = strFileName.substring(strFileName.length - 3, strFileName.length); 
     			strtype = strtype.toLowerCase(); 
     			if (strtype == "jpg" || strtype == "gif" || strtype == "bmp" || strtype == "png") { 
     				window.parent.document.getElementById("apiParam_pic").value = ""; return true; 
     			} else { alert("这种文件类型不允许上传！\r\n只允许上传这几种文件：jpg、gif、bmp、png\r\n请选择别的文件并重新上传。"); 
     			document.getElementById("image").focus();
     			return false; 
     			}   
     		} 
     	</script>
     </body> 
    </html>

方案2. 另外一种不刷新的方法当然就是ajax了。简单的一个post请求就可以了，现在有很多的ajax框架，也可以直接使用。

方案3. 前面两种做法其实都是将表单分为两步：1. 先上传图片(不刷新主页面); 2. 提交纯文本请求。这样编码就可以是简单的Content-Type:application/x-www-form-urlencoded。有没有可能文本和二进制内容一起上传呢？HTTP协议中的mime协议就是为这个制订的。
使用 Content-Type: multipart/form-data; boundary=${anything you like} 就可以上传多种格式的参数了。

关于Content-Type:application/x-www-form-urlencoded和Content-Type: multipart/form-data; boundary=${anything you like}的区别，
在StackOverFlow中有比较简洁的解释：http://stackoverflow.com/questions/4526273/what-does-enctype-multipart-form-data-mean

>>When you make a POST request, you have to encode the data that forms the body of the >>request in some way.
>>HTML forms provide two methods of encoding. The default is application/x-www-form->>urlencoded, which is more or less the same as a query string on the end of the URL. >>multipart/form-data is a more complicated encoding but one which allows entire files >>to be included in the data.