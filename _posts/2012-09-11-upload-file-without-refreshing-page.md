---
layout: post
title: 如何不刷新页面上传文件
---


### 场景：

一个表单页面：有很多字段，其中有一个是上传图片字段，如何提交这个表单页面。

### 解决方案

#### 方案1. iframe + hidden field

为了不刷新页面，一般来说要不就是对文件标签使用iframe和单独的form来提交。提交到iframe指定的action处理完成后，将文件上传路径回写到父页面的某个字段。需要在iframe中区分是否已经上传了文件。
如：http://apitest.buy.qq.com/apitools/imgUpload.xhtml?imgName=pic

    <html>
        <head>
            <meta name="description" content="腾讯电商" />
            <link rel="stylesheet" href="http://static.gtimg.com/css/open/open_api.css" />
            
            <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
            
            #if($title)
            <title>$title - API测试工具</title>
            #else
            <title>API测试工具</title>
            #end
        </head>
        <body>
            <form name="form1" id="form1" method="post"
              enctype="multipart/form-data" action="upload.xhtml" onsubmit="return upload();">
              <input type="file" name="file"     id="image"> 
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
                   var strtype = strFileName.substring(strFileName.length - 3,
                             strFileName.length);
                   strtype = strtype.toLowerCase();
                   if (strtype == "jpg" || strtype == "gif" || strtype == "bmp"
                             || strtype == "png") {
                        window.parent.document.getElementById("apiParam_pic").value = "";
                        return true;
                   } else {
                        alert("这种文件类型不允许上传！\r\n只允许上传这几种文件：jpg、gif、bmp、png\r\n请选择别的文件并重新上传。");
                        document.getElementById("image").focus();
                        return false;
                   }
            
              }
            
              #if($imgPath)
                  window.parent.document.getElementById("apiParam_pic").value ='$imgPath';
              #end
            
            </script>
        </body>
    </html>


然后controller里面接收，上传到临时目录，返回保存路径：

    @RequestMapping(value = "upload", method = RequestMethod.POST)
    public String uploadImage(FileItem file, Model model) {
        FileOutputStream fos = null;
        try {
            // 建立文件输出流
            String sp = imageSavePath;
            File uploadFilePath = new File(sp);
            // 如果该目录不存在,则创建之
            if (!uploadFilePath.exists()) {
                uploadFilePath.mkdirs();
            }

            String imageFileName = getImageFileName(file.getOriginalFilename());
            String imgPath = FilenameUtils.concat(sp, imageFileName);
            fos = new FileOutputStream(imgPath);
            IOUtils.write(file.getBytes(), fos);

            // 把上传图片路径放入model中，让页面可以获取下次作为参数上传。
            model.addAttribute("imgPath", imgPath);
        } catch (Exception e) {
            logger.error("upload image fail!", e);
        } finally {
            close(fos, file.getInputStream());
        }

        // set reture result to inputStream: url,
        return "apitools/imgUpload";
    }   


然后在要不刷新上传的页面用iframe引用upload页面：

    <iframe id="apirightframe" name="apirightframe" width="325px" height="28px" frameborder="0" scrolling="no" src="upload.xhtml?imgName=pic"></iframe>
    <input type="hidden" name="pic" id="apiParam_pic" value="test">

这样通过iframe上传后，该iframe会将parent window的apiParam_pic hidden field的value设置为上传路径。


#### 方案2.  ajax

另外一种不刷新的方法当然就是ajax了。简单的一个post请求就可以了，现在有很多的ajax框架，也可以直接使用。

#### 方案3. multipart与其他字段一起提交

前面两种做法其实都是将表单分为两步：1. 先上传图片(不刷新主页面); 2. 提交纯文本请求。这样编码就可以是简单的`Content-Type:application/x-www-form-urlencoded`。有没有可能文本和二进制内容一起上传呢？HTTP协议中的mime协议就是为这个制订的。
使用 `Content-Type: multipart/form-data; boundary=${anything you like}` 就可以上传多种格式的参数了。

关于`Content-Type:application/x-www-form-urlencoded`和`Content-Type: multipart/form-data; boundary=${anything you like}`的区别，在StackOverFlow中有比较简洁的解释：[What does enctype='multipart/form-data' mean?](http://stackoverflow.com/questions/4526273/what-does-enctype-multipart-form-data-mean)

>When you make a POST request, you have to encode the data that forms the body of the request in some way.
>HTML forms provide two methods of encoding. The default is application/x-www-form-urlencoded, which is more or less the same as a query string on the end of the URL. The other, multipart/form-data, is a more complicated encoding but one which allows entire files to be included in the data.
