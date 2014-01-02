---
title: 如何下载文件
layout: post
---

    // File file = xxx
    InputStream inStream = new FileInputStream(file);
    try {
        String fileName = file.getName();
        response.setHeader("Content-Disposition", "attachment; filename=" + fileName); // HttpServletResponse
        response.addHeader("Content-Length", "" + inStream.available());
        response.setContentType("application/octet-stream; charset=UTF-8");
        // 后面再来考虑支持断点续传。
        IOUtils.copy(inStream, response.getOutputStream());
    } finally {
        IOUtils.closeQuietly(inStream);
    }
    
