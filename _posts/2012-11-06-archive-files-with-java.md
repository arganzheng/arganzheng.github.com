---
title: 使用Java程序打jar/war包
layout: post
---

JDK自带的jar util库。于是可以这么简单封装一下：

	java.util.jar.JarOutputStream;
	java.util.zip.ZipOutputStream;


	package me.arganzheng.study.archive.util;

	import java.io.BufferedInputStream;
	import java.io.File;
	import java.io.FileInputStream;
	import java.io.FileNotFoundException;
	import java.io.FileOutputStream;
	import java.io.IOException;
	import java.util.jar.Attributes;
	import java.util.jar.JarEntry;
	import java.util.jar.JarOutputStream;
	import java.util.jar.Manifest;
	import java.util.zip.Deflater;
	import java.util.zip.ZipEntry;
	import java.util.zip.ZipOutputStream;

	import org.apache.commons.io.FilenameUtils;
	import org.apache.commons.lang.StringUtils;

	/**
	* ArchiveUtils类，用于创建、解压Jar/Zip/War/..文件
	*
	* @author arganzheng
	* @date 2012-10-25
	*/
	public class ArchiveUtils {

	    public static int       BUFFER_SIZE       = 10240;
	    /** Convert millis to seconds */
	    public static final int MILLIS_PER_SECOND = 1000;

	    public static class FileSet {

	        private File   dir;

	        private String regex;
	        private String replacement;

	        public File getDir() {
	            return dir;
	        }

	        public void setDir(File dir) {
	            this.dir = dir;
	        }

	        public String getRegex() {
	            return regex;
	        }

	        public void setRegex(String regex) {
	            this.regex = regex;
	        }

	        public String getReplacement() {
	            return replacement;
	        }

	        public void setReplacement(String replacement) {
	            this.replacement = replacement;
	        }

	        public FileSet(File dir, String regex, String replacement){
	            this.dir = dir;
	            this.regex = regex;
	            this.replacement = replacement;
	        }

	        public FileSet(String dir, String regex, String replacement){
	            this.dir = new File(dir);
	            this.regex = regex;
	            this.replacement = replacement;
	        }

	        public FileSet(String dir){
	            this.dir = new File(dir);
	        }

	        public FileSet(File dir){
	            this.dir = dir;
	        }

	        public FileSet(){
	        }
	    }

	    public static class JarUtils {

	        /**
	         * <pre>
	         * 创建jar文件:
	         *
	         * The 3 types of input files for the jar tool are
	         * 1. destination jar file
	         * 2. manifest file (optional) ，如果没有指定，那么会默认创建一个and is always the first entry in the jar file.
	         *  By default, it is named META-INF/MANIFEST.MF. The manifest file is the place where any meta-information about the archive is stored
	         * 3. files to be archived
	         *
	         * Typical usage is
	         *         % jar cf myjarfile *.class
	         *        
	         * If you have a pre-existing manifest file that you want the jar tool to use for the new jar archive, you can specify it using the -m option:
	         *         % jar cmf myManifestFile myJarFile *.class
	         * </pre>
	         *
	         * @param jarFile
	         * @param filesToJar
	         * @throws IOException
	         * @throws FileNotFoundException
	         */
	        public static void createJar(File jarFile, File manifestFile, FileSet[] filesToJar) throws IOException {
	            // if not specifiy the Manifest，generate it.
	            Manifest manifest = null;
	            if (manifestFile == null || !manifestFile.exists()) {
	                manifest = createManifest();
	            } else {
	                manifest = new Manifest(new FileInputStream(manifestFile));
	            }

	            if (filesToJar == null) {
	                return;
	            }

	            // open the archive file for writing
	            JarOutputStream target = new JarOutputStream(new FileOutputStream(jarFile), manifest);
	            // now add the files to be jar
	            for (FileSet fileset : filesToJar) {
	                addFileToJar(fileset, target);
	            }
	            target.close();
	        }

	        public static void createJar(File archiveFile, FileSet[] filesToJar) throws IOException {
	            createJar(archiveFile, null, filesToJar);
	        }

	        private static Manifest createManifest() {
	            Manifest manifest = new Manifest();

	            manifest.getMainAttributes().put(Attributes.Name.MANIFEST_VERSION, "1.0");

	            return manifest;
	        }

	        private static JarEntry createJarEntry(FileSet fileset) {
	            String name = getEntryName(fileset, false);
	            if (StringUtils.isBlank(name)) {
	                return null;
	            }

	            JarEntry jarEntry = new JarEntry(name);

	            long modTime = fileset.getDir().lastModified() / MILLIS_PER_SECOND;

	            jarEntry.setTime(modTime);

	            return jarEntry;
	        }

	        private static void addFileToJar(FileSet source, JarOutputStream target) throws IOException {
	            File dir = source.getDir();
	            if (dir == null || !dir.exists()) {
	                return;
	            }

	            BufferedInputStream in = null;
	            try {
	                if (dir.isDirectory()) {
	                    for (File nestedFile : dir.listFiles()) {
	                        FileSet nestedFileSet = new FileSet(nestedFile, source.getRegex(), source.getReplacement());
	                        addFileToJar(nestedFileSet, target);
	                    }
	                    return;
	                }

	                JarEntry entry = createJarEntry(source);
	                if (entry == null) {
	                    return;
	                }
	                target.putNextEntry(entry);

	                in = new BufferedInputStream(new FileInputStream(dir));
	                byte[] buffer = new byte[BUFFER_SIZE];
	                while (true) {
	                    int count = in.read(buffer);
	                    if (count == -1) {
	                        break;
	                    }
	                    target.write(buffer, 0, count);
	                }
	                target.closeEntry();
	            } finally {
	                if (in != null) in.close();
	            }
	        }
	    }

	    public static class ZipUtils {

	        public static void createZip(File zipFile, FileSet[] filesToZip) throws IOException {

	            if (filesToZip == null) {
	                return;
	            }

	            // create the zip file for writing
	            ZipOutputStream target = new ZipOutputStream(new FileOutputStream(zipFile));
	            target.setLevel(Deflater.DEFAULT_COMPRESSION);

	            // Compress the files
	            for (FileSet fileset : filesToZip) {
	                addFileToZip(fileset, target);
	            }

	            target.close();
	        }

	        private static void addFileToZip(FileSet source, ZipOutputStream target) throws IOException {
	            File dir = source.getDir();

	            if (dir == null || !dir.exists()) {
	                return;
	            }

	            BufferedInputStream in = null;
	            try {
	                if (dir.isDirectory()) {
	                    for (File nestedFile : dir.listFiles()) {
	                        FileSet nestedFileset = new FileSet(nestedFile, source.getRegex(), source.getReplacement());
	                        addFileToZip(nestedFileset, target);
	                    }
	                    return;
	                }

	                ZipEntry entry = createZipEntry(source);
	                if (entry == null) {
	                    return;
	                }
	                target.putNextEntry(entry);

	                in = new BufferedInputStream(new FileInputStream(dir));
	                byte[] buffer = new byte[BUFFER_SIZE];
	                while (true) {
	                    int count = in.read(buffer);
	                    if (count == -1) {
	                        break;
	                    }
	                    target.write(buffer, 0, count);
	                }
	                target.closeEntry();
	            } finally {
	                if (in != null) in.close();
	            }
	        }

	        private static ZipEntry createZipEntry(FileSet fileset) {
	            String name = getEntryName(fileset, false);
	            if (StringUtils.isBlank(name)) {
	                return null;
	            }
	            ZipEntry zipEntry = new ZipEntry(name);

	            long modTime = fileset.getDir().lastModified() / MILLIS_PER_SECOND;

	            zipEntry.setTime(modTime);

	            return zipEntry;
	        }
	    }

	    static String getEntryName(FileSet fileset, boolean preserveLeadingSlashes) {
	        File source = fileset.getDir();
	        String normalizeName = source.getPath();

	        if (fileset.getRegex() != null && fileset.getReplacement() != null) {
	            String regex = fileset.getRegex();
	            normalizeName = normalizeName.replace(regex, fileset.getReplacement());
	        }

	        normalizeName = FilenameUtils.normalizeNoEndSeparator(normalizeName);
	        while (!preserveLeadingSlashes && (normalizeName.startsWith("/") || normalizeName.startsWith("\\"))) {
	            normalizeName = normalizeName.substring(1);
	        }

	        return FilenameUtils.separatorsToUnix(normalizeName);
	    }
	}


但是发现有个问题，就是如果打包的文件名带有中文名，那么解压出来是乱码。[Non-UTF-8 encoding in ZIP file](http://t.cn/zlgJtST)
然后JDK7.0之后fix掉这个“bug”了，支持encoding： [Working with Zip Files](http://t.cn/zlgJtSY) 不错的文章。

但是JDK升级可不是我能决定的，于是换了一个支持encoding的库：apache的commons-compress库。但是发现有个小小的缺点，居然不支持manifest，需要自己手动构造和写入。。狂汗。。


	package me.arganzheng.study.archive.util;

	import java.io.BufferedInputStream;
	import java.io.BufferedOutputStream;
	import java.io.File;
	import java.io.FileInputStream;
	import java.io.FileNotFoundException;
	import java.io.FileOutputStream;
	import java.io.IOException;
	import java.util.jar.Attributes;
	import java.util.jar.JarFile;
	import java.util.jar.Manifest;
	import java.util.zip.Deflater;

	import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
	import org.apache.commons.compress.archivers.jar.JarArchiveOutputStream;
	import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
	import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
	import org.apache.commons.io.FilenameUtils;
	import org.apache.commons.io.IOUtils;
	import org.apache.commons.lang.StringUtils;

	/**
	* ArchiveUtils类，用于创建、解压Jar/Zip/War/..文件
	*
	* @author arganzheng
	* @date 2012-10-25
	*/
	public class ArchiveUtils {

	    public static int       BUFFER_SIZE       = 10240;
	    /** Convert millis to seconds */
	    public static final int MILLIS_PER_SECOND = 1000;

	    public static class FileSet {

	        private File   dir;

	        private String regex;
	        private String replacement;

	        public File getDir() {
	            return dir;
	        }

	        public void setDir(File dir) {
	            this.dir = dir;
	        }

	        public String getRegex() {
	            return regex;
	        }

	        public void setRegex(String regex) {
	            this.regex = regex;
	        }

	        public String getReplacement() {
	            return replacement;
	        }

	        public void setReplacement(String replacement) {
	            this.replacement = replacement;
	        }

	        public FileSet(File dir, String regex, String replacement){
	            this.dir = dir;
	            this.regex = regex;
	            this.replacement = replacement;
	        }

	        public FileSet(String dir, String regex, String replacement){
	            this.dir = new File(dir);
	            this.regex = regex;
	            this.replacement = replacement;
	        }

	        public FileSet(String dir){
	            this.dir = new File(dir);
	        }

	        public FileSet(File dir){
	            this.dir = dir;
	        }

	        public FileSet(){
	        }
	    }

	    public static class JarUtils {

	        public static void createJar(File archiveFile, FileSet[] filesToJar) throws IOException {
	            createJar(archiveFile, null, filesToJar);
	        }

	        /**
	         * <pre>
	         * 创建jar文件:
	         *
	         * The 3 types of input files for the jar tool are
	         * 1. destination jar file
	         * 2. manifest file (optional) ，如果没有指定，那么会默认创建一个and is always the first entry in the jar file.
	         *  By default, it is named META-INF/MANIFEST.MF. The manifest file is the place where any meta-information about the archive is stored
	         * 3. files to be archived
	         *
	         * Typical usage is
	         *         % jar cf myjarfile *.class
	         *        
	         * If you have a pre-existing manifest file that you want the jar tool to use for the new jar archive, you can specify it using the -m option:
	         *         % jar cmf myManifestFile myJarFile *.class
	         * </pre>
	         *
	         * @param jarFile
	         * @param filesToJar
	         * @throws IOException
	         * @throws FileNotFoundException
	         */
	        public static void createJar(File jarFile, File manifestFile, FileSet[] filesToJar) throws IOException {
	            // if not specifiy the Manifest，generate it.
	            Manifest manifest = null;
	            if (manifestFile == null || !manifestFile.exists()) {
	                manifest = createManifest();
	            } else {
	                manifest = new Manifest(new FileInputStream(manifestFile));
	            }

	            if (filesToJar == null) {
	                return;
	            }

	            // open the archive file for writing
	            JarArchiveOutputStream target = new JarArchiveOutputStream(new FileOutputStream(jarFile));

	            addManifestToJar(manifest, target);

	            // now add the files to be jar
	            for (FileSet fileset : filesToJar) {
	                addFileToJar(fileset, target);
	            }
	            target.close();
	        }

	        private static void addManifestToJar(Manifest manifest, JarArchiveOutputStream target) throws IOException {
	            if (manifest == null) {
	                throw new NullPointerException("manifest");
	            }

	            JarArchiveEntry e = new JarArchiveEntry(JarFile.MANIFEST_NAME);
	            target.putArchiveEntry(e);
	            manifest.write(new BufferedOutputStream(target));
	            target.closeArchiveEntry();
	        }

	        private static Manifest createManifest() {
	            Manifest manifest = new Manifest();

	            manifest.getMainAttributes().put(Attributes.Name.MANIFEST_VERSION, "1.0");

	            return manifest;
	        }

	        private static JarArchiveEntry createArchiveJarEntry(FileSet fileset) {
	            String name = getEntryName(fileset, false);
	            if (StringUtils.isBlank(name)) {
	                return null;
	            }

	            JarArchiveEntry jarEntry = new JarArchiveEntry(name);

	            long modTime = fileset.getDir().lastModified() / MILLIS_PER_SECOND;

	            jarEntry.setTime(modTime);

	            return jarEntry;
	        }

	        private static void addFileToJar(FileSet source, JarArchiveOutputStream target) throws IOException {
	            File dir = source.getDir();
	            if (dir == null || !dir.exists()) {
	                return;
	            }

	            BufferedInputStream in = null;
	            try {
	                if (dir.isDirectory()) {
	                    for (File nestedFile : dir.listFiles()) {
	                        FileSet nestedFileSet = new FileSet(nestedFile, source.getRegex(), source.getReplacement());
	                        addFileToJar(nestedFileSet, target);
	                    }
	                    return;
	                }

	                // not directory, dojar
	                JarArchiveEntry entry = createArchiveJarEntry(source);
	                if (entry == null) {
	                    return;
	                }
	                target.putArchiveEntry(entry);

	                in = new BufferedInputStream(new FileInputStream(dir));
	                IOUtils.copy(in, target);
	                target.closeArchiveEntry();
	            } finally {
	                IOUtils.closeQuietly(in);
	            }
	        }
	    }

	    public static class ZipUtils {

	        public static void createZip(File zipFile, FileSet[] filesToZip) throws IOException {

	            if (filesToZip == null) {
	                return;
	            }

	            // create the zip file for writing
	            ZipArchiveOutputStream target = new ZipArchiveOutputStream(new FileOutputStream(zipFile));
	            target.setLevel(Deflater.DEFAULT_COMPRESSION);

	            // Compress the files
	            for (FileSet fileset : filesToZip) {
	                addFileToZip(fileset, target);
	            }

	            target.close();
	        }

	        private static void addFileToZip(FileSet source, ZipArchiveOutputStream target) throws IOException {
	            File dir = source.getDir();

	            if (dir == null || !dir.exists()) {
	                return;
	            }

	            BufferedInputStream in = null;
	            try {
	                if (dir.isDirectory()) {
	                    for (File nestedFile : dir.listFiles()) {
	                        FileSet nestedFileset = new FileSet(nestedFile, source.getRegex(), source.getReplacement());
	                        addFileToZip(nestedFileset, target);
	                    }
	                    return;
	                }

	                ZipArchiveEntry entry = createZipArchiveEntry(source);
	                if (entry == null) {
	                    return;
	                }
	                target.putArchiveEntry(entry);
	                in = new BufferedInputStream(new FileInputStream(dir));
	                IOUtils.copy(in, target);
	                target.closeArchiveEntry();
	            } finally {
	                IOUtils.closeQuietly(in);
	            }
	        }

	        private static ZipArchiveEntry createZipArchiveEntry(FileSet fileset) {
	            String name = getEntryName(fileset, false);
	            if (StringUtils.isBlank(name)) {
	                return null;
	            }
	            ZipArchiveEntry zipEntry = new ZipArchiveEntry(name);

	            long modTime = fileset.getDir().lastModified() / MILLIS_PER_SECOND;

	            zipEntry.setTime(modTime);

	            return zipEntry;
	        }
	    }

	    static String getEntryName(FileSet fileset, boolean preserveLeadingSlashes) {
	        File source = fileset.getDir();
	        String normalizeName = source.getPath();

	        if (fileset.getRegex() != null && fileset.getReplacement() != null) {
	            String regex = fileset.getRegex();
	            normalizeName = normalizeName.replace(regex, fileset.getReplacement());
	        }

	        normalizeName = FilenameUtils.normalizeNoEndSeparator(normalizeName);
	        while (!preserveLeadingSlashes && (normalizeName.startsWith("/") || normalizeName.startsWith("\\"))) {
	            normalizeName = normalizeName.substring(1);
	        }

	        return FilenameUtils.separatorsToUnix(normalizeName);
	    }
	}



