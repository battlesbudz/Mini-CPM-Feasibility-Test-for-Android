package com.battlesbudz.moshitest;
import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
final class Io {
    static String read(Path path) throws IOException {return new String(Files.readAllBytes(path),StandardCharsets.UTF_8);}
    static void write(Path path,String value) throws IOException {Files.write(path,value.getBytes(StandardCharsets.UTF_8));}
    static byte[] bytes(InputStream in) throws IOException {ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);return out.toByteArray();}
}
