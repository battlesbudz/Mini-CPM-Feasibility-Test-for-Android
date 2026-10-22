package com.battlesbudz.minicpmv47;
import android.content.Context;
import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Locale;
// Hash-verified, resumable download of the MiniCPM-V 4.7 phone pack
// (the `-les` split: main gguf + mmproj + experts/ shards + manifest),
// following the same store pattern as the MiniCPM-o feasibility module.
// Files land in files/models/<repo path>, exactly the layout the native
// LES runtime expects (experts/ sibling of the main gguf).
final class ModelStore {
    interface Progress {void update(String text);}
    private final Context context;
    final File root;
    final JSONObject manifest;
    ModelStore(Context c)throws Exception{
        context=c;root=new File(c.getFilesDir(),"models");root.mkdirs();
        try(InputStream in=c.getAssets().open("models.json")){manifest=new JSONObject(new String(Io.bytes(in),java.nio.charset.StandardCharsets.UTF_8));}
    }
    boolean ready()throws Exception{
        File marker=new File(root,"verified.json");
        if(!marker.isFile()||!Io.read(marker.toPath()).equals(manifest.toString()))return false;
        JSONArray list=manifest.getJSONArray("files");
        for(int i=0;i<list.length();i++){JSONObject item=list.getJSONObject(i);if(new File(root,item.getString("path")).length()!=item.getLong("size"))return false;}
        return true;
    }
    void install(Progress progress)throws Exception{
        new File(root,"verified.json").delete();
        JSONArray list=manifest.getJSONArray("files");
        for(int i=0;i<list.length();i++){
            JSONObject item=list.getJSONObject(i);String path=item.getString("path");
            File target=new File(root,path);target.getParentFile().mkdirs();
            progress.update("Verifying "+path);
            if(target.isFile()&&verify(target,item))continue;
            File part=new File(target+".part");
            if(part.isFile()&&verify(part,item)){Files.move(part.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);continue;}
            long existing=part.isFile()?part.length():0;
            if(existing>item.getLong("size")){part.delete();existing=0;}
            if(root.getUsableSpace()<item.getLong("size")-existing+256L*1024*1024)throw new IOException("Not enough free storage to install "+path);
            progress.update("Installing "+path+" ("+(i+1)+" / "+list.length()+")");
            URL url=new URL("https://huggingface.co/"+manifest.getString("repository")+"/resolve/"+manifest.getString("revision")+"/"+path);
            HttpURLConnection connection=(HttpURLConnection)url.openConnection();connection.setConnectTimeout(30000);connection.setReadTimeout(30000);
            if(existing>0)connection.setRequestProperty("Range","bytes="+existing+"-");
            int code=connection.getResponseCode();
            if(code==200)existing=0;
            else if(code==206){String range=connection.getHeaderField("Content-Range");if(range==null||!range.startsWith("bytes "+existing+"-")){connection.disconnect();throw new IOException("Invalid resume response");}}
            else{connection.disconnect();throw new IOException("Download HTTP "+code+" for "+path);}
            InputStream source=connection.getInputStream();
            try(InputStream in=source;FileOutputStream out=new FileOutputStream(part,existing>0)){
                byte[] buffer=new byte[1024*1024];long total=existing,last=0;int n;
                while((n=in.read(buffer))!=-1){
                    if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("Installation stopped; download can resume");
                    total+=n;if(total>item.getLong("size"))throw new IOException("File exceeds manifest size");out.write(buffer,0,n);
                    if(total-last>32L*1024*1024){last=total;progress.update(String.format(Locale.US,"%s %.0f%%",path,100.0*total/item.getLong("size")));}
                }
                out.getFD().sync();
            }finally{connection.disconnect();}
            progress.update("Checking SHA-256 for "+path);
            if(!verify(part,item)){part.delete();throw new IOException("Checksum mismatch: "+path);}
            Files.move(part.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);
        }
        Io.write(new File(root,"verified.json").toPath(),manifest.toString());
        progress.update("Complete MiniCPM-V 4.7 pack verified. Ready for offline testing.");
    }
    private static boolean verify(File file,JSONObject item)throws Exception{
        if(file.length()!=item.getLong("size"))return false;
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream in=new FileInputStream(file)){byte[] b=new byte[1024*1024];int n;while((n=in.read(b))!=-1){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();digest.update(b,0,n);}}
        StringBuilder hex=new StringBuilder();for(byte b:digest.digest())hex.append(String.format(Locale.ROOT,"%02x",b&255));
        return hex.toString().equals(item.getString("sha256"));
    }
}
