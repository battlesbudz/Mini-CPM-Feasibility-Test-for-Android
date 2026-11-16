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
// Build 15: one file (the DSpark draft GGUF) lives in a different
// Hugging Face repo than the rest of the pack. A manifest entry may
// carry per-file "repository" / "revision" / "remotePath" overrides;
// the local path, size and SHA-256 stay in the entry as usual, and
// entries without overrides resolve against the manifest-level
// repository/revision exactly as before.
final class ModelStore {
    interface Progress {void update(String text);}
    private final Context context;
    final File root;
    final JSONObject manifest;
    ModelStore(Context c)throws Exception{
        context=c;root=new File(c.getFilesDir(),"models");root.mkdirs();
        try(InputStream in=c.getAssets().open("models.json")){manifest=new JSONObject(new String(Io.bytes(in),java.nio.charset.StandardCharsets.UTF_8));}
    }
    // Build 29: stage-1 shadow pack. Create/verify all links before the manifest commit point,
    // because models.json and ready() key on that manifest; partial provisioning must leave
    // no manifest in place. The bins link to verified originals; the verified pack is untouched.
    static void provisionHobbitAllHot(Context c)throws Exception{
        File root=new File(c.getFilesDir(),"models");
        File shadow=new File(root,"MiniCPM-V-4.7-35B-A3B/experts-hobbit-allhot");
        if(!shadow.isDirectory()&&!shadow.mkdirs())throw new IOException("Cannot create HOBBIT shadow directory");
        File manifest=new File(shadow,"manifest.json");
        Files.deleteIfExists(manifest.toPath());
        File originalExperts=new File(root,"MiniCPM-V-4.7-35B-A3B/experts");
        for(int i=0;i<40;i++){
            String name=String.format(Locale.US,"layer_%03d.bin",i);
            File link=new File(shadow,name);
            File target=new File(originalExperts,name);
            String targetAbs=target.getAbsolutePath();
            if(Files.isSymbolicLink(link.toPath())&&Files.readSymbolicLink(link.toPath()).toString().equals(targetAbs))continue;
            if(Files.exists(link.toPath(),LinkOption.NOFOLLOW_LINKS))Files.delete(link.toPath());
            android.system.Os.symlink(targetAbs,link.getAbsolutePath());
        }
        if(!manifest.isFile()||manifest.length()!=73805){
            File temp=File.createTempFile("hobbit-manifest-",".tmp",shadow);
            try(InputStream in=c.getAssets().open("hobbit_allhot_manifest.json");FileOutputStream out=new FileOutputStream(temp)){
                byte[] buffer=new byte[8192];int n;while((n=in.read(buffer))!=-1)out.write(buffer,0,n);out.getFD().sync();
            }catch(Exception e){temp.delete();throw e;}
            Files.move(temp.toPath(),manifest.toPath(),StandardCopyOption.REPLACE_EXISTING);
        }
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
            String repository=item.optString("repository",manifest.getString("repository"));
            String revision=item.optString("revision",manifest.getString("revision"));
            String remotePath=item.optString("remotePath",path);
            URL url=new URL("https://huggingface.co/"+repository+"/resolve/"+revision+"/"+remotePath);
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
