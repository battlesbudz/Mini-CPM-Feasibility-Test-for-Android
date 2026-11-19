package com.battlesbudz.minicpmv47;
import android.content.Context;
import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Locale;
import java.util.Map;
import java.util.HashMap;
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
    private static final long HOBBIT_BIN_BYTES=346030080L;
    private static final int HOBBIT_WINDOW_BYTES=1032192;
    private static final long HOBBIT_STRIDE_BYTES=1351680L;
    private static final int HOBBIT_LAYER_COUNT=40;
    private static final long HOBBIT_FREE_MARGIN=256L*1024*1024;
    static File hobbitShadowDir(File modelsRoot){
        return new File(new File(modelsRoot,"MiniCPM-V-4.7-35B-A3B"),"experts-hobbit");
    }
    private static String hex(byte[] bytes){
        StringBuilder out=new StringBuilder();for(byte b:bytes)out.append(String.format(Locale.ROOT,"%02x",b&255));return out.toString();
    }
    private static MessageDigest sha256()throws Exception{return MessageDigest.getInstance("SHA-256");}
    private static String digestFile(File file)throws Exception{
        MessageDigest digest=sha256();
        try(InputStream in=new FileInputStream(file)){byte[] b=new byte[1024*1024];int n;while((n=in.read(b))!=-1)digest.update(b,0,n);}
        return hex(digest.digest());
    }
    private static void syncDir(File dir){
        try{
            java.io.FileDescriptor fd=android.system.Os.open(dir.getAbsolutePath(),android.system.OsConstants.O_RDONLY,0);
            try{android.system.Os.fsync(fd);}finally{android.system.Os.close(fd);}
        }catch(Exception ignored){}
    }
    public static boolean hobbitShadowReady(Context c)throws Exception{
        File modelsRoot=new File(c.getFilesDir(),"models");
        File pack=new File(modelsRoot,"MiniCPM-V-4.7-35B-A3B");
        File indexFile=new File(pack,"hobbit-delta/cold_delta_index.json");
        if(!indexFile.isFile())return false;
        JSONObject index;
        try(InputStream in=new FileInputStream(indexFile)){index=new JSONObject(new String(Io.bytes(in),java.nio.charset.StandardCharsets.UTF_8));}
        catch(Exception e){return false;}
        File shadow=hobbitShadowDir(modelsRoot);
        File manifest=new File(shadow,"manifest.json"),construction=new File(shadow,"construction.json");
        if(!manifest.isFile()||!construction.isFile())return false;
        try{
            JSONObject constructed=new JSONObject(Io.read(construction.toPath()));
            if(!index.has("delta_sha256")||!constructed.has("delta_sha256")||
                !index.getString("delta_sha256").equals(constructed.getString("delta_sha256")))return false;
            for(int i=0;i<HOBBIT_LAYER_COUNT;i++)if(new File(shadow,String.format(Locale.US,"layer_%03d.bin",i)).length()!=HOBBIT_BIN_BYTES)return false;
            return true;
        }catch(Exception e){return false;}
    }
    public static void provisionHobbit(Context c,Progress progress)throws Exception{
        File modelsRoot=new File(c.getFilesDir(),"models");
        File pack=new File(modelsRoot,"MiniCPM-V-4.7-35B-A3B");
        File shadow=hobbitShadowDir(modelsRoot);
        File manifest=new File(shadow,"manifest.json");
        Files.deleteIfExists(manifest.toPath());
        File indexFile=new File(pack,"hobbit-delta/cold_delta_index.json");
        File deltaFile=new File(pack,"hobbit-delta/cold_delta.bin");
        if(!indexFile.isFile())throw new IOException("Missing "+indexFile+"; run Download model pack first");
        if(!deltaFile.isFile())throw new IOException("Missing "+deltaFile+"; run Download model pack first");
        JSONObject index;
        try(InputStream in=new FileInputStream(indexFile)){index=new JSONObject(new String(Io.bytes(in),java.nio.charset.StandardCharsets.UTF_8));}
        if(!"hobbit-cold-delta-1".equals(index.getString("format"))||index.getInt("window_bytes")!=HOBBIT_WINDOW_BYTES||index.getLong("stride_bytes")!=HOBBIT_STRIDE_BYTES)
            throw new IOException("Invalid HOBBIT cold delta index format or window geometry");
        if(!shadow.isDirectory()&&!shadow.mkdirs())throw new IOException("Cannot create HOBBIT shadow directory "+shadow);
        long expectedSize=index.getLong("delta_size");String expectedDigest=index.getString("delta_sha256");
        if(deltaFile.length()!=expectedSize||!expectedDigest.equals(digestFile(deltaFile)))throw new IOException("HOBBIT cold delta length or SHA-256 mismatch: "+deltaFile);
        if(!new ModelStore(c).ready())throw new IOException("Original model bins are not verified; run Download model pack first");
        long need=(HOBBIT_LAYER_COUNT+1)*HOBBIT_BIN_BYTES+HOBBIT_FREE_MARGIN;
        if(modelsRoot.getUsableSpace()<need)throw new IOException("HOBBIT construction needs about 14.5 GB of free storage (including safety margin)");
        File originalExperts=new File(pack,"experts");
        for(int i=0;i<HOBBIT_LAYER_COUNT;i++){
            String name=String.format(Locale.US,"layer_%03d.bin",i);
            File source=new File(originalExperts,name),target=new File(shadow,name);
            progress.update("HOBBIT pack: copying layer "+name.substring(6,9)+" of 40");
            if(!source.isFile())throw new IOException("Missing original expert bin "+source);
            File temp=File.createTempFile("hobbit-bin-",".tmp",shadow);
            try{
                try(InputStream in=new FileInputStream(source);FileOutputStream out=new FileOutputStream(temp)){byte[] b=new byte[1024*1024];int n;while((n=in.read(b))!=-1)out.write(b,0,n);out.getFD().sync();}
                if(temp.length()!=HOBBIT_BIN_BYTES)throw new IOException("Copied expert bin has wrong length: "+name);
                Files.move(temp.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
            }catch(Exception e){temp.delete();throw e;}
        }
        syncDir(shadow);
        JSONArray windows=index.getJSONArray("windows");
        int coldExperts=index.getInt("cold_experts");
        if(coldExperts!=8097)throw new IOException("HOBBIT production index cold_experts must be 8097");
        if(windows.length()!=coldExperts)throw new IOException("HOBBIT index window count does not equal cold_experts");
        byte[] record=new byte[HOBBIT_WINDOW_BYTES],readback=new byte[HOBBIT_WINDOW_BYTES];
        Map<Integer,RandomAccessFile> bins=new HashMap<>();
        int applied=0;
        try(InputStream in=new BufferedInputStream(new FileInputStream(deltaFile))){
            for(int i=0;i<windows.length();i++){
                JSONObject window=windows.getJSONObject(i);int layer=window.getInt("layer"),expert=window.getInt("expert");
                if(layer<0||layer>=HOBBIT_LAYER_COUNT||expert<0)throw new IOException("Invalid HOBBIT window layer/expert "+layer+"/"+expert);
                readFully(in,record);
                String recordHash=hex(sha256().digest(record));
                if(!recordHash.equals(window.getString("sha256")))throw new IOException("HOBBIT delta window SHA-256 mismatch at layer "+layer+" expert "+expert);
                RandomAccessFile bin=bins.get(layer);
                if(bin==null){bin=new RandomAccessFile(new File(shadow,String.format(Locale.US,"layer_%03d.bin",layer)),"rw");bins.put(layer,bin);}
                long offset=expert*HOBBIT_STRIDE_BYTES;
                if(offset+HOBBIT_WINDOW_BYTES>HOBBIT_BIN_BYTES)throw new IOException("HOBBIT window outside bin at layer "+layer+" expert "+expert);
                bin.seek(offset);bin.write(record);
                bin.seek(offset);bin.readFully(readback);
                if(!recordHash.equals(hex(sha256().digest(readback))))throw new IOException("HOBBIT shadow readback SHA-256 mismatch at layer "+layer+" expert "+expert);
                applied++;
                if(applied%200==0||applied==coldExperts)progress.update("HOBBIT pack: applied "+applied+" of "+coldExperts+" cold windows");
            }
            if(in.read()!=-1)throw new IOException("HOBBIT cold delta has trailing bytes");
            for(RandomAccessFile bin:bins.values())bin.getFD().sync();
        }finally{for(RandomAccessFile bin:bins.values())try{bin.close();}catch(IOException ignored){}}
        if(applied!=coldExperts)throw new IOException("HOBBIT applied window count mismatch: "+applied+" of "+coldExperts);
        for(int i=0;i<HOBBIT_LAYER_COUNT;i++)if(new File(shadow,String.format(Locale.US,"layer_%03d.bin",i)).length()!=HOBBIT_BIN_BYTES)throw new IOException("HOBBIT shadow bin length mismatch at layer "+i);
        File construction=new File(shadow,"construction.json");
        File constructionTemp=File.createTempFile("hobbit-construction-",".tmp",shadow);
        try{
            try(FileOutputStream out=new FileOutputStream(constructionTemp)){out.write(new JSONObject().put("delta_sha256",expectedDigest).put("windows",applied).toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));out.getFD().sync();}
            Files.move(constructionTemp.toPath(),construction.toPath(),StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
            syncDir(shadow);
        }catch(Exception e){constructionTemp.delete();throw e;}
        File placed=new File(shadow,"manifest.json"),manifestTemp=File.createTempFile("hobbit-manifest-",".tmp",shadow);
        try{
            try(InputStream in=c.getAssets().open("hobbit_production_manifest.json");FileOutputStream out=new FileOutputStream(manifestTemp)){byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);out.getFD().sync();}
            Files.move(manifestTemp.toPath(),placed.toPath(),StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
            syncDir(shadow);
            byte[] asset;
            try(InputStream in=c.getAssets().open("hobbit_production_manifest.json")){asset=Io.bytes(in);}
            if(placed.length()!=asset.length||!digestFile(placed).equals(hex(sha256().digest(asset))))throw new IOException("Placed HOBBIT production manifest failed byte verification");
        }catch(Exception e){manifestTemp.delete();placed.delete();throw e;}
    }
    private static void readFully(InputStream in,byte[] buffer)throws IOException{
        int offset=0;while(offset<buffer.length){int n=in.read(buffer,offset,buffer.length-offset);if(n<0)throw new EOFException("Truncated HOBBIT cold delta record");offset+=n;}
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
