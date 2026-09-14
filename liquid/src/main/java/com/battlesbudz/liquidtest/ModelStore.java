package com.battlesbudz.liquidtest;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Locale;
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
    void install(Uri folder,Progress progress)throws Exception{
        new File(root,"verified.json").delete();
        JSONArray list=manifest.getJSONArray("files");
        for(int i=0;i<list.length();i++){
            JSONObject item=list.getJSONObject(i);String path=item.getString("path");
            File target=new File(root,path);target.getParentFile().mkdirs();
            progress.update("Verifying "+path);
            if(target.isFile()&&verify(target,item))continue;
            File part=new File(target+".part");
            if(folder==null&&part.isFile()&&verify(part,item)){Files.move(part.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);continue;}
            long existing=folder==null&&part.isFile()?part.length():0;
            if(existing>item.getLong("size")){part.delete();existing=0;}
            if(root.getUsableSpace()<item.getLong("size")-existing+256L*1024*1024)throw new IOException("Not enough free storage to install "+path);
            progress.update("Installing "+path+" ("+(i+1)+" / "+list.length()+")");
            InputStream source;HttpURLConnection connection=null;
            if(folder==null){
                URL url=new URL("https://huggingface.co/"+manifest.getString("repository")+"/resolve/"+manifest.getString("revision")+"/"+path);
                connection=(HttpURLConnection)url.openConnection();connection.setConnectTimeout(30000);connection.setReadTimeout(30000);
                if(existing>0)connection.setRequestProperty("Range","bytes="+existing+"-");
                int code=connection.getResponseCode();
                if(code==200)existing=0;
                else if(code==206){String range=connection.getHeaderField("Content-Range");if(range==null||!range.startsWith("bytes "+existing+"-")){connection.disconnect();throw new IOException("Invalid resume response");}}
                else{connection.disconnect();throw new IOException("Download HTTP "+code+" for "+path);}
                source=connection.getInputStream();
            }else{existing=0;source=context.getContentResolver().openInputStream(find(folder,path));}
            if(source==null)throw new IOException("Cannot read "+path);
            try(InputStream in=source;FileOutputStream out=new FileOutputStream(part,existing>0)){
                byte[] buffer=new byte[1024*1024];long total=existing,last=0;int n;
                while((n=in.read(buffer))!=-1){
                    if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("Installation stopped; download can resume");
                    total+=n;if(total>item.getLong("size"))throw new IOException("File exceeds manifest size");out.write(buffer,0,n);
                    if(total-last>32L*1024*1024){last=total;progress.update(String.format(Locale.US,"%s %.0f%%",path,100.0*total/item.getLong("size")));}
                }
                out.getFD().sync();
            }finally{if(connection!=null)connection.disconnect();}
            progress.update("Checking SHA-256 for "+path);
            if(!verify(part,item)){part.delete();throw new IOException("Checksum mismatch: "+path);}
            Files.move(part.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);
        }
        Io.write(new File(root,"verified.json").toPath(),manifest.toString());
        progress.update("Complete voice pack verified. Ready for offline testing.");
    }
    private static boolean verify(File file,JSONObject item)throws Exception{
        if(file.length()!=item.getLong("size"))return false;
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream in=new FileInputStream(file)){byte[] b=new byte[1024*1024];int n;while((n=in.read(b))!=-1){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();digest.update(b,0,n);}}
        StringBuilder hex=new StringBuilder();for(byte b:digest.digest())hex.append(String.format(Locale.ROOT,"%02x",b&255));
        return hex.toString().equals(item.getString("sha256"));
    }
    private Uri find(Uri tree,String path)throws IOException{
        String id=DocumentsContract.getTreeDocumentId(tree);
        for(String name:path.split("/")){
            String found=null;Uri children=DocumentsContract.buildChildDocumentsUriUsingTree(tree,id);
            try(Cursor cursor=context.getContentResolver().query(children,new String[]{DocumentsContract.Document.COLUMN_DOCUMENT_ID,DocumentsContract.Document.COLUMN_DISPLAY_NAME},null,null,null)){
                while(cursor!=null&&cursor.moveToNext())if(name.equals(cursor.getString(1))){found=cursor.getString(0);break;}
            }
            if(found==null)throw new IOException("Selected folder is missing "+path);id=found;
        }
        return DocumentsContract.buildDocumentUriUsingTree(tree,id);
    }
}
