package com.battlesbudz.minicpmv47;
import android.content.*;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.zip.*;
import org.json.*;
// Build 26: the run export bundle. Battles's device pastes were
// tens of thousands of tokens, almost all of it the loader log
// embedded in report.json, and they flooded the chat. From this
// build on, every terminal run writes ONE zip he can attach
// instead: the small verdict-bearing files stay separate inside
// the zip, and the loader log is extracted into its own file so
// it exists for crash chasing but never has to be read otherwise.
// The manual Export button in MainActivity calls the same code
// on whatever state is on disk, so even a run whose process died
// or was killed leaves a sendable bundle.
public final class RunExporter {
    private static boolean exportedThisProcess;
    // Called from BenchmarkService.retire(), the one funnel every
    // terminal path (completed, error, user stop, watchdog
    // timeout) passes through before the process is killed.
    // Runs at most once per process so a timeout racing the
    // worker's own retire cannot write the bundle twice.
    static synchronized String exportOnce(Context context){
        if(exportedThisProcess)return null;
        exportedThisProcess=true;
        return export(context);
    }
    static String export(Context context){
        try{
            File files=context.getFilesDir();
            JSONObject status=readJson(new File(files,"status.json"));
            JSONObject report=readJson(new File(files,"report.json"));
            boolean partial=false;
            if(report==null){report=readJson(new File(files,"report-partial.json"));partial=report!=null;}
            String arm="unknown";
            if(status!=null)arm=status.optString("abArm",status.optString("mode","unknown"));
            if(report!=null)arm=report.optString("abArm",arm);
            String buildNum="?";
            String version=BuildConfig.VERSION_NAME;
            int dot=version.lastIndexOf("build.");
            if(dot>=0)buildNum=version.substring(dot+"build.".length());
            String stamp=new SimpleDateFormat("yyyyMMdd-HHmmss",Locale.US).format(new Date());
            String name="minicpm-build"+buildNum+"-"+arm.replaceAll("[^A-Za-z0-9_-]","_")+"-"+stamp+".zip";
            ByteArrayOutputStream bytes=new ByteArrayOutputStream();
            ZipOutputStream zip=new ZipOutputStream(bytes);
            putText(zip,"status.json",fileText(new File(files,"status.json")));
            if(report!=null){
                putText(zip,"report.json",report.toString(2));
                JSONObject slim=new JSONObject(report.toString());
                String loader=slim.optString("loaderLog","");
                slim.remove("loaderLog");
                putText(zip,"report-slim.json",slim.toString(2));
                putText(zip,"loader_log.txt",loader);
            }else{
                putText(zip,"report.json","");
                putText(zip,"report-slim.json","");
                putText(zip,"loader_log.txt","");
            }
            putText(zip,"summary.txt",summary(status,report,partial));
            putText(zip,"memory.json",fileText(new File(files,"last-run/memory.json")));
            putText(zip,"native_stderr_tail.txt",fileTail(new File(files,"native_stderr.log"),32*1024));
            File crash=new File(files,"native_crash.txt");
            if(crash.isFile()&&crash.length()>0)putText(zip,"native_crash.txt",fileText(crash));
            zip.close();
            byte[] data=bytes.toByteArray();
            // Public Downloads via MediaStore: no permission
            // needed under scoped storage, and the file is
            // reachable from the chat attachment picker / Files.
            ContentValues values=new ContentValues();
            values.put(MediaStore.Downloads.DISPLAY_NAME,name);
            values.put(MediaStore.Downloads.MIME_TYPE,"application/zip");
            values.put(MediaStore.Downloads.RELATIVE_PATH,Environment.DIRECTORY_DOWNLOADS);
            android.net.Uri uri=context.getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI,values);
            if(uri==null)throw new IOException("MediaStore insert returned null");
            try(OutputStream out=context.getContentResolver().openOutputStream(uri)){out.write(data);out.flush();}
            // Second copy in the app's external files exports
            // dir, so a bundle survives even if Downloads is
            // cleared or the MediaStore entry is hard to find.
            try{
                File dir=context.getExternalFilesDir("exports");
                if(dir!=null){dir.mkdirs();try(FileOutputStream out=new FileOutputStream(new File(dir,name))){out.write(data);out.getFD().sync();}}
            }catch(Exception ignored){}
            try{BenchmarkService.write(new File(files,"export-last.txt"),name);}catch(Exception ignored){}
            return name;
        }catch(Exception error){
            android.util.Log.w("MiniCPMV47","Run export failed",error);
            return null;
        }
    }
    private static JSONObject readJson(File f){
        try{String t=fileText(f);return t.isEmpty()?null:new JSONObject(t);}catch(Exception e){return null;}
    }
    private static String fileText(File f){
        try{return Io.read(f.toPath());}catch(Exception e){return "";}
    }
    private static String fileTail(File f,int maxBytes){
        try{
            if(!f.isFile()||f.length()==0)return "";
            long len=f.length();long skip=Math.max(0,len-maxBytes);
            try(RandomAccessFile raf=new RandomAccessFile(f,"r")){
                raf.seek(skip);byte[] buf=new byte[(int)(len-skip)];raf.readFully(buf);
                return new String(buf,StandardCharsets.UTF_8);
            }
        }catch(Exception e){return "";}
    }
    private static void putText(ZipOutputStream zip,String name,String text)throws IOException{
        zip.putNextEntry(new ZipEntry(name));
        zip.write((text==null?"":text).getBytes(StandardCharsets.UTF_8));
        zip.closeEntry();
    }
    private static String num(JSONObject o,String key){
        if(o==null||!o.has(key))return "";
        Object v=o.opt(key);return v==null?"":String.valueOf(v);
    }
    // The verdict summary: everything a run is judged on, in a
    // few hundred bytes of plain text. Token IDs are included in
    // full because cross-arm token-ID equality is the hard gate.
    private static String summary(JSONObject status,JSONObject report,boolean partial){
        StringBuilder s=new StringBuilder();
        JSONObject src=report!=null?report:status;
        s.append("MiniCPM-V 4.7 phone trial - run summary\n");
        s.append("build: ").append(src==null?BuildConfig.VERSION_NAME:src.optString("build",BuildConfig.VERSION_NAME)).append('\n');
        s.append("sourceCommit: ").append(src==null?BuildConfig.SOURCE_COMMIT:src.optString("sourceCommit",BuildConfig.SOURCE_COMMIT)).append('\n');
        s.append("device: ").append(src==null?"":src.optString("device","")).append('\n');
        if(status!=null){
            s.append("mode: ").append(status.optString("mode","")).append('\n');
            s.append("abArm: ").append(status.optString("abArm","")).append('\n');
            s.append("moePinPrewarm: ").append(status.optString("moePinPrewarm","")).append('\n');
            s.append("moeLfuAdmit: ").append(status.optString("moeLfuAdmit","")).append('\n');
            s.append("phase: ").append(status.optString("phase","")).append('\n');
            s.append("status: ").append(status.optString("status","")).append('\n');
        }
        if(report!=null){
            if(report.has("mode")&&status==null)s.append("mode: ").append(report.optString("mode")).append('\n');
            s.append("partial: ").append(partial).append('\n');
            s.append("verdict: ").append(report.optString("verdict","")).append('\n');
            s.append("peakWorkerPssKb: ").append(num(report,"peakWorkerPssKb")).append('\n');
            s.append("peakThermalStatus: ").append(num(report,"peakThermalStatus")).append('\n');
            s.append("moeExpertInstalls: ").append(num(report,"moeExpertInstalls")).append('\n');
            s.append("moeCacheHits: ").append(num(report,"moeCacheHits")).append('\n');
            s.append("moeCacheHitRatePct: ").append(num(report,"moeCacheHitRatePct")).append('\n');
            s.append("moeIoReadMiB: ").append(num(report,"moeIoReadMiB")).append('\n');
            s.append("moeIoCalls: ").append(num(report,"moeIoCalls")).append('\n');
            JSONArray cases=report.optJSONArray("cases");
            if(cases!=null)for(int i=0;i<cases.length();i++){
                JSONObject c=cases.optJSONObject(i);if(c==null)continue;
                s.append("case ").append(c.optInt("index")).append(":\n");
                s.append("  generatedTokens: ").append(c.optInt("generatedTokens")).append('\n');
                s.append("  decodeTokensPerSec: ").append(c.optDouble("decodeTokensPerSec")).append('\n');
                s.append("  prefillMs: ").append(c.optDouble("prefillMs")).append('\n');
                s.append("  ttftMs: ").append(c.optDouble("ttftMs")).append('\n');
                s.append("  endedByEos: ").append(c.optBoolean("endedByEos")).append('\n');
                if(c.has("dsparkRounds")){
                    s.append("  dsparkRounds: ").append(c.optInt("dsparkRounds")).append('\n');
                    s.append("  dsparkDraftedTokens: ").append(c.optInt("dsparkDraftedTokens")).append('\n');
                    s.append("  dsparkAcceptedTokens: ").append(c.optInt("dsparkAcceptedTokens")).append('\n');
                    s.append("  dsparkAcceptRatePct: ").append(c.optDouble("dsparkAcceptRatePct")).append('\n');
                }
                JSONArray ids=c.optJSONArray("generatedTokenIds");
                if(ids!=null)s.append("  generatedTokenIds: ").append(ids.toString()).append('\n');
            }
            JSONObject totals=report.optJSONObject("totals");
            if(totals!=null&&(cases==null||cases.length()==0)){
                s.append("totals: ").append(totals.toString()).append('\n');
            }
        }
        return s.toString();
    }
}
