package com.battlesbudz.minicpmtest;
import android.app.*;
import android.content.*;
import android.os.*;
import java.io.*;
import java.nio.file.*;
import java.util.concurrent.*;
import org.json.*;
public final class BenchmarkService extends Service {
    private final ScheduledExecutorService timers=Executors.newSingleThreadScheduledExecutor();
    private PowerManager.WakeLock wake;
    private volatile long heartbeat;
    private volatile boolean started;
    private volatile int peakPssKb,peakThermal;
    private File run;
    private JSONObject state;
    private int chargeStart;
    private volatile String currentPhase="Starting";
    @Override public IBinder onBind(Intent intent){return null;}
    private synchronized void state(String phase,String status){
        currentPhase=phase;
        try{state.put("phase",phase).put("status",status).put("updatedAtMs",System.currentTimeMillis()).put("pid",android.os.Process.myPid());write(new File(getFilesDir(),"status.json"),state.toString(2));}
        catch(Exception ignored){} // Retain the previous checkpoint on I/O failure.
    }
    static void write(File target,String value)throws IOException{
        File temp=new File(target+".tmp");
        try(FileOutputStream out=new FileOutputStream(temp)){out.write(value.getBytes(java.nio.charset.StandardCharsets.UTF_8));out.getFD().sync();}
        Files.move(temp.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        if(intent!=null&&"STOP".equals(intent.getAction())){if(state!=null)state("Stopped by user","CANCELLED");retire();return START_NOT_STICKY;}
        if(started)return START_NOT_STICKY;started=true;
        NotificationManager nm=getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel("bench","Model benchmark",NotificationManager.IMPORTANCE_LOW));
        PendingIntent stop=PendingIntent.getService(this,1,new Intent(this,BenchmarkService.class).setAction("STOP"),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);
        Notification notification=new Notification.Builder(this,"bench").setContentTitle("MiniCPM benchmark running").setContentText("Processing recorded audio locally").setSmallIcon(android.R.drawable.ic_media_play).addAction(new Notification.Action.Builder(null,"Stop",stop).build()).setOngoing(true).build();
        startForeground(1,notification);
        PowerManager pm=getSystemService(PowerManager.class);
        wake=pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"MiniCPM:benchmark");wake.acquire(60*60*1000L);
        heartbeat=SystemClock.elapsedRealtime();
        int frames=intent==null?60:intent.getIntExtra("frames",60);if(frames!=60&&frames!=300&&frames!=1800)frames=60;
        run=new File(getFilesDir(),"last-run");run.mkdirs();state=new JSONObject();
        try{state.put("startedAtMs",System.currentTimeMillis()).put("profile","cpu-memory-screen-v2").put("contextTokens",4096).put("batchTokens",256).put("microBatchTokens",64).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MANUFACTURER+" "+Build.MODEL).put("sdk",Build.VERSION.SDK_INT);}catch(Exception ignored){}
        state("Starting","RUNNING");
        chargeStart=getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        final long began=SystemClock.elapsedRealtime(),wallBudgetMs=12*60*1000L+frames*1500L;
        timers.scheduleWithFixedDelay(()->{
            try{
                android.os.Debug.MemoryInfo memory=new android.os.Debug.MemoryInfo();android.os.Debug.getMemoryInfo(memory);
                peakPssKb=Math.max(peakPssKb,memory.getTotalPss());peakThermal=Math.max(peakThermal,pm.getCurrentThermalStatus());
                ActivityManager.MemoryInfo system=new ActivityManager.MemoryInfo();getSystemService(ActivityManager.class).getMemoryInfo(system);
                JSONObject sample=new JSONObject().put("pid",android.os.Process.myPid()).put("updatedAtMs",System.currentTimeMillis())
                    .put("phase",currentPhase).put("workerPssKb",memory.getTotalPss()).put("peakWorkerPssKb",peakPssKb)
                    .put("systemAvailableBytes",system.availMem).put("systemTotalBytes",system.totalMem)
                    .put("systemLowMemoryThresholdBytes",system.threshold).put("systemLowMemory",system.lowMemory)
                    .put("thermalStatus",pm.getCurrentThermalStatus()).put("peakThermalStatus",peakThermal);
                write(new File(run,"memory.json"),sample.toString(2));
                try(FileOutputStream out=new FileOutputStream(new File(run,"memory-history.jsonl"),true)){
                    out.write((sample.toString()+"\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));out.getFD().sync();
                }
            }catch(Exception error){android.util.Log.w("MiniCPM","Memory checkpoint failed",error);}
            if(SystemClock.elapsedRealtime()-began>wallBudgetMs){state("Test exceeded its wall-clock budget","TIMEOUT");retire();}
            else if(SystemClock.elapsedRealtime()-heartbeat>10*60*1000L){state("Native worker stopped responding","TIMEOUT");retire();}
        },0,1,TimeUnit.SECONDS);
        final int total=frames;new Thread(()->execute(total),"native-benchmark").start();return START_NOT_STICKY;
    }
    private void execute(int frames){
        try{
            ModelStore models=new ModelStore(this);if(!models.ready())throw new IOException("Model pack is not verified");
            File chunks=new File(getCacheDir(),"input");
            for(int i=0;i<30;i++)if(!new File(chunks,String.format(java.util.Locale.ROOT,"%04d.wav",i)).isFile())throw new IOException("Record a test question first");
            JSONObject report=new JSONObject(NativeBench.run(models.root.getPath(),chunks.getPath(),run.getPath(),frames,message->{heartbeat=SystemClock.elapsedRealtime();state(message,"RUNNING");}));
            JSONArray decisions=report.optJSONArray("decisions");double[] latencies=new double[decisions==null?0:decisions.length()];
            for(int i=0;i<latencies.length;i++)latencies[i]=decisions.getJSONObject(i).getDouble("latencyMs");
            double p95=ScreenVerdict.p95(latencies),gap=0,end=0;boolean active=false;
            JSONArray audio=report.optJSONArray("audioChunks");
            if(audio!=null)for(int i=0;i<audio.length();i++){
                JSONObject chunk=audio.getJSONObject(i);double at=chunk.getDouble("atMs");if(active&&at>end)gap+=at-end;
                end=Math.max(active?end:at,at)+1000.0*chunk.getLong("samples")/chunk.getInt("sampleRate");active=!chunk.getBoolean("final");
            }
            String verdict=ScreenVerdict.classify(report.has("error"),frames,report.optInt("framesCompleted"),report.optInt("speakFrames"),report.optLong("audioSamples"),report.optLong("invalidSamples"),report.optDouble("audioRms",0),report.optBoolean("audioDrained"),p95,gap);
            report.put("verdict",verdict).put("decisionP95Ms",Double.isFinite(p95)?p95:JSONObject.NULL).put("estimatedPcmSupplyGapMs",gap).put("peakWorkerPssKb",peakPssKb).put("peakThermalStatus",peakThermal).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MODEL).put("batteryChargeChangeUah",getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)-chargeStart).put("limitations","CPU compute screen only. No live acoustic interruption, GPU comparison or tools tested. PCM supply estimate excludes device playback.");
            write(new File(getFilesDir(),"report.json"),report.toString(2));state(verdict,verdict);
        }catch(Throwable error){state(error.getClass().getSimpleName()+": "+error.getMessage(),"ERROR");}
        finally{retire();}
    }
    private void retire(){timers.shutdownNow();if(wake!=null&&wake.isHeld())wake.release();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf();android.os.Process.killProcess(android.os.Process.myPid());}
}
