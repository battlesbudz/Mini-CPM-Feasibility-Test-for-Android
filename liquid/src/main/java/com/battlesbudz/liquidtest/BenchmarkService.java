package com.battlesbudz.liquidtest;
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
        Notification notification=new Notification.Builder(this,"bench").setContentTitle("Liquid Voice benchmark running").setContentText("Processing recorded audio locally").setSmallIcon(android.R.drawable.ic_media_play).addAction(new Notification.Action.Builder(null,"Stop",stop).build()).setOngoing(true).build();
        startForeground(1,notification);
        PowerManager pm=getSystemService(PowerManager.class);
        wake=pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"Liquid Voice:benchmark");wake.acquire(60*60*1000L);
        heartbeat=SystemClock.elapsedRealtime();
        int profile=intent==null?1:intent.getIntExtra("profile",1);
        int voice=intent==null?0:intent.getIntExtra("voice",0);
        if(profile<0||profile>5)profile=1;if(voice<0||voice>2)voice=0;
        run=new File(getFilesDir(),"last-run");run.mkdirs();state=new JSONObject();
        try{state.put("startedAtMs",System.currentTimeMillis()).put("profile","liquid-hardware-compare-v2").put("contextTokens",4096).put("batchTokens",256).put("microBatchTokens",64).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MANUFACTURER+" "+Build.MODEL).put("sdk",Build.VERSION.SDK_INT);}catch(Exception ignored){}
        state("Starting","RUNNING");
        try{state.put("profileIndex",profile).put("voiceIndex",voice).put("initialThermalStatus",pm.getCurrentThermalStatus());}catch(Exception ignored){}
        state("Starting selected configuration","RUNNING");
        chargeStart=getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        final long began=SystemClock.elapsedRealtime(),wallBudgetMs=3*60*1000L;
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
            }catch(Exception error){android.util.Log.w("Liquid Voice","Memory checkpoint failed",error);}
            if(SystemClock.elapsedRealtime()-began>wallBudgetMs){state("Test exceeded its wall-clock budget","TIMEOUT");retire();}
            else if(SystemClock.elapsedRealtime()-heartbeat>3*60*1000L){state("Native worker stopped responding","TIMEOUT");retire();}
        },0,1,TimeUnit.SECONDS);
        final int selected=profile,selectedVoice=voice;new Thread(()->execute(selected,selectedVoice),"native-benchmark").start();return START_NOT_STICKY;
    }
    private void execute(int profile,int voice){
        try{
            ModelStore models=new ModelStore(this);if(profile!=5&&!models.ready())throw new IOException("Model pack is not verified");
            File chunks=new File(getCacheDir(),"input");
            if(profile!=5&&voice!=2&&!new File(chunks,"question.wav").isFile())throw new IOException("Record a test question first");
            JSONObject report=new JSONObject(NativeBench.run(models.root.getPath(),chunks.getPath(),run.getPath(),profile,voice,message->{heartbeat=SystemClock.elapsedRealtime();state(message,"RUNNING");}));
            String verdict=profile==5?(report.has("error")?"ERROR":report.optString("verdict","ERROR")):LiquidVerdict.classify(report.has("error"),report.optLong("audioSamples"),report.optDouble("audioRms",0),report.optString("text"));
            report.put("verdict",verdict).put("peakWorkerPssKb",peakPssKb).put("peakThermalStatus",peakThermal)
                .put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MODEL)
                .put("batteryChargeChangeUah",getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)-chargeStart)
                .put("limitations",profile==5?report.optString("limitations"):"Recorded-question smoke test. First PCM is callback latency after submission, excluding recording, model load and playback. Output needs human review. EOS versus generation cap is not exposed by upstream. No live interruption, multi-turn or tools tested.");
            report.put("initialThermalStatus",state.optInt("initialThermalStatus")).put("startedAtMs",state.optLong("startedAtMs"));
            JSONObject summary=new JSONObject(report.toString());summary.remove("audioChunks");
            try(FileOutputStream out=new FileOutputStream(new File(getFilesDir(),"comparison.jsonl"),true)){out.write((summary.toString()+"\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));out.getFD().sync();}
            write(new File(getFilesDir(),"report.json"),report.toString(2));state(verdict,verdict);
        }catch(Throwable error){state(error.getClass().getSimpleName()+": "+error.getMessage(),"ERROR");}
        finally{retire();}
    }
    private void retire(){timers.shutdownNow();if(wake!=null&&wake.isHeld())wake.release();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf();android.os.Process.killProcess(android.os.Process.myPid());}
}
