package com.battlesbudz.minicpmv47;
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
    private volatile JSONObject lastProgress;
    private volatile long lastProgressAtMs;
    @Override public IBinder onBind(Intent intent){return null;}
    private synchronized void state(String phase,String status){
        currentPhase=phase;
        try{state.put("phase",phase).put("status",status).put("updatedAtMs",System.currentTimeMillis()).put("pid",android.os.Process.myPid());write(new File(getFilesDir(),"status.json"),state.toString(2));}
        catch(Exception ignored){} // Retain the previous checkpoint on I/O failure.
    }
    private void onNativeEvent(String message){
        JSONObject progress=Progress.parse(message);
        if(progress==null){state(message,"RUNNING");return;}
        lastProgress=progress;lastProgressAtMs=System.currentTimeMillis();
        progressState(progress);
    }
    private synchronized void progressState(JSONObject progress){
        currentPhase=Progress.label(progress);
        try{Progress.merge(state,progress,System.currentTimeMillis());state.put("phase",currentPhase).put("status","RUNNING").put("updatedAtMs",System.currentTimeMillis()).put("pid",android.os.Process.myPid());write(new File(getFilesDir(),"status.json"),state.toString(2));}
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
        nm.createNotificationChannel(new NotificationChannel("bench","MiniCPM-V trial",NotificationManager.IMPORTANCE_LOW));
        PendingIntent stop=PendingIntent.getService(this,1,new Intent(this,BenchmarkService.class).setAction("STOP"),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);
        Notification notification=new Notification.Builder(this,"bench").setContentTitle("MiniCPM-V 4.7 trial running").setContentText("Answering fixed photo questions locally").setSmallIcon(android.R.drawable.ic_media_play).addAction(new Notification.Action.Builder(null,"Stop",stop).build()).setOngoing(true).build();
        startForeground(1,notification);
        PowerManager pm=getSystemService(PowerManager.class);
        wake=pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"MiniCPMV47:trial");wake.acquire(120*60*1000L);
        heartbeat=SystemClock.elapsedRealtime();
        run=new File(getFilesDir(),"last-run");run.mkdirs();state=new JSONObject();
        try{state.put("startedAtMs",System.currentTimeMillis()).put("profile","minicpm-v47-phone-trial-v1").put("contextTokens",4096).put("batchTokens",2048).put("maxNewTokens",512).put("threads",4).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MANUFACTURER+" "+Build.MODEL).put("sdk",Build.VERSION.SDK_INT);}catch(Exception ignored){}
        state("Starting","RUNNING");
        chargeStart=getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        final long began=SystemClock.elapsedRealtime(),wallBudgetMs=100*60*1000L;
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
                JSONObject progress=lastProgress;
                if(progress!=null){sample.put("stage",progress.optString("stage",""));if(progress.has("caseIndex"))sample.put("stageCase",progress.optInt("caseIndex"));if(progress.has("generatedTokens"))sample.put("generatedTokens",progress.optInt("generatedTokens"));sample.put("lastProgressAtMs",lastProgressAtMs);}
                write(new File(run,"memory.json"),sample.toString(2));
                try(FileOutputStream out=new FileOutputStream(new File(run,"memory-history.jsonl"),true)){
                    out.write((sample.toString()+"\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));out.getFD().sync();
                }
            }catch(Exception error){android.util.Log.w("MiniCPMV47","Memory checkpoint failed",error);}
            if(SystemClock.elapsedRealtime()-began>wallBudgetMs){state("Test exceeded its wall-clock budget","TIMEOUT");retire();}
            else if(SystemClock.elapsedRealtime()-heartbeat>30*60*1000L){state("Native worker stopped responding","TIMEOUT");retire();}
        },0,1,TimeUnit.SECONDS);
        new Thread(this::execute,"native-trial").start();return START_NOT_STICKY;
    }
    private void execute(){
        try{
            ModelStore models=new ModelStore(this);if(!models.ready())throw new IOException("Model pack is not verified");
            File image=new File(getFilesDir(),"test-bus.jpg");
            try(InputStream in=getAssets().open("test-bus.jpg")){Files.copy(in,image.toPath(),StandardCopyOption.REPLACE_EXISTING);}
            JSONObject report=new JSONObject(NativeBench.run(models.root.getPath(),image.getPath(),run.getPath(),message->{heartbeat=SystemClock.elapsedRealtime();onNativeEvent(message);}));
            JSONArray cases=report.optJSONArray("cases");int answered=0;
            if(cases!=null)for(int i=0;i<cases.length();i++){JSONObject c=cases.optJSONObject(i);if(c!=null&&c.optInt("generatedTokens",0)>0)answered++;}
            String verdict=TrialVerdict.classify(report.has("error"),report.optString("error"),answered,cases==null?0:cases.length());
            report.put("verdict",verdict).put("peakWorkerPssKb",peakPssKb).put("peakThermalStatus",peakThermal).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MODEL).put("batteryChargeChangeUah",getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)-chargeStart).put("limitations","Three text-only questions, then a single photo with three fixed questions, greedy sampling, thinking disabled. CPU path only (vendor demo defaults: 4 threads, 4096 context). First run on this device class; no prior Android LES result exists to compare against.");
            write(new File(getFilesDir(),"report.json"),report.toString(2));state(verdict,verdict);
        }catch(Throwable error){state(error.getClass().getSimpleName()+": "+error.getMessage(),"ERROR");}
        finally{retire();}
    }
    private void retire(){timers.shutdownNow();if(wake!=null&&wake.isHeld())wake.release();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf();android.os.Process.killProcess(android.os.Process.myPid());}
}
