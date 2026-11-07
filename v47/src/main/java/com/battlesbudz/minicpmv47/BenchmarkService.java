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
    private volatile String mode="full";
    private final JSONArray partialCases=new JSONArray();
    private final JSONObject runTotals=new JSONObject();
    @Override public IBinder onBind(Intent intent){return null;}
    // Build 15: the report/state arm labels for every mode, in one
    // place (build 14 inlined the quick_off ternaries in three
    // spots). quick_dspark is the DSpark arm: prefetch off like
    // quick_off, but its own abArm so the arms can never be
    // confused in a pasted diagnostics dump.
    private String lookaheadPrefetch(){return "quick_off".equals(mode)||"quick_dspark".equals(mode)||"quick_dspark_io8".equals(mode)||"quick_dspark_pin".equals(mode)||"quick_dspark_lfu".equals(mode)||"quick_dspark_pinlfu".equals(mode)||"quick_dspark_tailskip".equals(mode)?"off":"on";}
    private String abArm(){return "quick_dspark_tailskip".equals(mode)?"dspark_tailskip":"quick_dspark_pinlfu".equals(mode)?"dspark_pinlfu":"quick_dspark_pin".equals(mode)?"dspark_pin":"quick_dspark_lfu".equals(mode)?"dspark_lfu":"quick_dspark_io8".equals(mode)?"dspark_io8":"quick_dspark".equals(mode)?"dspark_on":"quick_off".equals(mode)?"lookahead_off":"lookahead_on";}
    // Build 24 (item 2): the two admission-policy flags, named in
    // the state and both reports so a device paste proves which
    // policy was live for the run.
    private String pinPrewarm(){return "quick_dspark_pin".equals(mode)||"quick_dspark_pinlfu".equals(mode)?"on":"off";}
    private String lfuAdmit(){return "quick_dspark_lfu".equals(mode)||"quick_dspark_pinlfu".equals(mode)?"on":"off";}
    // Build 27 (item 5): the tail-skip flag, carried the same way.
    private String tailSkip(){return "quick_dspark_tailskip".equals(mode)?"on":"off";}
    private synchronized void state(String phase,String status){
        currentPhase=phase;
        try{state.put("phase",phase).put("status",status).put("updatedAtMs",System.currentTimeMillis()).put("pid",android.os.Process.myPid());write(new File(getFilesDir(),"status.json"),state.toString(2));}
        catch(Exception ignored){} // Retain the previous checkpoint on I/O failure.
    }
    private void onNativeEvent(String message){
        // Build 12: a finished case arrives whole. Record it in the
        // incremental partial report and the cumulative totals before
        // anything else can overwrite the live status fields.
        if(message!=null&&message.startsWith("CASE_DONE ")){
            try{
                JSONObject kase=new JSONObject(message.substring("CASE_DONE ".length()));
                synchronized(this){partialCases.put(kase);Progress.accumulate(runTotals,kase);state.put("totals",new JSONObject(runTotals.toString()));}
                writePartialReport();
                state(currentPhase,"RUNNING");
            }catch(Exception ignored){}
            return;
        }
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
    // Build 12: incremental per-case report. report.json is only
    // written when the native run returns, and the watchdog kills the
    // process on a timeout — build 10's 100-minute run produced no
    // per-case results at all for exactly that reason. The native side
    // emits CASE_DONE per finished case; this file is rewritten after
    // each one, so a timeout loses at most the in-progress case. It is
    // deleted when the final report.json lands.
    private void writePartialReport(){
        try{
            JSONObject p=new JSONObject();
            p.put("schemaVersion",2).put("partial",true).put("mode",mode).put("moeLookaheadPrefetch",lookaheadPrefetch()).put("abArm",abArm()).put("moePinPrewarm",pinPrewarm()).put("moeLfuAdmit",lfuAdmit()).put("moeTailSkip",tailSkip()).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MODEL).put("startedAtMs",state==null?0:state.optLong("startedAtMs",0)).put("updatedAtMs",System.currentTimeMillis());
            synchronized(this){p.put("totals",new JSONObject(runTotals.toString()));p.put("cases",new JSONArray(partialCases.toString()));}
            write(new File(getFilesDir(),"report-partial.json"),p.toString(2));
        }catch(Exception ignored){}
    }
    static void write(File target,String value)throws IOException{
        File temp=new File(target+".tmp");
        try(FileOutputStream out=new FileOutputStream(temp)){out.write(value.getBytes(java.nio.charset.StandardCharsets.UTF_8));out.getFD().sync();}
        Files.move(temp.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING);
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        if(intent!=null&&"STOP".equals(intent.getAction())){if(state!=null)state("Stopped by user","CANCELLED");retire();return START_NOT_STICKY;}
        if(started)return START_NOT_STICKY;started=true;
        String requestedMode=intent==null?null:intent.getStringExtra("mode");
        mode="quick".equals(requestedMode)||"quick_off".equals(requestedMode)||"quick_dspark".equals(requestedMode)||"quick_dspark_io8".equals(requestedMode)||"quick_dspark_pin".equals(requestedMode)||"quick_dspark_lfu".equals(requestedMode)||"quick_dspark_pinlfu".equals(requestedMode)||"quick_dspark_tailskip".equals(requestedMode)?requestedMode:"full";
        NotificationManager nm=getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel("bench","MiniCPM-V trial",NotificationManager.IMPORTANCE_LOW));
        PendingIntent stop=PendingIntent.getService(this,1,new Intent(this,BenchmarkService.class).setAction("STOP"),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);
        Notification notification=new Notification.Builder(this,"bench").setContentTitle("MiniCPM-V 4.7 trial running").setContentText("Answering fixed photo questions locally").setSmallIcon(android.R.drawable.ic_media_play).addAction(new Notification.Action.Builder(null,"Stop",stop).build()).setOngoing(true).build();
        startForeground(1,notification);
        PowerManager pm=getSystemService(PowerManager.class);
        wake=pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"MiniCPMV47:trial");wake.acquire(120*60*1000L);
        heartbeat=SystemClock.elapsedRealtime();
        run=new File(getFilesDir(),"last-run");run.mkdirs();state=new JSONObject();
        try{state.put("startedAtMs",System.currentTimeMillis()).put("profile","minicpm-v47-phone-trial-v1").put("contextTokens",4096).put("batchTokens",2048).put("mode",mode).put("moeLookaheadPrefetch",lookaheadPrefetch()).put("abArm",abArm()).put("moePinPrewarm",pinPrewarm()).put("moeLfuAdmit",lfuAdmit()).put("moeTailSkip",tailSkip()).put("maxNewTokens",!"full".equals(mode)?32:512).put("threads",4).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MANUFACTURER+" "+Build.MODEL).put("sdk",Build.VERSION.SDK_INT);}catch(Exception ignored){}
        // Build 19: if a previous run died natively, the crash
        // recorder's file survives in the files dir (the process
        // that wrote it is gone, and this new run's native side
        // does not delete it). Carry a bounded copy into this
        // run's status.json so the record stays visible in every
        // diagnostics paste even after a new run starts.
        try{
            File crash=new File(getFilesDir(),"native_crash.txt");
            if(crash.isFile()&&crash.length()>0){
                String text=Io.read(crash.toPath());
                if(text.length()>4096)text=text.substring(text.length()-4096);
                state.put("previousNativeCrash",text);
            }
        }catch(Exception ignored){}
        state("Starting","RUNNING");
        writePartialReport();
        chargeStart=getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        final long began=SystemClock.elapsedRealtime(),wallBudgetMs=!"full".equals(mode)?10*60*1000L:100*60*1000L;
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
            JSONObject report=new JSONObject(NativeBench.run(models.root.getPath(),image.getPath(),run.getPath(),mode,message->{heartbeat=SystemClock.elapsedRealtime();onNativeEvent(message);}));
            JSONArray cases=report.optJSONArray("cases");int answered=0;
            if(cases!=null)for(int i=0;i<cases.length();i++){JSONObject c=cases.optJSONObject(i);if(c!=null&&c.optInt("generatedTokens",0)>0)answered++;}
            String verdict=TrialVerdict.classify(report.has("error"),report.optString("error"),answered,cases==null?0:cases.length());
            // Build 27: validate the tailskip arm's structured-output
            // probe with org.json's strict parser. The JSONObject
            // constructor rejects trailing text and non-objects, so a
            // true here means the probe answer was exactly one JSON
            // object and nothing else.
            if("quick_dspark_tailskip".equals(mode)){
                boolean probeValid=false;
                try{new JSONObject(report.optString("moeTailSkipProbeAnswer",""));probeValid=true;}catch(Exception ignored){}
                try{report.put("moeTailSkipProbeJsonValid",probeValid);}catch(Exception ignored){}
            }
            report.put("verdict",verdict).put("peakWorkerPssKb",peakPssKb).put("peakThermalStatus",peakThermal).put("build",BuildConfig.VERSION_NAME).put("sourceCommit",BuildConfig.SOURCE_COMMIT).put("device",Build.MODEL).put("batteryChargeChangeUah",getSystemService(BatteryManager.class).getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)-chargeStart).put("limitations","quick_dspark_tailskip".equals(mode)?"Quick mode, DSpark tail-skip A/B arm (build 27, item 5): Case 3 (the long bicycle text question) plus a JSON probe case, 32-token decode cap on Case 3 and 96 on the probe, 10-minute watchdog. Identical to the stock DSpark arm except LLAMA_MOE_TAIL_SKIP=1: in each DSpark verify batch, a missing expert routed only at rank 7 (the last of the top-8) by every row is not installed; its positions point at a reserved all-zero window slot, so its contribution that step is exactly zero. While that zero slot is armed the window has 31 usable slots instead of 32; a later batch needing 32 installs with no skippable expert would hit the runtime's fit guard (observed verify batches demand about 17-24 unique experts per layer). This arm CHANGES the math when it fires: judged on decodeTokensPerSec against the stock pair references, the skip telemetry (moeTailSkipBatches / moeTailSkipSkipped / moeTailSkipBytesNotReadMiB), Case 3 answer fluency (token IDs are expected to differ from the stock sequence), and moeTailSkipProbeJsonValid, which must be true. An A/B check only; it says nothing about the other cases.":"quick_dspark_pin".equals(mode)?"Quick mode, DSpark pin-prewarm A/B arm (build 24, item 2a): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog. Identical to the stock DSpark arm except LLAMA_MOE_PIN_PREWARM=1: the prewarm table's experts are eviction-protected in the slot runtime's victim scan, with a plain-LRU fallback if a whole window were protected, so an install is never refused. The runtime's admission log line (admission: pin_prewarm=on lfu_admit=off) is the engagement proof. Judged on decodeTokensPerSec, prefillMs as a secondary stage, answer fluency, and the installs/hits counters; generated token IDs must match the stock DSpark arm exactly. An A/B check only; it says nothing about the other cases.":"quick_dspark_lfu".equals(mode)?"Quick mode, DSpark LFU admission A/B arm (build 24, item 2b): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog. Identical to the stock DSpark arm except LLAMA_MOE_LFU_ADMIT=1: probationary admission - a miss always installs and computes, but lands in a probationary segment (2 of the layer's 32 window slots) unless its (layer, expert) demand count, aged by halving every 1,024 demand events, beats the weakest general resident's count; victim selection prefers probationary slots, then the lowest-count general resident, LRU breaks ties; a probationary expert demanded again while resident is promoted. The runtime's admission log line (admission: pin_prewarm=off lfu_admit=on) is the engagement proof. Judged on decodeTokensPerSec, prefillMs as a secondary stage, answer fluency, and the installs/hits counters; generated token IDs must match the stock DSpark arm exactly. An A/B check only; it says nothing about the other cases.":"quick_dspark_pinlfu".equals(mode)?"Quick mode, DSpark pin + LFU A/B arm (build 24, item 2 combined): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog. Identical to the stock DSpark arm except LLAMA_MOE_PIN_PREWARM=1 and LLAMA_MOE_LFU_ADMIT=1 together: the prewarm experts are eviction-protected AND new installs go through probationary LFU admission (see the pin and LFU arms' notes). The runtime's admission log line (admission: pin_prewarm=on lfu_admit=on) is the engagement proof. Judged on decodeTokensPerSec, prefillMs as a secondary stage, answer fluency, and the installs/hits counters; generated token IDs must match the stock DSpark arm exactly. An A/B check only; it says nothing about the other cases.":"quick_dspark_io8".equals(mode)?"Quick mode, DSpark I/O threads A/B arm (build 23): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog. Identical to the stock DSpark arm except LLAMA_MOE_IO_THREADS=8 instead of the fork default 4, widening the expert install read parallelism. Kept from build 22 on Battles's direction: in build 22's device runs, where this arm ran stacked on the since-removed per-layer window configuration, its two prefills were the fastest of the session, about 29 percent faster than the stock arm's, while its decode showed no proven gain; build 23 runs it on the stock configuration, so the combination is untested until a device run measures it. The runtime's init log line carries the engaged io_threads value. Judged on prefillMs, decodeTokensPerSec, answer fluency, and the installs/hits counters. An A/B speed check only; it says nothing about the other cases.":"quick_dspark".equals(mode)?"Quick mode, DSpark A/B arm (build 20): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog. Generation is speculative: the model's DSpark draft heads propose 3 tokens per round and the target verifies them in one batch with greedy acceptance, so every emitted token is the target's own pick; the ids need not match the plain greedy arms token for token (the runtime does not guarantee identical logits across batch sizes), and the arm is judged on answer fluency and decode speed. Lookahead prefetch OFF (the DSpark baseline is the lookahead OFF arm), decode window 32 with 4 verify rows in window mode; prompt, threads, context, slot bank and prewarm identical to the other quick checks. An A/B decode-speed check only; it says nothing about the other cases.":"quick_off".equals(mode)?"Quick mode, A/B lookahead OFF arm (build 14): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog, lookahead prefetch OFF via the runtime's own LLAMA_MOE_LOOKAHEAD_PREFETCH=0; every other setting identical to the normal quick decode check. An A/B decode-speed check only; it says nothing about the other cases.":"quick".equals(mode)?"Quick mode (build 12): Case 3 only (the long bicycle text question), 32-token decode cap, 10-minute watchdog. A decode-speed check only; it says nothing about the other cases. Same model pack and CPU path as the full trial.":"Three text-only questions, then a single photo with three fixed questions, greedy sampling, thinking disabled. CPU path only (vendor demo defaults: 4 threads, 4096 context). First run on this device class; no prior Android LES result exists to compare against.");
            write(new File(getFilesDir(),"report.json"),report.toString(2));new File(getFilesDir(),"report-partial.json").delete();state(verdict,verdict);
        }catch(Throwable error){state(error.getClass().getSimpleName()+": "+error.getMessage(),"ERROR");}
        finally{retire();}
    }
    // Build 26: retire() is the one funnel every terminal path
    // (completed, error, user stop, watchdog timeout) passes
    // through before the process is killed, so the export bundle
    // is written here, synchronously and at most once per
    // process. A run whose process dies outright leaves its
    // on-disk state for the manual Export button instead.
    private void retire(){try{RunExporter.exportOnce(this);}catch(Exception ignored){}timers.shutdownNow();if(wake!=null&&wake.isHeld())wake.release();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf();android.os.Process.killProcess(android.os.Process.myPid());}
}
