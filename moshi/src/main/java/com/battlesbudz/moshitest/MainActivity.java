package com.battlesbudz.moshitest;
import android.Manifest;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.media.*;
import android.net.Uri;
import android.os.*;
import android.widget.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.json.*;
public final class MainActivity extends Activity {
    private final Handler ui=new Handler(Looper.getMainLooper());
    private final ExecutorService work=Executors.newSingleThreadExecutor();
    private volatile boolean busy,destroyed;
    private long startingUntil;
    private Future<?> task;
    private TextView status;
    private Spinner duration,voice;
    private final ArrayList<Button> controls=new ArrayList<>();
    private volatile String local="Download or import the voice pack, then record a test question.";
    private final Runnable poll=new Runnable(){public void run(){refresh();ui.postDelayed(this,1000);}};
    @Override public void onCreate(Bundle saved){
        super.onCreate(saved);
        LinearLayout column=new LinearLayout(this);column.setOrientation(LinearLayout.VERTICAL);column.setPadding(32,24,32,24);
        ScrollView scroll=new ScrollView(this);scroll.setFitsSystemWindows(true);scroll.addView(column);setContentView(scroll);
        TextView title=new TextView(this);title.setText("Moshi Voice Feasibility Test\n"+BuildConfig.VERSION_NAME);title.setTextSize(24);column.addView(title);
        TextView scope=new TextView(this);scope.setText("CPU / Vulkan voice benchmark • Fully local after setup\n\nMoshi / Moshika Q4_K experimental voice engine. Trace CPU/GPU operations checks the first audio frame and model uploads using the existing recording and Mimi download. Allow a few minutes. The reference model has a female voice. This does not yet test live conversation or echo cancellation.\n");column.addView(scope);
        button(column,"Download Mimi codec only · 347 MB",()->install(null,true));
        button(column,"Download full Moshi pack · 4.68 GB",()->new AlertDialog.Builder(this).setTitle("Download model files?").setMessage("Downloads 4.68 GB from Hugging Face. Allow at least 6 GB free storage. Keep this screen open during setup; interrupted downloads can resume. Benchmarking then works offline.").setPositiveButton("Download",(d,w)->install(null,false)).setNegativeButton("Cancel",null).show());
        button(column,"Import existing model folder",()->startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION),10));
        button(column,"Record a question · 10 seconds",()->{if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},11);else record();});
        duration=new Spinner(this);duration.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"CPU · 4 threads","Vulkan model · CPU codec","Vulkan model + codec"}));column.addView(duration);duration.setSelection(0);
        voice=new Spinner(this);voice.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"1. Mimi codec replay · 10 sec","2. Moshi load-only · 750-frame context","3. Moshi voice replay · 20 sec","4. Compare CPU/GPU codec · all four routes","5. Trace CPU/GPU operations · first frame"}));column.addView(voice);
        voice.setSelection(3);
        voice.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener(){
            public void onItemSelected(android.widget.AdapterView<?> p,android.view.View v,int position,long id){refresh();}
            public void onNothingSelected(android.widget.AdapterView<?> p){}
        });
        button(column,"Run selected benchmark",this::startRun);
        Button stop=new Button(this);stop.setText("Stop current operation");column.addView(stop);stop.setOnClickListener(v->{if(task!=null)task.cancel(true);if(workerAlive())startService(new Intent(this,BenchmarkService.class).setAction("STOP"));local="Stop requested.";refresh();});
        button(column,"Play result / CPU reference",()->play("generated.pcm"));
        button(column,"Play GPU encoder → CPU decoder",()->play("gpu_encode_cpu_decode.pcm"));
        button(column,"Play CPU encoder → GPU decoder",()->play("cpu_encode_gpu_decode.pcm"));
        button(column,"Play GPU encoder → GPU decoder",()->play("gpu_encode_gpu_decode.pcm"));
        Button copy=new Button(this);copy.setText("Copy diagnostics");column.addView(copy);copy.setOnClickListener(v->{getSystemService(ClipboardManager.class).setPrimaryClip(ClipData.newPlainText("Moshi Voice diagnostics",diagnostics()));Toast.makeText(this,"Diagnostics copied",Toast.LENGTH_SHORT).show();});
        button(column,"Save native crash trace",()->startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("application/octet-stream").addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_TITLE,"moshi-native-crash.pb"),13));
        button(column,"Export diagnostics + test audio ZIP",()->startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("application/zip").addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_TITLE,"moshi-test.zip"),14));
        button(column,"Delete downloaded models",()->new AlertDialog.Builder(this).setTitle("Delete Moshi model downloads?").setPositiveButton("Delete",(d,w)->launch(()->{delete(new File(getFilesDir(),"models"));local="Model files deleted.";})).setNegativeButton("Cancel",null).show());
        button(column,"Delete recording and test output",()->{delete(new File(getCacheDir(),"input"));delete(new File(getFilesDir(),"last-run"));new File(getFilesDir(),"report.json").delete();new File(getFilesDir(),"status.json").delete();new File(getFilesDir(),"comparison.jsonl").delete();local="Recording and test output deleted. Model pack retained.";refresh();});
        status=new TextView(this);status.setTextIsSelectable(true);status.setPadding(0,24,0,24);column.addView(status);ui.post(poll);
        if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},12);
    }
    private void button(LinearLayout parent,String text,Runnable action){Button b=new Button(this);b.setText(text);parent.addView(b);controls.add(b);b.setOnClickListener(v->{if(!busy&&!workerAlive()&&SystemClock.elapsedRealtime()>=startingUntil)action.run();});}
    private interface Job {void run()throws Exception;}
    private void launch(Job job){busy=true;refresh();task=work.submit(()->{try{job.run();}catch(Exception e){local=e.getClass().getSimpleName()+": "+e.getMessage();}finally{busy=false;if(!destroyed)ui.post(this::refresh);}});}
    private void install(Uri folder,boolean codecOnly){launch(()->new ModelStore(this,codecOnly).install(folder,message->{local=message;if(!destroyed)ui.post(this::refresh);}));}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==14&&result==RESULT_OK&&data!=null&&data.getData()!=null){Uri destination=data.getData();launch(()->exportZip(destination));return;}if(request==13&&result==RESULT_OK&&data!=null&&data.getData()!=null){Uri destination=data.getData();launch(()->saveCrashTrace(destination));return;}if(request==10&&result==RESULT_OK&&data!=null){Uri uri=data.getData();if(uri!=null){getContentResolver().takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION);install(uri,false);}}}
    @Override public void onRequestPermissionsResult(int request,String[] names,int[] grants){super.onRequestPermissionsResult(request,names,grants);if(request==11&&grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED)record();}
    private void record(){launch(()->{
        if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED)throw new IOException("Microphone permission required");
        File folder=new File(getCacheDir(),"input");delete(folder);folder.mkdirs();
        int rate=24000;
        int size=AudioRecord.getMinBufferSize(rate,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
        if(size<=0){rate=48000;size=AudioRecord.getMinBufferSize(rate,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);}
        if(size<=0)throw new IOException("24/48 kHz recording unavailable");
        AudioRecord recorder=new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,rate,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT,Math.max(size,rate/5*2));
        short[] recording=new short[rate*10];int at=0;
        try{
            if(recorder.getState()!=AudioRecord.STATE_INITIALIZED)throw new IOException("Microphone unavailable");recorder.startRecording();
            while(at<recording.length){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("Recording cancelled");local="Speak now: "+(10-at/rate)+" seconds remaining.";int n=recorder.read(recording,at,Math.min(rate/10,recording.length-at));if(n<=0)throw new IOException("Microphone read failed: "+n);at+=n;}
        }finally{if(recorder.getRecordingState()==AudioRecord.RECORDSTATE_RECORDING)recorder.stop();recorder.release();}
        ByteBuffer full=ByteBuffer.allocate(480044).order(ByteOrder.LITTLE_ENDIAN);
        full.put(new byte[]{'R','I','F','F'}).putInt(480036).put(new byte[]{'W','A','V','E','f','m','t',' '}).putInt(16).putShort((short)1).putShort((short)1).putInt(24000).putInt(48000).putShort((short)2).putShort((short)16).put(new byte[]{'d','a','t','a'}).putInt(480000);
        for(int i=0;i<240000;i++)full.putShort(rate==24000?recording[i]:(short)((recording[i*2]+recording[i*2+1])/2));
        Files.write(new File(folder,"question.wav").toPath(),full.array());
        local="Question recorded. The test uses your 10-second question once. Recording stays on this device until deleted.";
    });}
    private void startRun(){
        try{if(!new ModelStore(this,(voice.getSelectedItemPosition()==0||voice.getSelectedItemPosition()>=3)).ready())throw new IOException("Install and verify the complete voice pack first.");if(voice.getSelectedItemPosition()!=1&&!new File(getCacheDir(),"input/question.wav").isFile())throw new IOException("Record a question first.");
            new File(getFilesDir(),"report.json").delete();new File(getFilesDir(),"status.json").delete();delete(new File(getFilesDir(),"last-run"));
            startingUntil=SystemClock.elapsedRealtime()+3000;startForegroundService(new Intent(this,BenchmarkService.class).putExtra("profile",duration.getSelectedItemPosition()).putExtra("voice",voice.getSelectedItemPosition()));local="Starting benchmark worker…";
        }catch(Exception e){startingUntil=0;local=e.getMessage();Toast.makeText(this,local,Toast.LENGTH_LONG).show();}refresh();
    }
    private boolean workerAlive(){var list=getSystemService(ActivityManager.class).getRunningAppProcesses();if(list!=null)for(var p:list)if(p.uid==android.os.Process.myUid()&&p.processName.equals(getPackageName()+":benchmark"))return true;return false;}
    private String read(String name){try{return Io.read(new File(getFilesDir(),name).toPath());}catch(IOException e){return "";}}
    private JSONObject reconciledState()throws JSONException{
        String raw=read("status.json");JSONObject saved=raw.isEmpty()?new JSONObject():new JSONObject(raw);
        if(!"RUNNING".equals(saved.optString("status"))||workerAlive())return saved;
        if(Build.VERSION.SDK_INT>=30){
            int pid=saved.optInt("pid",-1);long began=saved.optLong("startedAtMs",saved.optLong("updatedAtMs"));
            if(pid>0)for(var e:getSystemService(ActivityManager.class).getHistoricalProcessExitReasons(getPackageName(),pid,16)){
                if(e.getProcessName().equals(getPackageName()+":benchmark")&&ExitMatch.matches(pid,began,e.getPid(),e.getTimestamp())){
                    saved.put("lastCheckpointPhase",saved.optString("phase")).put("exitReason",e.getReason()).put("exitStatus",e.getStatus())
                        .put("exitAtMs",e.getTimestamp()).put("exitPssKb",e.getPss()).put("exitRssKb",e.getRss());
                    boolean low=e.getReason()==ApplicationExitInfo.REASON_LOW_MEMORY;
                    if(e.getReason()==ApplicationExitInfo.REASON_CRASH_NATIVE)saved.put("nativeCrash",true).put("crashTraceInstructions","Save native crash trace to export Android’s binary tombstone, if available.");
                    saved.put("status",low?"KILLED_LOW_MEMORY":"WORKER_EXITED")
                        .put("phase",low?"Android killed the worker under memory pressure. Test did not finish.":"Worker exited before completing the test.");
                    return saved;
                }
            }
        }
        // Exit history can arrive after process removal. Do not persist a guessed reason.
        return saved.put("status","WORKER_NOT_RUNNING").put("lastCheckpointPhase",saved.optString("phase"))
            .put("phase","Worker not running; waiting for Android exit details.");
    }
    private void saveCrashTrace(Uri destination)throws Exception{
        if(Build.VERSION.SDK_INT<31)throw new IOException("Native crash traces require Android 12 or newer.");
        JSONObject saved=new JSONObject(read("status.json"));int pid=saved.optInt("pid",-1);
        long began=saved.optLong("startedAtMs",saved.optLong("updatedAtMs"));
        if(pid<=0)throw new IOException("No benchmark process recorded.");
        for(var e:getSystemService(ActivityManager.class).getHistoricalProcessExitReasons(getPackageName(),pid,16)){
            if(!e.getProcessName().equals(getPackageName()+":benchmark")||!ExitMatch.matches(pid,began,e.getPid(),e.getTimestamp())||e.getReason()!=ApplicationExitInfo.REASON_CRASH_NATIVE)continue;
            try(InputStream in=e.getTraceInputStream()){
                if(in==null)throw new IOException("Android has not provided a trace for this crash. Try again shortly.");
                try(OutputStream out=getContentResolver().openOutputStream(destination,"wt")){
                    if(out==null)throw new IOException("Cannot open crash trace destination.");
                    byte[] bytes=new byte[8192];int n;while((n=in.read(bytes))!=-1){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("Export cancelled");out.write(bytes,0,n);}
                }
            }
            local="Native crash trace saved. Attach the .pb file with the copied diagnostics.";return;
        }
        throw new IOException("No native crash found for the current benchmark.");
    }
    private String diagnostics(){String s;try{s=reconciledState().toString(2);}catch(JSONException error){s=read("status.json");}String r=read("report.json"),failure=read("last-run/native.log"),exit="";if(failure.length()>10000)failure=failure.substring(failure.length()-10000);
        if(Build.VERSION.SDK_INT>=30){var list=getSystemService(ActivityManager.class).getHistoricalProcessExitReasons(getPackageName(),0,4);for(var e:list)if(e.getProcessName().endsWith(":benchmark"))exit+="\nWorker exit: pid="+e.getPid()+" pssKb="+e.getPss()+" rssKb="+e.getRss()+" reason="+e.getReason()+" status="+e.getStatus()+" time="+e.getTimestamp()+" description="+e.getDescription();}
        return "Moshi Voice Feasibility "+BuildConfig.VERSION_NAME+"\n"+Build.MANUFACTURER+" "+Build.MODEL+"\n"+local+"\n"+s+"\n"+r+"\n"+failure+"\nLast persisted memory sample:\n"+read("last-run/memory.json")+"\nSaved partial output:\n"+read("last-run/partial.json")+"\nComparison history:\n"+read("comparison.jsonl")+exit;
    }
    private void refresh(){if(destroyed||status==null)return;boolean running=workerAlive()||SystemClock.elapsedRealtime()<startingUntil;
        if(busy)getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);else getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);for(Button b:controls)b.setEnabled(!busy&&!running);duration.setEnabled(!busy&&!running&&voice.getSelectedItemPosition()<3);voice.setEnabled(!busy&&!running);String text=local;
        if(!busy)try{String raw=read("status.json");if(!raw.isEmpty()){JSONObject s=reconciledState();text=s.optString("status")+"\n"+s.optString("phase");if(s.optString("status").equals("RUNNING")&&!running)text="WORKER EXITED — test did not finish. Copy diagnostics for the exit reason.";}
            String memory=read("last-run/memory.json");if(!memory.isEmpty()){JSONObject m=new JSONObject(memory);text+="\nLast sampled worker memory: "+m.optInt("workerPssKb")/1024+" MB; peak: "+m.optInt("peakWorkerPssKb")/1024+" MB\nSystem available: "+m.optLong("systemAvailableBytes")/1048576+" MB";}
            String rawReport=read("report.json");if(!rawReport.isEmpty()){JSONObject r=new JSONObject(rawReport);if(r.optInt("mode")==4){text+="\n\n"+r.optString("verdict")+"\n"+r.optString("summary")+"\nExport the diagnostic ZIP for tensor analysis.";}else if(r.optInt("mode")==3){text+="\n\n"+r.optString("verdict")+"\n"+r.optString("summary")+"\nToken agreement: "+String.format(Locale.US,"%.1f%%",100*r.optDouble("tokenAgreement",0))+"\nCompared frames: "+r.optInt("inputFrames");}else text+="\n\n"+r.optString("verdict")+"\nFirst PCM after submission: "+r.opt("firstPcmMs")+" ms\nFrame processing: "+r.opt("processingFps")+" fps (12.5 needed)\nFrame p95: "+r.opt("frameP95Ms")+" ms\nLoad: "+r.opt("loadMs")+" ms\nWorker peak PSS: "+r.optInt("peakWorkerPssKb")/1024+" MB\nAnswer: "+r.optString("text")+"\n"+r.optString("error");}
        }catch(Exception e){text+="\nReading diagnostics: "+e.getMessage();}status.setText(text);
    }
    private void play(String filename){launch(()->{
        File pcm=new File(getFilesDir(),"last-run/"+filename);if(!pcm.isFile()||pcm.length()==0)throw new IOException("No generated speech yet");String metadata=read("report.json");if(metadata.isEmpty())metadata=read("last-run/partial.json");JSONObject report=metadata.isEmpty()?new JSONObject():new JSONObject(metadata);int rate=report.optInt("sampleRate",24000);
        int buffer=AudioTrack.getMinBufferSize(rate,AudioFormat.CHANNEL_OUT_MONO,AudioFormat.ENCODING_PCM_16BIT);if(buffer<=0)throw new IOException("Unsupported output sample rate");
        AudioTrack player=new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()).setAudioFormat(new AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_16BIT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build()).setBufferSizeInBytes(Math.max(buffer,8192)).setTransferMode(AudioTrack.MODE_STREAM).build();local="Playing generated speech.";
        try(InputStream in=new FileInputStream(pcm)){player.play();byte[] b=new byte[4096];int n;long written=0;while((n=in.read(b))!=-1){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();int off=0;while(off<n){int count=player.write(b,off,n-off);if(count<=0)throw new IOException("Playback failed");off+=count;written+=count;}}
            while(Integer.toUnsignedLong(player.getPlaybackHeadPosition())<written/2){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();Thread.sleep(20);}
        }finally{player.stop();player.release();}local="Playback finished.";
    });}
    private void exportZip(Uri destination)throws Exception{
        try(OutputStream target=getContentResolver().openOutputStream(destination,"wt");java.util.zip.ZipOutputStream zip=new java.util.zip.ZipOutputStream(target)){
            zip.putNextEntry(new java.util.zip.ZipEntry("diagnostics.txt"));zip.write(diagnostics().getBytes(java.nio.charset.StandardCharsets.UTF_8));zip.closeEntry();
            File run=new File(getFilesDir(),"last-run");File[] children=run.listFiles();
            if(children!=null)for(File f:children)if(f.isFile()){zip.putNextEntry(new java.util.zip.ZipEntry("last-run/"+f.getName()));Files.copy(f.toPath(),zip);zip.closeEntry();}
            File question=new File(getCacheDir(),"input/question.wav");if(question.isFile()){zip.putNextEntry(new java.util.zip.ZipEntry("question.wav"));Files.copy(question.toPath(),zip);zip.closeEntry();}
        }local="Test audio and diagnostics exported.";
    }
    private static void delete(File f){if(f.isDirectory()){File[] children=f.listFiles();if(children!=null)for(File child:children)delete(child);}f.delete();}
    @Override protected void onDestroy(){destroyed=true;ui.removeCallbacks(poll);work.shutdownNow();super.onDestroy();}
}
