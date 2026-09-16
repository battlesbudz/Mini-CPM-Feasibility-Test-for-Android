package com.battlesbudz.liquidtest;
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
        TextView title=new TextView(this);title.setText("Liquid Voice Feasibility Test\n"+BuildConfig.VERSION_NAME);title.setTextSize(24);column.addView(title);
        TextView scope=new TextView(this);scope.setText("CPU / Vulkan voice benchmark • Fully local after setup\n\nThis tests recorded audio through the complete Liquid LFM2.5-Audio-1.5B voice stack. It does not yet test live interruption or tools. A CPU failure does not rule out GPU acceleration.\n");column.addView(scope);
        button(column,"Download voice pack · 1.07 GB",()->new AlertDialog.Builder(this).setTitle("Download model files?").setMessage("Downloads 1.07 GB from Hugging Face. Allow at least 2 GB free storage. Keep this screen open during setup; interrupted downloads can resume. Benchmarking then works offline.").setPositiveButton("Download",(d,w)->install(null)).setNegativeButton("Cancel",null).show());
        button(column,"Import existing model folder",()->startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION),10));
        button(column,"Record a question · 10 seconds",()->{if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},11);else record();});
        duration=new Spinner(this);duration.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"CPU · 2 threads","CPU · 4 threads (baseline)","CPU · 6 threads","Vulkan · main model","Vulkan · main + audio","CPU vs Vulkan · correctness (no speech)"}));column.addView(duration);duration.setSelection(1);
        voice=new Spinner(this);voice.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"Default conversation voice","UK male conversation · experimental","UK male TTS · fixed sample"}));column.addView(voice);
        button(column,"Run selected benchmark",this::startRun);
        Button stop=new Button(this);stop.setText("Stop current operation");column.addView(stop);stop.setOnClickListener(v->{if(task!=null)task.cancel(true);if(workerAlive())startService(new Intent(this,BenchmarkService.class).setAction("STOP"));local="Stop requested.";refresh();});
        button(column,"Play generated speech",this::play);
        Button copy=new Button(this);copy.setText("Copy diagnostics");column.addView(copy);copy.setOnClickListener(v->{getSystemService(ClipboardManager.class).setPrimaryClip(ClipData.newPlainText("Liquid Voice diagnostics",diagnostics()));Toast.makeText(this,"Diagnostics copied",Toast.LENGTH_SHORT).show();});
        button(column,"Save native crash trace",()->startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("application/octet-stream").addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_TITLE,"liquid-native-crash.pb"),13));
        button(column,"Delete recording and test output",()->{delete(new File(getCacheDir(),"input"));delete(new File(getFilesDir(),"last-run"));new File(getFilesDir(),"report.json").delete();new File(getFilesDir(),"status.json").delete();new File(getFilesDir(),"comparison.jsonl").delete();local="Recording and test output deleted. Model pack retained.";refresh();});
        status=new TextView(this);status.setTextIsSelectable(true);status.setPadding(0,24,0,24);column.addView(status);ui.post(poll);
        if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},12);
    }
    private void button(LinearLayout parent,String text,Runnable action){Button b=new Button(this);b.setText(text);parent.addView(b);controls.add(b);b.setOnClickListener(v->{if(!busy&&!workerAlive())action.run();});}
    private interface Job {void run()throws Exception;}
    private void launch(Job job){busy=true;refresh();task=work.submit(()->{try{job.run();}catch(Exception e){local=e.getClass().getSimpleName()+": "+e.getMessage();}finally{busy=false;if(!destroyed)ui.post(this::refresh);}});}
    private void install(Uri folder){launch(()->new ModelStore(this).install(folder,message->{local=message;if(!destroyed)ui.post(this::refresh);}));}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==13&&result==RESULT_OK&&data!=null&&data.getData()!=null){Uri destination=data.getData();launch(()->saveCrashTrace(destination));return;}if(request==10&&result==RESULT_OK&&data!=null){Uri uri=data.getData();if(uri!=null){getContentResolver().takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION);install(uri);}}}
    @Override public void onRequestPermissionsResult(int request,String[] names,int[] grants){super.onRequestPermissionsResult(request,names,grants);if(request==11&&grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED)record();}
    private void record(){launch(()->{
        if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED)throw new IOException("Microphone permission required");
        File folder=new File(getCacheDir(),"input");delete(folder);folder.mkdirs();
        int size=AudioRecord.getMinBufferSize(16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);if(size<=0)throw new IOException("16 kHz recording unsupported");
        AudioRecord recorder=new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT,Math.max(size,6400));
        short[] recording=new short[160000];int at=0;
        try{
            if(recorder.getState()!=AudioRecord.STATE_INITIALIZED)throw new IOException("Microphone unavailable");recorder.startRecording();
            while(at<recording.length){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("Recording cancelled");local="Speak now: "+(10-at/16000)+" seconds remaining. Ask a question that invites an answer.";int n=recorder.read(recording,at,Math.min(1600,recording.length-at));if(n<=0)throw new IOException("Microphone read failed: "+n);at+=n;}
        }finally{if(recorder.getRecordingState()==AudioRecord.RECORDSTATE_RECORDING)recorder.stop();recorder.release();}
        for(int i=0;i<30;i++){
            ByteBuffer wav=ByteBuffer.allocate(32044).order(ByteOrder.LITTLE_ENDIAN);
            wav.put(new byte[]{'R','I','F','F'}).putInt(32036).put(new byte[]{'W','A','V','E','f','m','t',' '}).putInt(16).putShort((short)1).putShort((short)1).putInt(16000).putInt(32000).putShort((short)2).putShort((short)16).put(new byte[]{'d','a','t','a'}).putInt(32000);
            for(int j=0;j<16000;j++)wav.putShort(i<10?recording[i*16000+j]:0);
            Files.write(new File(folder,String.format(Locale.ROOT,"%04d.wav",i)).toPath(),wav.array());
        }
        ByteBuffer full=ByteBuffer.allocate(320044).order(ByteOrder.LITTLE_ENDIAN);
        byte[] header=Files.readAllBytes(new File(folder,"0000.wav").toPath());full.put(header,0,44);full.putInt(4,320036);full.putInt(40,320000);for(short x:recording)full.putShort(x);Files.write(new File(folder,"question.wav").toPath(),full.array());
        local="Question recorded. The test uses your 10-second question once. Recording stays on this device until deleted.";
    });}
    private void startRun(){
        try{if(duration.getSelectedItemPosition()!=5&&!new ModelStore(this).ready())throw new IOException("Install and verify the complete voice pack first.");if(duration.getSelectedItemPosition()!=5&&voice.getSelectedItemPosition()!=2&&!new File(getCacheDir(),"input/question.wav").isFile())throw new IOException("Record a question first.");
            new File(getFilesDir(),"report.json").delete();new File(getFilesDir(),"status.json").delete();delete(new File(getFilesDir(),"last-run"));
            startForegroundService(new Intent(this,BenchmarkService.class).putExtra("profile",duration.getSelectedItemPosition()).putExtra("voice",voice.getSelectedItemPosition()));local="Starting benchmark worker…";
        }catch(Exception e){local=e.getMessage();Toast.makeText(this,local,Toast.LENGTH_LONG).show();}refresh();
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
        return "Liquid Voice Feasibility "+BuildConfig.VERSION_NAME+"\n"+Build.MANUFACTURER+" "+Build.MODEL+"\n"+local+"\n"+s+"\n"+r+"\n"+failure+"\nLast persisted memory sample:\n"+read("last-run/memory.json")+"\nSaved partial output:\n"+read("last-run/partial.json")+"\nComparison history:\n"+read("comparison.jsonl")+exit;
    }
    private void refresh(){if(destroyed||status==null)return;boolean running=workerAlive();for(Button b:controls)b.setEnabled(!busy&&!running);duration.setEnabled(!busy&&!running);voice.setEnabled(!busy&&!running);String text=local;
        if(!busy)try{String raw=read("status.json");if(!raw.isEmpty()){JSONObject s=reconciledState();text=s.optString("status")+"\n"+s.optString("phase");if(s.optString("status").equals("RUNNING")&&!running)text="WORKER EXITED — test did not finish. Copy diagnostics for the exit reason.";}
            String memory=read("last-run/memory.json");if(!memory.isEmpty()){JSONObject m=new JSONObject(memory);text+="\nLast sampled worker memory: "+m.optInt("workerPssKb")/1024+" MB; peak: "+m.optInt("peakWorkerPssKb")/1024+" MB\nSystem available: "+m.optLong("systemAvailableBytes")/1048576+" MB";}
            String rawReport=read("report.json");if(!rawReport.isEmpty()){JSONObject r=new JSONObject(rawReport);text+="\n\n"+r.optString("verdict")+"\nFirst PCM after submission: "+r.opt("firstPcmMs")+" ms\nGeneration: "+r.opt("generationMs")+" ms\nWorker peak PSS: "+r.optInt("peakWorkerPssKb")/1024+" MB\nAnswer: "+r.optString("text")+"\n"+r.optString("error");}
        }catch(Exception e){text+="\nReading diagnostics: "+e.getMessage();}status.setText(text);
    }
    private void play(){launch(()->{
        File pcm=new File(getFilesDir(),"last-run/generated.pcm");if(!pcm.isFile())throw new IOException("No generated speech yet");String metadata=read("report.json");if(metadata.isEmpty())metadata=read("last-run/partial.json");JSONObject report=metadata.isEmpty()?new JSONObject():new JSONObject(metadata);int rate=report.optInt("sampleRate",24000);
        int buffer=AudioTrack.getMinBufferSize(rate,AudioFormat.CHANNEL_OUT_MONO,AudioFormat.ENCODING_PCM_16BIT);if(buffer<=0)throw new IOException("Unsupported output sample rate");
        AudioTrack player=new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()).setAudioFormat(new AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_16BIT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build()).setBufferSizeInBytes(Math.max(buffer,8192)).setTransferMode(AudioTrack.MODE_STREAM).build();local="Playing generated speech.";
        try(InputStream in=new FileInputStream(pcm)){player.play();byte[] b=new byte[4096];int n;long written=0;while((n=in.read(b))!=-1){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();int off=0;while(off<n){int count=player.write(b,off,n-off);if(count<=0)throw new IOException("Playback failed");off+=count;written+=count;}}
            while(Integer.toUnsignedLong(player.getPlaybackHeadPosition())<written/2){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();Thread.sleep(20);}
        }finally{player.stop();player.release();}local="Playback finished.";
    });}
    private static void delete(File f){if(f.isDirectory()){File[] children=f.listFiles();if(children!=null)for(File child:children)delete(child);}f.delete();}
    @Override protected void onDestroy(){destroyed=true;ui.removeCallbacks(poll);work.shutdownNow();super.onDestroy();}
}
