package com.battlesbudz.minicpmtest;
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
    private Spinner duration;
    private final ArrayList<Button> controls=new ArrayList<>();
    private volatile String local="Download or import the voice pack, then record a test question.";
    private final Runnable poll=new Runnable(){public void run(){refresh();ui.postDelayed(this,1000);}};
    @Override public void onCreate(Bundle saved){
        super.onCreate(saved);
        LinearLayout column=new LinearLayout(this);column.setOrientation(LinearLayout.VERTICAL);column.setPadding(32,24,32,24);
        ScrollView scroll=new ScrollView(this);scroll.setFitsSystemWindows(true);scroll.addView(column);setContentView(scroll);
        TextView title=new TextView(this);title.setText("MiniCPM Feasibility Test\n"+BuildConfig.VERSION_NAME);title.setTextSize(24);column.addView(title);
        TextView scope=new TextView(this);scope.setText("CPU voice benchmark • Fully local after setup\n\nThis tests recorded audio through the complete MiniCPM-o 4.5 voice stack. It does not yet test live interruption or tools. A CPU failure does not rule out GPU acceleration.\n");column.addView(scope);
        button(column,"Download voice pack · 7.78 GB",()->new AlertDialog.Builder(this).setTitle("Download model files?").setMessage("Downloads 7.78 GB from Hugging Face. Allow at least 9 GB free storage. Keep this screen open during setup; interrupted downloads can resume. Benchmarking then works offline.").setPositiveButton("Download",(d,w)->install(null)).setNegativeButton("Cancel",null).show());
        button(column,"Import existing model folder",()->startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION),10));
        button(column,"Record a question · 10 seconds",()->{if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},11);else record();});
        duration=new Spinner(this);duration.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"60-second smoke test","5-minute sustained test","30-minute thermal test"}));column.addView(duration);
        button(column,"Run offline CPU benchmark",this::startRun);
        Button stop=new Button(this);stop.setText("Stop current operation");column.addView(stop);stop.setOnClickListener(v->{if(task!=null)task.cancel(true);if(workerAlive())startService(new Intent(this,BenchmarkService.class).setAction("STOP"));local="Stop requested.";refresh();});
        button(column,"Play generated speech",this::play);
        Button copy=new Button(this);copy.setText("Copy diagnostics");column.addView(copy);copy.setOnClickListener(v->{getSystemService(ClipboardManager.class).setPrimaryClip(ClipData.newPlainText("MiniCPM diagnostics",diagnostics()));Toast.makeText(this,"Diagnostics copied",Toast.LENGTH_SHORT).show();});
        button(column,"Delete recording and test output",()->{delete(new File(getCacheDir(),"input"));delete(new File(getFilesDir(),"last-run"));new File(getFilesDir(),"report.json").delete();new File(getFilesDir(),"status.json").delete();local="Recording and test output deleted. Model pack retained.";refresh();});
        status=new TextView(this);status.setTextIsSelectable(true);status.setPadding(0,24,0,24);column.addView(status);ui.post(poll);
        if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},12);
    }
    private void button(LinearLayout parent,String text,Runnable action){Button b=new Button(this);b.setText(text);parent.addView(b);controls.add(b);b.setOnClickListener(v->{if(!busy&&!workerAlive())action.run();});}
    private interface Job {void run()throws Exception;}
    private void launch(Job job){busy=true;refresh();task=work.submit(()->{try{job.run();}catch(Exception e){local=e.getClass().getSimpleName()+": "+e.getMessage();}finally{busy=false;if(!destroyed)ui.post(this::refresh);}});}
    private void install(Uri folder){launch(()->new ModelStore(this).install(folder,message->{local=message;if(!destroyed)ui.post(this::refresh);}));}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==10&&result==RESULT_OK&&data!=null){Uri uri=data.getData();if(uri!=null){getContentResolver().takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION);install(uri);}}}
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
        local="Question recorded. Each test repeats your 10-second question followed by 20 seconds of silence. Recording stays on this device until deleted.";
    });}
    private void startRun(){
        try{if(!new ModelStore(this).ready())throw new IOException("Install and verify the complete voice pack first.");if(!new File(getCacheDir(),"input/0029.wav").isFile())throw new IOException("Record a question first.");
            new File(getFilesDir(),"report.json").delete();new File(getFilesDir(),"status.json").delete();delete(new File(getFilesDir(),"last-run"));
            startForegroundService(new Intent(this,BenchmarkService.class).putExtra("frames",new int[]{60,300,1800}[duration.getSelectedItemPosition()]));local="Starting benchmark worker…";
        }catch(Exception e){local=e.getMessage();}refresh();
    }
    private boolean workerAlive(){var list=getSystemService(ActivityManager.class).getRunningAppProcesses();if(list!=null)for(var p:list)if(p.uid==android.os.Process.myUid()&&p.processName.equals(getPackageName()+":benchmark"))return true;return false;}
    private String read(String name){try{return Io.read(new File(getFilesDir(),name).toPath());}catch(IOException e){return "";}}
    private String diagnostics(){String s=read("status.json"),r=read("report.json"),failure=read("last-run/native_failure.json"),exit="";
        if(Build.VERSION.SDK_INT>=30){var list=getSystemService(ActivityManager.class).getHistoricalProcessExitReasons(getPackageName(),0,4);for(var e:list)if(e.getProcessName().endsWith(":benchmark"))exit+="\nWorker exit: reason="+e.getReason()+" status="+e.getStatus()+" time="+e.getTimestamp()+" description="+e.getDescription();}
        return "MiniCPM Feasibility "+BuildConfig.VERSION_NAME+"\n"+Build.MANUFACTURER+" "+Build.MODEL+"\n"+local+"\n"+s+"\n"+r+"\n"+failure+exit;
    }
    private void refresh(){if(destroyed||status==null)return;boolean running=workerAlive();for(Button b:controls)b.setEnabled(!busy&&!running);duration.setEnabled(!busy&&!running);String text=local;
        if(!busy)try{String raw=read("status.json");if(!raw.isEmpty()){JSONObject s=new JSONObject(raw);text=s.optString("status")+"\n"+s.optString("phase");if(s.optString("status").equals("RUNNING")&&!running)text="WORKER EXITED — test did not finish. Copy diagnostics for the exit reason.";}
            String rawReport=read("report.json");if(!rawReport.isEmpty()){JSONObject r=new JSONObject(rawReport);text+="\n\n"+r.optString("verdict")+"\nDecision p95: "+r.opt("decisionP95Ms")+" ms\nPCM supply gaps: "+r.opt("estimatedPcmSupplyGapMs")+" ms\nWorker peak PSS: "+r.optInt("peakWorkerPssKb")/1024+" MB\nPeak thermal status: "+r.optInt("peakThermalStatus")+"\n"+r.optString("error");}
        }catch(Exception e){text+="\nReading diagnostics: "+e.getMessage();}status.setText(text);
    }
    private void play(){launch(()->{
        File pcm=new File(getFilesDir(),"last-run/generated.pcm");if(!pcm.isFile())throw new IOException("No generated speech yet");JSONObject report=new JSONObject(read("report.json"));int rate=report.optInt("sampleRate",24000);
        int buffer=AudioTrack.getMinBufferSize(rate,AudioFormat.CHANNEL_OUT_MONO,AudioFormat.ENCODING_PCM_16BIT);if(buffer<=0)throw new IOException("Unsupported output sample rate");
        AudioTrack player=new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()).setAudioFormat(new AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_16BIT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build()).setBufferSizeInBytes(Math.max(buffer,8192)).setTransferMode(AudioTrack.MODE_STREAM).build();local="Playing generated speech.";
        try(InputStream in=new FileInputStream(pcm)){player.play();byte[] b=new byte[4096];int n;long written=0;while((n=in.read(b))!=-1){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();int off=0;while(off<n){int count=player.write(b,off,n-off);if(count<=0)throw new IOException("Playback failed");off+=count;written+=count;}}
            while(Integer.toUnsignedLong(player.getPlaybackHeadPosition())<written/2){if(Thread.currentThread().isInterrupted())throw new InterruptedIOException();Thread.sleep(20);}
        }finally{player.stop();player.release();}local="Playback finished.";
    });}
    private static void delete(File f){if(f.isDirectory()){File[] children=f.listFiles();if(children!=null)for(File child:children)delete(child);}f.delete();}
    @Override protected void onDestroy(){destroyed=true;ui.removeCallbacks(poll);work.shutdownNow();super.onDestroy();}
}
