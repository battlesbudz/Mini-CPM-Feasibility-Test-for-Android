package com.battlesbudz.minicpmv47;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.os.*;
import android.widget.*;
import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;
import org.json.*;
public final class MainActivity extends Activity {
    private final Handler ui=new Handler(Looper.getMainLooper());
    private final ArrayList<Button> controls=new ArrayList<>();
    private TextView status;
    private String local="Download the model pack, then run the trial.";
    private boolean busy,destroyed;
    private AsyncTask<Void,String,String> task;
    @Override protected void onCreate(Bundle state){
        super.onCreate(state);
        if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{android.Manifest.permission.POST_NOTIFICATIONS},1);
        ScrollView scroll=new ScrollView(this);LinearLayout column=new LinearLayout(this);column.setOrientation(LinearLayout.VERTICAL);column.setPadding(32,32,32,32);scroll.addView(column);setContentView(scroll);
        TextView title=new TextView(this);title.setText("MiniCPM-V 4.7 phone trial");title.setTextSize(22);column.addView(title);
        TextView info=new TextView(this);info.setText("Tests whether MiniCPM-V 4.7 (35B) runs on this phone through OpenBMB's LES expert-streaming runtime. After the one-time 17.0 GB download everything runs offline. The trial asks three fixed text-only questions, then three fixed questions about one bundled street photo, and records load time, prefill and decode speed, memory and the answers. The quick decode check runs only the third question with a 32-token cap under a 10-minute limit, for a fast decode-speed verdict. The Lookahead OFF quick check repeats that exact run with lookahead prefetch switched off, so the two decode speeds can be compared head to head. The DSpark quick check repeats it again with speculative decoding: the model's DSpark draft heads propose tokens and the 35B verifies them in one batch, which is faster only if the guesses land; its answers are judged on fluency and speed. Every finished case is saved as it completes, so even a timed-out run keeps its per-case results.");column.addView(info);
        button(column,"Download model pack · 17.0 GB",()->new AlertDialog.Builder(this).setTitle("Download model files?").setMessage("Downloads 17.0 GB from Hugging Face (tc-mb/MiniCPM-V-Apps-gguf, plus the DSpark draft model from openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf). Allow at least 19 GB free storage and use Wi-Fi. Interrupted downloads can resume; every file is SHA-256 verified. If the pack is already installed, only files that are missing or changed download again.").setPositiveButton("Download",(d,w)->install()).setNegativeButton("Cancel",null).show());
        button(column,"Run phone trial",this::startTrial);
        button(column,"Quick decode check · Case 3 only",this::startQuick);
        button(column,"Quick decode check · Lookahead OFF (A/B)",this::startQuickOff);
        button(column,"Quick decode check · DSpark (A/B)",this::startQuickDspark);
        Button stop=new Button(this);stop.setText("Stop current operation");column.addView(stop);stop.setOnClickListener(v->{if(task!=null)task.cancel(true);if(workerAlive())startService(new Intent(this,BenchmarkService.class).setAction("STOP"));local="Stop requested.";refresh();});
        Button copy=new Button(this);copy.setText("Copy diagnostics");column.addView(copy);copy.setOnClickListener(v->{getSystemService(ClipboardManager.class).setPrimaryClip(ClipData.newPlainText("MiniCPM-V 4.7 diagnostics",diagnostics()));Toast.makeText(this,"Diagnostics copied",Toast.LENGTH_SHORT).show();});
        status=new TextView(this);status.setTextSize(13);column.addView(status);
        ui.postDelayed(new Runnable(){@Override public void run(){refresh();if(!destroyed)ui.postDelayed(this,2000);}},500);
        refresh();
    }
    private void button(LinearLayout parent,String text,Runnable action){Button b=new Button(this);b.setText(text);parent.addView(b);controls.add(b);b.setOnClickListener(v->{if(!busy&&!workerAlive())action.run();});}
    private void launch(AsyncTask<Void,String,String> work){busy=true;task=work;refresh();work.execute();}
    private void install(){
        launch(new AsyncTask<Void,String,String>(){
            @Override protected String doInBackground(Void...ignored){try{new ModelStore(MainActivity.this).install(message->publishProgress(message));return "Model pack verified. Ready to run the trial.";}catch(Throwable e){return "Setup failed: "+e.getMessage();}}
            @Override protected void onProgressUpdate(String...values){local=values[0];refresh();}
            @Override protected void onPostExecute(String result){busy=false;local=result;refresh();}
            @Override protected void onCancelled(){busy=false;local="Setup stopped. Partial downloads can resume.";refresh();}
        });
    }
    private void startTrial(){startRun("full","Starting trial worker…");}
    private void startQuick(){startRun("quick","Starting quick decode check…");}
    private void startQuickOff(){startRun("quick_off","Starting quick decode check (lookahead OFF, A/B)…");}
    private void startQuickDspark(){startRun("quick_dspark","Starting quick decode check (DSpark, A/B)…");}
    private void startRun(String mode,String starting){
        try{
            if(!new ModelStore(this).ready()){local="Install and verify the complete model pack first.";refresh();return;}
            startForegroundService(new Intent(this,BenchmarkService.class).putExtra("mode",mode));local=starting;
        }catch(Exception e){local="Cannot start: "+e.getMessage();}
        refresh();
    }
    private boolean workerAlive(){
        try{JSONObject s=new JSONObject(Io.read(new File(getFilesDir(),"status.json").toPath()));return "RUNNING".equals(s.optString("status"))&&System.currentTimeMillis()-s.optLong("updatedAtMs")<20000;}
        catch(Exception e){return false;}
    }
    private String fileText(String name){try{return Io.read(new File(getFilesDir(),name).toPath());}catch(Exception e){return "";}}
    private String diagnostics(){
        StringBuilder out=new StringBuilder();
        out.append("Build: ").append(BuildConfig.VERSION_NAME).append(" · Source: ").append(BuildConfig.SOURCE_COMMIT).append("\n");
        out.append("Device: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL).append(" · SDK ").append(Build.VERSION.SDK_INT).append("\n\n");
        String s=fileText("status.json");if(!s.isEmpty())out.append("STATUS\n").append(s).append("\n\n");
        String r=fileText("report.json");if(!r.isEmpty())out.append("REPORT\n").append(r).append("\n\n");
        String p=fileText("report-partial.json");if(!p.isEmpty())out.append("PARTIAL REPORT (latest run did not finish; per-case results saved so far)\n").append(p).append("\n\n");
        String m=fileText("last-run/memory.json");if(!m.isEmpty())out.append("MEMORY\n").append(m);
        return out.toString();
    }
    private void refresh(){
        if(destroyed||status==null)return;boolean running=workerAlive();
        for(Button b:controls)b.setEnabled(!busy&&!running);
        StringBuilder text=new StringBuilder(local);
        String s=fileText("status.json");
        if(!s.isEmpty()){try{JSONObject o=new JSONObject(s);text.append("\n\nPhase: ").append(o.optString("phase")).append("\nStatus: ").append(o.optString("status"));}catch(Exception ignored){}}
        String r=fileText("report.json");
        if(r.isEmpty())r=fileText("report-partial.json");
        if(!r.isEmpty()){try{JSONObject o=new JSONObject(r);if(o.has("verdict"))text.append("\n\nVerdict: ").append(o.optString("verdict"));else if(o.optBoolean("partial"))text.append("\n\nPartial results from an unfinished run:");JSONArray cases=o.optJSONArray("cases");if(cases!=null)for(int i=0;i<cases.length();i++){JSONObject c=cases.getJSONObject(i);text.append("\n\nQ").append(c.optInt("index")).append(": ").append(c.optString("prompt")).append("\n").append(c.optInt("generatedTokens")).append(" tokens · ").append(String.format(java.util.Locale.US,"%.2f",c.optDouble("decodeTokensPerSec"))).append(" tok/s\n").append(c.optString("answer"));}}catch(Exception ignored){}}
        status.setText(text.toString());
    }
    @Override protected void onDestroy(){destroyed=true;if(task!=null)task.cancel(true);super.onDestroy();}
}
