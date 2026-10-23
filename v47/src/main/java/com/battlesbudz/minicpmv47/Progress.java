package com.battlesbudz.minicpmv47;
import org.json.*;
// Live stage checkpoints from the native benchmark (build 9). The native
// run is one blocking JNI call; inside a case it now emits
// "PROGRESS {json}" events as stages start and finish and as decode
// produces tokens. Kept free of Android types so the parse/merge behavior
// stays unit-testable on the JVM, like TrialVerdict.
final class Progress {
    static final String PREFIX="PROGRESS ";
    private Progress(){}
    // Returns the checkpoint payload, or null for a plain phase event or a
    // malformed payload (a bad checkpoint must never break the run).
    static JSONObject parse(String message){
        if(message==null||!message.startsWith(PREFIX))return null;
        try{return new JSONObject(message.substring(PREFIX.length()));}
        catch(Exception error){return null;}
    }
    // Readable phase text for a checkpoint: the native label when present,
    // otherwise the stage name ("Case 2: vision encode").
    static String label(JSONObject progress){
        String label=progress.optString("label","");
        if(!label.isEmpty())return label;
        String stage=progress.optString("stage","working").replace('_',' ');
        int kase=progress.optInt("caseIndex",0);
        return kase>0?"Case "+kase+": "+stage:stage;
    }
    // Merges one checkpoint into the status object: every payload field is
    // copied across (the event kind lands under stageEvent; the label is
    // surfaced as the phase text, not a field), and the wall-clock stamps
    // are set here so status.json always says when the current stage began
    // (stageStartedAtMs) and when the last sign of life arrived
    // (lastProgressAtMs) — the two numbers that separate a hang from a
    // crawl when a run stalls.
    static void merge(JSONObject state,JSONObject progress,long nowMs)throws JSONException{
        JSONArray names=progress.names();
        if(names!=null){
            for(int i=0;i<names.length();i++){
                String key=names.getString(i);
                if("label".equals(key))continue;
                if("event".equals(key)){state.put("stageEvent",progress.get(key));continue;}
                state.put(key,progress.get(key));
            }
        }
        if("start".equals(progress.optString("event")))state.put("stageStartedAtMs",nowMs);
        state.put("lastProgressAtMs",nowMs);
    }
}
