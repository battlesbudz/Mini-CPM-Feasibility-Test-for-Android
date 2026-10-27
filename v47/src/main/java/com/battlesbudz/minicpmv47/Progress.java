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
    // Build 12: case measurements summed into the run totals as each
    // case completes. The status fields merged above stay latest-case
    // snapshots — they are the live in-progress view (diagnosis §16.1
    // proved they get overwritten case to case and must not be read as
    // totals); the totals object the service keeps in status.json and
    // the partial report is the cumulative sum. Rates and one-shot
    // timings (decodeTokensPerSec, ttftMs) are deliberately not summed.
    private static final String[] TOTAL_KEYS={
        "tokenizeMs","visionEvalMs","prefillMs","promptTokens",
        "imageTokens","generatedTokens","decodeWallMs"
    };
    static void accumulate(JSONObject totals,JSONObject kase){
        try{
            for(String key:TOTAL_KEYS){
                if(kase.has(key))totals.put(key,totals.optDouble(key,0)+kase.optDouble(key,0));
            }
            totals.put("casesCompleted",totals.optInt("casesCompleted",0)+1);
        }catch(Exception ignored){}
    }
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
