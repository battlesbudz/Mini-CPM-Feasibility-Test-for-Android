package com.battlesbudz.minicpmv47;
// Plain-language verdict for the MiniCPM-V 4.7 phone trial, computed from
// primitive report fields so it stays unit-testable on the JVM.
final class TrialVerdict {
    static String classify(boolean hasError,String error,int casesAnswered,int casesTotal){
        if(hasError)return "FAILED: "+(error==null||error.isEmpty()?"native error":error);
        if(casesTotal<=0||casesAnswered<=0)return "FAILED: model loaded but produced no answer tokens";
        if(casesAnswered<casesTotal)return "PARTIAL: "+casesAnswered+" of "+casesTotal+" cases answered";
        return "COMPLETED: all "+casesTotal+" cases answered";
    }
}
