package com.battlesbudz.liquidtest;
final class ExitMatch {
    private ExitMatch() {}
    static boolean matches(int runPid,long startedAtMs,int exitPid,long exitAtMs){
        return runPid>0 && startedAtMs>0 && runPid==exitPid && exitAtMs>=startedAtMs;
    }
}
