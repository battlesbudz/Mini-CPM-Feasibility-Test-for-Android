package com.battlesbudz.minicpmtest;
final class NativeBench {
    static {System.loadLibrary("minicpm_bench");}
    interface Listener {void onEvent(String message);}
    static native String run(String models,String chunks,String output,int frames,Listener listener);
}
