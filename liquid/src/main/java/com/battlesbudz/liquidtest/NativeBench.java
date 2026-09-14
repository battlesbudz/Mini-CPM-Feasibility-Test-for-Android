package com.battlesbudz.liquidtest;
final class NativeBench {
    static {System.loadLibrary("liquid_bench");}
    interface Listener {void onEvent(String message);}
    static native String run(String models,String chunks,String output,int profile,int voice,Listener listener);
}
