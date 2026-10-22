package com.battlesbudz.minicpmv47;
final class NativeBench {
    static {System.loadLibrary("minicpm_v47_bench");}
    interface Listener {void onEvent(String message);}
    static native String run(String models,String image,String output,Listener listener);
}
