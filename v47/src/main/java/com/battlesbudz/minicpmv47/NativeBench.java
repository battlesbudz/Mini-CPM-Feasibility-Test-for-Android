package com.battlesbudz.minicpmv47;
final class NativeBench {
    static {System.loadLibrary("minicpm_v47_bench");}
    interface Listener {void onEvent(String message);}
    // mode is "quick" (Case 3 only, 32-token cap) or "full" (all six
    // cases, trial defaults); added in build 12 for the quick mode.
    static native String run(String models,String image,String output,String mode,Listener listener);
}
