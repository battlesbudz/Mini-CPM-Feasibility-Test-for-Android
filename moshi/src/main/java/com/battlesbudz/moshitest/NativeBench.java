package com.battlesbudz.moshitest;
final class NativeBench {
    static { System.loadLibrary("moshi_bench"); }
    interface Listener { void onEvent(String message); }
    static native String run(String models, String input, String output,
        int mode, int backend, int contextFrames, Listener listener);
}
