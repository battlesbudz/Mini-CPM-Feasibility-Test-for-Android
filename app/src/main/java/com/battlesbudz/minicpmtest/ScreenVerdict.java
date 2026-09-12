package com.battlesbudz.minicpmtest;
import java.util.Arrays;
final class ScreenVerdict {
    static double p95(double[] values) {
        if(values.length==0)return Double.NaN;
        double[] copy=values.clone();Arrays.sort(copy);
        return copy[(int)Math.ceil(copy.length*.95)-1];
    }
    static String classify(boolean error,int expected,int completed,int speak,long samples,
                           long invalid,double rms,boolean drained,double p95,double supplyGap) {
        if(error)return "ERROR";
        if(!Double.isFinite(rms))return "INVALID_OUTPUT";
        if(expected<=0||completed!=expected||speak==0||samples<24000||!drained||rms<.00001)return "INCOMPLETE";
        if(!Double.isFinite(p95)||!Double.isFinite(supplyGap)||invalid>0)return "INVALID_OUTPUT";
        return p95<1000&&supplyGap<100?"PASS_CPU_SCREEN":"FAIL_CPU_SCREEN";
    }
}
