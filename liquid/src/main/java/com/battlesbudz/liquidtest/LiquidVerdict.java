package com.battlesbudz.liquidtest;
final class LiquidVerdict {
    static String classify(boolean error,long samples,double rms,String text){
        if(error)return "ERROR";
        if(samples<24000||!Double.isFinite(rms)||rms<0.00001||text.trim().isEmpty())return "INCOMPLETE_OUTPUT";
        return "OUTPUT_GENERATED_REVIEW_REQUIRED";
    }
}
