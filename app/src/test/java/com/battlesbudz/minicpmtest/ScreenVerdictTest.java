package com.battlesbudz.minicpmtest;
import org.junit.Test;
import static org.junit.Assert.*;
public class ScreenVerdictTest {
    @Test public void tinyAudioCannotPass(){assertEquals("INCOMPLETE",ScreenVerdict.classify(false,30,30,1,100,0,.1,true,5,0));}
    @Test public void invalidEnergyCannotPass(){assertEquals("INVALID_OUTPUT",ScreenVerdict.classify(false,30,30,4,24000,0,Double.NaN,true,5,0));}
    @Test public void silenceCannotPass(){assertEquals("INCOMPLETE",ScreenVerdict.classify(false,30,30,0,0,0,0,true,5,0));}
    @Test public void partialRunCannotPass(){assertEquals("INCOMPLETE",ScreenVerdict.classify(false,30,29,4,24000,0,.1,true,5,0));}
    @Test public void backlogFailsEvenWithSpeech(){assertEquals("FAIL_CPU_SCREEN",ScreenVerdict.classify(false,30,30,4,24000,0,.1,true,1400,0));}
    @Test public void generatedSupplyGapsFail(){assertEquals("FAIL_CPU_SCREEN",ScreenVerdict.classify(false,30,30,4,24000,0,.1,true,500,200));}
    @Test public void nonfiniteSamplesInvalidate(){assertEquals("INVALID_OUTPUT",ScreenVerdict.classify(false,30,30,4,24000,1,.1,true,500,0));}
    @Test public void validComputeScreenPasses(){assertEquals("PASS_CPU_SCREEN",ScreenVerdict.classify(false,30,30,4,24000,0,.1,true,500,0));}
    @Test public void percentileDoesNotMutate(){double[] a={4,1,3,2};assertEquals(4,ScreenVerdict.p95(a),0);assertEquals(4,a[0],0);}
}
