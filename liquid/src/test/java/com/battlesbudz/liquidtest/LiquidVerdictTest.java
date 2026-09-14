package com.battlesbudz.liquidtest;
import org.junit.Test;
import static org.junit.Assert.*;
public class LiquidVerdictTest {
 @Test public void doesNotCallOutputARealTimePass(){assertEquals("OUTPUT_GENERATED_REVIEW_REQUIRED",LiquidVerdict.classify(false,48000,0.1,"Hello"));}
 @Test public void rejectsSilentOrMissingSpeech(){assertEquals("INCOMPLETE_OUTPUT",LiquidVerdict.classify(false,48000,0,"Hello"));assertEquals("INCOMPLETE_OUTPUT",LiquidVerdict.classify(false,0,0.1,"Hello"));}
 @Test public void rejectsMissingTextOrInvalidRms(){assertEquals("INCOMPLETE_OUTPUT",LiquidVerdict.classify(false,48000,Double.NaN,"Hello"));assertEquals("INCOMPLETE_OUTPUT",LiquidVerdict.classify(false,48000,0.1,""));}
 @Test public void errorsCannotPass(){assertEquals("ERROR",LiquidVerdict.classify(true,48000,0.1,"Hello"));}
}
