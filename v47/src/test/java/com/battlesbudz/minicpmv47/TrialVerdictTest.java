package com.battlesbudz.minicpmv47;
import org.junit.Test;
import static org.junit.Assert.*;
public class TrialVerdictTest {
    @Test public void errorDominates() {
        assertEquals("FAILED: load failed", TrialVerdict.classify(true, "load failed", 3, 3));
    }
    @Test public void noAnswersIsFailure() {
        assertTrue(TrialVerdict.classify(false, "", 0, 3).startsWith("FAILED"));
    }
    @Test public void partialIsNamed() {
        assertEquals("PARTIAL: 2 of 3 cases answered", TrialVerdict.classify(false, "", 2, 3));
    }
    @Test public void allAnsweredCompletes() {
        assertEquals("COMPLETED: all 3 cases answered", TrialVerdict.classify(false, "", 3, 3));
    }
}
