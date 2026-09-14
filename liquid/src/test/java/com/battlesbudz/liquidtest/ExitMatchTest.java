package com.battlesbudz.liquidtest;
import org.junit.Test;
import static org.junit.Assert.*;
public class ExitMatchTest {
    @Test public void acceptsCurrentWorkerExit(){assertTrue(ExitMatch.matches(11759,1000,11759,2000));}
    @Test public void ignoresPreviousWorker(){assertFalse(ExitMatch.matches(11759,1000,11700,2000));}
    @Test public void ignoresReusedPidFromOldRun(){assertFalse(ExitMatch.matches(11759,1000,11759,900));}
    @Test public void rejectsMissingIdentity(){assertFalse(ExitMatch.matches(-1,1000,-1,2000));assertFalse(ExitMatch.matches(11759,0,11759,2000));}
}
