package com.battlesbudz.minicpmv47;
import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;
public class ProgressTest {
    @Test public void plainEventsAreNotProgress() {
        assertNull(Progress.parse("Case 1/3: Describe this image."));
        assertNull(Progress.parse(null));
    }
    @Test public void malformedProgressIsIgnored() {
        assertNull(Progress.parse("PROGRESS {not json"));
    }
    @Test public void mergeCopiesCheckpointFields() throws Exception {
        JSONObject state=new JSONObject();
        JSONObject progress=Progress.parse("PROGRESS {\"stage\":\"decode\",\"event\":\"progress\",\"caseIndex\":1,\"generatedTokens\":41,\"decodeElapsedMs\":9000.0,\"label\":\"Case 1: generating answer\"}");
        assertNotNull(progress);
        Progress.merge(state,progress,123456L);
        assertEquals("decode",state.getString("stage"));
        assertEquals("progress",state.getString("stageEvent"));
        assertEquals(1,state.getInt("caseIndex"));
        assertEquals(41,state.getInt("generatedTokens"));
        assertEquals(9000.0,state.getDouble("decodeElapsedMs"),0.001);
        assertEquals(123456L,state.getLong("lastProgressAtMs"));
        assertFalse(state.has("stageStartedAtMs"));
        assertFalse(state.has("label"));
    }
    @Test public void stageStartStampsStartTime() throws Exception {
        JSONObject state=new JSONObject();
        Progress.merge(state,Progress.parse("PROGRESS {\"stage\":\"vision_encode\",\"event\":\"start\",\"caseIndex\":2}"),777L);
        assertEquals(777L,state.getLong("stageStartedAtMs"));
        assertEquals(777L,state.getLong("lastProgressAtMs"));
        assertEquals("start",state.getString("stageEvent"));
    }
    @Test public void laterCheckpointsKeepEarlierStageMeasurements() throws Exception {
        JSONObject state=new JSONObject();
        Progress.merge(state,Progress.parse("PROGRESS {\"stage\":\"model_load\",\"event\":\"done\",\"modelLoadMs\":4200.0}"),1000L);
        Progress.merge(state,Progress.parse("PROGRESS {\"stage\":\"decode\",\"event\":\"progress\",\"caseIndex\":1,\"generatedTokens\":7}"),2000L);
        assertEquals(4200.0,state.getDouble("modelLoadMs"),0.001);
        assertEquals(7,state.getInt("generatedTokens"));
        assertEquals(2000L,state.getLong("lastProgressAtMs"));
    }
    @Test public void labelFallsBackToStageName() {
        JSONObject progress=Progress.parse("PROGRESS {\"stage\":\"vision_encode\",\"event\":\"start\",\"caseIndex\":2}");
        assertEquals("Case 2: vision encode",Progress.label(progress));
        JSONObject labeled=Progress.parse("PROGRESS {\"stage\":\"decode\",\"event\":\"start\",\"label\":\"Case 1: generating answer\"}");
        assertEquals("Case 1: generating answer",Progress.label(labeled));
    }
}
