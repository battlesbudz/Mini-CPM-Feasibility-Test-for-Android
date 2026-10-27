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
    @Test public void chunkProgressAdvancesWithoutRestampingStageStart() throws Exception {
        JSONObject state=new JSONObject();
        Progress.merge(state,Progress.parse("PROGRESS {\"stage\":\"vision_encode\",\"event\":\"start\",\"caseIndex\":4}"),1000L);
        Progress.merge(state,Progress.parse("PROGRESS {\"stage\":\"chunk_progress\",\"event\":\"progress\",\"caseIndex\":4,\"chunkIndex\":3,\"chunkTotal\":13,\"chunkIsImage\":true,\"imageChunksDone\":2}"),2500L);
        assertEquals("chunk_progress",state.getString("stage"));
        assertEquals(3,state.getInt("chunkIndex"));
        assertEquals(13,state.getInt("chunkTotal"));
        assertTrue(state.getBoolean("chunkIsImage"));
        assertEquals(2,state.getInt("imageChunksDone"));
        // A chunk checkpoint is a sign of life, not a new stage start:
        // the vision_encode start stamp survives, so a freeze on one
        // chunk index is visible against it.
        assertEquals(1000L,state.getLong("stageStartedAtMs"));
        assertEquals(2500L,state.getLong("lastProgressAtMs"));
    }
    @Test public void chunkProgressLabelFallsBackToStageName() {
        JSONObject progress=Progress.parse("PROGRESS {\"stage\":\"chunk_progress\",\"event\":\"progress\",\"caseIndex\":4,\"chunkIndex\":3,\"chunkTotal\":13}");
        assertEquals("Case 4: chunk progress",Progress.label(progress));
    }
    @Test public void accumulateSumsCaseMeasurements() throws Exception {
        JSONObject totals=new JSONObject();
        Progress.accumulate(totals,new JSONObject("{\"tokenizeMs\":10.0,\"prefillMs\":100.0,\"promptTokens\":43,\"imageTokens\":0,\"generatedTokens\":217,\"decodeWallMs\":790900.0,\"visionEvalMs\":0.0,\"answer\":\"Paris\"}"));
        Progress.accumulate(totals,new JSONObject("{\"tokenizeMs\":5.0,\"prefillMs\":200.0,\"promptTokens\":617,\"imageTokens\":574,\"generatedTokens\":100,\"decodeWallMs\":300000.0,\"visionEvalMs\":1233500.0}"));
        assertEquals(15.0,totals.getDouble("tokenizeMs"),0.001);
        assertEquals(300.0,totals.getDouble("prefillMs"),0.001);
        assertEquals(660,totals.getInt("promptTokens"));
        assertEquals(574,totals.getInt("imageTokens"));
        assertEquals(317,totals.getInt("generatedTokens"));
        assertEquals(1090900.0,totals.getDouble("decodeWallMs"),0.001);
        assertEquals(1233500.0,totals.getDouble("visionEvalMs"),0.001);
        assertEquals(2,totals.getInt("casesCompleted"));
        // Non-measurement fields (the verbatim answer) are never summed in.
        assertFalse(totals.has("answer"));
    }
    @Test public void accumulateIgnoresMissingKeys() throws Exception {
        JSONObject totals=new JSONObject();
        Progress.accumulate(totals,new JSONObject("{\"generatedTokens\":3}"));
        assertEquals(3,totals.getInt("generatedTokens"));
        assertEquals(1,totals.getInt("casesCompleted"));
        assertFalse(totals.has("prefillMs"));
    }
    @Test public void textPrefillMeasurementsMerge() throws Exception {
        JSONObject state=new JSONObject();
        Progress.merge(state,Progress.parse("PROGRESS {\"stage\":\"text_prefill\",\"event\":\"done\",\"caseIndex\":1,\"prefillMs\":42000.0,\"prefillTokensPerSec\":3.1,\"promptTokens\":130}"),9000L);
        assertEquals("text_prefill",state.getString("stage"));
        assertEquals(42000.0,state.getDouble("prefillMs"),0.001);
        assertEquals(3.1,state.getDouble("prefillTokensPerSec"),0.001);
        assertEquals(130,state.getInt("promptTokens"));
        assertEquals(9000L,state.getLong("lastProgressAtMs"));
    }
}
