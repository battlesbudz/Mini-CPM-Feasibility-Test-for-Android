package com.battlesbudz.minicpmv47;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

public final class ReportSelectionTest {
    private JSONObject fixture(String name) throws Exception {
        try (java.io.InputStream in = getClass().getResourceAsStream("/com/battlesbudz/minicpmv47/" + name)) {
            assertNotNull("missing fixture " + name, in);
            byte[] bytes = new byte[in.available()];
            int read = in.read(bytes);
            return new JSONObject(new String(bytes, 0, read, java.nio.charset.StandardCharsets.UTF_8));
        }
    }

    @Test public void staleBuild28ReportFallsBackToCurrentPartialAndStatusArm() throws Exception {
        ReportSelection selected = ReportSelection.select(fixture("b29-status.json"),
                fixture("b29-stale-report.json"), fixture("b29-current-partial.json"));
        assertNotNull(selected.report);
        assertTrue(selected.partial);
        assertFalse(selected.stale);
        assertTrue(selected.report.optBoolean("partial"));
        assertEquals("dspark_io8", selected.arm);
        assertFalse(selected.report.has("verdict"));
    }

    @Test public void foreignReportAndPartialProduceStaleBannerCase() throws Exception {
        ReportSelection selected = ReportSelection.select(fixture("b29-status.json"),
                fixture("b29-stale-report.json"), fixture("b29-foreign-partial.json"));
        assertNull(selected.report);
        assertFalse(selected.partial);
        assertTrue(selected.stale);
        assertEquals("dspark_io8", selected.arm);
    }

    @Test public void matchingFinalReportIsSelectedWithoutBanner() throws Exception {
        ReportSelection selected = ReportSelection.select(fixture("b29-status.json"),
                fixture("b29-current-report.json"), fixture("b29-current-partial.json"));
        assertNotNull(selected.report);
        assertFalse(selected.partial);
        assertFalse(selected.stale);
        assertEquals("PASS", selected.report.optString("verdict"));
    }

    @Test public void crashedRunningRunSelectsZeroCasePartial() throws Exception {
        ReportSelection selected = ReportSelection.select(fixture("b29-crashed-status.json"), null,
                fixture("b29-crashed-partial.json"));
        assertNotNull(selected.report);
        assertTrue(selected.partial);
        assertFalse(selected.stale);
        assertEquals(0, selected.report.optJSONArray("cases").length());
        assertEquals("dspark_hobbit", selected.arm);
    }
}
