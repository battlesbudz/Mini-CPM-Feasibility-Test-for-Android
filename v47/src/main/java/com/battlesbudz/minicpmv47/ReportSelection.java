package com.battlesbudz.minicpmv47;

import org.json.JSONObject;

/** Pure run-identity decision shared by export and the on-screen readers. */
final class ReportSelection {
    final JSONObject report;
    final boolean partial;
    final boolean stale;
    final String arm;

    private ReportSelection(JSONObject report, boolean partial, boolean stale, String arm) {
        this.report = report;
        this.partial = partial;
        this.stale = stale;
        this.arm = arm;
    }

    static ReportSelection select(JSONObject status, JSONObject report, JSONObject partialReport) {
        String arm = status == null ? "unknown" : status.optString("abArm", status.optString("mode", "unknown"));
        if (belongs(status, report)) return new ReportSelection(report, false, false, arm);
        if (belongs(status, partialReport)) return new ReportSelection(partialReport, true, false, arm);
        return new ReportSelection(null, false, true, arm);
    }

    private static boolean belongs(JSONObject status, JSONObject candidate) {
        if (status == null || candidate == null) return false;
        long started = status.optLong("startedAtMs", 0);
        if (started <= 0 || candidate.optLong("startedAtMs", 0) != started) return false;
        return consistent(status, candidate, "build") &&
                consistent(status, candidate, "sourceCommit") &&
                consistent(status, candidate, "abArm");
    }

    private static boolean consistent(JSONObject status, JSONObject candidate, String key) {
        if (!status.has(key) || !candidate.has(key)) return true;
        String left = status.optString(key, "");
        String right = candidate.optString(key, "");
        return left.isEmpty() || right.isEmpty() || left.equals(right);
    }
}
