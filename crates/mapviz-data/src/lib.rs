//! Streaming data sources for mapviz.
//!
//! Real-time feeds (WebSocket, server-sent events, user-driven push) are
//! first-class; static datasets (e.g. a GeoJSON file) are just a degenerate
//! streaming case. Sources produce typed features over time and feed a built-in
//! spatial index (R*-tree) for picking and viewport culling, plus a temporal
//! index for time scrubbing.
