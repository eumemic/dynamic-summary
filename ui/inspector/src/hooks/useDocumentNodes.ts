import { useEffect, useMemo, useRef, useState } from "react";
import { fetchNodesInSpan } from "../api/client";
import { NodeResponse, TelemetryEvent } from "../types";

const API_BASE =
  (import.meta.env.VITE_RAGZOOM_API_URL as string | undefined)?.replace(/\/$/, "") ??
  "";

export interface UseDocumentNodesOptions {
  documentId: string | null;
  spanStart: number;
  spanEnd: number;
  limit: number;
  minHeight?: number | null;
}

export interface UseDocumentNodesResult {
  nodes: NodeResponse[];
  totalMatching: number;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

const STREAM_ENDPOINT = (documentId: string) =>
  `${API_BASE}/documents/${encodeURIComponent(documentId)}/events`;

function spansOverlap(
  aStart: number,
  aEnd: number,
  bStart: number,
  bEnd: number
): boolean {
  return bStart < aEnd && bEnd > aStart;
}

export function useDocumentNodes(
  options: UseDocumentNodesOptions
): UseDocumentNodesResult {
  const { documentId, spanStart, spanEnd, limit, minHeight } = options;
  const [nodes, setNodes] = useState<NodeResponse[]>([]);
  const [totalMatching, setTotalMatching] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [sseEpoch, setSseEpoch] = useState(0);
  const nodeIdSetRef = useRef<Set<string>>(new Set());
  const previousRequestKeyRef = useRef<string | null>(null);
  const lastSseRefreshRef = useRef<number>(0);
  const lastQuerySignatureRef = useRef<{ key: string; timestamp: number } | null>(
    null
  );

  // Fetch span data whenever dependencies change or refresh is triggered.
  useEffect(() => {
    if (!documentId) {
      setNodes([]);
      setTotalMatching(0);
      setLoading(false);
      setError(null);
      return;
    }

    if (spanEnd <= spanStart) {
      return;
    }

    const requestKey = `${documentId}|${spanStart}|${spanEnd}|${limit}|${minHeight ?? ""}|${refreshToken}`;
    const dedupeKey = `${documentId}|${spanStart}|${spanEnd}|${limit}|${minHeight ?? ""}`;
    const now = Date.now();
    if (
      lastQuerySignatureRef.current &&
      lastQuerySignatureRef.current.key === dedupeKey &&
      now - lastQuerySignatureRef.current.timestamp < 120
    ) {
      return;
    }
    if (previousRequestKeyRef.current === requestKey) {
      return;
    }
    lastQuerySignatureRef.current = { key: dedupeKey, timestamp: now };
    previousRequestKeyRef.current = requestKey;

    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetchNodesInSpan(documentId, spanStart, spanEnd, limit, minHeight)
      .then((page) => {
        if (controller.signal.aborted) {
          return;
        }
        setNodes(page.nodes);
        setTotalMatching(page.total_matching);
        nodeIdSetRef.current = new Set(page.nodes.map((node) => node.node_id));
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [documentId, spanStart, spanEnd, limit, minHeight, refreshToken]);

  // Subscribe to SSE stream for incremental updates.
  useEffect(() => {
    if (!documentId) {
      return;
    }

    let cancelled = false;
    const source = new EventSource(STREAM_ENDPOINT(documentId));
    let reconnectTimeout: number | undefined;

    const scheduleReconnect = () => {
      if (cancelled) {
        return;
      }
      window.clearTimeout(reconnectTimeout);
      reconnectTimeout = window.setTimeout(() => {
        if (!cancelled) {
          setSseEpoch((value) => value + 1);
        }
      }, 2000);
    };

    source.onmessage = (event) => {
      if (cancelled) {
        return;
      }
      try {
        const payload = JSON.parse(event.data) as TelemetryEvent;
        if (payload.event === "node_committed") {
          const { span_start, span_end } = payload;
          if (spansOverlap(spanStart, spanEnd, span_start, span_end)) {
            const now = Date.now();
            if (now - lastSseRefreshRef.current > 150) {
              lastSseRefreshRef.current = now;
              setRefreshToken((token) => token + 1);
            }
          }
        } else if (payload.event === "nodes_deleted") {
          const incoming = payload.node_ids ?? [];
          const intersects = incoming.some((id) =>
            nodeIdSetRef.current.has(id)
          );
          if (intersects) {
            const now = Date.now();
            if (now - lastSseRefreshRef.current > 150) {
              lastSseRefreshRef.current = now;
              setRefreshToken((token) => token + 1);
            }
          }
        } else if (
          payload.event === "append_completed" ||
          payload.event === "append_failed"
        ) {
          const now = Date.now();
          if (now - lastSseRefreshRef.current > 150) {
            lastSseRefreshRef.current = now;
            setRefreshToken((token) => token + 1);
          }
        }
      } catch (err) {
        console.warn("Failed to parse event stream payload", err);
      }
    };

    source.onerror = () => {
      source.close();
      scheduleReconnect();
    };

    return () => {
      cancelled = true;
      window.clearTimeout(reconnectTimeout);
      source.close();
    };
  }, [documentId, spanStart, spanEnd, sseEpoch]);

  const refresh = () => setRefreshToken((token) => token + 1);

  return useMemo(
    () => ({
      nodes,
      totalMatching,
      loading,
      error,
      refresh,
    }),
    [nodes, totalMatching, loading, error]
  );
}
