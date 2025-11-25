import { useEffect, useMemo, useState } from "react";
import { fetchDocumentQueries } from "../api/client";
import { QueryListItem } from "../types";

const API_BASE =
  (import.meta.env.VITE_RAGZOOM_API_URL as string | undefined)?.replace(/\/$/, "") ??
  "";

interface UseQueryHistoryResult {
  queries: QueryListItem[];
  loading: boolean;
  error: string | null;
}

export function useQueryHistory(
  documentId: string | null,
  limit = 50
): UseQueryHistoryResult {
  const [queries, setQueries] = useState<QueryListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!documentId) {
      setQueries([]);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    fetchDocumentQueries(documentId, limit)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setQueries(payload.queries);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [documentId, limit]);

  useEffect(() => {
    if (!documentId) {
      return;
    }

    let cancelled = false;
    let source: EventSource | null = null;
    let reconnectTimer: number | null = null;

    const cleanup = () => {
      if (source) {
        source.close();
        source = null;
      }
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const connect = () => {
      cleanup();
      const endpoint = `${
        API_BASE === "" ? "" : API_BASE
      }/documents/${encodeURIComponent(documentId)}/queries/events?limit=${limit}`;
      source = new EventSource(endpoint);
      source.onmessage = (event) => {
        if (cancelled || !event.data) {
          return;
        }
        try {
          const payload = JSON.parse(event.data) as { queries: QueryListItem[] };
          setQueries(payload.queries);
          setError(null);
          setLoading(false);
        } catch {
          // Ignore parse errors
        }
      };
      source.onerror = () => {
        cleanup();
        if (cancelled) {
          return;
        }
        reconnectTimer = window.setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      cancelled = true;
      cleanup();
    };
  }, [documentId, limit]);

  return useMemo(() => ({ queries, loading, error }), [queries, loading, error]);
}
