import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DocumentTreeView from "./components/DocumentTreeView";
import { fetchDocuments } from "./api/client";
import { DocumentInfo, DocumentsStreamEvent } from "./types";

const MIN_NODE_LIMIT = 1;
const MAX_NODE_LIMIT = 2000;
const API_BASE =
  (import.meta.env.VITE_RAGZOOM_API_URL as string | undefined)?.replace(/\/$/, "") ??
  "";

interface QueryState {
  documentId: string | null;
  spanStart: number | null;
  spanEnd: number | null;
  limit: number | null;
  selectedNodeId: string | null;
}

function parseNonNegativeInt(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || Number.isNaN(parsed) || parsed < 0) {
    return null;
  }
  return parsed;
}

function readInitialQueryState(): QueryState {
  const params = new URLSearchParams(window.location.search);
  const documentId = params.get("document_id");
  const spanStart = parseNonNegativeInt(params.get("span_start"));
  const spanEndValue = parseNonNegativeInt(params.get("span_end"));
  const limitValue = parseNonNegativeInt(params.get("limit"));
  const selectedNodeId = params.get("node_id");

  const spanEnd =
    spanStart !== null &&
    spanEndValue !== null &&
    spanEndValue > spanStart
      ? spanEndValue
      : null;

  const limit =
    limitValue === null
      ? null
      : Math.min(Math.max(limitValue, MIN_NODE_LIMIT), MAX_NODE_LIMIT);

  return {
    documentId: documentId && documentId.length > 0 ? documentId : null,
    spanStart,
    spanEnd,
    limit,
    selectedNodeId:
      selectedNodeId && selectedNodeId.length > 0 ? selectedNodeId : null,
  };
}

interface ViewStatePayload {
  documentId: string;
  spanStart: number;
  spanEnd: number;
  limit: number;
  selectedNodeId: string | null;
}

export default function App() {
  const initialQueryState = useMemo(() => readInitialQueryState(), []);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialQueryState.documentId
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const initialSelectionAppliedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let source: EventSource | null = null;
    let reconnectTimer: number | null = null;

    const resolveSelection = (available: DocumentInfo[]) => {
      setSelectedId((current) => {
        if (available.length === 0) {
          initialSelectionAppliedRef.current = false;
          return null;
        }
        if (
          current &&
          available.some((doc) => doc.document_id === current)
        ) {
          initialSelectionAppliedRef.current = true;
          return current;
        }
        if (!initialSelectionAppliedRef.current) {
          if (
            initialQueryState.documentId &&
            available.some(
              (doc) => doc.document_id === initialQueryState.documentId
            )
          ) {
            initialSelectionAppliedRef.current = true;
            return initialQueryState.documentId;
          }
        }
        initialSelectionAppliedRef.current = true;
        return available[0].document_id;
      });
    };

    const loadDocuments = async () => {
      setLoading(true);
      try {
        const payload = await fetchDocuments();
        if (cancelled) {
          return;
        }
        setDocuments(payload.documents);
        setError(null);
        resolveSelection(payload.documents);
      } catch (err) {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    const cleanupSource = () => {
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
      if (cancelled) {
        return;
      }
      cleanupSource();
      const endpoint =
        API_BASE === "" ? "/documents/events" : `${API_BASE}/documents/events`;
      const nextSource = new EventSource(endpoint);
      source = nextSource;

      nextSource.onmessage = (event) => {
        if (cancelled || !event.data) {
          return;
        }
        try {
          const payload = JSON.parse(event.data) as DocumentsStreamEvent;
          if (payload.event !== "documents_changed") {
            return;
          }
          setDocuments(payload.documents);
          setError(null);
          resolveSelection(payload.documents);
          setLoading(false);
        } catch (err) {
          console.warn("Failed to parse documents stream payload", err);
        }
      };

      nextSource.onerror = () => {
        if (cancelled) {
          return;
        }
        if (source) {
          source.close();
          source = null;
        }
        if (reconnectTimer !== null) {
          return;
        }
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = null;
          connect();
        }, 2000);
      };
    };

    void loadDocuments();
    connect();

    return () => {
      cancelled = true;
      cleanupSource();
    };
  }, [initialQueryState]);

  const handleViewStateChange = useCallback(
    ({
      documentId,
      spanStart,
      spanEnd,
      limit,
      selectedNodeId,
    }: ViewStatePayload) => {
      const params = new URLSearchParams(window.location.search);
      params.set("document_id", documentId);
      params.set("span_start", Math.floor(spanStart).toString());
      params.set("span_end", Math.floor(spanEnd).toString());
      params.set("limit", Math.floor(limit).toString());
      if (selectedNodeId) {
        params.set("node_id", selectedNodeId);
      } else {
        params.delete("node_id");
      }
      const search = params.toString();
      const nextUrl = `${window.location.pathname}${
        search ? `?${search}` : ""
      }${window.location.hash}`;
      window.history.replaceState(null, "", nextUrl);
    },
    []
  );

  const initialSpanStart =
    selectedId && selectedId === initialQueryState.documentId
      ? initialQueryState.spanStart
      : null;
  const initialSpanEnd =
    selectedId && selectedId === initialQueryState.documentId
      ? initialQueryState.spanEnd
      : null;
  const initialNodeId =
    selectedId && selectedId === initialQueryState.documentId
      ? initialQueryState.selectedNodeId
      : null;
  const initialLimit = initialQueryState.limit;

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Documents</h1>
        {loading && <p>Loading…</p>}
        {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}
        <ul className="document-list">
          {documents.map((doc) => (
            <li
              key={doc.document_id}
              className={`document-item${
                selectedId === doc.document_id ? " document-item--active" : ""
              }`}
              onClick={() => setSelectedId(doc.document_id)}
            >
              <strong>{doc.document_id}</strong>
              <div style={{ fontSize: "0.8rem", opacity: 0.75 }}>
                Nodes: {doc.node_count}
              </div>
            </li>
          ))}
        </ul>
      </aside>
      <main className="content">
        {selectedId ? (
          <DocumentTreeView
            documentId={selectedId}
            initialSpanStart={initialSpanStart}
            initialSpanEnd={initialSpanEnd}
            initialLimit={initialLimit}
            initialSelectedNodeId={initialNodeId}
            onStateChange={handleViewStateChange}
          />
        ) : (
          <p>Select a document to inspect the tree.</p>
        )}
      </main>
    </div>
  );
}
