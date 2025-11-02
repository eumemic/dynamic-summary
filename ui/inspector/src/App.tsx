import { useEffect, useState } from "react";
import DocumentTreeView from "./components/DocumentTreeView";
import { fetchDocuments } from "./api/client";
import { DocumentInfo } from "./types";

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    fetchDocuments()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setDocuments(payload.documents);
        if (payload.documents.length > 0) {
          setSelectedId((previous) =>
            previous ?? payload.documents[0].document_id
          );
        }
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
  }, []);

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
          <DocumentTreeView documentId={selectedId} />
        ) : (
          <p>Select a document to inspect the tree.</p>
        )}
      </main>
    </div>
  );
}
