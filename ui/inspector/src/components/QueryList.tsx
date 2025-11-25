import { useMemo } from "react";
import { useQueryHistory } from "../hooks/useQueryHistory";

interface QueryListProps {
  documentId: string;
  activeQueryId: string | null;
  onSelect: (queryId: string) => void;
  limit?: number;
  showHeader?: boolean;
}

export default function QueryList({
  documentId,
  activeQueryId,
  onSelect,
  limit = 50,
  showHeader = true,
}: QueryListProps) {
  const { queries, loading, error } = useQueryHistory(documentId, limit);

  const content = useMemo(() => {
    if (loading) {
      return <p className="muted">Loading queries…</p>;
    }
    if (error) {
      return <p className="error-text">{error}</p>;
    }
    if (queries.length === 0) {
      return <p className="muted">No queries logged yet.</p>;
    }
    return (
      <ul>
        {queries.map((query) => (
          <li
            key={query.id}
            className={`query-list__item${
              activeQueryId === query.id ? " query-list__item--active" : ""
            }`}
            onClick={() => onSelect(query.id)}
          >
            <div className="query-list__meta">
              <span className="query-list__time">
                {new Date(query.created_at).toLocaleString()}
              </span>
              {typeof query.budget_tokens === "number" && (
                <span className="pill">
                  Budget {query.budget_tokens.toLocaleString()}
                </span>
              )}
            </div>
            <div className="query-list__text" title={query.query_text}>
              {query.query_text}
            </div>
          </li>
        ))}
      </ul>
    );
  }, [queries, loading, error, activeQueryId, onSelect]);

  return (
    <div className="query-list">
      {showHeader && <div style={{ marginBottom: "0.5rem" }}>{loading && <span className="muted">Loading…</span>}</div>}
      {content}
    </div>
  );
}
