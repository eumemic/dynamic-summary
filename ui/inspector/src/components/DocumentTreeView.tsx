import { useMemo, useState } from "react";
import { useDocumentNodes } from "../hooks/useDocumentNodes";

interface DocumentTreeViewProps {
  documentId: string;
}

const clampLimit = (value: number) => Math.max(1, Math.min(2000, value));
const clampSpan = (value: number) => Math.max(0, value);

export default function DocumentTreeView({
  documentId,
}: DocumentTreeViewProps) {
  const [spanStart, setSpanStart] = useState(0);
  const [spanEnd, setSpanEnd] = useState(2000);
  const [limit, setLimit] = useState(200);
  const [minHeight, setMinHeight] = useState<number | null>(null);

  const span = useMemo(() => {
    if (spanEnd <= spanStart) {
      return { start: spanStart, end: spanStart + 1 };
    }
    return { start: spanStart, end: spanEnd };
  }, [spanStart, spanEnd]);

  const { nodes, totalMatching, loading, error, refresh } = useDocumentNodes({
    documentId,
    spanStart: span.start,
    spanEnd: span.end,
    limit,
    minHeight,
  });

  return (
    <section>
      <header className="controls">
        <fieldset>
          <legend>Span (character offsets)</legend>
          <label>
            Start
            <input
              type="number"
              value={spanStart}
              min={0}
              onChange={(event) =>
                setSpanStart(clampSpan(Number(event.target.value)))
              }
            />
          </label>
          <label>
            End
            <input
              type="number"
              value={spanEnd}
              min={spanStart + 1}
              onChange={(event) =>
                setSpanEnd(
                  clampSpan(Math.max(Number(event.target.value), spanStart + 1))
                )
              }
            />
          </label>
        </fieldset>
        <fieldset>
          <legend>Rendering</legend>
          <label>
            Node budget
            <input
              type="number"
              min={1}
              max={2000}
              value={limit}
              onChange={(event) =>
                setLimit(clampLimit(Number(event.target.value)))
              }
            />
          </label>
          <label>
            Min height
            <input
              type="number"
              min={0}
              placeholder="Any"
              value={minHeight ?? ""}
              onChange={(event) => {
                const raw = event.target.value;
                setMinHeight(raw === "" ? null : Math.max(0, Number(raw)));
              }}
            />
          </label>
          <button type="button" onClick={refresh}>
            Refresh
          </button>
        </fieldset>
      </header>

      <div className="status">
        {loading && <span>Loading nodes… </span>}
        {error && <span style={{ color: "#ff6b6b" }}>{error}</span>}
        {!loading && !error && (
          <span>
            Showing {nodes.length} of {totalMatching} nodes covering span [
            {span.start}, {span.end}).
          </span>
        )}
      </div>

      <div className="node-grid">
        {nodes.map((node) => {
          const createdLabel = node.created_at
            ? new Date(node.created_at).toLocaleString()
            : "—";
          return (
            <article className="node-card" key={node.node_id}>
              <div className="node-card__meta">
                <span>Height: {node.height}</span>
                <span>
                  Span: [{node.span_start}, {node.span_end})
                </span>
                <span>Tokens: {node.token_count}</span>
                <span>Level: {node.level_index}</span>
                <span>Pinned: {node.is_pinned ? "yes" : "no"}</span>
                <span>Created: {createdLabel}</span>
              </div>
              <p className="node-card__text">{node.text}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
