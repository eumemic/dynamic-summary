import { NodeResponse } from "../types";

interface NodeDetailsPanelProps {
  node: NodeResponse | null;
}

export default function NodeDetailsPanel({ node }: NodeDetailsPanelProps) {
  if (!node) {
    return (
      <aside className="node-details">
        <p>Click a node to view its details.</p>
      </aside>
    );
  }

  const createdLabel = node.created_at
    ? new Date(node.created_at).toLocaleString()
    : "—";

  return (
    <aside className="node-details" aria-live="polite">
      <header>
        <h2>Node {node.node_id}</h2>
      </header>
      <dl>
        <div>
          <dt>Span</dt>
          <dd>
            [{node.span_start}, {node.span_end})
          </dd>
        </div>
        <div>
          <dt>Height</dt>
          <dd>{node.height}</dd>
        </div>
        <div>
          <dt>Tokens</dt>
          <dd>{node.token_count}</dd>
        </div>
        <div>
          <dt>Level</dt>
          <dd>{node.level_index}</dd>
        </div>
        <div>
          <dt>Pinned</dt>
          <dd>{node.is_pinned ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{createdLabel}</dd>
        </div>
        <div>
          <dt>Parent</dt>
          <dd>{node.parent_id ?? "—"}</dd>
        </div>
        <div>
          <dt>Left Child</dt>
          <dd>{node.left_child_id ?? "—"}</dd>
        </div>
        <div>
          <dt>Right Child</dt>
          <dd>{node.right_child_id ?? "—"}</dd>
        </div>
      </dl>
      {node.preceding_context_summary && (
        <section className="node-details__context">
          <h3>Preceding Context Summary</h3>
          <p>{node.preceding_context_summary}</p>
        </section>
      )}
      <section className="node-details__text">
        <h3>Text</h3>
        <p>{node.text || "(empty)"}</p>
      </section>
    </aside>
  );
}
