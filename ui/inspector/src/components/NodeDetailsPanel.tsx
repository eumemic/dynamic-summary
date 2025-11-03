import { NodeResponse } from "../types";

interface NodeDetailsPanelProps {
  node: NodeResponse | null;
  hoveredNode: NodeResponse | null;
}

export default function NodeDetailsPanel({
  node,
  hoveredNode,
}: NodeDetailsPanelProps) {
  const active = node ?? hoveredNode;

  if (!active) {
    return (
      <aside className="node-details">
        <p>Select a node to inspect its contents.</p>
      </aside>
    );
  }

  const createdLabel = active.created_at
    ? new Date(active.created_at).toLocaleString()
    : "—";

  return (
    <aside className="node-details" aria-live="polite">
      <header>
        <h2>Node {active.node_id}</h2>
        {node && hoveredNode && node.node_id !== hoveredNode.node_id && (
          <small>Previewing node {hoveredNode.node_id}</small>
        )}
      </header>
      <dl>
        <div>
          <dt>Span</dt>
          <dd>
            [{active.span_start}, {active.span_end})
          </dd>
        </div>
        <div>
          <dt>Height</dt>
          <dd>{active.height}</dd>
        </div>
        <div>
          <dt>Tokens</dt>
          <dd>{active.token_count}</dd>
        </div>
        <div>
          <dt>Level</dt>
          <dd>{active.level_index}</dd>
        </div>
        <div>
          <dt>Pinned</dt>
          <dd>{active.is_pinned ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{createdLabel}</dd>
        </div>
        <div>
          <dt>Parent</dt>
          <dd>{active.parent_id ?? "—"}</dd>
        </div>
        <div>
          <dt>Left Child</dt>
          <dd>{active.left_child_id ?? "—"}</dd>
        </div>
        <div>
          <dt>Right Child</dt>
          <dd>{active.right_child_id ?? "—"}</dd>
        </div>
      </dl>
      <section className="node-details__text">
        <h3>Text</h3>
        <p>{active.text || "(empty)"}</p>
      </section>
    </aside>
  );
}
