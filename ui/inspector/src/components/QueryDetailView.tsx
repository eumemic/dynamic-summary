import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchNodesBatch, fetchQueryDetail } from "../api/client";
import { NodeResponse, QueryDetailResponse } from "../types";
import TreeCanvas from "./TreeCanvas";
import NodeDetailsPanel from "./NodeDetailsPanel";

interface QueryDetailViewProps {
  queryId: string | null;
  onBack?: () => void;
  onDocumentResolved?: (documentId: string) => void;
  selectedNodeId?: string | null;
  onSelectNode?: (nodeId: string | null) => void;
}

type NodeMap = Map<string, NodeResponse>;

export default function QueryDetailView({
  queryId,
  onBack,
  onDocumentResolved,
  selectedNodeId,
  onSelectNode,
}: QueryDetailViewProps) {
  const [detail, setDetail] = useState<QueryDetailResponse | null>(null);
  const [nodes, setNodes] = useState<NodeMap>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setNodes(new Map());
    setHoveredNodeId(null);
    if (!queryId) {
      return;
    }

    setLoading(true);
    setError(null);
    fetchQueryDetail(queryId)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setDetail(payload);
        setError(null);
        onDocumentResolved?.(payload.query.document_id);
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
  }, [queryId, onDocumentResolved]);

  useEffect(() => {
    let cancelled = false;
    if (!detail) {
      return;
    }
    const ordered = [...detail.nodes].sort(
      (a, b) => a.position - b.position
    );
    const ids = ordered.map((node) => node.node_id);
    if (ids.length === 0) {
      return;
    }
    fetchNodesBatch(detail.query.document_id, ids)
      .then((result) => {
        if (cancelled) {
          return;
        }
        const map = new Map<string, NodeResponse>();
        result.forEach((node) => map.set(node.node_id, node));
        setNodes(map);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      });

    return () => {
      cancelled = true;
    };
  }, [detail]);

  const seedSet = useMemo(() => {
    if (!detail) {
      return new Set<string>();
    }
    return new Set(
      detail.nodes.filter((node) => node.is_seed).map((node) => node.node_id)
    );
  }, [detail]);

  const orderedNodes = useMemo(() => {
    if (!detail) {
      return [];
    }
    return [...detail.nodes].sort((a, b) => a.position - b.position);
  }, [detail]);

  const tilingNodes = useMemo(() => {
    if (!detail) {
      return [];
    }
    return orderedNodes
      .map((entry) => nodes.get(entry.node_id))
      .filter((node): node is NodeResponse => Boolean(node));
  }, [orderedNodes, nodes, detail]);

  const scoreMap = useMemo(() => {
    const map = new Map<string, number>();
    if (detail) {
      for (const entry of detail.nodes) {
        map.set(entry.node_id, entry.score);
      }
    }
    return map;
  }, [detail]);

  const documentSpanEnd = useMemo(() => {
    if (tilingNodes.length === 0) {
      return 1;
    }
    return tilingNodes.reduce(
      (acc, node) => (node.span_end > acc ? node.span_end : acc),
      0
    );
  }, [tilingNodes]);

  const stitchedBlocks = useMemo(() => {
    return orderedNodes.map((entry) => ({
      entry,
      node: nodes.get(entry.node_id) ?? null,
    }));
  }, [orderedNodes, nodes]);

  const handleSelectNode = useCallback(
    (node: NodeResponse) => {
      onSelectNode?.(node.node_id);
    },
    [onSelectNode]
  );

  const activeSelectedNodeId = selectedNodeId ?? null;
  const selectedNode = activeSelectedNodeId
    ? nodes.get(activeSelectedNodeId) ?? null
    : null;

  useEffect(() => {
    if (nodes.size === 0) {
      return;
    }
    if (!activeSelectedNodeId) {
      return;
    }
    if (!nodes.has(activeSelectedNodeId) && onSelectNode) {
      onSelectNode(null);
    }
  }, [activeSelectedNodeId, nodes, onSelectNode]);

  if (!queryId) {
    return <p className="muted">Select a query to view its tiling.</p>;
  }

  if (loading && !detail) {
    return <p className="muted">Loading query…</p>;
  }

  if (error && !detail) {
    return <p className="error-text">{error}</p>;
  }

  if (!detail) {
    return <p className="muted">No detail available.</p>;
  }

  const createdLabel = new Date(detail.query.created_at).toLocaleString();

  return (
    <div className="query-detail__content">
      <header className="query-detail__meta">
        <div>
          <div className="pill pill--ghost">Query ID: {detail.query.id}</div>
          <h3>{detail.query.query_text}</h3>
          <p className="muted">
            {createdLabel} • Nodes in tiling: {orderedNodes.length} • Seeds:{" "}
            {seedSet.size}
          </p>
        </div>
        <div className="query-detail__metrics">
          {typeof detail.query.budget_tokens === "number" && (
            <div className="pill">
              Budget {detail.query.budget_tokens.toLocaleString()}
            </div>
          )}
          {typeof detail.query.num_seeds === "number" && (
            <div className="pill pill--ghost">
              Seeds requested {detail.query.num_seeds}
            </div>
          )}
          {onBack && (
            <button type="button" className="link-button" onClick={onBack}>
              Back to document
            </button>
          )}
        </div>
      </header>
      {error && <p className="error-text">{error}</p>}

      <div className="query-detail__viz">
        <TreeCanvas
          nodes={tilingNodes}
          spanStart={0}
          spanEnd={documentSpanEnd}
          maxSpanEnd={documentSpanEnd}
          selectedNodeId={activeSelectedNodeId}
          hoveredNodeId={hoveredNodeId}
          onHover={(id) => setHoveredNodeId(id)}
          onSelect={handleSelectNode}
          seedNodeIds={seedSet}
        />
      </div>

      <div className="query-detail__body">
        <div className="query-detail__stitched">
          <h4>Tiling Text</h4>
          <div className="stitched-list">
            {stitchedBlocks.map(({ entry, node }) => {
              const isSelected = node?.node_id === activeSelectedNodeId;
              const isSeed = seedSet.has(entry.node_id);
              const score = scoreMap.get(entry.node_id) ?? 0;
              return (
                <div
                  key={entry.node_id}
                  className={`stitched-card${
                    isSelected ? " stitched-card--selected" : ""
                  }${isSeed ? " stitched-card--seed" : ""}`}
                  onClick={() => {
                    if (node && onSelectNode) {
                      onSelectNode(node.node_id);
                    }
                  }}
                >
                  <div className="stitched-card__header">
                    <div>
                      <strong>Node {entry.node_id}</strong>
                      {isSeed && <span className="pill">Seed</span>}
                    </div>
                    <span className="muted">
                      Score {score.toFixed(3)} • order {entry.position + 1}/
                      {orderedNodes.length}
                    </span>
                  </div>
                  <div className="stitched-card__text">
                    {node ? node.text || "(empty node)" : "(node missing)"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <NodeDetailsPanel node={selectedNode} />
      </div>
    </div>
  );
}
